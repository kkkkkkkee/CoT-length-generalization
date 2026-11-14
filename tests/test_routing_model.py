import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import Task
from sparse_model import SparseClauseTransformer, SparseModelConfig
from sparse_probe import temporary_q_structure
from routing_model import (RoutingClauseTransformer, convert_model, load_routing_model,
                           routing_parameter_count, routing_parameter_counts)


def setup():
    torch.manual_seed(21001)
    task = Task("S5", 8)
    base = SparseClauseTransformer(SparseModelConfig(task.vocab_size, 8, hidden=32))
    with torch.no_grad():
        base.Q43.normal_()
        base.Q44.normal_()
    sample = task.generate(5, 4, torch.Generator().manual_seed(21002))
    return task, base, sample


@pytest.mark.parametrize("mode,count", [("row_tied", 32), ("diagonal", 16), ("dense", 128)])
def test_true_parameter_count_conversion_and_ffn_freeze(mode, count):
    _, base, sample = setup()
    source = {k: v.clone() for k, v in base.state_dict().items()}
    converted = convert_model(base, mode)
    assert routing_parameter_count(converted) == count
    assert routing_parameter_counts(converted)["trainable_coefficients"] == count
    assert all(not p.requires_grad for p in converted.ff.parameters())
    for k, v in converted.ff.state_dict().items():
        assert torch.equal(v, base.ff.state_dict()[k])
    for key in ["Q43", "Q44"]:
        assert torch.equal(getattr(converted, key).diag(), getattr(base, key).diag())
    for key, value in base.state_dict().items():
        assert torch.equal(value, source[key])
    if mode == "dense":
        torch.testing.assert_close(converted(sample['clauses']), base(sample['clauses']))


def test_row_tied_conversion_matches_independent_rowmean_intervention():
    _, base, sample = setup()
    converted = convert_model(base, "row_tied")
    with temporary_q_structure(base, "q_row_mean_background"):
        expected, attention = base(sample["clauses"], return_attention=True)
        actual, converted_attention = converted(sample["clauses"], return_attention=True)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(converted_attention, attention)


@pytest.mark.parametrize("mode", ["dense", "row_tied", "diagonal"])
def test_checkpoint_roundtrip_and_dense_legacy(tmp_path, mode):
    _, base, sample = setup()
    model = convert_model(base, mode)
    path = tmp_path / f"{mode}.pt"
    state = {"model": model.state_dict(), "model_config": model.config_dict(),
             "stage": "attention", "args": {"task": "S5"}}
    if mode != "dense":
        state['routing_mode'] = mode
    torch.save(state, path)
    loaded, metadata = load_routing_model(path)
    assert loaded.routing_mode == mode
    assert routing_parameter_count(loaded) == routing_parameter_count(model)
    torch.testing.assert_close(loaded(sample['clauses']), model(sample['clauses']), rtol=0, atol=0)
    assert all(not p.requires_grad for p in loaded.ff.parameters())


@pytest.mark.parametrize("mode", ["row_tied", "diagonal"])
def test_gradients_update_low_dimensional_parameters_and_keep_ties_and_ffn(mode):
    task, base, sample = setup()
    model = convert_model(base, mode)
    before_hash = model.frozen_state_hash()
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.03)
    for _ in range(4):
        logits = model.forward_queries(sample['clauses'][:, :-1], torch.tensor([5]))[:, 0, 4]
        loss = F.cross_entropy(logits, sample['clauses'][:, 6, 4])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        for p in model.routing_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.ff.parameters())
        optim.step()
    for name, value in before.items():
        assert not torch.equal(dict(model.named_parameters())[name], value)
    assert before_hash == model.frozen_state_hash()
    for key in ['Q43', 'Q44']:
        q = getattr(model, key).detach()
        for i in range(task.nvars):
            off = q[i, torch.arange(task.nvars) != i]
            torch.testing.assert_close(off, off[0].expand_as(off), rtol=0, atol=0)
            if mode == 'diagonal':
                assert off.count_nonzero() == 0


def test_zero_routing_uniform_attention_causality_and_cached_generation():
    task, _, sample = setup()
    model = RoutingClauseTransformer(SparseModelConfig(task.vocab_size, 8, hidden=32))
    _, a = model.forward_queries(sample['clauses'], torch.tensor([4, 5]), return_attention=True)
    torch.testing.assert_close(a[:, :, 0, :5], torch.full_like(a[:, :, 0, :5], .2))
    assert a[:, :, 0, 5:].count_nonzero() == 0
    torch.testing.assert_close(a[:, :, 1, :6], torch.full_like(a[:, :, 1, :6], 1 / 6))
    changed = sample['clauses'].clone()
    changed[:, 6:] = torch.randint(task.vocab_size, changed[:, 6:].shape)
    torch.testing.assert_close(model(sample['clauses'])[:, :6], model(changed)[:, :6])
    full = sample['clauses'][:, :5]
    incoming, cache = full, None
    for _ in range(4):
        expected = model(full)[:, -1]
        actual, cache = model.decode_last(incoming, cache)
        torch.testing.assert_close(actual, expected)
        incoming = actual.argmax(-1)[:, None]
        full = torch.cat((full, incoming), 1)


def test_stage_configuration_and_single_variable_degenerate_case():
    model = RoutingClauseTransformer(SparseModelConfig(4, 1, hidden=8), "row_tied")
    assert routing_parameter_count(model) == 2
    assert model.background43 is None and model.background44 is None
    model.configure_stage('local')
    assert all(not p.requires_grad for p in model.routing_parameters())
    assert all(p.requires_grad for p in model.ff.parameters())
    model.configure_stage('attention')
    assert routing_parameter_counts(model)['trainable_coefficients'] == 2
    with pytest.raises(ValueError):
        model.configure_stage('other')


def test_conversion_and_loading_preserve_float_dtype_and_caller_rng(tmp_path):
    _, base, sample = setup()
    base = base.double()
    rng = torch.random.get_rng_state().clone()
    model = convert_model(base, 'row_tied')
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(p.dtype == torch.float64 for p in model.parameters())
    path = tmp_path / 'double.pt'
    torch.save({'model': model.state_dict(), 'model_config': model.config_dict(),
                'routing_mode': 'row_tied', 'stage': 'attention'}, path)
    loaded, _ = load_routing_model(path)
    assert all(p.dtype == torch.float64 for p in loaded.parameters())
    torch.testing.assert_close(loaded(sample['clauses']), model(sample['clauses']), rtol=0, atol=0)
