import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import Task
from sparse_model import SparseClauseTransformer, SparseModelConfig


def make_model():
    torch.manual_seed(372)
    task = Task("S5", nvars=8)
    model = SparseClauseTransformer(SparseModelConfig(task.vocab_size, task.nvars, hidden=48))
    sample = task.generate(4, 4, torch.Generator().manual_seed(931))
    return task, model, sample


def test_zero_q_gives_causal_uniform_attention_and_blank_is_zero():
    task, model, sample = make_model()
    clauses = sample["clauses"][:, :-1]
    positions = torch.tensor([0, 4, 6])
    logits, attention = model.forward_queries(clauses, positions, return_attention=True)
    expected = torch.zeros_like(attention)
    for j, position in enumerate(positions):
        expected[:, :, j, :position + 1] = 1 / (position + 1)
    torch.testing.assert_close(attention, expected)

    # Independently materialize the fixed orthogonal embedding and compare.
    embedded = F.one_hot(clauses, task.vocab_size).float()
    embedded[..., 0] = 0
    mixed = attention[:, 0] @ embedded.flatten(-2)
    expected_logits = model.ff(mixed).reshape_as(logits)
    torch.testing.assert_close(logits, expected_logits)


def test_selected_queries_equal_full_forward_and_cannot_see_future():
    _, model, sample = make_model()
    with torch.no_grad():
        model.Q43.normal_()
        model.Q44.normal_()
    clauses = sample["clauses"][:, :-1]
    positions = torch.tensor([[4, 5], [5, 6], [4, 6], [5, 7]])
    full, full_attention = model(clauses, temperature=1.7, return_attention=True)
    selected, selected_attention = model.forward_queries(clauses, positions, temperature=1.7,
                                                        return_attention=True)
    expected = full.gather(1, positions[..., None, None].expand(-1, -1, 5, full.shape[-1]))
    expected_attention = full_attention.gather(2, positions[:, None, :, None].expand(-1, 1, -1, clauses.shape[1]))
    torch.testing.assert_close(selected, expected)
    torch.testing.assert_close(selected_attention, expected_attention)
    changed = clauses.clone()
    changed[:, 6:] = torch.randint(model.config.vocab_size, changed[:, 6:].shape)
    torch.testing.assert_close(model(clauses)[:, :6], model(changed)[:, :6])


def test_attention_training_keeps_local_map_bitwise_fixed_and_updates_both_q_blocks():
    task, model, sample = make_model()
    model.configure_stage("attention")
    frozen_before = model.frozen_state_hash()
    ff_before = {k: v.detach().clone() for k, v in model.ff.state_dict().items()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.3)
    logits = model.forward_queries(sample["clauses"][:, :-1], torch.tensor([5]))[:, 0, 4]
    # L=4: query 5 is the first answer, whose next target is y2 at clause 6.
    loss = F.cross_entropy(logits, sample["clauses"][:, 6, 4])
    loss.backward()
    assert model.Q43.grad is not None and model.Q43.grad.abs().sum() > 0
    assert model.Q44.grad is not None and model.Q44.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.ff.parameters())
    optimizer.step()
    assert model.frozen_state_hash() == frozen_before
    assert model.Q43.detach().abs().sum() > 0 and model.Q44.detach().abs().sum() > 0
    for key, value in model.ff.state_dict().items():
        assert torch.equal(value, ff_before[key])


def test_local_training_keeps_q_zero():
    _, model, sample = make_model()
    model.configure_stage("local")
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    logits = model.forward_queries(sample["clauses"][:, :-1], torch.tensor([4]))[:, 0]
    F.cross_entropy(logits.flatten(0, 1), sample["clauses"][:, 5].flatten()).backward()
    assert model.Q43.grad is None and model.Q44.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.ff.parameters())
    optimizer.step()
    assert torch.count_nonzero(model.Q43) == 0 and torch.count_nonzero(model.Q44) == 0


def test_cached_rollout_matches_full_prefix_even_for_invalid_variable_predictions():
    _, model, sample = make_model()
    with torch.no_grad():
        model.Q43.normal_()
        model.Q44.normal_()
    for temperature in [0.5, 1.0, 2.0]:
        full = sample["clauses"][:, :5]
        incoming = full
        cache = None
        for _ in range(4):
            expected = model.forward_queries(full, torch.tensor([full.shape[1] - 1]), temperature)[:, 0]
            actual, cache = model.decode_last(incoming, cache, temperature)
            torch.testing.assert_close(actual, expected)
            # No output vocabulary masks or oracle variable/blank replacement.
            incoming = actual.argmax(-1)[:, None]
            full = torch.cat((full, incoming), dim=1)


def test_oracle_attention_uses_only_the_chosen_rule_and_current_state():
    task, model, sample = make_model()
    clauses = sample["clauses"][:, :-1]
    positions = torch.tensor([4, 5])
    override = torch.zeros(len(clauses), 1, 2, clauses.shape[1])
    override[:, 0, 0, [0, 4]] = 0.5
    override[:, 0, 1, [1, 5]] = 0.5
    actual, attention = model.forward_queries(clauses, positions, return_attention=True,
                                             attention_override=override)
    inputs = (F.one_hot(clauses[:, [0, 1]], task.vocab_size)
              + F.one_hot(clauses[:, [4, 5]], task.vocab_size)).float() * 0.5
    inputs[..., 0] = 0
    expected = model.ff(inputs.flatten(-2)).reshape_as(actual)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(attention, override)
    invalid = override.clone()
    invalid[:, 0, 0, 4] = 0
    invalid[:, 0, 0, 7] = 0.5
    with pytest.raises(ValueError, match="future"):
        model.forward_queries(clauses, positions, attention_override=invalid)


def test_invalid_query_variable_does_not_alias_a_real_variable():
    task, model, sample = make_model()
    with torch.no_grad():
        model.Q43.normal_()
        model.Q44.normal_()
    clauses = sample["clauses"][:, :5].clone()
    clauses[:, -1, 3] = task.action_offset  # Action, not a variable.
    _, attention = model.forward_queries(clauses, torch.tensor([4]), return_attention=True)
    torch.testing.assert_close(attention, torch.full_like(attention, 0.2))
