"""Semantic checks of mass/balance interventions, without GPU computation."""
import json

import pytest
import torch

from src.data import Task
from src.sparse_model import SparseClauseTransformer, SparseModelConfig
from src.sparse_probe import CONDITIONS, Q_CONDITIONS, intervene_attention, probe_model, save_results, temporary_q_structure


def attention_fixture():
    # Rule index 1, query index 4, future key index 5.
    a = torch.tensor([[[[0.10, 0.30, 0.20, 0.15, 0.25, 0.0]]]], dtype=torch.float64)
    return a, torch.tensor([1]), torch.tensor([4])


def test_independent_mass_and_balance_and_background_ratios():
    a, rule, query = attention_fixture()
    for condition, m, r in [("reconstruct_sham", .55, .30 / .55),
                            ("balance_only", .55, .5),
                            ("concentrate_only", .99, .30 / .55),
                            ("both", .99, .5)]:
        result, diag = intervene_attention(a, rule, query, condition)
        torch.testing.assert_close(result.sum(-1), torch.ones_like(result[..., 0]))
        assert (result >= 0).all()
        assert result[..., 5].item() == 0
        assert diag["relevant_mass"].item() == pytest.approx(m)
        assert diag["rule_share"].item() == pytest.approx(r)
        torch.testing.assert_close(result[..., [0, 2, 3]] / result[..., [0]].clamp_min(1e-20),
                                   a[..., [0, 2, 3]] / a[..., [0]])
    sham, _ = intervene_attention(a, rule, query, "reconstruct_sham")
    torch.testing.assert_close(sham, a)


def test_random_pair_is_feasible_and_reports_its_actual_unmatched_amplitude():
    a, rule, query = attention_fixture()
    a = a.expand(12, -1, -1, -1).clone()
    result, diag = intervene_attention(a, rule, query, "random_pair_both", seed=15)
    assert (result >= 0).all()
    torch.testing.assert_close(result.sum(-1), torch.ones_like(result[..., 0]))
    assert (result[..., 5] == 0).all()
    assert ((result - a).abs() > 1e-12).sum(-1).eq(2).all()
    torch.testing.assert_close(diag["l1_change"], (result - a).abs().sum(-1))
    assert (diag["l1_change"] <= diag["requested_l1"] + 1e-12).all()
    assert diag["random_transfer_capped"].any()
    assert not torch.allclose(diag["l1_change"], diag["requested_l1"])
    left, _ = intervene_attention(a[:5], rule, query, "random_pair_both", seed=15)
    right, _ = intervene_attention(a[5:], rule, query, "random_pair_both", seed=15, sample_offset=5)
    torch.testing.assert_close(result, torch.cat((left, right)))


def test_no_background_cannot_receive_mass_and_zero_background_fallback_is_counted():
    a = torch.tensor([[[[0.4, 0.6]]]])
    with pytest.raises(ValueError, match="without background"):
        intervene_attention(a, [0], [1], "both")
    b = torch.tensor([[[[0.4, 0.0, 0.6]]]])
    result, diag = intervene_attention(b, [0], [2], "both")
    assert diag["zero_background_uniform_fallback"].item()
    assert result[..., 1].item() == pytest.approx(.01)
    torch.testing.assert_close(result.sum(-1), torch.ones_like(result[..., 0]))


def test_future_attention_is_rejected():
    a, rule, query = attention_fixture()
    a[..., 5] = .05
    a[..., 0] -= .05
    with pytest.raises(ValueError, match="future"):
        intervene_attention(a, rule, query, "both")


def test_probe_sham_is_paired_and_serializes_exact_panels(tmp_path):
    torch.manual_seed(29)
    task = Task("C5", nvars=10)
    model = SparseClauseTransformer(SparseModelConfig(task.vocab_size, task.nvars, hidden=24))
    with torch.no_grad():
        model.Q43.normal_()
        model.Q44.normal_()
    result = probe_model(model, task, lengths=[4], samples=7, batch=3, seed=75)
    panel = result["panels"][0]
    assert panel["sham_max_abs_logit_difference"] < 1e-6
    assert len(panel["sample_ids"]) == 7
    rows = {row["condition"]: row for row in panel["conditions"]}
    assert tuple(rows) == CONDITIONS
    assert rows["original"]["per_sample"] == rows["reconstruct_sham"]["per_sample"]
    assert rows["reconstruct_sham"]["paired_with_original"]["clause_chain_accuracy_difference"] == 0
    assert rows["both"]["attention"]["relevant_mass"]["mean"] == pytest.approx(.99)
    assert rows["both"]["attention"]["rule_share"]["mean"] == pytest.approx(.5)
    # Independent regeneration confirms exact recorded input/target provenance.
    again = probe_model(model, task, lengths=[4], samples=7, batch=7, seed=75)
    assert again["panels"][0]["sample_clauses_sha256"] == panel["sample_clauses_sha256"]
    again_random = next(row for row in again["panels"][0]["conditions"] if row["condition"] == "random_pair_both")
    assert again_random["per_sample_attention"] == rows["random_pair_both"]["per_sample_attention"]
    save_results(tmp_path, result)
    saved = json.loads((tmp_path / "probe.json").read_text())
    assert saved["panels"][0]["sample_clauses_sha256"] == panel["sample_clauses_sha256"]
    assert len((tmp_path / "probe.csv").read_text().splitlines()) == len(CONDITIONS) + 1


def test_intervened_query_cannot_read_its_target_or_future_history():
    torch.manual_seed(12)
    task = Task("C5", nvars=10)
    model = SparseClauseTransformer(SparseModelConfig(task.vocab_size, task.nvars, hidden=24))
    clauses = task.generate(3, 4, torch.Generator().manual_seed(13))["clauses"][:, :-1]
    query = torch.tensor([5])  # Predict y2; y2 is at position 6, causally hidden.
    _, original_attention = model.forward_queries(clauses, query, return_attention=True)
    weights, _ = intervene_attention(original_attention, [1], query, "both")
    actual = model.forward_queries(clauses, query, attention_override=weights)
    altered = clauses.clone()
    altered[:, 6:] = torch.randint(task.vocab_size, altered[:, 6:].shape)
    expected = model.forward_queries(altered, query, attention_override=weights)
    torch.testing.assert_close(actual, expected)


def test_q_structure_preserves_diagonal_and_restores_even_after_exception():
    task = Task("C5", nvars=4)
    model = SparseClauseTransformer(SparseModelConfig(task.vocab_size, task.nvars, hidden=24))
    original = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    with torch.no_grad():
        model.Q43.copy_(original)
        model.Q44.copy_(-original)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    diagonal_mask = torch.eye(4, dtype=torch.bool)
    for condition in Q_CONDITIONS:
        with temporary_q_structure(model, condition):
            for name, sign in [("Q43", 1), ("Q44", -1)]:
                q = getattr(model, name)
                torch.testing.assert_close(q.diagonal(), sign * original.diagonal())
                if condition == "q_diagonal_only":
                    assert (q[~diagonal_mask] == 0).all()
                else:
                    for row in range(4):
                        expected = sign * original[row][~diagonal_mask[row]].mean()
                        torch.testing.assert_close(q[row][~diagonal_mask[row]], expected.expand(3))
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key])
    with pytest.raises(RuntimeError):
        with temporary_q_structure(model, "q_diagonal_only"):
            raise RuntimeError("interrupted mechanism probe")
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
