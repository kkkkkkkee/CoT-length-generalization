"""Independent checks of sparse-experiment diagnostic semantics and batching."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import Task
from sparse_experiment import evaluate
from sparse_model import SparseClauseTransformer, SparseModelConfig


def small_model():
    torch.manual_seed(1927)
    task = Task("C5", nvars=8)
    model = SparseClauseTransformer(
        SparseModelConfig(task.vocab_size, task.nvars, hidden=16)
    ).eval()
    return task, model


def test_zero_q_is_a_tie_not_successful_rule_retrieval():
    task, model = small_model()
    rows = evaluate(model, task, [1, 2, 4], samples=7,
                    rollout_samples=0, seed=261, batch_size=3)
    for row in rows:
        length = row["length"]
        attention = row["attention"]
        # Query j sees L predicates plus the initial answer and j old answers.
        expected_mass = sum(1 / (length + j + 1) for j in range(length)) / length
        assert attention["target_rule_mass"] == pytest.approx(expected_mass, abs=1e-7)
        assert attention["current_state_mass"] == pytest.approx(expected_mass, abs=1e-7)
        assert attention["relevant_mass"] == pytest.approx(2 * expected_mass, abs=1e-7)
        assert attention["absolute_rule_state_gap"] == 0
        if length > 1:
            assert attention["rule_strict_top1_among_predicates"] == 0
            assert attention["rule_top_tie_rate"] == 1
            assert attention["largest_background_mass"] == pytest.approx(expected_mass, abs=1e-7)
        else:
            assert attention["rule_strict_top1_among_predicates"] == 1
            assert attention["rule_top_tie_rate"] == 0
            assert attention["largest_background_mass"] == 0


def test_oracle_diagnostics_and_routing_match_half_rule_half_current_answer(monkeypatch):
    task, model = small_model()
    seen = []
    original = model.forward_queries

    def inspect_queries(clauses, positions, **kwargs):
        override = kwargs["attention_override"]
        length = positions.shape[1]
        expected = torch.zeros_like(override)
        for j in range(length):
            expected[:, 0, j, j] = 0.5
            expected[:, 0, j, length + j] = 0.5
        torch.testing.assert_close(override, expected)
        result = original(clauses, positions, **kwargs)
        # Independently construct the exact local-computation inputs, including
        # blank=zero rather than a conventional one-hot blank embedding.
        mixed = (
            F.one_hot(clauses[:, :length], task.vocab_size)
            + F.one_hot(clauses[:, length:], task.vocab_size)
        ).float() * 0.5
        mixed[..., 0] = 0
        expected_logits = model.ff(mixed.flatten(-2)).reshape_as(result[0])
        torch.testing.assert_close(result[0], expected_logits)
        seen.append(len(clauses))
        return result

    monkeypatch.setattr(model, "forward_queries", inspect_queries)
    row = evaluate(model, task, [4], samples=7, rollout_samples=5,
                   seed=261, batch_size=3, oracle=True)[0]
    assert seen == [3, 3, 1]
    assert row["oracle_attention"] is True
    assert "rollout" not in row  # Oracle teacher-forcing is not free rollout.
    assert row["teacher_forced"]["samples"] == 7
    assert row["attention"] == {
        "target_rule_mass": 0.5,
        "current_state_mass": 0.5,
        "relevant_mass": 1.0,
        "absolute_rule_state_gap": 0.0,
        "rule_strict_top1_among_predicates": 1.0,
        "largest_background_mass": 0.0,
        "rule_top_tie_rate": 0.0,
    }


def test_unequal_final_batch_aggregates_counts_and_rollout_subset_correctly():
    task, model = small_model()
    # A deterministic constant output makes the expected correct counts depend
    # on the actual test examples rather than on fragile random-model argmaxes.
    with torch.no_grad():
        for parameter in model.ff.parameters():
            parameter.zero_()
        bias = model.ff[-1].bias.view(5, task.vocab_size)
        bias[:3, 0] = 1
        bias[3, 1] = 1
        bias[4, task.value_offset] = 1

    seed, samples, length, rollout_samples = 271, 11, 4, 5
    small = evaluate(model, task, [length], samples=samples,
                     rollout_samples=rollout_samples, seed=seed, batch_size=3)[0]
    whole = evaluate(model, task, [length], samples=samples,
                     rollout_samples=rollout_samples, seed=seed, batch_size=32)[0]
    assert small["panel_sha256"] == whole["panel_sha256"]
    truth = task.generate(samples, length, torch.Generator().manual_seed(seed + length))["clauses"]
    target = truth[:, length + 1:]
    for kind, count in [("teacher_forced", samples), ("rollout", rollout_samples)]:
        actual = small[kind]
        expected_values = target[:count, :, 4].eq(task.value_offset)
        expected_variables = target[:count, :, 3].eq(1)
        expected_clauses = expected_values & expected_variables
        assert actual["samples"] == count
        assert actual["counts"] == whole[kind]["counts"]
        assert actual["counts"]["value_correct"] == int(expected_values.sum())
        assert actual["counts"]["variable_correct"] == int(expected_variables.sum())
        assert actual["counts"]["clause_correct"] == int(expected_clauses.sum())
        assert actual["counts"]["final_value_correct"] == int(expected_values[:, -1].sum())
        assert actual["metrics"]["value_accuracy"] == pytest.approx(
            expected_values.double().mean().item()
        )
        assert actual["metrics"]["clause_chain_accuracy"] == pytest.approx(
            expected_clauses.all(-1).double().mean().item()
        )
        assert sum(actual["counts"]["first_value_error_histogram"]) == count
        assert sum(actual["counts"]["first_clause_error_histogram"]) == count
        for key, value in actual["by_position"].items():
            assert value == pytest.approx(whole[kind]["by_position"][key])
    assert small["attention"] == pytest.approx(whole["attention"], abs=1e-7)
