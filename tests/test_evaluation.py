"""Checks target alignment, real feedback, metric distinctions and no-GT filtering."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import Task, permute_predicates
from model import ClauseTransformer, ModelConfig
from evaluate import (teacher_forced_metrics, greedy_rollout, rollout_metrics,
                      aggregate_batches, permutation_check)
from generate_pseudo import generate_pseudo_dataset
from distractors import add_distractors


class Oracle(nn.Module):
    """Independent semantic oracle; attention itself is exactly uniform causal."""
    def __init__(self, task, corrupt=False):
        super().__init__()
        self.task = task
        self.dummy = nn.Parameter(torch.zeros(()))
        self.corrupt = corrupt

    def forward(self, clauses, temperature=1., return_attention=False):
        b, t, _ = clauses.shape
        output = torch.zeros((b, t, 5), dtype=torch.long, device=clauses.device)
        for row in range(b):
            predicates = clauses[row][clauses[row, :, 1] >= self.task.action_offset]
            for position in range(t):
                current = clauses[row, position]
                variable = current[3]
                valid = predicates[predicates[:, 2].eq(variable)]
                if len(valid) == 1 and self.task.value_offset <= current[4] < self.task.vocab_size:
                    predicate = valid[0]
                    action = int(predicate[1]) - self.task.action_offset
                    state = int(current[4]) - self.task.value_offset
                    output[row, position, 3] = predicate[0]
                    output[row, position, 4] = self.task.value_offset + self.task.action_table[action, state]
                if self.corrupt and variable > 0:
                    # Corrupt slot0 even though it does not affect the first value.
                    # Its feedback causes subsequent outputs to be invalid.
                    output[row, position, 0] = 1
                    if current[0] != 0:
                        output[row, position, 4] = 0
        logits = torch.full((b, t, 5, self.task.vocab_size), -20., device=clauses.device)
        logits.scatter_(-1, output[..., None], 20.)
        if return_attention:
            attention = torch.ones((t, t), device=clauses.device).tril()
            attention /= attention.sum(-1, keepdim=True)
            return logits, attention.expand(b, 2, -1, -1)
        return logits


def test_oracle_alignment_and_attention():
    task = Task("C6", 16)
    sample = task.generate(7, 4, torch.Generator().manual_seed(101))
    metrics = teacher_forced_metrics(Oracle(task), sample)
    assert metrics["counts"]["token_correct"] == 7 * 4 * 5
    assert metrics["metrics"]["value_chain_accuracy"] == 1.
    expected = torch.tensor([1 / k for k in range(5, 9)]).mean().item()
    assert metrics["attention"]["mean_per_head"]["target_predicate_weight"] == pytest.approx([expected, expected])
    assert metrics["attention"]["mean_per_head"]["relevant_weight"] == pytest.approx([2 * expected] * 2)
    pred = greedy_rollout(Oracle(task), sample["clauses"][:, :5], 4)
    assert torch.equal(pred, sample["clauses"][:, 5:])


def test_unconstrained_rollout_feeds_all_five_slots_back():
    task = Task("C6", 16)
    sample = task.generate(3, 4, torch.Generator().manual_seed(103))
    pred = greedy_rollout(Oracle(task, corrupt=True), sample["clauses"][:, :5], 4)
    # First value remains right, but a wrong marker is fed back and ruins value2.
    assert torch.equal(pred[:, 0, 4], sample["clauses"][:, 5, 4])
    assert torch.all(pred[:, 0, 0] == 1)
    assert torch.all(pred[:, 1, 4] == 0)
    result = rollout_metrics(pred, sample["clauses"][:, 5:])
    assert result["counts"]["first_value_error_histogram"] == [0, 3, 0, 0, 0]
    assert result["counts"]["first_clause_error_histogram"] == [3, 0, 0, 0, 0]
    assert result["metrics"]["final_value_accuracy"] == 0.


def test_final_answer_and_chain_metrics_are_distinct():
    targets = torch.zeros(3, 4, 5, dtype=torch.long)
    pred = targets.clone()
    pred[0, 0, 4] = 1  # recovers by the final answer
    pred[1, -1, 4] = 1
    pred[2, 1, 3] = 1  # values all right; variable wrong
    result = rollout_metrics(pred, targets)
    assert result["metrics"]["final_value_accuracy"] == pytest.approx(2 / 3)
    assert result["metrics"]["value_chain_accuracy"] == pytest.approx(1 / 3)
    assert result["metrics"]["clause_chain_accuracy"] == 0.
    assert result["counts"]["first_value_error_histogram"] == [1, 0, 0, 1, 1]


def test_aggregation_weights_uneven_last_batch():
    target = torch.zeros(3, 2, 5, dtype=torch.long)
    pred = target.clone()
    pred[2, :, 4] = 1
    full = rollout_metrics(pred, target)
    merged = aggregate_batches([rollout_metrics(pred[:2], target[:2]), rollout_metrics(pred[2:], target[2:])])
    assert merged["samples"] == 3
    assert merged["counts"] == full["counts"]
    assert merged["metrics"]["value_accuracy"] == pytest.approx(full["metrics"]["value_accuracy"])
    assert merged["metrics"]["first_value_error_censored_mean"] == pytest.approx(full["metrics"]["first_value_error_censored_mean"])


def test_predicate_permutation_invariance_and_diagnostic_alignment():
    torch.manual_seed(19)
    task = Task("C6", 16)
    sample = task.generate(3, 4, torch.Generator().manual_seed(106))
    permuted = permute_predicates(sample, torch.Generator().manual_seed(107))
    model = ClauseTransformer(ModelConfig(task.vocab_size, token_dim=4)).eval()
    result = permutation_check(model, sample, permuted)
    assert result["max_abs_logit_difference"] < 2e-6
    assert result["argmax_token_agreement"] == 1.
    original_diag = teacher_forced_metrics(model, sample)["attention"]["mean_per_head"]
    shuffled_diag = teacher_forced_metrics(model, permuted)["attention"]["mean_per_head"]
    assert original_diag["target_predicate_weight"] == pytest.approx(shuffled_diag["target_predicate_weight"], abs=1e-7)


def test_pseudolabels_retain_wrong_samples_without_truth_correction():
    task = Task("C6", 16)
    payload = generate_pseudo_dataset(Oracle(task, corrupt=True), task, 4, 5, batch=2,
                                      seed=111, device="cpu")
    assert payload["clauses"].shape == payload["ground_truth_clauses"].shape == (5, 9, 5)
    assert torch.equal(payload["clauses"][:, :5], payload["ground_truth_clauses"][:, :5])
    assert not torch.equal(payload["clauses"][:, 5:], payload["ground_truth_clauses"][:, 5:])
    assert torch.all(payload["clauses"][:, 5, 0] == 1)
    assert payload["metadata"]["retained_samples"] == 5
    assert payload["metadata"]["filtered_samples"] == 0
    assert payload["audit"]["metrics"]["clause_chain_accuracy"] == 0.


def test_cached_rollout_matches_full_prefix_feedback():
    torch.manual_seed(29)
    task = Task("C6", 16)
    sample = task.generate(3, 4, torch.Generator().manual_seed(119))
    model = ClauseTransformer(ModelConfig(task.vocab_size, token_dim=4)).eval()
    class FullPrefix(nn.Module):
        def forward(self, clauses, temperature=1.):
            return model(clauses, temperature=temperature)
    prompt = sample["clauses"][:, :5]
    for temperature in (0.5, 1., 4.):
        cached = greedy_rollout(model, prompt, 4, temperature)
        full = greedy_rollout(FullPrefix(), prompt, 4, temperature)
        assert torch.equal(cached, full)


def test_explicit_query_positions_with_distractors_use_correct_targets():
    task = Task("C6", 16)
    sample = task.generate(3, 4, torch.Generator().manual_seed(123))
    extended = add_distractors(sample, task, 7, torch.Generator().manual_seed(124))
    result = teacher_forced_metrics(Oracle(task), extended)
    assert result["length"] == 4
    assert result["counts"]["token_correct"] == 3 * 4 * 5
    expected = sum(7 / k for k in range(12, 16)) / 4
    assert result["attention"]["mean_per_head"]["added_distractor_weight"] == pytest.approx([expected] * 2)
