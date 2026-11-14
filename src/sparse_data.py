"""Separate prefixes, value targets, and supervision masks for sparse CoT studies.

The full input tensor contains every available training prefix. A model MUST
apply a causal mask: the query for answer step j is the preceding answer at
position L+j-1, while its target is stored separately. No gradient through
discrete teacher generation is implied. `random` chooses one fixed uniformly
random step in 1..L for each example, rather than resampling each epoch.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn.functional as F


def tensor_hash(value: torch.Tensor) -> str:
    """Content hash includes shape and dtype as well as tensor bytes."""
    x = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(x.shape), str(x.dtype))).encode())
    digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def _rate(value: torch.Tensor) -> float:
    return value.double().mean().item()


def build_supervision(
    sample: dict[str, torch.Tensor],
    task: Any,
    mode: str = "second",
    label_source: str = "truth",
    teacher_clauses: torch.Tensor | None = None,
    noise_kind: str = "none",
    noise_rate: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Construct value-only supervision without mutating any source tensor.

    Return tensors on the source clauses' device:
      inputs: [N,2L,5], targets/mask/query_positions: [N,L].
      selected_steps/query_positions/targets: [N,J], J=1 or L.
    `selected_steps` uses ZERO-based answer indices (1 means answer y2).

    Teacher mode uses the supplied full greedy trace verbatim: every predicted
    slot is retained, including invalid value/variable choices. Only the
    predicates and given initial answer must match the original prompt.

    Controlled noise is deliberately restricted to truth/second, isolating
    one intervention: target changes only y2's loss label; prefix_inconsistent
    changes only input y1; prefix_consistent also changes y2's label to g2(y1').
    Variables and all other tokens remain unchanged. Every replacement is a
    different valid state. The same seed gives nested corruption sets across
    rates and identical replacements across prefix conditions.
    """
    if mode not in {"second", "random", "all"}:
        raise ValueError("mode must be second, random, or all")
    if label_source not in {"truth", "teacher"}:
        raise ValueError("label_source must be truth or teacher")
    if noise_kind not in {"none", "target", "prefix_inconsistent", "prefix_consistent"}:
        raise ValueError("unsupported noise_kind")
    if not math.isfinite(noise_rate) or not 0 <= noise_rate <= 1:
        raise ValueError("noise_rate must be finite and in [0,1]")
    if noise_kind == "none" and noise_rate != 0:
        raise ValueError("nonzero noise_rate requires an explicit noise_kind")
    if noise_kind != "none" and (label_source != "truth" or mode != "second"):
        raise ValueError("controlled noise currently requires truth labels and second-step supervision")
    truth = sample["clauses"]
    if truth.ndim != 3 or truth.shape[-1] != 5 or truth.shape[1] % 2 != 1:
        raise ValueError("sample clauses must have shape [N,2L+1,5]")
    n, length = truth.shape[0], (truth.shape[1] - 1) // 2
    if n < 1 or length < 1 or truth.dtype != torch.long:
        raise ValueError("sample must contain nonempty long-integer LEGO traces")
    if mode == "second" and length < 2:
        raise ValueError("second-step supervision requires L>=2")
    if bool(((truth < 0) | (truth >= task.vocab_size)).any()):
        raise ValueError("sample contains a token outside the task vocabulary")
    if label_source == "teacher":
        if teacher_clauses is None or teacher_clauses.shape != truth.shape:
            raise ValueError("teacher_clauses must contain a complete trace matching sample shape")
        if teacher_clauses.dtype != torch.long or teacher_clauses.device != truth.device:
            raise ValueError("teacher_clauses must have source dtype and device")
        if not torch.equal(teacher_clauses[:, :length + 1], truth[:, :length + 1]):
            raise ValueError("teacher trace must preserve predicates and the given initial answer")
        if bool(((teacher_clauses < 0) | (teacher_clauses >= task.vocab_size)).any()):
            raise ValueError("teacher token outside vocabulary")
        source = teacher_clauses
    else:
        if teacher_clauses is not None:
            raise ValueError("teacher_clauses supplied with truth label_source")
        source = truth
    # These copies must stay separate: changing loss labels cannot silently
    # change a later training prefix, and prefix corruption cannot change labels.
    inputs = source[:, :-1].clone()
    targets = source[:, length + 1:, 4].clone()
    query_positions = torch.arange(length, 2 * length, device=truth.device).expand(n, -1).clone()
    if mode == "all":
        selected_steps = torch.arange(length, device=truth.device).expand(n, -1).clone()
    elif mode == "second":
        selected_steps = torch.ones((n, 1), device=truth.device, dtype=torch.long)
    else:
        gen = torch.Generator().manual_seed(seed + 1000003)
        selected_steps = torch.randint(length, (n, 1), generator=gen).to(truth.device)
    mask = torch.zeros((n, length), dtype=torch.bool, device=truth.device)
    mask.scatter_(1, selected_steps, True)

    # Independent draws make rates nested, and ensure neither replacement
    # values nor selected supervision positions depend on noise_rate.
    noise_gen = torch.Generator().manual_seed(seed + 2000003)
    noise_scores = torch.rand(n, generator=noise_gen, dtype=torch.float64).to(truth.device)
    offsets = torch.randint(1, task.nstates, (n,), generator=noise_gen).to(truth.device)
    corrupted = noise_scores < noise_rate
    if noise_kind == "target":
        original = targets[:, 1] - task.value_offset
        replacement = (original + offsets) % task.nstates + task.value_offset
        targets[corrupted, 1] = replacement[corrupted]
    elif noise_kind in {"prefix_inconsistent", "prefix_consistent"}:
        original = inputs[:, length + 1, 4] - task.value_offset
        replacement = (original + offsets) % task.nstates
        inputs[corrupted, length + 1, 4] = replacement[corrupted] + task.value_offset
        if noise_kind == "prefix_consistent":
            actions = sample["actions"][:, 1].to(truth.device)
            repaired = task.action_table.to(truth.device)[actions, replacement] + task.value_offset
            targets[corrupted, 1] = repaired[corrupted]

    selected_queries = query_positions.gather(1, selected_steps)
    selected_targets = targets.gather(1, selected_steps)
    previous_state = inputs[:, length:2 * length, 4] - task.value_offset
    valid_state = (previous_state >= 0) & (previous_state < task.nstates)
    # Invalid generated state tokens are counted inconsistent, never silently
    # replaced for training. Clamping only keeps this audit lookup in bounds.
    actions = sample["actions"].to(truth.device)
    if actions.shape != (n, length):
        raise ValueError("sample actions must match [N,L]")
    local_target = task.action_table.to(truth.device)[actions, previous_state.clamp(0, task.nstates - 1)] + task.value_offset
    locally_consistent = valid_state & targets.eq(local_target)
    routing_correct = inputs[:, length:2 * length, 3].eq(truth[:, length:2 * length, 3])
    valid_supervised = mask & valid_state
    valid_routed_supervised = valid_supervised & routing_correct
    target_wrong = targets.ne(truth[:, length + 1:, 4])
    prefix_wrong = inputs[:, length:2 * length, 4].ne(truth[:, length:2 * length, 4])
    selected_variables = inputs[:, length:2 * length, 3][mask]
    valid_query_variable = (selected_variables >= 1) & (selected_variables <= task.nvars)
    query_variable_counts = torch.bincount(
        selected_variables[valid_query_variable] - 1, minlength=task.nvars)
    selected_actions, selected_states = actions[mask], previous_state[mask]
    valid_selected_states = valid_state[mask]
    # The action is the intended ground-truth predicate's action, even when a
    # teacher supplies the wrong routing variable. The routing audit above
    # therefore remains essential to interpreting this exposure histogram.
    action_state_counts = torch.bincount(
        selected_actions[valid_selected_states] * task.nstates + selected_states[valid_selected_states],
        minlength=task.nactions * task.nstates).reshape(task.nactions, task.nstates)
    invalid_state_action_counts = torch.bincount(
        selected_actions[~valid_selected_states], minlength=task.nactions)
    # For answer y_j, j-1 generated/provided intermediate answers are visible.
    # Distinct prefix answers count each example's longest selected prefix once;
    # occurrences count reused prefix information separately for each query.
    distinct_prefix_answers = selected_steps.amax(dim=1)
    repeated_prefix_answers = selected_steps.sum(dim=1)
    audit = {
        "task": task.name, "length": length, "examples": n,
        "mode": mode, "label_source": label_source,
        "supervised_value_labels": int(mask.sum()),
        "labels_per_example": selected_steps.shape[1],
        "supervision_positions": "zero-based indices in selected_steps; y2 has index 1",
        "random_position_policy": "one fixed draw per example from 1..L" if mode == "random" else None,
        "seed": seed, "noise_kind": noise_kind, "requested_noise_rate": noise_rate,
        "corrupted_examples": int(corrupted.sum()), "actual_noise_rate": _rate(corrupted),
        "target_error_rate_all_steps": _rate(target_wrong),
        "target_error_rate_supervised": _rate(target_wrong[mask]),
        "prefix_state_error_rate_supervised": _rate(prefix_wrong[mask]),
        "prefix_variable_error_rate_supervised": _rate(~routing_correct[mask]),
        "invalid_prefix_state_rate_supervised": _rate(~valid_state[mask]),
        "intended_action_local_inconsistency_rate_supervised": _rate(~locally_consistent[mask]),
        "valid_prefix_state_supervised_labels": int(valid_supervised.sum()),
        "local_inconsistency_rate_given_valid_prefix_state": _rate(~locally_consistent[valid_supervised]) if bool(valid_supervised.any()) else None,
        "valid_state_correct_routing_supervised_labels": int(valid_routed_supervised.sum()),
        "local_inconsistency_rate_given_valid_state_and_correct_routing": _rate(~locally_consistent[valid_routed_supervised]) if bool(valid_routed_supervised.any()) else None,
        "local_consistency_and_correct_routing_rate_supervised": _rate((locally_consistent & routing_correct)[mask]),
        "selected_query_variable_counts": query_variable_counts.cpu().tolist(),
        "selected_query_variable_counts_indexing": "entry i counts actual input query variable token i+1; invalid tokens excluded",
        "selected_query_variables_seen": int((query_variable_counts > 0).sum()),
        "selected_query_variables_unseen": int((query_variable_counts == 0).sum()),
        "selected_query_invalid_variable_count": int((~valid_query_variable).sum()),
        "selected_true_action_input_state_counts": action_state_counts.cpu().tolist(),
        "selected_true_action_input_state_counts_indexing": "[zero-based intended true action][zero-based actual input state]; invalid state tokens excluded; routing can still be wrong",
        "selected_true_action_input_state_pairs_seen": int((action_state_counts > 0).sum()),
        "selected_true_action_input_state_pairs_unseen": int((action_state_counts == 0).sum()),
        "selected_invalid_input_state_count": int((~valid_selected_states).sum()),
        "selected_invalid_input_state_counts_by_true_action": invalid_state_action_counts.cpu().tolist(),
        "source_full_clause_trajectory_correct_rate": _rate(source[:, length + 1:].eq(truth[:, length + 1:]).all(dim=(1, 2))),
        "source_value_trajectory_correct_rate": _rate(source[:, length + 1:, 4].eq(truth[:, length + 1:, 4]).all(dim=1)),
        "source_clause_sha256": tensor_hash(truth),
        "label_source_clause_sha256": tensor_hash(source),
        "inputs_sha256": tensor_hash(inputs), "targets_sha256": tensor_hash(targets),
        "mask_sha256": tensor_hash(mask),
        "ground_truth_usage": "training source" if label_source == "truth" else "audit only; no filtering or correction",
        "prefix_information": "all five available preceding answer slots are retained, regardless of loss mask",
        "input_information_accounting": {
            "prefix_answer_source": label_source,
            "predicate_clauses_per_example": length,
            "given_initial_answers_per_example": 1,
            "distinct_intermediate_prefix_answers_total": int(distinct_prefix_answers.sum()),
            "distinct_intermediate_prefix_answers_per_example_mean": _rate(distinct_prefix_answers),
            "distinct_intermediate_prefix_answers_per_example_min": int(distinct_prefix_answers.min()),
            "distinct_intermediate_prefix_answers_per_example_max": int(distinct_prefix_answers.max()),
            "intermediate_prefix_answer_occurrences_across_selected_queries": int(repeated_prefix_answers.sum()),
            "source_full_generated_answers_supplied": n * length if label_source == "teacher" else None,
            "cost_caveat": "loss label count is not total annotation or generation cost: preceding answers supply five slots of information each, even with no loss on them; teacher cost includes generating the supplied trace; exact FLOPs/calls must be measured by the driver",
        },
        "local_consistency_definition": "target equals intended true action applied to input prefix state; invalid prefix states count inconsistent in unconditional rate and are excluded in conditional rate",
    }
    return {"inputs": inputs, "targets": targets, "mask": mask,
            "query_positions": query_positions, "selected_steps": selected_steps,
            "selected_query_positions": selected_queries, "selected_targets": selected_targets,
            "corrupted": corrupted, "noise_scores": noise_scores, "audit": audit}


def select_supervised(supervision: dict[str, Any], indices=None) -> dict[str, torch.Tensor]:
    """Return paired queries and value targets for efficient selected-query loss."""
    index = slice(None) if indices is None else indices
    return {"inputs": supervision["inputs"][index],
            "query_positions": supervision["selected_query_positions"][index],
            "targets": supervision["selected_targets"][index],
            "step_indices": supervision["selected_steps"][index]}


def masked_value_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean CE per selected value label; masks never supervise other slots.

    Accept full-clause logits [N,L,5,V] or value logits [N,L,V]. The mean
    normalization does not equate information budgets; count labels separately.
    """
    if logits.ndim == 4:
        if logits.shape[-2] != 5:
            raise ValueError("full-clause logits must contain five slots")
        logits = logits[:, :, 4, :]
    if logits.ndim != 3 or logits.shape[:2] != targets.shape or mask.shape != targets.shape:
        raise ValueError("expected logits [N,L,V] and matching targets/mask [N,L]")
    if mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("mask must be boolean and select at least one value")
    return F.cross_entropy(logits[mask], targets[mask])
