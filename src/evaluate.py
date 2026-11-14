"""Independent, unconstrained LEGO evaluation and attention diagnostics.

`temperature` follows model.py: it multiplies attention logits (inverse
softmax temperature).  Every greedy rollout feeds back all five predicted
slots.  No vocabulary masks, oracle variables, or oracle marker slots are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from data import Task, permute_predicates
from model import load_model


def _mean(value):
    return value.double().mean().item()


def _jsonable(value):
    return value.detach().cpu().tolist()


def _correctness(predictions, targets, prefix):
    """Integer counts plus rates, with sequence as the independent unit."""
    if predictions.shape != targets.shape or predictions.ndim != 3 or predictions.shape[-1] != 5:
        raise ValueError("predictions and targets must have identical shape (batch,length,5)")
    b, length, _ = targets.shape
    if b < 1 or length < 1:
        raise ValueError("evaluation requires nonempty batches and traces")
    match = predictions.eq(targets)
    value, variable, clause = match[:, :, 4], match[:, :, 3], match.all(-1)
    counts = {
        "token_correct": int(match.sum()), "value_correct": int(value.sum()),
        "variable_correct": int(variable.sum()), "clause_correct": int(clause.sum()),
        "value_chain_correct": int(value.all(-1).sum()),
        "variable_chain_correct": int(variable.all(-1).sum()),
        "clause_chain_correct": int(clause.all(-1).sum()),
        "final_value_correct": int(value[:, -1].sum()),
        "slot_correct": _jsonable(match.sum((0, 1))),
    }
    metrics = {
        "token_accuracy": _mean(match), "value_accuracy": _mean(value),
        "variable_accuracy": _mean(variable), "clause_accuracy": _mean(clause),
        "value_chain_accuracy": _mean(value.all(-1)),
        "variable_chain_accuracy": _mean(variable.all(-1)),
        "clause_chain_accuracy": _mean(clause.all(-1)),
        "final_value_accuracy": _mean(value[:, -1]),
        "slot_accuracy": _jsonable(match.double().mean((0, 1))),
    }
    by_position = {
        "value_accuracy": _jsonable(value.double().mean(0)),
        "variable_accuracy": _jsonable(variable.double().mean(0)),
        "clause_accuracy": _jsonable(clause.double().mean(0)),
    }
    # Histograms use steps 1..L; the last bin is right-censored (no error).
    positions = torch.arange(1, length + 1, device=targets.device)[None]
    for kind, correct in (("value", value), ("clause", clause)):
        first = torch.where(correct, length + 1, positions).amin(1)
        counts[f"first_{kind}_error_histogram"] = _jsonable(torch.bincount(first - 1, minlength=length + 1))
        metrics[f"first_{kind}_error_censored_mean"] = _mean(first)
    return {"samples": b, "length": length, "kind": prefix,
            "counts": counts, "metrics": metrics, "by_position": by_position}


@torch.inference_mode()
def teacher_forced_metrics(model, sample, temperature=1.0):
    clauses = sample["clauses"]
    query_positions = sample.get("query_positions")
    if query_positions is None:
        length = (clauses.shape[1] - 1) // 2
        if clauses.shape[1] != 2 * length + 1:
            raise ValueError("a complete sample must contain 2L+1 clauses or explicit query_positions")
        query_positions = torch.arange(length, 2 * length, device=clauses.device).expand(len(clauses), -1)
    else:
        length = query_positions.shape[1]
        if query_positions.shape[0] != len(clauses) or int(query_positions.max()) >= clauses.shape[1] - 1:
            raise ValueError("query_positions must have shape (batch,L), each followed by its target clause")
    logits, attention = model(clauses[:, :-1], temperature=temperature, return_attention=True)
    logits = logits.gather(1, query_positions[..., None, None].expand(-1, -1, 5, logits.shape[-1]))
    targets = clauses.gather(1, (query_positions + 1)[..., None].expand(-1, -1, 5))
    result = _correctness(logits.argmax(-1), targets, "teacher_forced")
    logp = logits.double().log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    result["metrics"].update({
        "value_joint_probability": logp[:, :, 4].sum(-1).exp().mean().item(),
        "all_slots_joint_probability": logp.sum((-1, -2)).exp().mean().item(),
        "value_mean_log_probability": logp[:, :, 4].mean().item(),
    })
    b, heads = attention.shape[:2]
    a = attention.gather(2, query_positions[:, None, :, None].expand(-1, heads, -1, attention.shape[-1])).double()
    predicate_positions = sample.get("target_predicate_positions", sample.get("predicate_positions"))
    if predicate_positions is None:
        predicate_positions = torch.arange(length, device=clauses.device).expand(b, -1)
    pred_index = predicate_positions[:, None, :, None].expand(-1, heads, -1, -1)
    self_index = query_positions[:, None, :, None].expand(b, heads, -1, -1)
    target_predicate = a.gather(-1, pred_index).squeeze(-1)
    own_answer = a.gather(-1, self_index).squeeze(-1)
    entropy = -(a * a.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(-1)
    # log attention ratios equal score differences; target and competitors share
    # a softmax denominator. Exclude self and target from the distractor maximum.
    log_a = a.clamp_min(torch.finfo(torch.float64).tiny).log()
    other_predicates = log_a.clone()
    is_answer = torch.arange(a.shape[-1], device=a.device)[None, :] >= query_positions[:, :1]
    other_predicates.masked_fill_(is_answer[:, None, None, :], -torch.inf)
    other_predicates.scatter_(-1, pred_index, -torch.inf)
    distractors = log_a.clone()
    distractors.scatter_(-1, pred_index, -torch.inf)
    distractors.scatter_(-1, self_index, -torch.inf)
    # Exclude future keys rather than treating their zero probability as finite.
    future = torch.arange(a.shape[-1], device=a.device)[None, None, :] > query_positions[:, :, None]
    distractors.masked_fill_(future[:, None], -torch.inf)
    log_target = target_predicate.clamp_min(torch.finfo(torch.float64).tiny).log()
    pred_margin = log_target - other_predicates.amax(-1)
    distractor_margin = log_target - distractors.amax(-1)
    quantities = {
        "target_predicate_weight": target_predicate,
        "own_answer_weight": own_answer,
        "relevant_weight": target_predicate + own_answer,
        "entropy_nats": entropy,
    }
    if "distractor_positions" in sample and sample["distractor_positions"].shape[1]:
        extra_indices = sample["distractor_positions"][:, None, None, :].expand(-1, heads, length, -1)
        quantities["added_distractor_weight"] = a.gather(-1, extra_indices).sum(-1)
    if length > 1:
        quantities["target_predicate_vs_best_other_predicate_log_margin"] = pred_margin
        quantities["target_predicate_vs_best_distractor_log_margin"] = distractor_margin
    result["attention"] = {
        "mean_per_head": {key: _jsonable(value.mean((0, 2))) for key, value in quantities.items()},
        "by_position_per_head": {key: _jsonable(value.mean(0)) for key, value in quantities.items()},
    }
    return result


@torch.inference_mode()
def greedy_rollout(model, prompt, length, temperature=1.0):
    """Generate `length` clauses; prompt contains predicates plus initial answer."""
    if prompt.ndim != 3 or prompt.shape[-1] != 5 or length < 1:
        raise ValueError("prompt must be (batch,prefix,5) and length must be positive")
    context = prompt.clone()
    generated = []
    cache = None
    cached_decode = getattr(model, "decode_last", None)
    for _ in range(length):
        if cached_decode is not None:
            logits, cache = cached_decode(context, cache=cache, temperature=temperature)
            next_clause = logits.argmax(-1)
        else:
            next_clause = model(context, temperature=temperature)[:, -1].argmax(-1)
        generated.append(next_clause)
        context = next_clause[:, None] if cached_decode is not None else torch.cat((context, next_clause[:, None]), 1)
    return torch.stack(generated, 1)


def rollout_metrics(predictions, targets):
    return _correctness(predictions, targets, "unconstrained_full_clause_rollout")


def _combine(values, weights, counts=False, key=""):
    first = values[0]
    if isinstance(first, dict):
        return {k: _combine([v[k] for v in values], weights, counts or k == "counts", k) for k in first}
    if isinstance(first, list):
        return [_combine([v[i] for v in values], weights, counts, key) for i in range(len(first))]
    if counts or key == "samples":
        return sum(values)
    if isinstance(first, (float, int)):
        if key.startswith("max_"):
            return max(values)
        if key == "length":
            return first
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)
    return first


def aggregate_batches(results):
    if not results:
        raise ValueError("cannot aggregate an empty result list")
    return _combine(results, [r["samples"] for r in results])


def _slice_sample(sample, start, stop, device):
    return {key: value[start:stop].to(device) for key, value in sample.items()}


@torch.inference_mode()
def permutation_check(model, original, permuted, temperature=1.0):
    length = (original["clauses"].shape[1] - 1) // 2
    a = model(original["clauses"][:, :-1], temperature=temperature)[:, length:]
    b = model(permuted["clauses"][:, :-1], temperature=temperature)[:, length:]
    return {"samples": len(a),
            "argmax_token_agreement": _mean(a.argmax(-1).eq(b.argmax(-1))),
            "max_abs_logit_difference": (a - b).abs().max().item(),
            "mean_abs_logit_difference": (a - b).abs().double().mean().item()}


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_outputs(path, result):
    path = Path(path)
    if path.suffix == ".json":
        json_path, csv_path = path, path.with_suffix(".rows.csv")
    else:
        json_path, csv_path = path / "evaluation.json", path / "rows.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
    tmp.replace(json_path)
    rows = []
    for item in result["evaluations"]:
        row = {"length": item["length"], "inverse_temperature": item["inverse_temperature"]}
        row.update({key: item[key] for key in ("context_extra", "distractor_kind", "prefix_clauses") if key in item})
        for kind in ("teacher_forced", "rollout"):
            if kind in item:
                row[f"{kind}_samples"] = item[kind]["samples"]
                row.update({f"{kind}_{key}": value for key, value in item[kind]["metrics"].items() if isinstance(value, (int, float))})
        for key, values in item["teacher_forced"]["attention"]["mean_per_head"].items():
            row.update({f"head{head}_{key}": value for head, value in enumerate(values)})
        if "predicate_permutation" in item:
            row.update({f"permutation_{key}": value for key, value in item["predicate_permutation"].items()})
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--rollout-samples", type=int, default=128)
    parser.add_argument("--lengths", nargs="+", type=int, default=[5, 10, 20, 40, 80, 160])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.5, 1, 2, 4])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=730001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.samples, args.batch) < 1 or args.rollout_samples < 0:
        parser.error("samples and batch must be positive; rollout-samples must be nonnegative")
    if any(x <= 0 or not math.isfinite(x) for x in args.temperatures):
        parser.error("inverse temperatures must be positive and finite")
    torch.set_num_threads(4)
    model, state = load_model(args.checkpoint, args.device)
    model.eval()
    task = Task(state["args"]["task"], state["args"]["nvars"])
    result = {"schema_version": 1, "checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_sha256": _hash_file(args.checkpoint), "checkpoint_epoch": state.get("epoch"),
              "task": task.name, "config": vars(args),
              "definitions": {"temperature": "attention logits multiplier (inverse temperature)",
                 "teacher_forced": "all answers are conditioned on ground-truth prior clauses",
                 "rollout": "all five predicted slots are fed back without masks or oracle scaffolding",
                 "joint_probability": "product of teacher-forced correct-token probabilities; diagnostic, not greedy accuracy",
                 "first_error_histogram": "bins correspond to steps 1..L; last bin means no error (right-censored)",
                 "attention_log_margin": "log attention ratio; multiplied score difference at the specified inverse temperature"},
              "evaluations": []}
    start_time = time.monotonic()
    permutation_lengths = {min(args.lengths), max(args.lengths)} | ({20} & set(args.lengths))
    for length in args.lengths:
        panel_count = max(args.samples, args.rollout_samples)
        sample = task.generate(panel_count, length, torch.Generator().manual_seed(args.seed + length))
        permuted = permute_predicates(sample, torch.Generator().manual_seed(args.seed + 10000 + length))
        for temperature in args.temperatures:
            item = {"length": length, "inverse_temperature": temperature}
            teacher_results, permutation_results = [], []
            for offset in range(0, args.samples, args.batch):
                chunk = _slice_sample(sample, offset, min(offset + args.batch, args.samples), args.device)
                teacher_results.append(teacher_forced_metrics(model, chunk, temperature))
                if length in permutation_lengths and temperature == 1:
                    reordered = _slice_sample(permuted, offset, min(offset + args.batch, args.samples), args.device)
                    permutation_results.append(permutation_check(model, chunk, reordered, temperature))
            item["teacher_forced"] = aggregate_batches(teacher_results)
            if permutation_results:
                item["predicate_permutation"] = aggregate_batches(permutation_results)
            do_rollout = temperature == 1 or (length == max(args.lengths) and temperature in (2, 4))
            if do_rollout and args.rollout_samples:
                rollout_results = []
                for offset in range(0, args.rollout_samples, args.batch):
                    chunk = _slice_sample(sample, offset, min(offset + args.batch, args.rollout_samples), args.device)
                    generated = greedy_rollout(model, chunk["clauses"][:, :length + 1], length, temperature)
                    rollout_results.append(rollout_metrics(generated, chunk["clauses"][:, length + 1:]))
                item["rollout"] = aggregate_batches(rollout_results)
            result["evaluations"].append(item)
            result["elapsed_seconds"] = time.monotonic() - start_time
            _write_outputs(args.output, result)
            print(json.dumps({"length": length, "inverse_temperature": temperature,
                "tf_value_chain": item["teacher_forced"]["metrics"]["value_chain_accuracy"],
                "rollout_final_value": item.get("rollout", {}).get("metrics", {}).get("final_value_accuracy"),
                "seconds": result["elapsed_seconds"]}), flush=True)
    result["complete"] = True
    _write_outputs(args.output, result)


if __name__ == "__main__":
    main()
