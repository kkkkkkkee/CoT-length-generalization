"""Paired oracle attention interventions for the finite sparse-supervision model.

This is a teacher-forced mechanism diagnostic, not an inference algorithm:
interventions use the TRUE logical rule position. No target labels enter the
model. Predictions cover all five slots, with a causal mask at every query.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path

import torch

try:
    from .data import Task
    from .sparse_data import tensor_hash
    from .sparse_model import load_model
except ImportError:  # Direct command-line execution from src/.
    from data import Task
    from sparse_data import tensor_hash
    from sparse_model import load_model


ATTENTION_CONDITIONS = ("original", "reconstruct_sham", "balance_only", "concentrate_only",
                        "both", "random_pair_both")
Q_CONDITIONS = ("q_diagonal_only", "q_row_mean_background")
CONDITIONS = ATTENTION_CONDITIONS + Q_CONDITIONS


@contextmanager
def temporary_q_structure(model, condition):
    """Temporarily modify both Q blocks, preserving diagonals and restoring state.

    These post-training structural diagnostics use no attention override. The
    original matrices are restored even when an exception interrupts a probe.
    """
    if condition not in Q_CONDITIONS:
        raise ValueError("unknown Q structure intervention")
    originals = {name: getattr(model, name).detach().clone() for name in ("Q43", "Q44")}
    try:
        with torch.no_grad():
            for name, original in originals.items():
                size = original.shape[0]
                diagonal = original.diagonal().clone()
                if condition == "q_diagonal_only":
                    altered = torch.diag(diagonal)
                elif size > 1:
                    offdiagonal_mean = (original.sum(-1) - diagonal) / (size - 1)
                    altered = offdiagonal_mean[:, None].expand_as(original).clone()
                    altered.diagonal().copy_(diagonal)
                else:
                    altered = original.clone()
                getattr(model, name).copy_(altered)
        yield
    finally:
        with torch.no_grad():
            for name, original in originals.items():
                getattr(model, name).copy_(original)


def _positions(value, batch, queries, device):
    result = torch.as_tensor(value, dtype=torch.long, device=device)
    if result.ndim == 1:
        result = result[None].expand(batch, -1)
    if result.shape != (batch, queries):
        raise ValueError("positions must have shape [J] or [B,J]")
    return result


def intervene_attention(attention, rule_positions, query_positions, condition,
                        concentration=0.99, seed=0, sample_offset=0):
    """Return valid causal weights and per-query intervention diagnostics.

    Relevant mass m=a_rule+a_self; ratio r=a_rule/m. The background keeps its
    internal proportions. If background mass underflows to exactly zero but
    background keys exist, its undefined proportions use an explicitly counted
    uniform fallback. A positive desired background mass with NO background
    keys raises, rather than introducing an unreported token.

    random_pair_both transfers weight from a uniformly random visible donor
    key to a different random visible recipient. Requested L1 is the both
    intervention's L1, capped by donor mass for feasibility. ACTUAL L1 and the
    fraction of capped rows are recorded. This is NOT exactly matched noise.
    Per-sample RNG makes it invariant to evaluation batch size.
    """
    if attention.ndim != 4 or condition not in ATTENTION_CONDITIONS:
        raise ValueError("expected attention [B,H,J,T] and a known condition")
    if not 0 <= concentration <= 1:
        raise ValueError("concentration must be in [0,1]")
    b, h, j, t = attention.shape
    rule = _positions(rule_positions, b, j, attention.device)
    query = _positions(query_positions, b, j, attention.device)
    if bool(((rule < 0) | (query < 0) | (rule >= t) | (query >= t) | (rule > query) | (rule == query)).any()):
        raise ValueError("rule and current-answer keys must be distinct visible positions")
    a = attention.double()
    visible = torch.arange(t, device=a.device)[None, None, None, :] <= query[:, None, :, None]
    visible = visible.expand(b, h, j, t)
    if not torch.isfinite(a).all() or bool((a < 0).any()):
        raise ValueError("attention must be finite and nonnegative")
    if bool((a.masked_select(~visible) != 0).any()):
        raise ValueError("attention cannot include future keys")
    if not torch.allclose(a.sum(-1), torch.ones_like(a[..., 0]), atol=1e-6, rtol=1e-6):
        raise ValueError("attention rows must sum to one")
    ri = rule[:, None, :, None].expand(b, h, j, 1)
    qi = query[:, None, :, None].expand(b, h, j, 1)
    ap, ay = a.gather(-1, ri).squeeze(-1), a.gather(-1, qi).squeeze(-1)
    m = ap + ay
    r = torch.where(m > 0, ap / m.clamp_min(torch.finfo(a.dtype).tiny), 0.5)
    background = visible.clone().scatter_(-1, ri, False).scatter_(-1, qi, False)
    background_sum = (a * background).sum(-1)
    background_count = background.sum(-1)
    fallback = torch.zeros_like(m, dtype=torch.bool)
    capped = torch.zeros_like(m, dtype=torch.bool)
    requested_l1 = torch.zeros_like(m)
    if condition == "original":
        changed = a.clone()
    elif condition == "random_pair_both":
        goal, _ = intervene_attention(attention, rule, query, "both", concentration,
                                      seed, sample_offset)
        requested_l1 = (goal.double() - a).abs().sum(-1)
        # Generate donor/receiver draws by sample ID, without relying on batch
        # grouping. Exactly two keys change; no projection or renormalization.
        draws = torch.stack([torch.rand((h, j, 2), dtype=torch.float64,
            generator=torch.Generator().manual_seed(seed + 104729 * (sample_offset + i)))
            for i in range(b)]).to(a.device)
        count = query[:, None, :].expand(b, h, j) + 1
        donor = (draws[..., 0] * count).long()
        distance = 1 + (draws[..., 1] * (count - 1)).long()
        recipient = (donor + distance) % count
        donor_mass = a.gather(-1, donor[..., None]).squeeze(-1)
        transfer = torch.minimum(requested_l1 / 2, donor_mass)
        capped = transfer + 1e-12 < requested_l1 / 2
        changed = a.clone()
        changed.scatter_add_(-1, donor[..., None], -transfer[..., None])
        changed.scatter_add_(-1, recipient[..., None], transfer[..., None])
    else:
        desired_m = torch.full_like(m, concentration) if condition in {"concentrate_only", "both"} else m
        desired_r = torch.full_like(r, 0.5) if condition in {"balance_only", "both"} else r
        desired_background = (1 - desired_m).clamp_min(0)
        if bool(((background_count == 0) & (desired_background > 1e-12)).any()):
            raise ValueError("cannot allocate background mass without background keys")
        fallback = (background_sum == 0) & (background_count > 0) & (desired_background > 0)
        proportions = a * background / background_sum.clamp_min(torch.finfo(a.dtype).tiny)[..., None]
        uniform = background / background_count.clamp_min(1)[..., None]
        proportions = torch.where(fallback[..., None], uniform, proportions)
        changed = proportions * desired_background[..., None]
        changed.scatter_(-1, ri, (desired_m * desired_r)[..., None])
        changed.scatter_(-1, qi, (desired_m * (1 - desired_r))[..., None])
        requested_l1 = (changed - a).abs().sum(-1)
    changed = changed.to(attention.dtype)
    actual = changed.double()
    new_ap = actual.gather(-1, ri).squeeze(-1)
    new_ay = actual.gather(-1, qi).squeeze(-1)
    new_m = new_ap + new_ay
    new_r = torch.where(new_m > 0, new_ap / new_m.clamp_min(torch.finfo(a.dtype).tiny), 0.5)
    diagnostics = {
        "original_relevant_mass": m, "original_rule_share": r,
        "relevant_mass": new_m, "rule_share": new_r,
        "rule_weight": new_ap, "current_answer_weight": new_ay,
        "absolute_rule_state_gap": (new_ap - new_ay).abs(),
        "l1_change": (actual - a).abs().sum(-1), "requested_l1": requested_l1,
        "random_transfer_capped": capped,
        "zero_background_uniform_fallback": fallback,
        "original_zero_relevant_mass": m == 0,
    }
    return changed, diagnostics


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _correctness(predictions, targets):
    match = predictions.eq(targets)
    value, variable, clause = match[..., 4], match[..., 3], match.all(-1)
    n, length = value.shape
    arrays = {"value_correct": value, "variable_correct": variable,
              "clause_correct": clause}
    metrics = {}
    for name, correct in list(arrays.items()):
        prefix = name.removesuffix("_correct")
        metrics[f"{prefix}_step_accuracy"] = correct.double().mean().item()
        metrics[f"{prefix}_chain_accuracy"] = correct.all(-1).double().mean().item()
        first = torch.where(correct, length + 1,
                            torch.arange(1, length + 1)[None]).amin(-1)
        arrays[f"first_{prefix}_error_or_censored"] = first
    metrics["token_accuracy"] = match.double().mean().item()
    return metrics, arrays


@torch.inference_mode()
def probe_model(model, task, lengths=(8, 16), samples=512, seed=1950001,
                batch=32, concentration=0.99, conditions=CONDITIONS):
    """Run fixed independent panels with paired conditions, never training."""
    if samples < 1 or batch < 1 or any(length < 2 or length >= task.nvars for length in lengths):
        raise ValueError("positive samples/batch and 2<=length<nvars required")
    if not {"original", "reconstruct_sham"}.issubset(conditions):
        raise ValueError("original and reconstruct_sham controls are required")
    device = next(model.parameters()).device
    model.eval()
    panels = []
    for length in lengths:
        sample = task.generate(samples, length, torch.Generator().manual_seed(seed + length))
        # Targets are kept on CPU and never supplied to attention intervention
        # or forward_queries. Full input history remains causally masked.
        targets = sample["clauses"][:, length + 1:].clone()
        all_inputs = sample["clauses"][:, :-1].clone()
        predictions = {c: [] for c in conditions}
        diagnostic_batches = {c: {} for c in conditions}
        sham_logit_delta = 0.0
        for offset in range(0, samples, batch):
            inputs = all_inputs[offset:offset + batch].to(device)
            queries = torch.arange(length, 2 * length, device=device)
            rules = torch.arange(length, device=device)
            original_logits, original_attention = model.forward_queries(inputs, queries, return_attention=True)
            for condition in conditions:
                if condition in Q_CONDITIONS:
                    with temporary_q_structure(model, condition):
                        logits, changed_attention = model.forward_queries(inputs, queries, return_attention=True)
                    _, diag = intervene_attention(changed_attention, rules, queries, "original")
                    _, original_diag = intervene_attention(original_attention, rules, queries, "original")
                    diag["original_relevant_mass"] = original_diag["original_relevant_mass"]
                    diag["original_rule_share"] = original_diag["original_rule_share"]
                    diag["l1_change"] = (changed_attention.double() - original_attention.double()).abs().sum(-1)
                    diag["original_zero_relevant_mass"] = original_diag["original_zero_relevant_mass"]
                else:
                    weights, diag = intervene_attention(original_attention, rules, queries, condition,
                        concentration=concentration, seed=seed + 31 * length, sample_offset=offset)
                    logits = original_logits if condition == "original" else model.forward_queries(
                        inputs, queries, attention_override=weights)
                if condition == "reconstruct_sham":
                    sham_logit_delta = max(sham_logit_delta, (logits - original_logits).abs().max().item())
                predictions[condition].append(logits.argmax(-1).cpu())
                for key, value in diag.items():
                    diagnostic_batches[condition].setdefault(key, []).append(value.cpu())
        rows = []
        original_arrays = None
        for condition in conditions:
            pred = torch.cat(predictions[condition])
            metrics, arrays = _correctness(pred, targets)
            if condition == "original":
                original_arrays = arrays
            diag = {key: torch.cat(parts) for key, parts in diagnostic_batches[condition].items()}
            attention_summary = {key: {"mean": value.double().mean().item(),
                "min": value.min().item(), "max": value.max().item(),
                "by_position_mean": value.double().mean((0, 1)).tolist()}
                for key, value in diag.items()}
            rows.append({"condition": condition, "metrics": metrics,
                "per_sample": {key: value.to(torch.int64).tolist() for key, value in arrays.items()},
                "attention": attention_summary,
                "per_sample_attention": {key: value[:, 0].tolist() for key, value in diag.items()}})
        # Compute paired changes even if a caller supplied a different order.
        original_arrays = next(row["per_sample"] for row in rows if row["condition"] == "original")
        original_chain = torch.tensor(original_arrays["clause_correct"], dtype=torch.bool).all(-1)
        for row in rows:
            current_chain = torch.tensor(row["per_sample"]["clause_correct"], dtype=torch.bool).all(-1)
            row["paired_with_original"] = {
                "original_wrong_now_correct": int((~original_chain & current_chain).sum()),
                "original_correct_now_wrong": int((original_chain & ~current_chain).sum()),
                "both_correct": int((original_chain & current_chain).sum()),
                "both_wrong": int((~original_chain & ~current_chain).sum()),
                "clause_chain_accuracy_difference": (current_chain.double() - original_chain.double()).mean().item(),
            }
        panels.append({"length": length, "samples": samples, "data_seed": seed + length,
            "sample_clauses_sha256": tensor_hash(sample["clauses"]),
            "inputs_sha256": tensor_hash(all_inputs), "targets_sha256": tensor_hash(targets),
            "sample_ids": list(range(samples)), "sham_max_abs_logit_difference": sham_logit_delta,
            "conditions": rows})
    return {"schema_version": 1, "experiment": "oracle_attention_mass_balance_probe",
        "task": task.name, "nvars": task.nvars, "samples": samples, "seed": seed,
        "batch": batch, "concentration": concentration, "panels": panels,
        "definitions": {
            "scope": "Teacher-forced causal five-slot predictions only; no free rollout was run",
            "oracle": "true rule positions are used for mechanism diagnosis, not a deployable inference method",
            "chain": "all five slots at all steps correct; value-chain alone is not full-clause trajectory correctness",
            "background": "all other visible keys preserve relative attention, except counted uniform fallback when original background mass is exactly zero",
            "random_control": "random_pair_both requests both intervention L1 but caps transfer at a random donor mass; actual L1 is recorded; this is NOT exactly amplitude matched",
            "q_structure": "q_diagonal_only zeros both Q blocks' offdiagonals; q_row_mean_background replaces each row's offdiagonals by that row's offdiagonal mean. Diagonals remain fixed; original Q is restored; no attention override is applied. These are post-training structure diagnostics, not a claim of a cost-free algorithmic improvement",
            "sham": "reconstruct original m and r through the intervention code; compare logits and predictions with original",
            "targets": "targets remain outside the model and intervention; causal mask hides later answer slots in supplied training-style history",
            "first_error": "steps numbered 1..L; L+1 means no error (right censored)",
            "zero_relevant_mass": "r defaults to 0.5 only where original relevant mass is exactly zero; counted in diagnostics",
        }, "complete": True}


def save_results(output, result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "probe.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(target)
    rows = []
    for panel in result["panels"]:
        for row in panel["conditions"]:
            flat = {"length": panel["length"], "samples": panel["samples"],
                    "condition": row["condition"], **row["metrics"],
                    **row["paired_with_original"]}
            flat.update({f"attention_{k}_mean": v["mean"] for k, v in row["attention"].items()})
            flat["sample_clauses_sha256"] = panel["sample_clauses_sha256"]
            rows.append(flat)
    with (output / "probe.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1950001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--concentration", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.output / "probe.json").exists():
        parser.error("refusing to overwrite an existing probe.json")
    torch.set_num_threads(4)
    model, state = load_model(args.checkpoint, args.device)
    task = Task(state["args"]["task"], model.config.nvars)
    result = probe_model(model, task, args.lengths, args.samples, args.seed,
                         args.batch, args.concentration)
    result.update({"checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _hash_file(args.checkpoint),
        "model_config": model.config_dict(),
        "source_sha256": {name: _hash_file(Path(__file__).with_name(name))
            for name in ["sparse_probe.py", "sparse_model.py", "sparse_data.py", "data.py"]}})
    save_results(args.output, result)
    print(json.dumps({"output": str(args.output), "complete": True,
        "panels": [{"length": p["length"], "sham_max_abs_logit_difference": p["sham_max_abs_logit_difference"]}
                   for p in result["panels"]]}))


if __name__ == "__main__":
    main()
