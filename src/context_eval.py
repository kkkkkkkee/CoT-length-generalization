"""Separate reasoning depth from disconnected context volume in LEGO models.

Compatible/incompatible distractors are conditioned on the oracle FIRST
transition only. They are controlled mechanism probes, not label-free tasks.
The base problem panel is paired across all conditions and inverse temperatures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from data import Task
from distractors import add_distractors
from evaluate import (teacher_forced_metrics, greedy_rollout, rollout_metrics,
                      aggregate_batches, _slice_sample, _hash_file, _write_outputs)
from model import load_model


def first_step_metrics(result):
    """Expose first-step rates while retaining the full by-position profile."""
    for kind in ("value", "variable", "clause"):
        result["metrics"][f"first_step_{kind}_accuracy"] = result["by_position"][f"{kind}_accuracy"][0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--counts", nargs="+", type=int, default=[0, 5, 20, 80, 160])
    parser.add_argument("--kinds", nargs="+", choices=["random", "compatible", "incompatible"],
                        default=["random", "compatible", "incompatible"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[1, 2, 4])
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--rollout-samples", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=930001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.samples, args.batch, args.length) < 1 or args.rollout_samples < 0 or min(args.counts) < 0:
        parser.error("invalid sample, batch, length or distractor count")
    torch.set_num_threads(4)
    model, state = load_model(args.checkpoint, args.device)
    model.eval()
    task = Task(state["args"]["task"], state["args"]["nvars"])
    panel_count = max(args.samples, args.rollout_samples)
    sample = task.generate(panel_count, args.length, torch.Generator().manual_seed(args.seed))
    result = {"schema_version": 1, "experiment": "disconnected_context_volume",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _hash_file(args.checkpoint), "checkpoint_epoch": state.get("epoch"),
        "task": task.name, "config": vars(args),
        "definitions": {
            "context_extra": "number of disconnected predicate clauses inserted before the answer block",
            "compatible": "oracle-designed distractor actions producing the correct FIRST transition value from its true initial state",
            "incompatible": "oracle-designed distractor actions producing an incorrect FIRST transition value from its true initial state",
            "compatibility_scope": "only the first logical transition; later steps are not conditioned",
            "OOD": "distractors reuse two off-trace variables with no initial value; outside original distinct-variable distribution",
            "zero_count": "shared baseline appears only once per inverse temperature, with kind=random",
            "temperature": "multiplier of attention logits (inverse temperature)",
            "pairing": "same base problem panel across counts, kinds and inverse temperatures",
            "rollout": "all five predicted slots are fed back; no type masks or oracle scaffolding",
        }, "evaluations": []}
    start_time = time.monotonic()
    for count in args.counts:
        kinds = ["random"] if count == 0 else args.kinds
        for kind in kinds:
            extended = add_distractors(sample, task, count,
                torch.Generator().manual_seed(args.seed + count * 100 + ["random", "compatible", "incompatible"].index(kind)), kind)
            prefix_length = args.length + count + 1
            for temperature in args.temperatures:
                item = {"length": args.length, "context_extra": count, "distractor_kind": kind,
                        "inverse_temperature": temperature, "prefix_clauses": prefix_length}
                teacher_results = []
                for offset in range(0, args.samples, args.batch):
                    chunk = _slice_sample(extended, offset, min(offset + args.batch, args.samples), args.device)
                    teacher_results.append(first_step_metrics(teacher_forced_metrics(model, chunk, temperature)))
                item["teacher_forced"] = aggregate_batches(teacher_results)
                if kind == "random" and temperature == 1 and args.rollout_samples:
                    rollout_results = []
                    for offset in range(0, args.rollout_samples, args.batch):
                        chunk = _slice_sample(extended, offset, min(offset + args.batch, args.rollout_samples), args.device)
                        generated = greedy_rollout(model, chunk["clauses"][:, :prefix_length], args.length, temperature)
                        targets = chunk["clauses"][:, prefix_length:]
                        rollout_results.append(first_step_metrics(rollout_metrics(generated, targets)))
                    item["rollout"] = aggregate_batches(rollout_results)
                result["evaluations"].append(item)
                result["elapsed_seconds"] = time.monotonic() - start_time
                _write_outputs(args.output, result)
                print(json.dumps({"length": args.length, "context_extra": count, "distractor_kind": kind,
                    "inverse_temperature": temperature,
                    "first_value": item["teacher_forced"]["metrics"]["first_step_value_accuracy"],
                    "value_chain": item["teacher_forced"]["metrics"]["value_chain_accuracy"],
                    "seconds": result["elapsed_seconds"]}), flush=True)
    result["complete"] = True
    _write_outputs(args.output, result)


if __name__ == "__main__":
    main()
