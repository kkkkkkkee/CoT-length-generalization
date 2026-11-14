"""Evaluate fixed independent panels across training checkpoints.

Defaults target ID saturation followed by OOD/margin improvement: epochs
0,5,10,15,20,25,50,100,200,300 at lengths 5,40,160, inverse temperature 1.
This script does not train or perform sequential rollout. Full-clause TF
trajectory correctness equals the deterministic greedy exact-trajectory event;
the separate value/variable trajectory metrics retain ground-truth prefixes.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import torch

from data import Task
from evaluate import aggregate_batches, teacher_forced_metrics
from local_rule import evaluate_local_rule
from model import load_model


DEFAULT_EPOCHS = [0, 5, 10, 15, 20, 25, 50, 100, 200, 300]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_outputs(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    rows = []
    for item in result["evaluations"]:
        panel = item["teacher_forced"]
        row = {"epoch": item["epoch"], "length": item["length"],
               "samples": panel["samples"], "inverse_temperature": 1.0}
        row.update({key: value for key, value in panel["metrics"].items()
                    if isinstance(value, (int, float))})
        for quantity, values in panel["attention"]["mean_per_head"].items():
            row.update({f"head{index}_{quantity}": value for index, value in enumerate(values)})
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    csv_path = path.with_suffix(".csv")
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Directory containing checkpoint-NNNN.pt files")
    parser.add_argument("--output", help="JSON path; default RUN/dynamics.json")
    parser.add_argument("--epochs", nargs="+", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lengths", nargs="+", type=int, default=[5, 40, 160])
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--long-samples", type=int, default=128,
                        help="Samples for lengths >= 160")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=830001,
                        help="Independent panel seed; actual length seed is seed+length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-rule", action="store_true",
                        help="Also run L1 local-rule diagnostic on the last available selected checkpoint")
    parser.add_argument("--n-variable-pairs", type=int, default=8)
    args = parser.parse_args()
    if min(args.samples, args.long_samples, args.batch, args.n_variable_pairs) < 1:
        parser.error("sample counts, batch and n-variable-pairs must be positive")
    if min(args.epochs) < 0 or min(args.lengths) < 1:
        parser.error("epochs must be nonnegative and lengths positive")
    torch.set_num_threads(4)
    run = Path(args.run)
    output = Path(args.output) if args.output else run / "dynamics.json"
    checkpoints = [(epoch, run / f"checkpoint-{epoch:04d}.pt") for epoch in sorted(set(args.epochs))]
    available = [(epoch, path) for epoch, path in checkpoints if path.is_file()]
    missing = [epoch for epoch, path in checkpoints if not path.is_file()]
    if not available:
        parser.error("No requested checkpoints exist")
    if missing:
        print(json.dumps({"missing_checkpoint_epochs": missing}), flush=True)

    first_model, first_state = load_model(available[0][1], args.device)
    task = Task(first_state["args"]["task"], first_state["args"]["nvars"])
    panels = {}
    panel_info = {}
    for length in sorted(set(args.lengths)):
        n = args.long_samples if length >= 160 else args.samples
        sample = task.generate(n, length, torch.Generator().manual_seed(args.seed + length))
        panels[length] = sample
        panel_info[str(length)] = {
            "samples": n, "seed": args.seed + length,
            "clauses_sha256": hashlib.sha256(sample["clauses"].numpy().tobytes()).hexdigest(),
        }
    source_dir = Path(__file__).parent
    result = {
        "schema_version": 1, "run": str(run.resolve()), "task": task.name,
        "config": vars(args), "training_seed": first_state["args"].get("seed"),
        "training_length": first_state["args"].get("length"),
        "panels": panel_info, "checkpoints": [], "evaluations": [],
        "missing_checkpoint_epochs": missing,
        "source_sha256": {name: file_hash(source_dir / name)
                          for name in ("dynamics.py", "evaluate.py", "model.py", "data.py", "local_rule.py")},
        "definitions": {
            "fixed_panels": "Each length uses the same independent examples at every checkpoint; panels differ across lengths.",
            "temperature": "Fixed attention logits multiplier (inverse temperature) of 1.",
            "value_variable_chain": "All steps correct with ground-truth prior full clauses; not separate free-rollout success.",
            "clause_chain": "All slots at all steps correct; equals deterministic full greedy exact-trajectory event.",
            "attention": "Per-head target predicate, current answer, relevant mass, entropy and log-attention-ratio margins from evaluate.py.",
            "dynamics": "Correlation of OOD accuracy and retrieval margins over training is observational; use interventions for causality.",
        },
    }
    start = time.monotonic()
    for index, (epoch, path) in enumerate(available):
        if index == 0:
            model, state = first_model, first_state
            del first_model, first_state
        else:
            model, state = load_model(path, args.device)
        model.eval()
        if (state["args"]["task"].lower() != task.name.lower()
                or state["args"]["nvars"] != task.nvars
                or model.config.vocab_size != task.vocab_size):
            raise ValueError(f"Checkpoint {path} has inconsistent task metadata")
        checkpoint_hash = file_hash(path)
        result["checkpoints"].append({"epoch": epoch, "path": str(path.resolve()), "sha256": checkpoint_hash})
        for length, sample in panels.items():
            batches = []
            n = len(sample["clauses"])
            for offset in range(0, n, args.batch):
                batch = {key: value[offset:offset + args.batch].to(args.device) for key, value in sample.items()}
                batches.append(teacher_forced_metrics(model, batch, temperature=1.0))
            metrics = aggregate_batches(batches)
            result["evaluations"].append({"epoch": epoch, "length": length, "inverse_temperature": 1.0,
                                           "teacher_forced": metrics})
            result["elapsed_seconds"] = time.monotonic() - start
            write_outputs(output, result)
            print(json.dumps({"epoch": epoch, "length": length,
                              "value_accuracy": metrics["metrics"]["value_accuracy"],
                              "value_chain_accuracy": metrics["metrics"]["value_chain_accuracy"],
                              "clause_chain_accuracy": metrics["metrics"]["clause_chain_accuracy"],
                              "seconds": result["elapsed_seconds"]}), flush=True)
        if args.local_rule and index == len(available) - 1:
            local = evaluate_local_rule(model, task, n_variable_pairs=args.n_variable_pairs,
                                        seed=args.seed + 10000, batch_size=args.batch)
            local.update({"checkpoint": str(path.resolve()), "checkpoint_sha256": checkpoint_hash,
                          "training_epoch": epoch, "training_length": state["args"].get("length")})
            result["local_rule_last_selected_checkpoint"] = local
        del model, state
    result["complete"] = not missing
    result["all_available_panels_evaluated"] = True
    result["elapsed_seconds"] = time.monotonic() - start
    write_outputs(output, result)


if __name__ == "__main__":
    main()
