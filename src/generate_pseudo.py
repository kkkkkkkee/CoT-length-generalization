"""Generate unfiltered, full-clause greedy pseudo-labels for a longer LEGO task.

Ground truth is stored solely for post-hoc auditing. The training entry point
reads only `clauses`; no example is selected, corrected, or filtered using truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from data import Task
from evaluate import aggregate_batches, greedy_rollout, rollout_metrics, _hash_file
from model import load_model


@torch.inference_mode()
def generate_pseudo_dataset(model, task, length, samples, batch=32, seed=830001,
                            temperature=1.0, device=None):
    if min(length, samples, batch) < 1:
        raise ValueError("length, samples and batch must be positive")
    if device is None:
        device = next(model.parameters()).device
    truth = task.generate(samples, length, torch.Generator().manual_seed(seed))
    pseudo = truth["clauses"].clone()
    diagnostics = []
    for offset in range(0, samples, batch):
        stop = min(offset + batch, samples)
        prompt = truth["clauses"][offset:stop, :length + 1].to(device)
        generated = greedy_rollout(model, prompt, length, temperature)
        pseudo[offset:stop, length + 1:] = generated.cpu()
        diagnostics.append(rollout_metrics(generated.cpu(), truth["clauses"][offset:stop, length + 1:]))
    return {
        "clauses": pseudo,
        "ground_truth_clauses": truth["clauses"],
        "ground_truth_variables": truth["variables"],
        "ground_truth_actions": truth["actions"],
        "ground_truth_states": truth["states"],
        "audit": aggregate_batches(diagnostics),
        "metadata": {"task": task.name, "nvars": task.nvars, "length": length,
            "samples": samples, "batch": batch, "seed": seed,
            "inverse_temperature": temperature,
            "label_source": "unconstrained greedy full-clause autoregression",
            "ground_truth_usage": "post-hoc quality audit only; no filtering or correction",
            "retained_samples": samples, "filtered_samples": 0,
            "training_field": "clauses"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=830001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, state = load_model(args.checkpoint, args.device)
    model.eval()
    task = Task(state["args"]["task"], state["args"]["nvars"])
    start = time.monotonic()
    payload = generate_pseudo_dataset(model, task, args.length, args.samples, args.batch,
                                     args.seed, args.temperature, args.device)
    payload["metadata"].update({"checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _hash_file(args.checkpoint), "checkpoint_epoch": state.get("epoch"),
        "seconds": time.monotonic() - start})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(output)
    output.with_suffix(".audit.json").write_text(json.dumps({"metadata": payload["metadata"],
                                                           "audit": payload["audit"]}, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "metadata": payload["metadata"],
                      "metrics": payload["audit"]["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
