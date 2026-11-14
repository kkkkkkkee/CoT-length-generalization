"""Exhaustive action/state evaluation in a two-clause (L=1) context.

This is a short-context local-competency diagnostic, not a direct FFN truth
table: the model still performs attention, normalization and residual updates.
Success/failure here need not match the decision regions used in long contexts.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import Task
from model import load_model


@torch.no_grad()
def evaluate_local_rule(model, task, n_variable_pairs=8, seed=73001, batch_size=256):
    if n_variable_pairs < 1 or batch_size < 1:
        raise ValueError("n_variable_pairs and batch_size must be positive")
    device = next(model.parameters()).device
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    variable_pairs = task.generate(n_variable_pairs, 1, generator)["variables"]

    # Pair-major enumeration: every action/state combination shares the same
    # sampled variable pairs, so comparisons do not change those nuisance inputs.
    combinations = task.nactions * task.nstates
    actions = torch.arange(task.nactions).repeat_interleave(task.nstates).repeat(n_variable_pairs)
    initial = torch.arange(task.nstates).repeat(task.nactions * n_variable_pairs)
    variables = variable_pairs.repeat_interleave(combinations, dim=0)
    final = task.action_table[actions, initial]
    clauses = torch.zeros((len(actions), 3, 5), dtype=torch.long)
    clauses[:, 0, 0] = variables[:, 1]
    clauses[:, 0, 1] = actions + task.action_offset
    clauses[:, 0, 2] = variables[:, 0]
    clauses[:, 1:, 3] = variables
    clauses[:, 1, 4] = initial + task.value_offset
    clauses[:, 2, 4] = final + task.value_offset

    predictions, value_losses = [], []
    for batch in clauses.split(batch_size):
        batch = batch.to(device)
        logits = model(batch[:, :2])[:, -1]
        predictions.append(logits.argmax(-1).cpu())
        value_losses.append(F.cross_entropy(logits[:, 4], batch[:, 2, 4], reduction="none").cpu())
    predicted = torch.cat(predictions)
    targets = clauses[:, 2]
    value_ok = predicted[:, 4].eq(targets[:, 4])
    variable_ok = predicted[:, 3].eq(targets[:, 3])
    clause_ok = predicted.eq(targets).all(-1)
    shape = (n_variable_pairs, task.nactions, task.nstates)

    def errors(ok):
        return (~ok).reshape(shape).sum(0).tolist()

    return {
        "task": task.name,
        "length": 1,
        "nvars": task.nvars,
        "nactions": task.nactions,
        "nstates": task.nstates,
        "seed": seed,
        "n_variable_pairs": n_variable_pairs,
        "variable_pairs_token_ids": variable_pairs.tolist(),
        "examples": len(clauses),
        "value_accuracy": value_ok.float().mean().item(),
        "variable_accuracy": variable_ok.float().mean().item(),
        "full_clause_accuracy": clause_ok.float().mean().item(),
        "value_cross_entropy": torch.cat(value_losses).mean().item(),
        "action_state_value_error_counts": errors(value_ok),
        "action_state_variable_error_counts": errors(variable_ok),
        "action_state_clause_error_counts": errors(clause_ok),
        "action_state_next_state": task.action_table.tolist(),
        "matrix_axes": "rows: zero-based action index; columns: zero-based initial state",
        "matrix_denominator_per_cell": n_variable_pairs,
        "interpretation": (
            "Exhaustive action/state combinations in L=1 contexts, using sampled variable pairs. "
            "This measures short-context local competency, not the isolated FFN or all variable pairs. "
            "L=1 may differ from the training context; success or failure need not match the "
            "decision regions reached during longer-context reasoning."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-variable-pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_model(args.checkpoint, args.device)
    training_args = checkpoint["args"]
    task = Task(training_args["task"], nvars=training_args["nvars"])
    if model.config.vocab_size != task.vocab_size:
        raise ValueError("Checkpoint model vocabulary does not match its stored task configuration")
    result = evaluate_local_rule(model, task, args.n_variable_pairs, args.seed, args.batch)
    checkpoint_path = Path(args.checkpoint)
    result.update({
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "training_epoch": checkpoint.get("epoch"),
        "training_length": training_args.get("length"),
        "training_seed": training_args.get("seed"),
        "device": args.device,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "task", "examples", "value_accuracy", "variable_accuracy", "full_clause_accuracy", "value_cross_entropy"
    )}))


if __name__ == "__main__":
    main()
