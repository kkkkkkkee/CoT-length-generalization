"""Create reproducible PNG/PDF figures from training and evaluation JSON.

Only recorded observations are plotted. Bands are +/- one sample standard
deviation across independently trained seeds, not confidence intervals over
individual reasoning steps. Pilots are excluded by the default run glob.
Usage: python src/analyze.py --root . --run-glob 'base_*'
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fnmatch
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"C6": "#0072B2", "C5": "#009E73", "S5": "#D55E00", "aliasC5": "#CC79A7"}
METRICS = {
    "value_step": "Teacher-forced value accuracy",
    "value_chain": "Teacher-forced value-chain accuracy",
    "clause_chain": "Exact full-clause chain accuracy",
    "value_accuracy": "Teacher-forced value accuracy",
    "value_chain_accuracy": "Teacher-forced value-chain accuracy",
    "clause_chain_accuracy": "Exact full-clause chain accuracy",
    "final_value_accuracy": "Free-rollout final-value accuracy",
    "value_joint_probability": "Joint probability of true values",
}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_history(path):
    """Ignore an incomplete final line if plotting while training is active."""
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def collect_runs(root, run_glob):
    records = []
    for path in sorted((root / "runs").glob("*/history.jsonl")):
        if not fnmatch.fnmatch(path.parent.name, run_glob):
            continue
        config = read_json(path.parent / "config.json") or {}
        records.append({"path": path.parent, "name": path.parent.name,
                        "task": config.get("task", path.parent.name),
                        "seed": config.get("seed"), "train_length": config.get("length"),
                        "config": config, "history": read_history(path)})
    return records


def collect_evaluations(root, run_glob):
    """Read the evaluator schema and enrich each row with training metadata."""
    records = []
    for path in sorted((root / "runs").rglob("*.json")):
        if "smoke" in path.name.lower():
            continue
        if path.name in {"config.json", "progress.json", "complete.json"}:
            continue
        document = read_json(path)
        if not isinstance(document, dict):
            continue
        evaluations = document.get("evaluations")
        if not isinstance(evaluations, list):
            continue
        checkpoint = Path(document.get("checkpoint") or document.get("run_dir") or document.get("run") or path.parent)
        run_dir = checkpoint.parent if checkpoint.suffix == ".pt" else checkpoint
        config = read_json(run_dir / "config.json") or {}
        run_name = run_dir.name
        # Some analyses store metadata beside, instead of inside, the run.
        if not fnmatch.fnmatch(run_name, run_glob):
            continue
        for index, row in enumerate(evaluations):
            if not isinstance(row, dict):
                continue
            records.append({**row, "task": document.get("task", config.get("task", run_name)),
                            "seed": config.get("seed", document.get("training_seed")), "run": run_name,
                            "analysis_kind": "dynamics" if "checkpoints" in document else "evaluation",
                            "source": str(path), "row_index": index,
                            "checkpoint_epoch": row.get("epoch", document.get("checkpoint_epoch")),
                            "train_length": config.get("length"),
                            "inverse_temperature": row.get("inverse_temperature", 1.0)})
    return records


def summarize(points):
    grouped = defaultdict(list)
    for x, y in points:
        if y is not None and np.isfinite(y):
            grouped[x].append(float(y))
    x = np.array(sorted(grouped))
    mean = np.array([np.mean(grouped[v]) for v in x])
    std = np.array([np.std(grouped[v], ddof=1) if len(grouped[v]) > 1 else 0 for v in x])
    return x, mean, std


def draw(ax, points, label, color=None, linestyle="-", marker=None):
    x, mean, std = summarize(points)
    if not len(x):
        return
    line, = ax.plot(x, mean, label=label, color=color, linestyle=linestyle,
                    marker=marker, markersize=3.8, linewidth=1.7)
    if np.any(std):
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.15, linewidth=0)


def format_accuracy(ax, xlabel):
    ax.set(xlabel=xlabel, ylim=(-0.025, 1.025))
    ax.grid(alpha=0.18)


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training(runs, out):
    if not runs:
        return []
    tasks = sorted({r["task"] for r in runs})
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.0 * len(tasks), 3.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        selected = [r for r in runs if r["task"] == task]
        points = [(h["epoch"], h.get("loss")) for r in selected for h in r["history"]]
        draw(ax, points, task, COLORS.get(task))
        ax.set(title=task, xlabel="Training epoch", ylabel="Cross-entropy loss", yscale="log")
        ax.grid(alpha=0.18)
    fig.tight_layout()
    save(fig, out, "training_loss")
    names = ["training_loss"]
    for metric in ["value_step", "value_chain", "clause_chain"]:
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.0 * len(tasks), 3.2), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            selected = [r for r in runs if r["task"] == task]
            lengths = sorted({e["length"] for r in selected for h in r["history"] for e in h.get("eval", [])})
            for length in lengths:
                points = [(h["epoch"], e.get(metric)) for r in selected for h in r["history"] for e in h.get("eval", []) if e["length"] == length]
                draw(ax, points, f"L = {length}")
            format_accuracy(ax, "Training epoch")
            ax.set(title=task, ylabel=METRICS[metric])
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        name = f"training_{metric}"
        save(fig, out, name)
        names.append(name)
    return names


def is_context(row):
    return any(k in row for k in ["count", "distractor_count", "distractors", "context_extra"])


def context_count(row):
    return row.get("count", row.get("distractor_count", row.get("distractors", row.get("context_extra"))))


def latest_static(rows):
    """Select final available checkpoint per run, avoiding dynamics mixtures."""
    candidates = [r for r in rows if not is_context(r) and r.get("analysis_kind") != "dynamics" and "teacher_forced" in r and "length" in r]
    epochs = defaultdict(int)
    for r in candidates:
        epochs[r["run"]] = max(epochs[r["run"]], r.get("checkpoint_epoch") or 0)
    return [r for r in candidates if (r.get("checkpoint_epoch") or 0) == epochs[r["run"]]]


def plot_length(rows, out):
    selected = [r for r in latest_static(rows) if r["inverse_temperature"] == 1]
    if not selected:
        return []
    panels = [("teacher_forced", "value_accuracy"), ("teacher_forced", "clause_chain_accuracy"), ("rollout", "final_value_accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    for ax, (kind, metric) in zip(axes, panels):
        for task in sorted({r["task"] for r in selected}):
            task_rows = [r for r in selected if r["task"] == task and kind in r]
            points = [(r["length"], r[kind]["metrics"].get(metric)) for r in task_rows]
            draw(ax, points, task, COLORS.get(task), marker="o")
        format_accuracy(ax, "Reasoning length L")
        ax.set(xscale="log", ylabel=METRICS[metric])
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "length_generalization")
    return ["length_generalization"]


def plot_temperature(rows, out):
    selected = latest_static(rows)
    temperatures = {r["inverse_temperature"] for r in selected}
    if len(temperatures) < 2:
        return []
    tasks = sorted({r["task"] for r in selected})
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.3), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        task_rows = [r for r in selected if r["task"] == task]
        for length in sorted({r["length"] for r in task_rows}):
            points = [(r["inverse_temperature"], r["teacher_forced"]["metrics"].get("clause_chain_accuracy")) for r in task_rows if r["length"] == length]
            draw(ax, points, f"L = {length}", marker="o")
        format_accuracy(ax, "Attention inverse temperature")
        ax.set(title=task, ylabel="Exact full-clause chain accuracy")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, out, "temperature_intervention")
    return ["temperature_intervention"]


def plot_attention(rows, out):
    selected = [r for r in latest_static(rows) if r["inverse_temperature"] == 1 and "attention" in r["teacher_forced"]]
    if not selected:
        return []
    tasks = sorted({r["task"] for r in selected})
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.2), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        task_rows = [r for r in selected if r["task"] == task]
        nheads = len(task_rows[0]["teacher_forced"]["attention"]["mean_per_head"]["target_predicate_weight"])
        for head in range(nheads):
            for key, label, style in [("target_predicate_weight", "target predicate", "-"), ("own_answer_weight", "current answer", "--")]:
                points = [(r["length"], r["teacher_forced"]["attention"]["mean_per_head"][key][head]) for r in task_rows]
                draw(ax, points, f"Head {head + 1}: {label}", linestyle=style, color=f"C{head}", marker="o")
        format_accuracy(ax, "Reasoning length L")
        ax.set(title=task, ylabel="Mean attention mass", xscale="log")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, out, "attention_vs_length")
    return ["attention_vs_length"]


def context_value(row, quantity, head=0):
    tf = row.get("teacher_forced", {})
    if quantity == "accuracy":
        if "first_step_accuracy" in row:
            return row["first_step_accuracy"]
        values = tf.get("by_position", {}).get("value_accuracy", [])
        return values[0] if values else None
    if "target_predicate_mass" in row:
        mass = row["target_predicate_mass"]
        return mass[head] if isinstance(mass, list) else mass
    values = tf.get("attention", {}).get("by_position_per_head", {}).get("target_predicate_weight", [])
    return values[head][0] if head < len(values) else None


def plot_context(rows, out):
    selected = [r for r in rows if is_context(r) and r["inverse_temperature"] == 1]
    if not selected:
        return []
    tasks = sorted({r["task"] for r in selected})
    nheads = max((len(r.get("teacher_forced", {}).get("attention", {}).get("by_position_per_head", {}).get("target_predicate_weight", [])) for r in selected), default=1)
    nheads = max(1, nheads)
    fig, axes = plt.subplots(1 + nheads, len(tasks), figsize=(4 * len(tasks), 3.0 * (1 + nheads)), squeeze=False)
    for column, task in enumerate(tasks):
        task_rows = [r for r in selected if r["task"] == task]
        kinds = sorted({r.get("kind", r.get("distractor_kind", "random")) for r in task_rows})
        for kind in kinds:
            kind_rows = [r for r in task_rows if r.get("kind", r.get("distractor_kind", "random")) == kind]
            # Context evaluation records one shared M=0 baseline; all action
            # compatibility conditions coincide in the absence of distractors.
            if not any(context_count(r) == 0 for r in kind_rows):
                kind_rows += [r for r in task_rows if context_count(r) == 0]
            for row_index in range(1 + nheads):
                quantity = "accuracy" if row_index == 0 else "mass"
                points = [(context_count(r), context_value(r, quantity, max(0, row_index - 1))) for r in kind_rows]
                draw(axes[row_index, column], points, kind, marker="o")
        for row_index in range(1 + nheads):
            format_accuracy(axes[row_index, column], "Number of disconnected predicates M")
            axes[row_index, column].legend(frameon=False, fontsize=8)
        axes[0, column].set(title=task, ylabel="First-step value accuracy")
        for head in range(nheads):
            axes[1 + head, column].set(ylabel=f"Head {head + 1}: first-step target attention mass")
    fig.tight_layout()
    save(fig, out, "context_intervention")
    return ["context_intervention"]


def plot_dynamics(rows, out):
    selected = [r for r in rows if r.get("analysis_kind") == "dynamics" and "teacher_forced" in r]
    if not selected:
        return []
    tasks = sorted({r["task"] for r in selected})
    fig, axes = plt.subplots(3, len(tasks), figsize=(4.1 * len(tasks), 9.0), squeeze=False)
    for column, task in enumerate(tasks):
        task_rows = [r for r in selected if r["task"] == task]
        lengths = sorted({r["length"] for r in task_rows})
        short, long = lengths[0], lengths[-1]
        for metric, label in [("variable_accuracy", "Next variable"), ("value_accuracy", "Next value")]:
            points = [(r["checkpoint_epoch"], r["teacher_forced"]["metrics"].get(metric)) for r in task_rows if r["length"] == short]
            draw(axes[0, column], points, label, marker="o")
        for length in lengths:
            points = [(r["checkpoint_epoch"], r["teacher_forced"]["metrics"].get("clause_chain_accuracy")) for r in task_rows if r["length"] == length]
            draw(axes[1, column], points, f"L = {length}", marker="o")
        long_rows = [r for r in task_rows if r["length"] == long]
        nheads = len(long_rows[0]["teacher_forced"]["attention"]["mean_per_head"]["target_predicate_weight"])
        for head in range(nheads):
            for key, label, style in [("target_predicate_weight", "target", "-"), ("own_answer_weight", "self", "--")]:
                points = [(r["checkpoint_epoch"], r["teacher_forced"]["attention"]["mean_per_head"][key][head]) for r in long_rows]
                draw(axes[2, column], points, f"Head {head + 1}: {label}", color=f"C{head}", linestyle=style, marker="o")
        axes[0, column].set(title=task, ylabel=f"Teacher-forced accuracy, L = {short}")
        axes[1, column].set(ylabel="Exact full-clause chain accuracy")
        axes[2, column].set(ylabel=f"Attention mass, L = {long}")
        for row in range(3):
            format_accuracy(axes[row, column], "Training epoch")
            axes[row, column].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, out, "mechanism_dynamics")
    return ["mechanism_dynamics"]


def write_table(rows, out):
    flat = []
    for row in rows:
        base = {k: row.get(k) for k in ["task", "run", "seed", "checkpoint_epoch", "length", "inverse_temperature", "context_extra", "distractor_kind", "source"]}
        for kind in ["teacher_forced", "rollout"]:
            for key, value in row.get(kind, {}).get("metrics", {}).items():
                if isinstance(value, (float, int)):
                    base[f"{kind}_{key}"] = value
        flat.append(base)
    if flat:
        fields = list(dict.fromkeys(k for row in flat for k in row))
        with (out / "evaluation_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-glob", default="base_*")
    args = parser.parse_args()
    out = args.output or args.root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    runs = collect_runs(args.root, args.run_glob)
    rows = collect_evaluations(args.root, args.run_glob)
    figures = plot_training(runs, out)
    figures += plot_length(rows, out) + plot_temperature(rows, out) + plot_attention(rows, out) + plot_context(rows, out) + plot_dynamics(rows, out)
    write_table(rows, out)
    metadata = {"run_glob": args.run_glob, "runs": [r["name"] for r in runs], "evaluation_rows": len(rows),
                "figures": figures, "bands": "mean +/- one sample standard deviation across training seeds; omitted for one seed",
                "notes": ["No interpolation or fabricated results", "Teacher-forced value chains are not free-rollout final-answer accuracy",
                          "Exact full-clause teacher-forced chain success equals deterministic unconstrained rollout whole-chain success",
                          "Incomplete ongoing runs may appear; consult config.json and complete.json",
                          "Files whose names contain smoke are excluded from scientific figures",
                          "Attention heads are indexed as stored and are not aligned across seeds"]}
    (out / "plot_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
