"""Hierarchical summaries of the planned routing round-two experiments.

Only direct run names from this campaign-plan.json are read. Model seeds and
dataset/noise replicates are retained separately; nine cells are not treated
as nine independent model initializations. Initial evaluations are AFTER
routing conversion/projection and BEFORE optimization, not the original
unconverted parent. No confidence intervals are inferred from these cells.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DESIGN_FIELDS = ["task", "nvars", "model", "routing_mode", "initialization_routing_mode",
    "length", "samples", "pool_samples", "steps", "batch", "lr", "mode", "label_source",
    "noise_kind", "noise_rate", "eval_seed", "validation_seed", "eval_samples",
    "validation_samples", "rollout_samples", "eval_every", "budget_label", "source_sha256"]
METRIC = "teacher_forced.clause_chain_accuracy"
MODE_COLORS = {"dense": "#0072B2", "row_tied": "#009E73", "diagonal": "#D55E00"}
SEED_COLORS = ["#0072B2", "#D55E00", "#009E73"]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:12]


def scalar_metrics(document):
    result = {}
    for kind in ["teacher_forced", "rollout"]:
        for name, value in document.get(kind, {}).get("metrics", {}).items():
            if isinstance(value, (float, int)):
                result[f"{kind}.{name}"] = value
    for name, value in document.get("attention", {}).items():
        if isinstance(value, (float, int)):
            result[f"attention.{name}"] = value
    return result


def condition(config):
    if config.get("label_source") == "teacher":
        return "teacher"
    kind = config.get("noise_kind", "none")
    if kind == "none":
        return "truth"
    return f"{kind}{round(config['noise_rate'] * 100):02d}"


def collect(campaign):
    plan = read_json(campaign / "campaign-plan.json")
    if not isinstance(plan, dict) or not isinstance(plan.get("runs"), list):
        raise ValueError("A campaign-plan.json with explicit runs is required")
    rows, runs, skipped = [], [], []
    for spec in plan["runs"]:
        name = spec["name"]
        if Path(name).name != name:
            skipped.append({"run": name, "reason": "not a direct campaign child"})
            continue
        directory = campaign / name
        config = read_json(directory / "config.json")
        complete = read_json(directory / "complete.json")
        if not config or not complete or complete.get("complete") is not True:
            skipped.append({"run": name, "reason": "not completed"})
            continue
        if complete.get("frozen_sha256_before") != complete.get("frozen_sha256_after"):
            skipped.append({"run": name, "reason": "frozen hash mismatch"})
            continue
        documents = {"initial": read_json(directory / "initial-evaluation.json"),
                     "final": read_json(directory / "evaluation.json")}
        if any(not d or d.get("complete") is not True for d in documents.values()):
            skipped.append({"run": name, "reason": "initial or final evaluation incomplete"})
            continue
        expected_phases = {"initial": "after_conversion_before_training", "final": "after_training"}
        if any(documents[p].get("phase") != expected_phases[p] for p in documents):
            skipped.append({"run": name, "reason": "unexpected evaluation phase semantics"})
            continue
        design = {key: config.get(key) for key in DESIGN_FIELDS}
        audit = read_json(directory / "data-audit.json") or {}
        metadata = {"run": name, "model_seed": config["seed"], "replicate": config["replicate"],
                    "data_seed": config["data_seed"], "noise_seed": config["noise_seed"],
                    "condition": condition(config), "design": design,
                    "config": config, "complete": complete, "audit": audit}
        run_rows = []
        for phase, document in documents.items():
            for item in document.get("evaluations", []):
                metrics = scalar_metrics(item)
                if METRIC not in metrics:
                    continue
                run_rows.append({"run": name, "model_seed": config["seed"],
                    "replicate": config["replicate"], "data_seed": config["data_seed"],
                    "noise_seed": config["noise_seed"], "condition": condition(config),
                    "routing_mode": config["routing_mode"], "lr": config["lr"],
                    "label_source": config["label_source"], "noise_kind": config["noise_kind"],
                    "noise_rate": config["noise_rate"], "phase": phase,
                    "phase_semantics": document["phase"], "length": item["length"],
                    "panel_sha256": item.get("panel_sha256"),
                    "checkpoint_sha256": document.get("checkpoint_sha256"),
                    "evaluation_source": str((directory / ("initial-evaluation.json" if phase == "initial" else "evaluation.json")).resolve()),
                    "design": design, "metrics": metrics,
                    "routing_statistics": document.get("routing_statistics", {})})
        initial_panels = {x["length"]: x["panel_sha256"] for x in run_rows if x["phase"] == "initial"}
        final_panels = {x["length"]: x["panel_sha256"] for x in run_rows if x["phase"] == "final"}
        if not initial_panels or initial_panels != final_panels:
            skipped.append({"run": name, "reason": "initial/final evaluation panels not paired"})
            continue
        runs.append(metadata)
        rows.extend(run_rows)
    return plan, runs, rows, skipped


def hierarchical_statistics(members, metric):
    """Average dataset/noise replicates within model seed before across seeds."""
    by_seed = defaultdict(list)
    for row in members:
        if metric in row["metrics"]:
            by_seed[row["model_seed"]].append(row)
    duplicate = []
    for seed, entries in by_seed.items():
        reps = [entry["replicate"] for entry in entries]
        if len(set(reps)) != len(reps):
            duplicate.append(seed)
    if duplicate:
        return {"aggregation_skipped": True, "duplicate_seed_replicate_cells": duplicate}
    seed_means = {str(seed): statistics.mean(entry["metrics"][metric] for entry in entries)
                  for seed, entries in sorted(by_seed.items())}
    if not seed_means:
        return None
    all_values = [entry["metrics"][metric] for entries in by_seed.values() for entry in entries]
    return {"mean": statistics.mean(seed_means.values()), "model_seed_means": seed_means,
        "model_seed_mean_range": [min(seed_means.values()), max(seed_means.values())],
        "cell_range": [min(all_values), max(all_values)], "n_model_seeds": len(seed_means),
        "n_cells": len(all_values),
        "replicates_per_model_seed": {str(seed): sorted(entry["replicate"] for entry in entries)
                                      for seed, entries in sorted(by_seed.items())},
        "cells": [{"run": entry["run"], "model_seed": seed, "replicate": entry["replicate"],
                   "data_seed": entry["data_seed"], "noise_seed": entry["noise_seed"],
                   "value": entry["metrics"][metric]} for seed, entries in sorted(by_seed.items()) for entry in entries],
        "aggregation": "equal weight per model seed after averaging its dataset/noise replicates; range is not a confidence interval"}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = {"design": row["design"], "phase": row["phase"], "length": row["length"]}
        groups[canonical(key)].append(row)
    result = []
    for key, members in sorted(groups.items()):
        metrics = {metric: hierarchical_statistics(members, metric)
                   for metric in sorted({m for row in members for m in row["metrics"]})}
        result.append({"group_id": digest(json.loads(key)), **json.loads(key),
                       "runs": [row["run"] for row in members], "metrics": metrics})
    return result


def paired_changes(rows):
    keyed = {(row["run"], row["length"], row["phase"]): row for row in rows}
    output = []
    for row in rows:
        if row["phase"] != "final":
            continue
        initial = keyed[(row["run"], row["length"], "initial")]
        output.append({**row, "phase": "final_minus_initial",
                       "phase_semantics": "training change relative to after-conversion initialization",
                       "metrics": {key: value - initial["metrics"][key] for key, value in row["metrics"].items()
                                   if key in initial["metrics"]}})
    return output


def flat_row(row):
    return {**{k: v for k, v in row.items() if k not in {"metrics", "design", "routing_statistics"}},
            "design_json": canonical(row["design"]), "routing_statistics_json": canonical(row["routing_statistics"]),
            **row["metrics"]}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["run"])
        writer.writeheader()
        writer.writerows(rows)


def select(rows, **wanted):
    return [row for row in rows if all(row.get(key) == value for key, value in wanted.items())]


def polish(ax):
    ax.set_ylim(-3, 103)
    ax.set_ylabel("Exact full-clause chain accuracy (%)")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, output, name):
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_noise(rows, output, status=""):
    conditions = ["truth", "target05", "target20", "prefix_inconsistent20", "prefix_consistent20"]
    labels = ["Clean", "Wrong target\n5%", "Wrong target\n20%", "Wrong prefix /\noriginal target 20%", "Wrong prefix /\nconsistent target 20%"]
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    coverage = {}
    for x, cond in enumerate(conditions):
        selected = select(rows, phase="final", routing_mode="dense", lr=.03, condition=cond, length=8)
        stats = hierarchical_statistics(selected, METRIC)
        coverage[cond] = stats
        if not stats or stats.get("aggregation_skipped"):
            continue
        for seed in range(3):
            cells = [c for c in stats["cells"] if c["model_seed"] == seed]
            if not cells:
                continue
            center = x + (seed - 1) * .19
            offsets = [(i - (len(cells) - 1) / 2) * .045 for i in range(len(cells))]
            ax.scatter([center + d for d in offsets], [100*c["value"] for c in cells],
                       color=SEED_COLORS[seed], s=22, alpha=.25)
            ax.scatter(center, 100*stats["model_seed_means"][str(seed)], color=SEED_COLORS[seed], s=53,
                       label=f"Model seed {seed}: replicate mean" if x == 0 else None)
        ax.plot([x-.31, x+.31], [100*stats["mean"]]*2, color="black", linewidth=2)
    ax.set_xticks(range(len(labels)), labels)
    polish(ax)
    ax.set_title("Independent data/noise repetitions: dense routing, L4 training → L8 test" + status)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(.5, .015, "Large dots: model-seed means; faint dots: dataset/noise replicates; black bars: mean of model-seed means. No IID confidence interval.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0,.06,1,1])
    save(fig, output, "noise_repetitions")
    return coverage


def figure_training(rows, output, status=""):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    coverage = {}
    for rowindex, length in enumerate([8, 16]):
        for column, source in enumerate(["truth", "teacher"]):
            ax = axes[rowindex, column]
            for x, mode in enumerate(["dense", "row_tied", "diagonal"]):
                selected = select(rows, routing_mode=mode, lr=.03, condition=source, length=length)
                stats = {phase: hierarchical_statistics(select(selected, phase=phase), METRIC)
                         for phase in ["initial", "final"]}
                coverage[f"{source}_L{length}_{mode}"] = stats
                if not all(stats.values()) or any(s.get("aggregation_skipped") for s in stats.values()):
                    continue
                xx = [x-.23, x+.23]
                for seed in range(3):
                    key = str(seed)
                    if not all(key in s["model_seed_means"] for s in stats.values()):
                        continue
                    yy = [100*stats[phase]["model_seed_means"][key] for phase in ["initial", "final"]]
                    ax.plot(xx, yy, color=MODE_COLORS[mode], alpha=.28, linewidth=1, marker="o", markersize=3)
                means = [100*stats[phase]["mean"] for phase in ["initial", "final"]]
                ax.plot(xx, means, color=MODE_COLORS[mode], linewidth=2.2, marker="o", markersize=6)
            polish(ax)
            ax.set_xticks([0, 1, 2], ["Dense\n3 reps/seed", "Row-tied\n3 reps/seed", "Diagonal\nrep 0 only"])
            ax.set_title(f"{'True' if source == 'truth' else 'Teacher'} labels, L{length} test")
            ax.set_xlim(-.5, 2.5)
    fig.suptitle("Routing constraint: after-conversion initial → final training accuracy (lr=0.03)" + status, fontsize=12)
    fig.text(.5,.015,"Each segment runs left (AFTER projection, before training) → right (after training). Faint lines: model-seed means; bold line: their mean.",
             ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.95])
    save(fig, output, "routing_initial_to_final")
    return coverage


def figure_lr(rows, output, status=""):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    coverage = {}
    for ax, cond, label in zip(axes, ["truth", "target05", "target20"], ["Clean", "5% wrong targets", "20% wrong targets"]):
        selected = select(rows, phase="final", routing_mode="dense", condition=cond, length=8, replicate=0)
        stats = {str(lr): hierarchical_statistics(select(selected, lr=lr), METRIC) for lr in [.003,.03]}
        coverage[cond] = stats
        if all(stats.values()) and not any(s.get("aggregation_skipped") for s in stats.values()):
            for seed in range(3):
                key = str(seed)
                if all(key in s["model_seed_means"] for s in stats.values()):
                    yy = [100*stats[str(lr)]["model_seed_means"][key] for lr in [.003,.03]]
                    ax.plot([0,1], yy, color=SEED_COLORS[seed], marker="o", alpha=.6, linewidth=1)
            ax.scatter([0,1],[100*stats[str(lr)]["mean"] for lr in [.003,.03]], color="black", marker="_", s=350, linewidths=2.5,zorder=4)
        polish(ax)
        ax.set_title(label)
        ax.set_xticks([0,1],["lr = 0.003", "lr = 0.03"])
        ax.set_xlim(-.3,1.3)
    fig.suptitle("Learning-rate control: dense routing, replicate 0, L4 → L8, 1000 updates" + status,fontsize=12)
    fig.text(.5,.015,"Lines pair the same model seed and dataset/noise draw. Black marks are means; no early stopping or test-based checkpoint selection.",
             ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.93])
    save(fig, output, "learning_rate_control")
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/routing-round2-2026-11-09")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/routing-round2-2026-11-09")
    args = parser.parse_args()
    plan, runs, rows, skipped = collect(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_run_phase_length.csv", [flat_row(row) for row in rows])
    changes = paired_changes(rows)
    write_csv(args.output / "paired_training_changes.csv", [flat_row(row) for row in changes])
    summary = {"groups": aggregate(rows), "paired_training_change_groups": aggregate(changes),
               "definitions": {"hierarchy": "dataset/noise replicates within model seed; equal weight across model-seed means",
                   "initial": "after conversion/projection and before training, NOT the unconverted original parent",
                   "scope": "only direct planned round2 runs; recursive experiments excluded",
                   "uncertainty": "observed model-mean and cell ranges only, no IID confidence intervals"}}
    (args.output / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Guard figures against silently combining distinct implementation designs.
    hashes = {canonical(row['design'].get('source_sha256')) for row in rows}
    warnings = []
    figures = []
    plot_data = {}
    if len(hashes) > 1:
        warnings.append("Multiple source implementations found. Raw/grouped outputs written; overview figures skipped to avoid mixing.")
    elif rows:
        status = (f"\nPARTIAL SNAPSHOT: {len(runs)}/{len(plan['runs'])} planned runs complete"
                  if len(runs) != len(plan['runs']) else "")
        plot_data["noise_repetitions"] = figure_noise(rows, args.output, status)
        plot_data["routing_initial_to_final"] = figure_training(rows, args.output, status)
        plot_data["learning_rate_control"] = figure_lr(rows, args.output, status)
        figures = list(plot_data)
    (args.output / "overview_numbers.json").write_text(json.dumps(plot_data, indent=2) + "\n")
    manifest = {"campaign": str(args.runs.resolve()), "planned_runs": len(plan['runs']),
        "completed_runs": len(runs), "rows": len(rows), "all_planned_complete": len(runs) == len(plan['runs']),
        "included_runs": [run['run'] for run in runs], "skipped": skipped, "warnings": warnings,
        "figures": figures, "hierarchy": summary['definitions'],
        "incomplete_policy": "Skip unfinished runs; displayed means use only available model-seed/replicate cells, with exact coverage in overview_numbers.json",
        "diagonal_design": "Only dataset/noise replicate 0; not nine independent cells",
        "run_sources": [{"run": run['run'], "model_seed": run['model_seed'], "replicate": run['replicate'],
                         "data_seed": run['data_seed'], "noise_seed": run['noise_seed'],
                         "config": str((args.runs / run['run'] / 'config.json').resolve())} for run in runs]}
    (args.output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ['planned_runs','completed_runs','rows','all_planned_complete','figures','warnings']}, indent=2))


if __name__ == "__main__":
    main()
