"""Summarize completed sparse-supervision runs without pooling unlike budgets.

Usage:
    python src/analyze_sparse.py
    python src/analyze_sparse.py --runs runs/sparse-supervision-2026-11-09

Each final-evaluation row remains available in per_seed_evaluation.csv. The
aggregate JSON reports arithmetic means and observed seed ranges, not confidence
intervals. Figures are separated by training length, data size, budget, optimizer
configuration, model, source version and initialization/teacher design. Ongoing
or invalid runs are recorded in the manifest and skipped.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CAMPAIGN = "sparse-supervision-2026-11-09"
DESIGN_FIELDS = [
    "stage", "task", "nvars", "hidden", "model", "length", "samples", "pool_samples",
    "steps", "batch", "lr", "mode", "label_source", "noise_kind", "noise_rate",
    "budget_label", "eval_seed", "eval_samples", "rollout_samples",
    "local_gate", "local_min_steps", "trainable_parameters",
]
METRICS = [
    ("teacher_forced.clause_chain_accuracy", "Exact full-clause chain accuracy"),
    ("teacher_forced.value_accuracy", "Teacher-forced value accuracy"),
    ("rollout.final_value_accuracy", "Free-rollout final-value accuracy"),
]
MODE_NAMES = {"second": "Second step", "random": "One fixed random step", "all": "All steps"}


def read_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def identifier(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:12]


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(value):
    return isinstance(value, (str, int, float, bool)) or value is None


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def flatten(value, prefix=""):
    """Keep arrays as JSON strings rather than silently dropping count vectors."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    else:
        result[prefix] = value if scalar(value) else canonical(value)
    return result


def parent_directory(checkpoint, run_directory):
    path = Path(checkpoint)
    # Configs normally contain absolute paths. Relative paths created by the
    # runner are relative to its project cwd, not necessarily the run directory.
    if not path.is_absolute():
        project = Path(__file__).resolve().parents[1]
        candidates = [project / path, run_directory / path]
        path = next((p for p in candidates if (p.parent / "config.json").exists()), candidates[0])
    return path.parent.resolve()


def parent_design(checkpoint, run_directory, visited):
    if checkpoint is None:
        return None
    parent = parent_directory(checkpoint, run_directory)
    if parent in visited:
        return {"unresolved_parent": str(parent), "reason": "cyclic lineage"}
    config = read_json(parent / "config.json")
    if config is None:
        # Never pool two unknown parent models solely because other settings
        # match. Including the path conservatively keeps them separate.
        return {"unresolved_parent": str(parent)}
    return design(config, parent, visited | {parent})


def design(config, run_directory, visited=None):
    visited = visited or {run_directory.resolve()}
    result = {key: config.get(key) for key in DESIGN_FIELDS}
    sources = config.get("source_sha256")
    if isinstance(sources, dict):
        sources = sources.copy()
        if config.get("stage") == "local":
            # The local branch does not call build_supervision or
            # select_supervised. sparse_data's imported tensor_hash is only a
            # provenance helper, not a training/evaluation computation. Keep
            # every other hash (including unknown future files) strict.
            sources.pop("sparse_data.py", None)
    result["effective_source_sha256"] = sources
    result["initialization_design"] = parent_design(config.get("init"), run_directory, visited)
    # --teacher may be passed to a paired truth-label run for bookkeeping, but
    # that branch never loads it. Its identity cannot alter a truth-label run.
    result["teacher_design"] = (parent_design(config.get("teacher"), run_directory, visited)
                                if config.get("label_source") == "teacher" else None)
    return result


def randomization_lineage(config, run_directory, visited=None):
    """Record seed realizations without treating them as experimental factors."""
    visited = visited or {run_directory.resolve()}
    result = {"seed": config.get("seed"), "data_seed": config.get("data_seed")}
    for key in ["init", "teacher"]:
        checkpoint = config.get(key)
        if checkpoint is None:
            result[key] = None
            continue
        parent = parent_directory(checkpoint, run_directory)
        parent_config = read_json(parent / "config.json")
        if parent in visited or parent_config is None:
            result[key] = {"unresolved_parent": str(parent)}
        else:
            result[key] = {"run_directory": str(parent),
                           **randomization_lineage(parent_config, parent, visited | {parent})}
    return result


def collect(runs_root):
    runs, evaluations, histories, skipped = [], [], [], []
    if not runs_root.exists():
        return runs, evaluations, histories, [{"run": str(runs_root), "reason": "run directory does not exist"}]
    for directory in sorted(p.parent for p in runs_root.rglob("config.json")):
        name = str(directory.relative_to(runs_root))
        config = read_json(directory / "config.json")
        complete = read_json(directory / "complete.json")
        evaluation = read_json(directory / "evaluation.json")
        audit = read_json(directory / "data-audit.json")
        if config is None or complete is None or complete.get("complete") is not True:
            skipped.append({"run": name, "reason": "missing/invalid config or no complete marker"})
            continue
        if evaluation is None or evaluation.get("complete") is not True or not isinstance(evaluation.get("evaluations"), list):
            skipped.append({"run": name, "reason": "missing/invalid/incomplete final evaluation"})
            continue
        if audit is None:
            skipped.append({"run": name, "reason": "missing or invalid data audit"})
            continue
        before, after = complete.get("frozen_sha256_before"), complete.get("frozen_sha256_after")
        if before is not None and after != before:
            skipped.append({"run": name, "reason": "frozen parameter hashes disagree"})
            continue
        group_config = design(config, directory)
        run = {
            "run": name, "run_directory": str(directory.resolve()), "seed": config.get("seed"),
            "design": group_config, "config": config, "complete": complete, "audit": audit,
            "randomization": randomization_lineage(config, directory),
            "evaluation_sha256": file_sha256(directory / "evaluation.json"),
            "evaluation_checkpoint_sha256": evaluation.get("checkpoint_sha256"),
            "warnings": [],
        }
        if config.get("stage") == "local" and complete.get("local_gate_passed") is not True:
            run["warnings"].append("Local pretraining gate failed; oracle curves are diagnostic, not an admitted foundation.")
        if before is None:
            run["warnings"].append("No frozen-parameter hash audit in completion metadata.")
        rows = []
        for index, row in enumerate(evaluation["evaluations"]):
            if not isinstance(row, dict) or not isinstance(row.get("length"), int):
                continue
            tf = row.get("teacher_forced", {})
            if not isinstance(tf.get("metrics"), dict):
                continue
            metrics = {}
            for kind in ["teacher_forced", "rollout"]:
                for key, value in row.get(kind, {}).get("metrics", {}).items():
                    if finite_number(value):
                        metrics[f"{kind}.{key}"] = value
            for key, value in row.get("attention", {}).items():
                if finite_number(value):
                    metrics[f"attention.{key}"] = value
            rows.append({
                "run": name, "seed": config.get("seed"), "design": group_config,
                "data_seed": config.get("data_seed"), "randomization": run["randomization"],
                "evaluation_length": row["length"], "oracle_attention": row.get("oracle_attention", False),
                "actual_steps": complete.get("steps"),
                "local_gate_passed": complete.get("local_gate_passed"),
                "metrics": metrics, "raw": row, "row_index": index,
                "source": str((directory / "evaluation.json").resolve()),
            })
        if not rows:
            skipped.append({"run": name, "reason": "no usable final evaluation rows"})
            continue
        runs.append(run)
        evaluations.extend(rows)
        history_path = directory / "history.jsonl"
        if history_path.exists():
            for number, line in enumerate(history_path.read_text().splitlines(), 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    run["warnings"].append(f"Ignored invalid history line {number}.")
                    continue
                if isinstance(record, dict):
                    histories.append({"run": name, "seed": config.get("seed"),
                                      "design_id": identifier(group_config), **flatten(record)})
    return runs, evaluations, histories, skipped


def grouping_config(row):
    return {"design": row["design"], "evaluation_length": row["evaluation_length"],
            "oracle_attention": row["oracle_attention"], "actual_steps": row["actual_steps"],
            "local_gate_passed": row["local_gate_passed"]}


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[canonical(grouping_config(row))].append(row)
    results = []
    for key, members in sorted(grouped.items()):
        seeds = [member["seed"] for member in members]
        duplicate_seeds = sorted({seed for seed in seeds if seeds.count(seed) > 1}, key=str)
        statistics_by_metric = {}
        # Duplicated seed/config runs are retained in raw tables but never
        # counted as independent training seeds or silently averaged together.
        if not duplicate_seeds:
            for metric in sorted({k for row in members for k in row["metrics"]}):
                pairs = [(row["seed"], row["metrics"][metric]) for row in members if metric in row["metrics"]]
                values = [value for _, value in pairs]
                statistics_by_metric[metric] = {
                    "n": len(values), "seeds": [seed for seed, _ in pairs],
                    "mean": statistics.mean(values), "min": min(values), "max": max(values),
                }
        results.append({
            "group_id": identifier(json.loads(key)), **json.loads(key),
            "runs": [row["run"] for row in members], "seeds": seeds,
            "data_seeds": sorted({row.get("data_seed") for row in members}, key=str),
            "replicates": [{"run": row["run"], "seed": row["seed"],
                            "data_seed": row.get("data_seed"),
                            "randomization_lineage": row.get("randomization")}
                           for row in members],
            "duplicate_seeds": duplicate_seeds, "metrics": statistics_by_metric,
            "aggregation_skipped": bool(duplicate_seeds),
        })
    return results


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["run"])
        writer.writeheader()
        writer.writerows(rows)


def figure_context(group, remove_design=(), remove_evaluation=False):
    config = {k: v for k, v in group["design"].items() if k not in remove_design}
    result = {"design": config, "oracle_attention": group["oracle_attention"],
              "actual_steps": group["actual_steps"], "local_gate_passed": group["local_gate_passed"]}
    if not remove_evaluation:
        result["evaluation_length"] = group["evaluation_length"]
    return result


def context_title(context):
    d = context["design"]
    text = (f"{d.get('task')} | stage={d.get('stage')} | train L={d.get('length')} | "
            f"N={d.get('samples')} | steps={context.get('actual_steps')}\n"
            f"budget={d.get('budget_label')}")
    if "evaluation_length" in context:
        text += f" | test L={context['evaluation_length']}"
    if context["oracle_attention"]:
        text += " | ORACLE attention"
    if context.get("local_gate_passed") is False:
        text += " | local gate FAILED"
    return text


def save_plot(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_series(ax, points, label):
    points = sorted(points, key=lambda item: item[0])
    x = [point[0] for point in points]
    y = [point[1]["mean"] for point in points]
    n = sorted({point[1]["n"] for point in points})
    line, = ax.plot(x, y, marker="o", linewidth=1.5,
                    label=f"{label} (n={','.join(map(str, n))})")
    if any(point[1]["n"] > 1 for point in points):
        ax.fill_between(x, [point[1]["min"] for point in points],
                        [point[1]["max"] for point in points], color=line.get_color(), alpha=.13)


def plot_lengths(groups, output):
    comparisons = defaultdict(list)
    for group in groups:
        if group["aggregation_skipped"]:
            continue
        # Comparing modes and label sources is intentional. A ground-truth run
        # has no teacher, so teacher design cannot define the shared panel. It
        # remains a distinct curve identity (and an aggregate grouping factor),
        # while initialization design stays fixed for all compared students.
        context = figure_context(group, ["mode", "label_source", "teacher_design"], remove_evaluation=True)
        comparisons[canonical(context)].append(group)
    manifests = []
    for key, members in sorted(comparisons.items()):
        context = json.loads(key)
        available = [(metric, title) for metric, title in METRICS if any(metric in m["metrics"] for m in members)]
        if not available:
            continue
        fig, axes = plt.subplots(1, len(available), figsize=(5.2 * len(available), 4), squeeze=False)
        for ax, (metric, title) in zip(axes[0], available):
            lines = defaultdict(list)
            teachers = {canonical(m["design"].get("teacher_design")) for m in members
                        if m["design"].get("teacher_design") is not None}
            for member in members:
                if metric in member["metrics"]:
                    d = member["design"]
                    mode = MODE_NAMES.get(d.get("mode"), str(d.get("mode")))
                    label = f"{mode} / {d.get('label_source')}"
                    teacher_key = canonical(d.get("teacher_design"))
                    if len(teachers) > 1 and d.get("teacher_design") is not None:
                        label += f" / teacher {identifier(d['teacher_design'])}"
                    lines[(label, teacher_key)].append((member["evaluation_length"], member["metrics"][metric]))
            for (label, _), points in sorted(lines.items()):
                draw_series(ax, points, label)
            ax.set(xlabel="Evaluation chain length", ylabel=title, ylim=(-.025, 1.025), xscale="log", xlim=None)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, frameon=False)
        fig.suptitle(context_title(context), fontsize=10)
        fig.tight_layout()
        stem = f"length_{identifier(context)}"
        save_plot(fig, output, stem)
        manifests.append({"kind": "length", "files": [f"{stem}.png", f"{stem}.pdf"],
                          "context": context, "groups": [g["group_id"] for g in members],
                          "teacher_designs": {identifier(g["design"]["teacher_design"]): g["design"]["teacher_design"]
                              for g in members if g["design"].get("teacher_design") is not None}})
    return manifests


def plot_noise(groups, output):
    comparisons = defaultdict(list)
    for group in groups:
        if group["aggregation_skipped"] or group["design"].get("stage") != "attention":
            continue
        context = figure_context(group, ["noise_kind", "noise_rate"])
        comparisons[canonical(context)].append(group)
    manifests = []
    for key, members in sorted(comparisons.items()):
        if not any((m["design"].get("noise_rate") or 0) > 0 for m in members):
            continue
        context = json.loads(key)
        available = [(metric, title) for metric, title in METRICS if any(metric in m["metrics"] for m in members)]
        fig, axes = plt.subplots(1, len(available), figsize=(5.2 * len(available), 4), squeeze=False)
        for ax, (metric, title) in zip(axes[0], available):
            lines = defaultdict(list)
            baseline = None
            for member in members:
                if metric not in member["metrics"]:
                    continue
                kind = member["design"].get("noise_kind", "none")
                rate = member["design"].get("noise_rate") or 0
                if kind == "none" and rate == 0:
                    baseline = member["metrics"][metric]
                else:
                    lines[str(kind)].append((rate, member["metrics"][metric]))
            if baseline is not None:
                ax.axhline(baseline["mean"], linestyle="--", color="0.4", linewidth=1,
                           label=f"clean reference (n={baseline['n']})")
            for label, points in sorted(lines.items()):
                # A clean reference may anchor a line only if no explicit
                # zero-noise observation for that intervention exists.
                if baseline is not None and not any(x == 0 for x, _ in points):
                    points = [(0, baseline), *points]
                draw_series(ax, points, label)
            ax.set(xlabel="Requested corruption probability", ylabel=title, ylim=(-.025, 1.025))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, frameon=False)
        d = context["design"]
        fig.suptitle(context_title(context) + f"\nmode={d.get('mode')} | labels={d.get('label_source')}", fontsize=10)
        fig.tight_layout()
        stem = f"noise_{identifier(context)}"
        save_plot(fig, output, stem)
        manifests.append({"kind": "noise", "files": [f"{stem}.png", f"{stem}.pdf"],
                          "context": context, "groups": [g["group_id"] for g in members]})
    return manifests


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=project / "runs" / DEFAULT_CAMPAIGN)
    parser.add_argument("--output", type=Path, default=project / "figures" / DEFAULT_CAMPAIGN)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs, rows, histories, skipped = collect(args.runs)
    grouped = aggregate(rows)
    run_lookup = {run["run"]: run for run in runs}
    flat_rows = []
    for row in rows:
        run = run_lookup[row["run"]]
        flat_rows.append({"run": row["run"], "seed": row["seed"],
            "design_id": identifier(row["design"]), "evaluation_length": row["evaluation_length"],
            "source": row["source"], "evaluation_sha256": run["evaluation_sha256"],
            **flatten(run["config"], "config"), **flatten(run["complete"], "complete"),
            **flatten(run["audit"], "audit"), **flatten(row["raw"], "evaluation")})
    run_rows = [{"run": run["run"], "seed": run["seed"], "run_directory": run["run_directory"],
                 "design_id": identifier(run["design"]), "warnings": canonical(run["warnings"]),
                 **flatten(run["config"], "config"), **flatten(run["complete"], "complete"),
                 **flatten(run["audit"], "audit")} for run in runs]
    write_csv(args.output / "per_seed_evaluation.csv", flat_rows)
    write_csv(args.output / "per_seed_runs.csv", run_rows)
    write_csv(args.output / "history.csv", histories)
    aggregate_document = {"schema_version": 1, "statistics": "arithmetic mean and observed min/max across distinct training seeds; not confidence intervals",
                          "groups": grouped}
    (args.output / "aggregate.json").write_text(json.dumps(aggregate_document, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    plots = [] if args.no_plots else plot_lengths(grouped, args.output) + plot_noise(grouped, args.output)
    manifest = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_directory": str(args.runs.resolve()), "output_directory": str(args.output.resolve()),
        "completed_runs": [run["run"] for run in runs], "skipped_runs": skipped,
        "evaluation_rows": len(rows), "history_rows": len(histories), "aggregate_groups": len(grouped),
        "run_warnings": {run["run"]: run["warnings"] for run in runs if run["warnings"]},
        "coverage": [{"design_id": identifier(run["design"]), "run": run["run"],
                      "seed": run["seed"], "stage": run["config"].get("stage"),
                      "data_seed": run["config"].get("data_seed"),
                      "mode": run["config"].get("mode"), "label_source": run["config"].get("label_source"),
                      "train_length": run["config"].get("length"), "samples": run["config"].get("samples"),
                      "pool_samples": run["config"].get("pool_samples"),
                      "budget_label": run["config"].get("budget_label")} for run in runs],
        "plots": plots, "table_files": ["per_seed_evaluation.csv", "per_seed_runs.csv", "history.csv", "aggregate.json"],
        "notes": [
            "Only runs with complete final evaluation and a data audit are included; coverage may be partial.",
            "No pooling across training lengths, dataset sizes, budget labels, optimizer settings, model/source versions or parent experimental designs.",
            "Training seed and data_seed are random realizations, not design factors, including in initialization/teacher ancestry. Aggregate groups record seed/data_seed sets and full randomization lineage; original configs remain in CSV.",
            "pool_samples is a design factor: sampling a prefix of a common pool is not silently pooled with another pool size.",
            "Source exception for local stage only: effective_source_sha256 omits sparse_data.py because its supervision builders are not used in local training (tensor_hash only records provenance). Driver, model, data, local_rule, evaluate and all other hashes remain strict; attention-stage sparse_data hashes remain strict. Full original hashes are retained in config CSV.",
            "For truth labels, an unused --teacher argument is excluded from effective experimental design; its original value and hash remain in config CSV.",
            "The intended contrasts within a length figure are supervision mode and label source. Student initialization designs match; differing teacher designs are distinct curves, never pooled.",
            "Noise figures hold supervision mode and label source fixed and vary noise kind/rate; a clean reference may be shared.",
            "Shading is observed minimum to maximum across distinct training seeds, not a confidence interval.",
            "Duplicate seed/config groups are kept in raw CSV but excluded from aggregates and figures.",
            "TF full-clause whole-chain success equals the deterministic greedy exact-trajectory event; TF value-only metrics do not equal rollout final accuracy.",
            "Oracle attention and failed local gates are explicitly separated and cannot be presented as autonomous inference success.",
            "The manifest lists the current generated figures; any older unlisted files in the output directory are not part of this snapshot.",
        ],
    }
    (args.output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"completed_runs": len(runs), "skipped_runs": len(skipped), "evaluation_rows": len(rows),
                      "plots": len(plots), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
