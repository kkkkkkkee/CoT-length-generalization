"""Three-panel, three-seed overview of the completed sparse-supervision panel.

Reads final run JSON directly. Requires the 81-run campaign completion marker,
all specified three-seed cells, the same L4->L8 endpoint, frozen parameters and
matched evaluation panels. This is separate from the older end-to-end baseline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CAMPAIGN = "sparse-supervision-2026-11-09"
MODES = ["second", "random", "all"]
MODE_NAMES = ["Second step", "Random step", "All steps"]
SOURCES = ["truth", "teacher"]
SOURCE_NAMES = {"truth": "True labels", "teacher": "Teacher labels"}
COLORS = {"truth": "#0072B2", "teacher": "#D55E00",
          "target": "#D55E00", "prefix_inconsistent": "#CC79A7",
          "prefix_consistent": "#009E73"}
NOISE_NAMES = {"target": "Wrong target", "prefix_inconsistent": "Wrong prefix / original target",
               "prefix_consistent": "Wrong prefix / consistent target"}
VALUE_METRIC = "clause_chain_accuracy"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(root, name, mode, source, samples, seed, noise_kind="none", noise_rate=0):
    directory = root / name
    config = read(directory / "config.json")
    complete = read(directory / "complete.json")
    evaluation = read(directory / "evaluation.json")
    audit = read(directory / "data-audit.json")
    assert complete.get("complete") is True and evaluation.get("complete") is True, name
    expected = {"stage": "attention", "task": "S5", "length": 4, "samples": samples,
                "mode": mode, "label_source": source, "seed": seed,
                "noise_kind": noise_kind, "noise_rate": noise_rate}
    for key, value in expected.items():
        assert config.get(key) == value, f"{name}: {key}={config.get(key)}, expected {value}"
    assert complete["frozen_sha256_before"] == complete["frozen_sha256_after"], name
    assert complete["steps"] == 1000, name
    assert Path(config["init"]).parent.name == f"warm_L2_second_s{seed}", name
    rows = [row for row in evaluation["evaluations"] if row["length"] == 8]
    assert len(rows) == 1 and not rows[0].get("oracle_attention"), name
    row = rows[0]
    assert row["teacher_forced"]["samples"] == 1024, name
    return {"run": name, "seed": seed, "mode": mode, "source": source, "samples": samples,
            "noise_kind": noise_kind, "requested_noise_rate": noise_rate,
            "actual_noise_rate": audit.get("actual_noise_rate", 0),
            "chain_accuracy": row["teacher_forced"]["metrics"][VALUE_METRIC],
            "rollout_final_accuracy": row.get("rollout", {}).get("metrics", {}).get("final_value_accuracy"),
            "unique_labels": complete["unique_labels"], "labels_presented": complete["labels_presented"],
            "examples_presented": complete["examples_presented"], "steps": complete["steps"],
            "budget_label": config["budget_label"], "batch": config["batch"],
            "pool_samples": config.get("pool_samples"), "data_seed": config.get("data_seed"),
            "panel_sha256": row["panel_sha256"], "model_config": config["model"],
            "source_sha256": config["source_sha256"],
            "evaluation_path": str((directory / "evaluation.json").resolve()),
            "evaluation_sha256": sha(directory / "evaluation.json")}


def stats(records):
    assert sorted(record["seed"] for record in records) == [0, 1, 2]
    values = [record["chain_accuracy"] for record in records]
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values),
            "seeds": [record["seed"] for record in records], "values": values,
            "runs": [record["run"] for record in records], "n": 3}


def categorical_panel(ax, cells, title, extra_labels=None):
    for source in SOURCES:
        offset = -.16 if source == "truth" else .16
        for index, mode in enumerate(MODES):
            summary = cells[(source, mode)]
            x = index + offset
            xs = [x + delta for delta in [-.045, 0, .045]]
            ax.scatter(xs, [100 * value for value in summary["values"]], s=27,
                       facecolors="none", edgecolors=COLORS[source], linewidths=1, zorder=3)
            ax.scatter([x], [100 * summary["mean"]], color=COLORS[source], marker="_",
                       s=260, linewidths=2.4, zorder=4)
    labels = MODE_NAMES if extra_labels is None else [f"{label}\n{extra}" for label, extra in zip(MODE_NAMES, extra_labels)]
    ax.set(xticks=range(3), xticklabels=labels, title=title, ylim=(-3, 103),
           ylabel="L8 exact full-clause chain accuracy (%)")
    ax.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], color=COLORS[s], marker="_", linewidth=0,
                      markersize=14, markeredgewidth=2, label=SOURCE_NAMES[s]) for s in SOURCES]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower left")


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=project / "runs" / CAMPAIGN)
    parser.add_argument("--output", type=Path, default=project / "figures" / CAMPAIGN)
    args = parser.parse_args()
    campaign = read(args.runs / "campaign-complete.json")
    assert campaign.get("complete") is True and not campaign.get("failures"), "Campaign incomplete or failed"
    planned = {spec["name"] for spec in read(args.runs / "campaign-plan.json")["runs"]}
    assert planned == set(campaign["completed"]), "Campaign completion does not cover its plan"
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}

    def take(name, **kwargs):
        assert name in planned, f"Unplanned run: {name}"
        if name not in records:
            records[name] = load_run(args.runs, name, **kwargs)
        return records[name]

    main_cells, budget_cells, noise_cells = {}, {}, {}
    for source in SOURCES:
        for mode in MODES:
            main_rows = [take(f"extend_L4_N4096_{source}_{mode}_s{seed}", mode=mode,
                              source=source, samples=4096, seed=seed) for seed in range(3)]
            main_cells[(source, mode)] = stats(main_rows)
            if mode == "all":
                budget_rows = [take(f"labels_L4_N1024_{source}_all_s{seed}", mode=mode,
                                    source=source, samples=1024, seed=seed) for seed in range(3)]
            else:
                budget_rows = main_rows
            assert all(row["unique_labels"] == 4096 and row["labels_presented"] == 128000
                       for row in budget_rows), "Value-label budgets do not match"
            budget_cells[(source, mode)] = stats(budget_rows)
    for kind in NOISE_NAMES:
        noise_cells[(kind, 0)] = main_cells[("truth", "second")]
        for rate in [.05, .2, .4]:
            rows = [take(f"noise_L4_{kind}_r{round(rate*100):02d}_s{seed}", mode="second",
                         source="truth", samples=4096, seed=seed, noise_kind=kind,
                         noise_rate=rate) for seed in range(3)]
            noise_cells[(kind, rate)] = stats(rows)
    # These factors are fixed for every displayed cell. Budget-panel N/batch
    # differences are intentional and explicitly exposed in labels and CSV.
    for key in ["panel_sha256", "pool_samples", "data_seed", "model_config", "source_sha256"]:
        assert len({json.dumps(row[key], sort_keys=True) for row in records.values()}) == 1, f"Mixed {key} in overview"

    parent_rows = []
    for seed in range(3):
        name = f"warm_L2_second_s{seed}"
        assert name in planned
        directory = args.runs / name
        config = read(directory / "config.json")
        complete = read(directory / "complete.json")
        evaluation = read(directory / "evaluation.json")
        assert complete.get("complete") and evaluation.get("complete")
        assert config["length"] == 2 and config["mode"] == "second" and config["seed"] == seed
        assert complete["frozen_sha256_before"] == complete["frozen_sha256_after"]
        row = next(row for row in evaluation["evaluations"] if row["length"] == 8)
        assert row["panel_sha256"] == next(iter(records.values()))["panel_sha256"]
        parent_rows.append({"run": name, "seed": seed,
                            "chain_accuracy": row["teacher_forced"]["metrics"][VALUE_METRIC],
                            "evaluation_path": str((directory / "evaluation.json").resolve()),
                            "evaluation_sha256": sha(directory / "evaluation.json")})
    parent_summary = stats(parent_rows)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.35))
    categorical_panel(axes[0], main_cells, "A  Equal examples and updates\nN = 4096; 1000 updates")
    axes[0].axhline(100 * parent_summary["mean"], linestyle="--", color="0.35", linewidth=1.2,
                    label=f"L2 parent, no further training: {100 * parent_summary['mean']:.1f}%")
    handles, labels = axes[0].get_legend_handles_labels()
    handles = [Line2D([], [], color=COLORS[s], marker="_", linewidth=0,
                      markersize=14, markeredgewidth=2, label=SOURCE_NAMES[s]) for s in SOURCES] + handles
    axes[0].legend(handles=handles, fontsize=7.6, frameon=False, loc="lower left")
    categorical_panel(axes[1], budget_cells, "B  Equal value-label budgets\n4096 unique / 128000 presented labels",
                      ["N = 4096", "N = 4096", "N = 1024"])
    ax = axes[2]
    for kind in NOISE_NAMES:
        rates = [0, .05, .2, .4]
        summaries = [noise_cells[(kind, rate)] for rate in rates]
        x = [100 * rate for rate in rates]
        ax.plot(x, [100 * cell["mean"] for cell in summaries], color=COLORS[kind], marker="o",
                markersize=4, linewidth=1.5, label=NOISE_NAMES[kind])
        ax.fill_between(x, [100 * cell["min"] for cell in summaries],
                        [100 * cell["max"] for cell in summaries], color=COLORS[kind], alpha=.13)
    ax.set(title="C  Noise type at fixed sparse supervision\nSecond step; N = 4096; true-label baseline",
           xlabel="Requested corruption probability (%)", ylabel="L8 exact full-clause chain accuracy (%)",
           xticks=[0, 5, 20, 40], ylim=(-3, 103))
    ax.grid(alpha=.2)
    ax.legend(fontsize=7.3, frameon=False, loc="lower left")
    fig.suptitle("S5: frozen local computation, L4 training → L8 evaluation", fontsize=12, y=1.015)
    fig.text(.5, .005, "Three training seeds. A/B: hollow points = individual seeds; horizontal marks = means. "
             "C: mean and observed seed range, not a confidence interval.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(args.output / f"sparse_overview.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    flat = []
    for panel, cells in [("equal_examples", main_cells), ("equal_value_labels", budget_cells)]:
        for (source, mode), summary in cells.items():
            flat.append({"panel": panel, "source": source, "mode": mode, "noise_kind": "none",
                         "requested_noise_rate": 0, **summary})
    for (kind, rate), summary in noise_cells.items():
        flat.append({"panel": "noise", "source": "truth", "mode": "second",
                     "noise_kind": kind, "requested_noise_rate": rate, **summary})
    document = {"schema_version": 1, "campaign": str(args.runs.resolve()),
                "metric": "teacher_forced.metrics.clause_chain_accuracy",
                "training_length": 4, "evaluation_length": 8, "samples_per_seed": 1024,
                "statistics": "means and observed minimum/maximum across seeds 0,1,2; not CI",
                "cells": flat, "runs": list(records.values()),
                "untrained_parent_baseline": {"summary": parent_summary, "runs": parent_rows},
                "notes": ["Exact full-clause success is equivalent to the deterministic full greedy exact-trajectory event.",
                          "This figure uses only planned runs from this sparse-supervision campaign.",
                          "Panel A matches examples and updates, not supervised label counts.",
                          "Panel B matches unique and presented value labels; example counts and batches differ.",
                          "Panel C shares its zero-noise baseline across three noise types.",
                          "Panel A includes the same-seed L2 parents before L4 training as a horizontal mean baseline; the three parent values are retained.",
                          "Teacher-generated input histories are additional information not counted as target labels.",
                          "Only one length expansion is tested here; multi-round stability remains untested."]}
    (args.output / "sparse_overview.json").write_text(json.dumps(document, indent=2) + "\n")
    with (args.output / "sparse_overview_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows([{k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in flat])
    print(json.dumps({"figure": str((args.output / "sparse_overview.png").resolve()),
                      "runs": len(records), "cells": len(flat), "tables": flat}, indent=2))


if __name__ == "__main__":
    main()
