"""Launch the prespecified large-scale routing campaign on one GPU.

The campaign is deliberately sequential.  It first creates five additional
admitted local FFN seeds, then runs the main replication and optimization/noise
panels.  Every run has a unique directory and an immutable command in the
manifest.  Existing completed runs are reused; incomplete directories are
never overwritten.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "runs/sparse-supervision-2026-11-09"
OUT = ROOT / "runs/large-scale-2026-11-10"
LOCAL = OUT / "local"

SEEDS = list(range(8))
REPLICATES = list(range(3))
EVAL_LENGTHS = [2, 4, 8, 16, 32, 64]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def parent_for(seed: int) -> Path:
    if seed < 3:
        return OLD / f"local_full_s{seed}" / "checkpoint-final.pt"
    return LOCAL / f"local_full_s{seed}" / "checkpoint-final.pt"


def teacher_for(seed: int) -> Path:
    if seed < 3:
        return OLD / f"warm_L2_second_s{seed}" / "checkpoint-final.pt"
    return OUT / f"warm_L2_second_s{seed}" / "checkpoint-final.pt"


def condition_args(condition: str, parent: Path) -> tuple[list[str], str]:
    if condition == "truth_clean":
        return ["--label-source", "truth", "--noise-kind", "none", "--noise-rate", "0"], "truth"
    if condition == "teacher_clean":
        return ["--label-source", "teacher", "--teacher", parent,
                "--noise-kind", "none", "--noise-rate", "0"], "teacher"
    if condition.startswith("target_"):
        rate = float(condition.split("_")[-1])
        return ["--label-source", "truth", "--noise-kind", "target", "--noise-rate", str(rate)], "truth"
    if condition.startswith("prefix_inconsistent_"):
        rate = float(condition.split("_")[-1])
        return ["--label-source", "truth", "--noise-kind", "prefix_inconsistent",
                "--noise-rate", str(rate)], "truth"
    if condition.startswith("prefix_consistent_"):
        rate = float(condition.split("_")[-1])
        return ["--label-source", "truth", "--noise-kind", "prefix_consistent",
                "--noise-rate", str(rate)], "truth"
    raise ValueError(condition)


def routing_command(name: str, seed: int, replicate: int, routing: str,
                    length: int, lr: float, condition: str,
                    phase: str, steps: int = 1000,
                    init: Path | None = None, teacher: Path | None = None) -> dict:
    parent = init or parent_for(seed)
    cargs, source = condition_args(condition, teacher or parent)
    # Noise/data streams are nested by seed and replicate, but are shared
    # across routing/learning-rate conditions for paired comparisons.
    data_seed = 4810001 + 10000 * seed + 100 * replicate + length
    noise_seed = 4910001 + 10000 * seed + 100 * replicate + length
    args: list[object] = [sys.executable, "src/routing_experiment.py",
        "--output", OUT / name, "--init", parent, "--routing-mode", routing,
        "--replicate", replicate, "--seed", seed, "--length", length,
        "--samples", 4096, "--pool-samples", 16384, "--steps", steps,
        "--batch", 128, "--lr", lr, "--mode", "second", *cargs,
        "--data-seed", data_seed, "--noise-seed", noise_seed,
        "--eval-seed", 5840001, "--validation-seed", 5740001,
        "--eval-samples", 1024, "--validation-samples", 512,
        "--rollout-samples", 128, "--eval-lengths", *EVAL_LENGTHS,
        # Endpoint runs retain initial/final checkpoints while smoke runs
        # below use the default interval to expose training dynamics.
        "--eval-every", 1000, "--budget-label",
        f"large_scale_{phase}_paired_seed_replicate"]
    return {"name": name, "phase": phase, "seed": seed,
            "replicate": replicate, "routing": routing, "length": length,
            "lr": lr, "condition": condition, "label_source": source,
            "command": [str(x) for x in args]}


def build_specs() -> list[dict]:
    specs: list[dict] = []
    # Phase A: eight local seeds, three independent data/noise replicates,
    # four paired conditions.  This is the primary robustness panel.
    for seed in SEEDS:
        for rep in REPLICATES:
            for routing in ("dense", "row_tied"):
                for condition in ("truth_clean", "teacher_clean", "target_0.2",
                                  "prefix_inconsistent_0.2"):
                    name = f"A_{routing}_{condition}_s{seed}_r{rep}"
                    specs.append(routing_command(name, seed, rep, routing, 4,
                                                  .03, condition, "A",
                                                  teacher=teacher_for(seed)
                                                  if condition == "teacher_clean" else None))
    # Phase B: fixed replicate 0 isolates optimization and noise dose.  The
    # two prefix controls are kept at lr=.03 to avoid multiplying the main
    # dose panel unnecessarily.
    for seed in SEEDS:
        for routing in ("dense", "row_tied"):
            for lr in (.003, .01, .03):
                for rate in (0., .05, .20, .40):
                    condition = "target_0" if rate == 0. else f"target_{rate}"
                    name = f"B_{routing}_lr{lr:g}_{condition}_s{seed}"
                    specs.append(routing_command(name, seed, 0, routing, 4,
                                                  lr, condition, "B"))
                for kind in ("prefix_inconsistent", "prefix_consistent"):
                    condition = f"{kind}_0.2"
                    name = f"B_{routing}_lr{lr:g}_{condition}_s{seed}"
                    specs.append(routing_command(name, seed, 0, routing, 4,
                                                  lr, condition, "B"))
    return specs


def local_specs() -> list[dict]:
    result = []
    for seed in range(3, 8):
        name = f"local_full_s{seed}"
        output = LOCAL / name
        result.append({"name": name, "phase": "local_pretraining", "seed": seed,
            "command": [str(x) for x in [sys.executable, "src/sparse_experiment.py", "--output", output,
                "--stage", "local", "--seed", seed, "--samples", 32768,
                "--steps", 16000, "--batch", 256, "--lr", .001,
                "--eval-every", 500, "--data-seed", 1810001 + 1000 * seed]]})
    return result


def warm_specs() -> list[dict]:
    result = []
    for seed in range(3, 8):
        name = f"warm_L2_second_s{seed}"
        result.append(routing_command(name, seed, 0, "dense", 2, .03,
                                      "truth_clean", "warm",
                                      init=parent_for(seed)))
    return result


def smoke_specs() -> list[dict]:
    result = []
    for seed in (0, 1):
        for routing in ("dense", "row_tied"):
            for condition in ("truth_clean", "teacher_clean", "target_0.2"):
                name = f"SMOKE_{routing}_{condition}_s{seed}"
                result.append(routing_command(name, seed, 0, routing, 4,
                                              .03, condition, "smoke", steps=25,
                                              teacher=teacher_for(seed)
                                              if condition == "teacher_clean" else None))
    return result


def ensure_free_space() -> None:
    free = shutil.disk_usage(OUT.parent).free
    if free < 10 * 1024**3:
        raise RuntimeError(f"Refusing campaign with only {free / 2**30:.1f} GiB free")


def run_one(spec: dict) -> None:
    output = Path(spec["command"][spec["command"].index("--output") + 1])
    complete = output / "complete.json"
    if complete.exists():
        result = json.loads(complete.read_text())
        if not result.get("complete"):
            raise RuntimeError(f"completion marker is false: {output}")
        if spec["phase"] == "local_pretraining" and not result.get("local_gate_passed"):
            raise RuntimeError(f"local admission gate failed: {output}")
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    log = OUT / "logs" / f"{spec['name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print("START", spec["name"], flush=True)
    with log.open("w") as handle:
        subprocess.run([str(x) for x in spec["command"]], cwd=ROOT,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    if not complete.exists() or not json.loads(complete.read_text()).get("complete"):
        raise RuntimeError(f"run did not complete: {output}")
    if spec["phase"] == "local_pretraining":
        if not json.loads(complete.read_text()).get("local_gate_passed"):
            raise RuntimeError(f"local admission gate failed: {output}")
    print("DONE", spec["name"], flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_free_space()
    smoke = smoke_specs()
    local = local_specs()
    warm = warm_specs()
    specs = build_specs()
    manifest = {"created_at": time.time(), "smoke": smoke, "local_pretraining": local,
                "warmup": warm,
                "runs": specs, "counts": {"local": len(local),
                    "smoke": len(smoke),
                    "warmup": len(warm),
                    "phase_A": sum(s["phase"] == "A" for s in specs),
                    "phase_B": sum(s["phase"] == "B" for s in specs)},
                "primary_endpoint": "L8/L16/L64 teacher-forced full-clause chain accuracy",
                "design": "paired local-model seed x data/noise replicate; FFN frozen; second-step value supervision",
                "storage_policy": "endpoint runs use eval-every=1000; initial/final checkpoints retained"}
    write_json(OUT / "campaign-plan.json", manifest)
    all_specs = smoke + local + warm + specs
    completed: list[str] = []
    failures: list[dict] = []
    start = time.monotonic()
    write_json(OUT / "campaign-progress.json", {"complete": False,
        "total": len(all_specs), "completed": completed, "failures": failures})
    for spec in all_specs:
        try:
            run_one(spec)
            completed.append(spec["name"])
        except Exception as exc:
            failures.append({"run": spec["name"], "error": repr(exc)})
            write_json(OUT / "campaign-progress.json", {"complete": False,
                "total": len(all_specs), "completed": completed,
                "failures": failures, "seconds": time.monotonic() - start})
            raise
        write_json(OUT / "campaign-progress.json", {"complete": False,
            "total": len(all_specs), "completed": completed,
            "failures": failures, "seconds": time.monotonic() - start,
            "current": spec["name"]})
        ensure_free_space()
    write_json(OUT / "campaign-complete.json", {"complete": True,
        "total": len(all_specs), "completed": completed, "failures": failures,
        "seconds": time.monotonic() - start})
    write_json(OUT / "campaign-progress.json", {"complete": True,
        "total": len(all_specs), "completed": completed, "failures": failures,
        "seconds": time.monotonic() - start})


if __name__ == "__main__":
    main()
