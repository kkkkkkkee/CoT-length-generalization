"""Phase D: longer-budget structural-constraint comparison.

This campaign compares dense routing, two soft row-tying penalties, and hard
row tying under the same frozen local FFNs, data, seeds, and evaluation panel.
It is intentionally separate from the completed Phase C and pair-coverage
campaigns. Completed run directories are skipped; incomplete directories are
never overwritten.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/phase-d-structural-2026-11-10"
SEEDS = range(4)
REPLICATES = range(2)
TREATMENTS = (
    ("dense", "dense", 0.0),
    ("soft_0.1", "dense", 0.1),
    ("soft_1", "dense", 1.0),
    ("row_tied", "row_tied", 0.0),
)
SOURCES = (
    "run_phase_d_structural.py", "src/routing_experiment.py",
    "src/routing_model.py", "src/routing_augmentation.py",
    "src/sparse_model.py", "src/sparse_experiment.py",
    "src/sparse_data.py", "src/data.py", "src/evaluate.py",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def parent(seed: int) -> Path:
    base = ROOT / ("runs/sparse-supervision-2026-11-09" if seed < 3
                   else "runs/large-scale-2026-11-10/local")
    return base / f"local_full_s{seed}/checkpoint-final.pt"


def spec(seed: int, replicate: int, treatment, smoke: bool = False) -> dict:
    label, routing, penalty = treatment
    phase = "smoke" if smoke else "main"
    name = f"{phase}_{label}_s{seed}_r{replicate}"
    steps = 25 if smoke else 2000
    command = [
        sys.executable, "src/routing_experiment.py", "--output", str(OUT / name),
        "--init", str(parent(seed)), "--seed", str(seed),
        "--replicate", str(replicate), "--routing-mode", routing,
        "--row-tie-lambda", str(penalty), "--q-l2-lambda", "0",
        "--length", "4", "--samples", "4096", "--pool-samples", "16384",
        "--steps", str(steps), "--batch", "128", "--lr", ".03",
        "--mode", "second", "--label-source", "truth",
        "--noise-kind", "none", "--noise-rate", "0",
        "--data-seed", str(8310001 + seed * 10000 + replicate * 100),
        "--noise-seed", str(8410001 + seed * 10000 + replicate * 100),
        "--augmentation-seed", str(8420001 + seed * 10000 + replicate * 100),
        "--eval-seed", "8510001", "--validation-seed", "8610001",
        "--eval-samples", str(128 if smoke else 1024),
        "--validation-samples", str(128 if smoke else 512),
        "--rollout-samples", str(16 if smoke else 128),
        "--eval-lengths", "2", "4", "8", "16", "32", "64",
        "--eval-every", str(25 if smoke else 2000),
        "--budget-label", "phase_d_equal_data_longer_budget",
    ]
    return {"name": name, "phase": phase, "treatment": label,
            "seed": seed, "replicate": replicate, "command": command}


def run_one(item: dict) -> None:
    directory = OUT / item["name"]
    marker = directory / "complete.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if not result.get("complete"):
            raise RuntimeError(f"invalid completion marker: {directory}")
        return
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete directory: {directory}")
    if shutil.disk_usage(OUT).free < 8 * 2**30:
        raise RuntimeError("less than 8 GiB free")
    log = OUT / "logs" / f"{item['name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print("START", item["name"], flush=True)
    with log.open("w") as handle:
        subprocess.run(item["command"], cwd=ROOT, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    result = json.loads(marker.read_text())
    if (not result.get("complete")
            or result["frozen_sha256_before"] != result["frozen_sha256_after"]
            or result["ff_sha256_before"] != result["ff_sha256_after"]):
        raise RuntimeError(f"invalid result or frozen state: {directory}")
    print("DONE", item["name"], flush=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    smoke = [spec(seed, 0, treatment, True)
             for seed in (0, 1) for treatment in (TREATMENTS[0], TREATMENTS[2])]
    main_specs = [spec(seed, rep, treatment)
                  for seed in SEEDS for rep in REPLICATES
                  for treatment in TREATMENTS]
    selected = smoke if args.smoke_only else main_specs
    manifest = {
        "runs": main_specs, "smoke": smoke,
        "source_sha256": {p: sha(ROOT / p) for p in SOURCES},
        "init_sha256": {str(s): sha(parent(s)) for s in SEEDS},
        "design": "4 local FFN seeds x 2 data replicates x dense/soft .1/soft 1/row_tied; clean L4 second-step labels; Q=0; 2000 Adam updates at lr=.03",
        "primary_endpoint": "L8/L16/L64 TF and greedy full-clause chain accuracy; history attention margin and row residual",
        "limits": ["Finite Adam/GELU protocol, not theorem architecture or proof.",
                   "Hard tying changes parameterization and optimizer geometry.",
                   "Longer budget is a fixed 2000-update comparison, not asymptotic convergence."],
    }
    plan = OUT / "campaign-plan.json"
    if plan.exists():
        if json.loads(plan.read_text()) != manifest:
            raise RuntimeError("existing Phase D manifest differs")
    else:
        write(plan, manifest)
        for p in SOURCES:
            dest = OUT / "source" / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / p, dest)
    write(OUT / ("smoke-manifest.json" if args.smoke_only else "run-manifest.json"), selected)
    completed = []
    started = time.monotonic()
    for item in selected:
        try:
            run_one(item)
            completed.append(item["name"])
            write(OUT / "campaign-progress.json", {
                "complete": False, "total": len(selected),
                "completed": completed, "current": item["name"],
                "failures": [], "elapsed_seconds": time.monotonic() - started,
                "updated_at_unix": time.time(),
            })
        except Exception as exc:
            write(OUT / "campaign-progress.json", {
                "complete": False, "total": len(selected),
                "completed": completed, "current": item["name"],
                "failures": [{"run": item["name"], "error": repr(exc)}],
                "elapsed_seconds": time.monotonic() - started,
            })
            raise
    result = {"complete": True, "total": len(selected),
              "completed": completed, "failures": [],
              "elapsed_seconds": time.monotonic() - started}
    write(OUT / "campaign-progress.json", result)
    write(OUT / ("smoke-complete.json" if args.smoke_only else "campaign-complete.json"), result)


if __name__ == "__main__":
    main()
