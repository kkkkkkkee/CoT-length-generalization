"""Run the eight-seed Phase C length-curriculum follow-up.

This campaign is isolated from the completed large-scale and convergence
campaigns.  It creates fresh L2 warmups for both dense and row_tied routing,
then runs L4 -> L8 -> L16 -> L32 under self-teacher, truth-curriculum, and
continue-L2 controls.  Every command is recorded before execution, incomplete
directories are never overwritten, and source hashes are checked between runs.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "runs/sparse-supervision-2026-11-09"
LARGE = ROOT / "runs/large-scale-2026-11-10"
OUT = ROOT / "runs/phase-c-length-curriculum-2026-11-10"
SEEDS = tuple(range(8))
ROUTINGS = ("dense", "row_tied")
REGIMES = ("self", "truth", "continue_L2")
CURRICULUM = (4, 8, 16, 32)
EVAL_LENGTHS = (2, 4, 8, 16, 32, 64)
SOURCE_FILES = (
    "run_phase_c_length_curriculum.py",
    "src/routing_experiment.py",
    "src/routing_model.py",
    "src/sparse_experiment.py",
    "src/sparse_model.py",
    "src/sparse_data.py",
    "src/data.py",
    "src/evaluate.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def local_parent(seed: int) -> Path:
    if seed < 3:
        return OLD / f"local_full_s{seed}" / "checkpoint-final.pt"
    return LARGE / "local" / f"local_full_s{seed}" / "checkpoint-final.pt"


def warm_name(routing: str, seed: int) -> str:
    return f"warm_{routing}_s{seed}"


def stage_name(routing: str, regime: str, stage: int, seed: int) -> str:
    return f"{routing}_{regime}_stage{stage}_s{seed}"


def base_command(name: str, seed: int, init: Path, routing: str,
                 length: int, data_seed: int, noise_seed: int,
                 steps: int = 1000) -> list[str]:
    return [str(x) for x in [
        sys.executable, "src/routing_experiment.py", "--output", OUT / name,
        "--init", init, "--routing-mode", routing, "--seed", seed,
        "--replicate", 0, "--length", length, "--samples", 4096,
        "--pool-samples", 16384, "--steps", steps, "--batch", 128,
        "--lr", .03, "--mode", "second", "--label-source", "truth",
        "--noise-kind", "none", "--noise-rate", 0,
        "--data-seed", data_seed, "--noise-seed", noise_seed,
        "--eval-seed", 7240001, "--validation-seed", 7140001,
        "--eval-samples", 1024 if steps > 25 else 128,
        "--validation-samples", 512 if steps > 25 else 128,
        "--rollout-samples", 128 if steps > 25 else 16,
        "--eval-lengths", *EVAL_LENGTHS,
        "--eval-every", 1000 if steps > 25 else 25,
        "--budget-label", "phase_c_length_curriculum_equal_updates",
    ]]


def warm_specs() -> list[dict]:
    specs = []
    for routing in ROUTINGS:
        for seed in SEEDS:
            name = warm_name(routing, seed)
            cmd = base_command(name, seed, local_parent(seed), routing, 2,
                               7210001 + 10000 * seed,
                               7220001 + 10000 * seed)
            specs.append({"name": name, "phase": "warmup", "routing": routing,
                          "seed": seed, "length": 2, "command": cmd})
    return specs


def curriculum_specs() -> list[dict]:
    specs = []
    for routing in ROUTINGS:
        for regime in REGIMES:
            for stage, curriculum_length in enumerate(CURRICULUM, 1):
                for seed in SEEDS:
                    name = stage_name(routing, regime, stage, seed)
                    parent_name = (warm_name(routing, seed) if stage == 1 else
                                   stage_name(routing, regime, stage - 1, seed))
                    parent = OUT / parent_name / "checkpoint-final.pt"
                    train_length = 2 if regime == "continue_L2" else curriculum_length
                    cmd = base_command(
                        name, seed, parent, routing, train_length,
                        7230001 + 100000 * seed + 1000 * stage + (0 if regime == "self" else 100),
                        7240001 + 100000 * seed + 1000 * stage + (0 if regime == "self" else 100),
                    )
                    if regime == "self":
                        cmd[cmd.index("--label-source") + 1] = "teacher"
                        cmd += ["--teacher", str(parent)]
                    specs.append({"name": name, "phase": "curriculum",
                                  "routing": routing, "regime": regime,
                                  "seed": seed, "stage": stage,
                                  "curriculum_length": curriculum_length,
                                  "training_length": train_length,
                                  "command": cmd})
    return specs


def smoke_specs() -> list[dict]:
    specs = []
    for seed in (0, 1):
        for routing in ROUTINGS:
            for condition in ("truth", "teacher", "target"):
                name = f"SMOKE_{routing}_{condition}_s{seed}"
                cmd = base_command(name, seed, local_parent(seed), routing, 4,
                                   7250001 + 100 * seed,
                                   7260001 + 100 * seed, steps=25)
                if condition == "teacher":
                    cmd[cmd.index("--label-source") + 1] = "teacher"
                    cmd += ["--teacher", str(local_parent(seed))]
                elif condition == "target":
                    cmd[cmd.index("--noise-kind") + 1] = "target"
                    cmd[cmd.index("--noise-rate") + 1] = "0.2"
                specs.append({"name": name, "phase": "smoke", "seed": seed,
                              "routing": routing, "condition": condition,
                              "command": cmd})
    return specs


def ensure_space() -> None:
    free = shutil.disk_usage(OUT.parent).free
    if free < 10 * 2**30:
        raise RuntimeError(f"Refusing campaign with only {free / 2**30:.1f} GiB free")


def run_one(spec: dict) -> None:
    output = Path(spec["command"][spec["command"].index("--output") + 1])
    marker = output / "complete.json"
    if marker.exists():
        if not json.loads(marker.read_text()).get("complete"):
            raise RuntimeError(f"invalid completion marker: {output}")
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    log = OUT / "logs" / f"{spec['name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print("START", spec["name"], flush=True)
    with log.open("w") as handle:
        subprocess.run(spec["command"], cwd=ROOT, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    if not marker.exists() or not json.loads(marker.read_text()).get("complete"):
        raise RuntimeError(f"run did not complete: {output}")
    print("DONE", spec["name"], flush=True)


def check_warm_gate() -> bool:
    """Require every row_tied L2 warmup to pass independent L4 validation."""
    validation = []
    for seed in SEEDS:
        history = OUT / warm_name("row_tied", seed) / "history.jsonl"
        if not history.exists():
            raise RuntimeError(f"missing row_tied warmup history: {history}")
        rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        if not rows:
            raise RuntimeError(f"empty row_tied warmup history: {history}")
        panel = rows[-1].get("panel", {})
        item = panel.get("4", panel.get(4))
        if item is None:
            raise RuntimeError(f"warmup history lacks L4 panel: {history}")
        validation.append({"seed": seed, "step": rows[-1].get("step"),
                           "L4_chain": item.get("clause_chain_accuracy")})
    passed = all(row["L4_chain"] is not None and row["L4_chain"] >= .95
                 for row in validation)
    write_json(OUT / "gate.json", {"passed": passed, "threshold": .95,
                                   "all_seeds_required": True,
                                   "validation": validation})
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true",
                        help="run the 12 smoke commands and stop")
    parser.add_argument("--max-runs", type=int,
                        help="run at most this many selected commands, then stop")
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    return args


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_space()
    smoke = smoke_specs()
    warm = warm_specs()
    curriculum = curriculum_specs()
    all_specs = smoke + warm + curriculum
    selected_specs = smoke if args.smoke_only else all_specs
    if args.max_runs is not None:
        selected_specs = selected_specs[:args.max_runs]
    manifest = {
        "created_at": time.time(),
        "runs": all_specs,
        "counts": {"smoke": len(smoke), "warmup": len(warm),
                    "curriculum": len(curriculum), "total": len(all_specs)},
        "design": "Eight local FFN seeds, dense/row_tied, L2 warmup then L4-L32 self/truth/continue-L2; every stage 4096 examples and 1000 Adam updates",
        "primary_endpoint": "Final L64 TF and 128-example greedy full-clause chain accuracy",
        "source_sha256": {p: sha(ROOT / p) for p in SOURCE_FILES},
        "limits": ["This tests the finite Adam/GELU protocol, not the theorem setting",
                   "continue-L2 matches update and value-label presentations, not FLOPs",
                   "All stages are retained; no final-test seed selection"],
    }
    plan = OUT / "campaign-plan.json"
    if plan.exists():
        old = json.loads(plan.read_text())
        if old.get("source_sha256") != manifest["source_sha256"] or old.get("counts") != manifest["counts"]:
            raise RuntimeError("Existing manifest differs; use a new output directory")
    else:
        write_json(plan, manifest)
    completed, failures = [], []
    started = time.monotonic()
    gate_checked = False
    for spec in selected_specs:
        if spec["phase"] == "curriculum" and not gate_checked:
            gate_checked = True
            if not check_warm_gate():
                result = {"complete": True, "gate_passed": False,
                          "total": len(selected_specs), "planned_total": len(all_specs),
                          "completed": completed, "failures": failures,
                          "seconds": time.monotonic() - started}
                write_json(OUT / "gate-failed.json", result)
                write_json(OUT / "campaign-progress.json", result)
                print("ROW_TIED WARMUP GATE FAILED; curriculum not started", flush=True)
                return
        if any(sha(ROOT / p) != h for p, h in manifest["source_sha256"].items()):
            raise RuntimeError("Source changed during campaign")
        try:
            run_one(spec)
            completed.append(spec["name"])
        except Exception as exc:
            failures.append({"run": spec["name"], "error": repr(exc)})
            write_json(OUT / "campaign-progress.json", {"complete": False,
                "total": len(selected_specs), "planned_total": len(all_specs),
                "completed": completed,
                "failures": failures, "seconds": time.monotonic() - started})
            raise
        write_json(OUT / "campaign-progress.json", {"complete": False,
            "total": len(selected_specs), "planned_total": len(all_specs),
            "completed": completed,
            "failures": failures, "current": spec["name"],
            "seconds": time.monotonic() - started})
        ensure_space()
    result = {"complete": True, "total": len(selected_specs),
              "planned_total": len(all_specs),
              "completed": completed, "failures": failures,
              "seconds": time.monotonic() - started}
    marker = "campaign-complete.json" if len(selected_specs) == len(all_specs) else "partial-complete.json"
    write_json(OUT / marker, result)
    write_json(OUT / "campaign-progress.json", result)


if __name__ == "__main__":
    main()
