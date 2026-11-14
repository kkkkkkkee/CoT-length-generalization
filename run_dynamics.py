"""Serialize fixed-panel dynamics evaluation after each base run completes."""
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
MANIFEST = json.loads((RUNS / "base_manifest.json").read_text())


def atomic_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def main():
    jobs = [RUNS / f"base_{task}_s{seed}"
            for task in MANIFEST["tasks"] for seed in MANIFEST["seeds"]]
    progress = {"completed": [], "failures": [], "total": len(jobs)}
    started = time.monotonic()
    for run in jobs:
        while not (run / "complete.json").exists():
            print(json.dumps({"waiting_for_training": run.name}), flush=True)
            time.sleep(15)
        existing = run / "dynamics.json"
        if existing.exists() and json.loads(existing.read_text()).get("complete"):
            progress["completed"].append(run.name)
            continue
        command = [sys.executable, str(ROOT / "src" / "dynamics.py"),
                   "--run", str(run), "--device", "cuda", "--local-rule"]
        print(json.dumps({"starting": run.name, "command": command}), flush=True)
        with (run / "dynamics.log").open("w") as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            progress["failures"].append({"run": run.name, "returncode": result.returncode})
        else:
            report = json.loads(existing.read_text())
            if report.get("complete"):
                progress["completed"].append(run.name)
            else:
                progress["failures"].append({"run": run.name, "reason": "incomplete panel"})
        progress["elapsed_seconds"] = time.monotonic() - started
        atomic_json(RUNS / "dynamics_campaign_progress.json", progress)
        print(json.dumps(progress), flush=True)
    progress["complete"] = len(progress["completed"]) == len(jobs)
    progress["elapsed_seconds"] = time.monotonic() - started
    atomic_json(RUNS / "dynamics_campaign_progress.json", progress)
    if progress["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
