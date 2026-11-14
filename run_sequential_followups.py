from pathlib import Path
import subprocess, time
root=Path(__file__).resolve().parent
phase=root/'runs/phase-c-length-curriculum-2026-11-10/campaign-complete.json'
while not phase.exists():
    time.sleep(15)
subprocess.run(['python','run_pair_coverage_control.py','--smoke-only'],cwd=root,check=True)
subprocess.run(['python','run_pair_coverage_control.py'],cwd=root,check=True)
