import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
for task in ["C6","S5","C5","aliasC5"]:
    for seed in [0,1,2]:
        d=ROOT/"runs"/f"base_{task}_s{seed}"
        while not (d/"complete.json").exists():
            time.sleep(1)
        path=d/"context/evaluation.json"
        if path.exists() and json.loads(path.read_text()).get("complete"):
            continue
        print("START",d.name,flush=True)
        with (d/"context.log").open("w") as f:
            subprocess.run([sys.executable,"src/context_eval.py","--checkpoint",str(d/"checkpoint-0300.pt"),
                "--output",str(d/"context")],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print("DONE",d.name,flush=True)
(ROOT/"runs/context_complete.json").write_text('{"status":"complete"}')
