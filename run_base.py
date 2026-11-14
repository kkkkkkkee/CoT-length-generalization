"""Run the preregistered four-task/three-seed base panel sequentially on one GPU."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
manifest = {"tasks": ["C6", "S5", "C5", "aliasC5"], "seeds": [0,1,2],
            "epochs": 300, "samples":8192,"batch":128,"lr":1e-4,
            "nvars":256,"token_dim":64,"length":5}
(root/"runs/base_manifest.json").write_text(json.dumps(manifest,indent=2))
for task in manifest["tasks"]:
    for seed in manifest["seeds"]:
        out = root / "runs" / f"base_{task}_s{seed}"
        if (out / "complete.json").exists():
            continue
        print(f"START {task} seed={seed}",flush=True)
        with (root / "runs" / f"base_{task}_s{seed}.log").open("w") as log:
            subprocess.run([sys.executable,"src/train.py","--task",task,"--seed",str(seed),
                "--epochs","300","--eval-every","5","--checkpoint-every","25",
                "--output",str(out)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f"DONE {task} seed={seed}",flush=True)
