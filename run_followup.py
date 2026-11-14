"""Evaluate base panel; complete the S5 recursive pseudo-label curriculum."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def run(args, log):
    log.parent.mkdir(parents=True,exist_ok=True)
    print("START", " ".join(map(str,args)), flush=True)
    with log.open("w") as f:
        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def wait_for(path):
    last = 0
    while not path.exists():
        if time.monotonic()-last > 30:
            print("WAIT",path,flush=True)
            last=time.monotonic()
        time.sleep(1)


def evaluate(run_dir, rollout=512, temps=(0.5,1,2,4)):
    checkpoint=run_dir/"checkpoint-0300.pt"
    path=run_dir/"eval/evaluation.json"
    if path.exists() and json.loads(path.read_text()).get("complete"):
        return
    run(["src/evaluate.py","--checkpoint",checkpoint,"--output",run_dir/"eval",
        "--samples",1024,"--rollout-samples",rollout,"--temperatures",*temps],run_dir/"evaluate.log")


for task in ["C6","S5","C5","aliasC5"]:
    for seed in [0,1,2]:
        d=ROOT/"runs"/f"base_{task}_s{seed}"
        wait_for(d/"complete.json")
        evaluate(d)
        if not (d/"local-rule.json").exists():
            run(["src/local_rule.py","--checkpoint",d/"checkpoint-0300.pt","--output",d/"local-rule.json"],d/"local-rule.log")

for seed in [0,1,2]:
    parent=ROOT/"runs"/f"base_S5_s{seed}"/"checkpoint-0300.pt"
    for length in [10,20,40]:
        d=ROOT/"runs"/f"self_S5_s{seed}_L{length}"
        pseudo=ROOT/"runs"/f"pseudo_S5_s{seed}_L{length}.pt"
        if not pseudo.exists():
            run(["src/generate_pseudo.py","--checkpoint",parent,"--length",length,
                "--samples",8192,"--seed",830001+1000*seed+length,"--output",pseudo],
                ROOT/"runs"/f"pseudo_S5_s{seed}_L{length}.log")
        if not (d/"complete.json").exists():
            run(["src/train.py","--task","S5","--seed",seed,"--length",length,
                "--epochs",300,"--eval-every",25,"--checkpoint-every",100,
                "--resume",parent,"--pseudo-data",pseudo,"--output",d],
                ROOT/"runs"/f"self_S5_s{seed}_L{length}.log")
        evaluate(d,rollout=512,temps=(1,))
        parent=d/"checkpoint-0300.pt"

(ROOT/"runs/followup_complete.json").write_text(json.dumps({"status":"complete"}))
