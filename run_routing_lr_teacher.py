"""Exploratory lower-LR teacher control, motivated by the noise sensitivity panel."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'runs/sparse-supervision-2026-11-09'
OUT = ROOT / 'runs/routing-lr-teacher-2026-11-09'
WAIT = ROOT / 'runs/routing-recursive-2026-11-09/campaign-complete.json'
OUT.mkdir(parents=True, exist_ok=True)
runs = []
for seed in range(3):
    parent = OLD / f'warm_L2_second_s{seed}/checkpoint-final.pt'
    name = f'dense_lr003_teacher_s{seed}_r0'
    command = [sys.executable, 'src/routing_experiment.py', '--output', str(OUT / name),
        '--init', str(parent), '--teacher', str(parent), '--routing-mode', 'dense',
        '--seed', str(seed), '--replicate', '0', '--length', '4', '--samples', '4096',
        '--steps', '1000', '--batch', '128', '--lr', '.003', '--mode', 'second',
        '--label-source', 'teacher', '--data-seed', str(2810001 + 10000 * seed),
        '--noise-seed', str(2910001 + 10000 * seed), '--eval-seed', '2940001',
        '--validation-seed', '2840001', '--eval-samples', '1024', '--rollout-samples', '128',
        '--eval-lengths', '2', '4', '8', '16', '32', '64']
    runs.append({'name': name, 'seed': seed, 'replicate': 0, 'routing_mode': 'dense',
                 'condition': 'teacher', 'lr': .003, 'command': command})
(OUT / 'campaign-plan.json').write_text(json.dumps({'runs': runs, 'total': 3,
    'status': 'exploratory follow-up; not part of the original 87-run panel',
    'motivation': 'Check whether the favorable effect of lower LR on true/noisy labels also changes real-teacher self-training',
    'comparison': 'Same seed/replicate0 parent, teacher, data, mask and evaluation as main-panel dense .03 and row_tied .03'}, indent=2))
while not WAIT.exists():
    time.sleep(2)
completed = []
for spec in runs:
    if not (OUT / spec['name'] / 'complete.json').exists():
        print('START', spec['name'], flush=True)
        with (OUT / f"{spec['name']}.log").open('w') as log:
            subprocess.run(spec['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    completed.append(spec['name'])
    print('DONE', spec['name'], flush=True)
(OUT / 'campaign-complete.json').write_text(json.dumps({'complete': True, 'completed': completed,
                                                       'total': 3, 'failures': []}, indent=2))
