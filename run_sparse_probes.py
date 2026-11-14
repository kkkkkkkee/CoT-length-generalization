"""Run post-training mechanism probes once the first training panel completes."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'runs/sparse-supervision-2026-11-09'
OUT = CAMPAIGN / 'probes'
OUT.mkdir(exist_ok=True)
names = [pattern.format(seed=seed) for seed in range(3) for pattern in [
    'extend_L4_N4096_truth_second_s{seed}',
    'extend_L4_N4096_teacher_second_s{seed}',
    'extend_L4_N4096_truth_all_s{seed}',
    'noise_L4_target_r20_s{seed}',
    'noise_L4_prefix_inconsistent_r20_s{seed}',
    'noise_L4_prefix_consistent_r20_s{seed}',
]]
plan = {'models': names, 'lengths': [8, 16], 'samples': 512, 'seed': 1950001,
        'concentration': .99, 'interpretation': 'Exploratory paired mechanism diagnostics on a fresh panel; true rule locations used in attention interventions',
        'selection': 'All three seeds; fixed representative conditions, no selection by probe outcome'}
(OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
while not (CAMPAIGN / 'campaign-complete.json').exists():
    progress = CAMPAIGN / 'campaign-progress.json'
    if progress.exists() and json.loads(progress.read_text()).get('failures'):
        raise RuntimeError('Training campaign failed; mechanism probes stopped')
    time.sleep(2)
completed = []
start = time.monotonic()
for name in names:
    output = OUT / name
    if not (output / 'probe.json').exists():
        command = [sys.executable, 'src/sparse_probe.py', '--checkpoint',
                   str(CAMPAIGN / name / 'checkpoint-final.pt'), '--output', str(output),
                   '--lengths', '8', '16', '--samples', '512', '--seed', '1950001', '--device', 'cuda']
        print('START', name, flush=True)
        with (OUT / f'{name}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    completed.append(name)
    print('DONE', name, flush=True)
    tmp = OUT / 'progress.tmp'
    tmp.write_text(json.dumps({'completed': completed, 'total': len(names),
        'complete': len(completed) == len(names), 'seconds': time.monotonic() - start}, indent=2))
    tmp.replace(OUT / 'progress.json')
