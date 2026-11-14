"""Fixed exploratory panel: soft row tying, ID augmentation, and norm control."""
from __future__ import annotations
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/convergence-followup-2026-11-10'
SEEDS = range(4)
REPLICATES = range(2)
TREATMENTS = [
    ('dense', 'dense', 0., False, 0.),
    ('soft_0.001', 'dense', .001, False, 0.),
    ('soft_0.01', 'dense', .01, False, 0.),
    ('soft_0.1', 'dense', .1, False, 0.),
    ('soft_1', 'dense', 1., False, 0.),
    ('row_tied', 'row_tied', 0., False, 0.),
    ('rename', 'dense', 0., True, 0.),
    ('rename_soft_0.01', 'dense', .01, True, 0.),
    ('q_l2_0.01', 'dense', 0., False, .01),
]
SOURCES = ['run_convergence_followup.py', 'src/routing_experiment.py',
           'src/routing_model.py', 'src/routing_augmentation.py',
           'src/sparse_model.py', 'src/sparse_experiment.py',
           'src/sparse_data.py', 'src/data.py', 'src/evaluate.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def parent(seed):
    base = ('runs/sparse-supervision-2026-11-09' if seed < 3
            else 'runs/large-scale-2026-11-10/local')
    return ROOT / base / f'local_full_s{seed}/checkpoint-final.pt'


def spec(seed, rep, treatment, smoke=False):
    label, mode, penalty, rename, l2 = treatment
    phase = 'smoke' if smoke else 'main'
    name = f'{phase}_{label}_s{seed}_r{rep}'
    steps = 25 if smoke else 1000
    command = [sys.executable, 'src/routing_experiment.py',
        '--output', OUT/name, '--init', parent(seed), '--seed', seed,
        '--replicate', rep, '--routing-mode', mode,
        '--row-tie-lambda', penalty, '--q-l2-lambda', l2,
        '--augmentation-seed', 6210001 + seed*10000 + rep*100,
        '--length', 4, '--samples', 4096, '--pool-samples', 16384,
        '--steps', steps, '--batch', 128, '--lr', .03,
        '--mode', 'second', '--label-source', 'truth',
        '--data-seed', 6310001 + seed*10000 + rep*100,
        '--noise-seed', 6410001 + seed*10000 + rep*100,
        '--eval-seed', 6510001, '--validation-seed', 6610001,
        '--eval-samples', 128 if smoke else 1024,
        '--validation-samples', 128 if smoke else 512,
        '--rollout-samples', 16 if smoke else 128,
        '--eval-lengths', 2, 4, 8, 16, 32, 64,
        '--eval-every', 25 if smoke else 250,
        '--budget-label', 'exploratory_equal_abstract_traces_examples_updates']
    if rename:
        command.append('--rename-variables')
    return dict(name=name, phase=phase, treatment=label, seed=seed,
                replicate=rep, command=[str(x) for x in command])


def run_one(item):
    directory = OUT/item['name']
    marker = directory/'complete.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        if not result.get('complete'):
            raise RuntimeError(f'invalid completion: {directory}')
        return
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f'refusing to overwrite incomplete run: {directory}')
    if shutil.disk_usage(OUT).free < 8*2**30:
        raise RuntimeError('Less than 8 GiB free')
    print('START', item['name'], flush=True)
    log = OUT/'logs'/f"{item['name']}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open('w') as handle:
        subprocess.run(item['command'], cwd=ROOT, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    result = json.loads(marker.read_text())
    if (not result.get('complete') or result['frozen_sha256_before'] != result['frozen_sha256_after']
            or result['ff_sha256_before'] != result['ff_sha256_after']):
        raise RuntimeError(f'invalid result/frozen weights: {directory}')
    print('DONE', item['name'], flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    main_specs = [spec(s, r, t) for s in SEEDS for r in REPLICATES for t in TREATMENTS]
    smoke_specs = [spec(s, 0, t, True) for s in (0, 1)
                   for t in (TREATMENTS[0], TREATMENTS[2], TREATMENTS[6])]
    manifest = {'runs': main_specs, 'smoke': smoke_specs,
        'source_sha256': {p: sha(ROOT/p) for p in SOURCES},
        'init_sha256': {str(s): sha(parent(s)) for s in SEEDS},
        'design': 'Exploratory 4 fixed local model seeds x 2 independent datasets x 9 treatments. Clean S5 L4 second-step labels; Q=0; frozen FFN. No selection by final outcomes.',
        'primary_endpoint': 'L8/L16/L64 TF and free-generation full-clause chain accuracy',
        'normalization': 'Penalty = mean over two Q blocks of mean squared offdiagonal within-row residuals. Q L2 = mean squared entries.',
        'limits': ['Finite GELU+Adam model, not exact theorem architecture or supervision.',
                   'Fresh ID renaming adds tokenized exposures, keeping abstract traces and value-label count fixed.',
                   'Hard tying changes optimizer geometry; finite soft lambda does not reproduce tied Adam trajectories.',
                   'Four model seeds and two data replicates are exploratory.']}
    plan = OUT/'campaign-plan.json'
    if plan.exists():
        if json.loads(plan.read_text()) != manifest:
            raise RuntimeError('Manifest/source differs; use a new campaign directory')
    else:
        write(plan, manifest)
        for p in SOURCES:
            dest = OUT/'source'/p
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/p, dest)
    completed = []
    start = time.monotonic()
    all_specs = smoke_specs + main_specs
    def progress(current=None, failed=None, complete=False):
        write(OUT/'campaign-progress.json', dict(complete=complete,
              total=len(all_specs), completed=completed, current=current,
              failures=[] if failed is None else [failed],
              elapsed_seconds=time.monotonic()-start, updated_at_unix=time.time()))
    for item in all_specs:
        progress(item['name'])
        try:
            # Prevent source drift during a campaign without a VCS checkout.
            if any(sha(ROOT/p) != h for p,h in manifest['source_sha256'].items()):
                raise RuntimeError('Source changed during campaign')
            run_one(item)
            completed.append(item['name'])
        except Exception as exc:
            progress(item['name'], {'run': item['name'], 'error': repr(exc)})
            raise
        progress(item['name'])
    progress(complete=True)
    write(OUT/'campaign-complete.json', json.loads((OUT/'campaign-progress.json').read_text()))

if __name__ == '__main__':
    main()
