"""Manifest and launcher for a controlled query-key coverage follow-up.

This is a new campaign entry point.  It invokes
``src/pair_coverage_experiment.py`` and leaves all historical campaign
directories and source snapshots untouched.  The default matrix is 4 local
FFN seeds x 2 data replicates x 4 paired treatments (32 full runs); smoke mode
uses four 25-step runs.  No run is started merely by importing this module.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/pair-coverage-control-2026-11-10'
SEEDS = range(4)
REPLICATES = range(2)
TREATMENTS = [
    ('uniform_dense', 'uniform', 'dense'),
    ('balanced_dense', 'balanced', 'dense'),
    ('uniform_row_tied', 'uniform', 'row_tied'),
    ('balanced_row_tied', 'balanced', 'row_tied'),
]
SOURCES = [
    'run_pair_coverage_control.py', 'src/pair_coverage_experiment.py',
    'src/routing_experiment.py', 'src/routing_model.py',
    'src/routing_augmentation.py', 'src/sparse_model.py',
    'src/sparse_experiment.py', 'src/sparse_data.py', 'src/data.py',
    'src/evaluate.py',
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def parent(seed):
    if seed < 3:
        return ROOT / 'runs/sparse-supervision-2026-11-09' / f'local_full_s{seed}' / 'checkpoint-final.pt'
    return ROOT / 'runs/large-scale-2026-11-10/local' / f'local_full_s{seed}' / 'checkpoint-final.pt'


def spec(seed, rep, treatment, smoke=False):
    label, coverage, routing = treatment
    phase = 'smoke' if smoke else 'main'
    name = f'{phase}_{label}_s{seed}_r{rep}'
    steps = 25 if smoke else 1000
    command = [sys.executable, 'src/pair_coverage_experiment.py',
               '--coverage-mode', coverage, '--output', str(OUT / name),
               '--init', str(parent(seed)), '--seed', str(seed),
               '--replicate', str(rep), '--routing-mode', routing,
               '--length', '4', '--samples', '4096', '--pool-samples', '16384',
               '--steps', str(steps), '--batch', '128', '--lr', '.03',
               '--mode', 'second', '--label-source', 'truth',
               '--data-seed', str(6710001 + seed * 10000 + rep * 100),
               '--noise-seed', str(6810001 + seed * 10000 + rep * 100),
               '--augmentation-seed', str(6820001 + seed * 10000 + rep * 100),
               '--eval-seed', '6910001', '--validation-seed', '6920001',
               '--eval-samples', str(128 if smoke else 1024),
               '--validation-samples', str(128 if smoke else 512),
               '--rollout-samples', str(16 if smoke else 128),
               '--eval-lengths', '2', '4', '8', '16', '32', '64',
               '--eval-every', str(25 if smoke else 250),
               '--budget-label', 'coverage_control_equal_examples_updates']
    return {'name': name, 'phase': phase, 'treatment': label,
            'coverage_mode': coverage, 'routing_mode': routing, 'seed': seed,
            'replicate': rep, 'command': command}


def run_one(item):
    directory = OUT / item['name']
    marker = directory / 'complete.json'
    if marker.exists():
        result = json.loads(marker.read_text())
        if not result.get('complete'):
            raise RuntimeError(f'invalid completion: {directory}')
        return
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f'refusing to overwrite incomplete run: {directory}')
    if shutil.disk_usage(OUT).free < 8 * 2**30:
        raise RuntimeError('Less than 8 GiB free')
    log = OUT / 'logs' / f"{item['name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print('START', item['name'], flush=True)
    with log.open('w') as handle:
        subprocess.run(item['command'], cwd=ROOT, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    result = json.loads(marker.read_text())
    if (not result.get('complete')
            or result['frozen_sha256_before'] != result['frozen_sha256_after']
            or result['ff_sha256_before'] != result['ff_sha256_after']):
        raise RuntimeError(f'invalid result/frozen weights: {directory}')
    print('DONE', item['name'], flush=True)


def main(argv=None):
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true',
                        help='run the four short smoke jobs instead of the full matrix')
    parser.add_argument('--dry-run', action='store_true',
                        help='write/print the manifest without launching jobs')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args(argv)
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = [spec(s, 0, t, smoke=True) for s, t in
             [(0, TREATMENTS[0]), (0, TREATMENTS[1]),
              (1, TREATMENTS[0]), (1, TREATMENTS[1])]]
    full = [spec(s, r, t) for s in SEEDS for r in REPLICATES for t in TREATMENTS]
    all_specs = smoke if args.smoke_only else full
    manifest = {
        'runs': full, 'smoke': smoke,
        'source_sha256': {p: sha(ROOT / p) for p in SOURCES},
        'init_sha256': {str(s): sha(parent(s)) for s in SEEDS},
        'design': '4 admitted local FFN seeds x 2 replicates x uniform/balanced pair coverage x dense/row_tied; clean L4 second-step labels; Q=0; 1000 Adam steps at lr=.03',
        'balanced_definition': 'query variable variables[:,1] cycles through IDs; visible nonmatching source slots use distinct cyclic offsets, with random completion of remaining chain variables; actions/states remain uniform',
        'primary_endpoint': 'L8/L16/L64 TF and greedy full-clause chain accuracy',
        'limits': ['Balanced exposure changes the variable-chain distribution and must be audited against uniform action/state and query counts.',
                   'Pair coverage is an exposure intervention, not a proof of causal sufficiency.',
                   'All students start from Q=0; this is not the L2-warm recursive protocol.'],
    }
    plan = OUT / 'campaign-plan.json'
    if plan.exists() and json.loads(plan.read_text()) != manifest:
        raise RuntimeError('Existing manifest differs; use a new output directory')
    if not plan.exists():
        write(plan, manifest)
        for source in SOURCES:
            dest = OUT / 'source' / source
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, dest)
    write(OUT / ('smoke-manifest.json' if args.smoke_only else 'run-manifest.json'), all_specs)
    if args.dry_run:
        print(json.dumps({'output': str(OUT), 'jobs': len(all_specs)}, indent=2))
        return
    lock = (OUT / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed = []
    start = time.monotonic()
    for item in all_specs:
        try:
            run_one(item)
            completed.append(item['name'])
            write(OUT / 'campaign-progress.json', {
                'total': len(all_specs), 'completed': completed,
                'updated_at_unix': time.time(),
                'elapsed_seconds': time.monotonic() - start})
        except Exception as exc:
            write(OUT / 'campaign-progress.json', {
                'total': len(all_specs), 'completed': completed,
                'failure': {'name': item['name'], 'error': repr(exc)},
                'updated_at_unix': time.time(),
                'elapsed_seconds': time.monotonic() - start})
            raise


if __name__ == '__main__':
    main()

