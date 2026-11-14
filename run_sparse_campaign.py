"""Run the prespecified first sparse-supervision panel sequentially on one GPU."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/sparse-supervision-2026-11-09'
OUT.mkdir(parents=True, exist_ok=True)
planned = []


def add(name, args, phase):
    planned.append({'name': name, 'phase': phase, 'command': [sys.executable,
        'src/sparse_experiment.py', '--output', str(OUT / name), *map(str, args)]})


for seed in range(3):
    name = f'local_full_s{seed}'
    add(name, ['--stage', 'local', '--seed', seed, '--samples', 32768,
        '--steps', 16000, '--batch', 256, '--lr', .001, '--eval-every', 500,
        '--data-seed', 1810001 + 1000 * seed], 'local_pretraining')
for seed in range(3):
    for mode in ['second', 'random', 'all']:
        add(f'warm_L2_{mode}_s{seed}', ['--stage', 'attention', '--seed', seed,
            '--init', OUT / f'local_full_s{seed}/checkpoint-final.pt', '--length', 2,
            '--samples', 4096, '--steps', 1000, '--batch', 128, '--lr', .03,
            '--mode', mode, '--data-seed', 1820003], 'clean_L2')
for seed in range(3):
    parent = OUT / f'warm_L2_second_s{seed}/checkpoint-final.pt'
    for samples in [512, 4096]:
        for source in ['truth', 'teacher']:
            for mode in ['second', 'random', 'all']:
                add(f'extend_L4_N{samples}_{source}_{mode}_s{seed}', [
                    '--stage', 'attention', '--seed', seed, '--init', parent,
                    '--teacher', parent, '--length', 4, '--samples', samples,
                    '--steps', 1000, '--batch', 128, '--lr', .03,
                    '--mode', mode, '--label-source', source,
                    '--data-seed', 1820005], 'paired_L4')
    for source in ['truth', 'teacher']:
        add(f'labels_L4_N1024_{source}_all_s{seed}', ['--stage', 'attention',
            '--seed', seed, '--init', parent, '--teacher', parent,
            '--length', 4, '--samples', 1024, '--steps', 1000, '--batch', 32,
            '--lr', .03, '--mode', 'all', '--label-source', source,
            '--data-seed', 1820005, '--budget-label',
            'matched_to_N4096_single_step_unique_and_presented_value_labels'], 'equal_labels')
    for kind in ['target', 'prefix_inconsistent', 'prefix_consistent']:
        for rate in [.05, .20, .40]:
            add(f'noise_L4_{kind}_r{round(rate*100):02d}_s{seed}', [
                '--stage', 'attention', '--seed', seed, '--init', parent,
                '--length', 4, '--samples', 4096, '--steps', 1000, '--batch', 128,
                '--lr', .03, '--mode', 'second', '--data-seed', 1820005,
                '--noise-kind', kind, '--noise-rate', rate], 'noise_decomposition')


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


write(OUT / 'campaign-plan.json', {'runs': planned, 'total': len(planned),
    'seeds': [0, 1, 2], 'description': 'First paired finite-data/noise panel; pilots preserved separately'})
completed, failures = [], []
start = time.monotonic()
for spec in planned:
    directory = OUT / spec['name']
    try:
        if not (directory / 'complete.json').exists():
            print('START', spec['name'], flush=True)
            with (OUT / f"{spec['name']}.log").open('w') as handle:
                subprocess.run(spec['command'], cwd=ROOT, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
        result = json.loads((directory / 'complete.json').read_text())
        if not result.get('complete'):
            raise RuntimeError('Run did not complete')
        if spec['phase'] == 'local_pretraining' and not result.get('local_gate_passed'):
            raise RuntimeError('Local admission gate failed; dependent experiments will not run')
        completed.append(spec['name'])
        print('DONE', spec['name'], flush=True)
    except Exception as exc:
        failures.append({'run': spec['name'], 'error': str(exc)})
        print('FAILED', spec['name'], repr(exc), flush=True)
        write(OUT / 'campaign-progress.json', {'completed': completed, 'failures': failures,
            'total': len(planned), 'complete': False, 'seconds': time.monotonic() - start})
        raise
    write(OUT / 'campaign-progress.json', {'completed': completed, 'failures': failures,
        'total': len(planned), 'complete': len(completed) == len(planned),
        'seconds': time.monotonic() - start})
write(OUT / 'campaign-complete.json', {'complete': True, 'completed': completed,
    'failures': failures, 'total': len(planned), 'seconds': time.monotonic() - start})
