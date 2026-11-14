"""Prespecified from-zero routing warmup and paired recursive curricula.

The performance gate is evaluated on a separate validation panel for ALL
row_tied seeds. Once admitted, every dense and row_tied branch runs all stages,
including failures; no checkpoint or seed is selected by final-test outcomes.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'runs/sparse-supervision-2026-11-09'
PRIMARY = ROOT / 'runs/routing-round2-2026-11-09'
OUT = ROOT / 'runs/routing-recursive-2026-11-09'


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def build_plan():
    specs = []
    for routing in ['dense', 'row_tied']:
        for seed in range(3):
            name = f'warm_{routing}_s{seed}'
            args = ['--routing-mode', routing, '--init', OLD / f'local_full_s{seed}/checkpoint-final.pt',
                    '--length', 2, '--data-seed', 3010001 + 10000 * seed,
                    '--noise-seed', 3110001 + 10000 * seed]
            specs.append({'name': name, 'phase': 'warmup', 'routing': routing,
                          'seed': seed, 'regime': 'truth', 'length': 2, 'args': args})
    for stage, length in enumerate([4, 8, 16, 32], 1):
        for routing in ['dense', 'row_tied']:
            for seed in range(3):
                for regime in ['self', 'truth', 'continue_L2']:
                    name = f'{routing}_{regime}_stage{stage}_s{seed}'
                    parent_name = f'warm_{routing}_s{seed}' if stage == 1 else f'{routing}_{regime}_stage{stage - 1}_s{seed}'
                    parent = OUT / parent_name / 'checkpoint-final.pt'
                    training_length = 2 if regime == 'continue_L2' else length
                    args = ['--routing-mode', routing, '--init', parent, '--length', training_length,
                            '--data-seed', 3010001 + 10000 * seed + 100 * stage,
                            '--noise-seed', 3110001 + 10000 * seed + 100 * stage,
                            '--label-source', 'teacher' if regime == 'self' else 'truth']
                    if regime == 'self':
                        args += ['--teacher', parent]
                    specs.append({'name': name, 'phase': 'recursive', 'routing': routing, 'seed': seed,
                        'regime': regime, 'length': training_length, 'curriculum_length': length,
                        'stage': stage, 'args': args})
    for spec in specs:
        args = spec.pop('args')
        spec['command'] = [sys.executable, 'src/routing_experiment.py', '--output', str(OUT / spec['name']),
            '--seed', str(spec['seed']), '--samples', '4096', '--pool-samples', '16384',
            '--steps', '1000', '--batch', '128', '--lr', '.03', '--mode', 'second',
            '--eval-seed', '3240001', '--validation-seed', '3140001',
            '--eval-lengths', '2', '4', '8', '16', '32', '64',
            '--eval-samples', '1024', '--rollout-samples', '128', *map(str, args)]
    return specs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = build_plan()
    write(OUT / 'campaign-plan.json', {'runs': specs, 'total': len(specs),
        'gate': 'Each of the three row_tied warmup seeds must reach validation L4 exact-clause-chain >= .95; fixed final step only',
        'primary_endpoint': 'Final L64 exact-clause-chain; stagewise train-before/after at twice curriculum length',
        'control': 'continue_L2 matches optimizer steps and supervised value presentations, not FLOPs',
        'routing_prior': 'row_tied encodes equality of variable identity explicitly'})
    while not (PRIMARY / 'campaign-complete.json').exists():
        progress = PRIMARY / 'campaign-progress.json'
        if progress.exists() and json.loads(progress.read_text()).get('failures'):
            raise RuntimeError('Primary campaign failed; recursive work not started')
        time.sleep(2)
    completed, failures = [], []
    gate = None
    start = time.monotonic()
    for spec in specs:
        if spec['phase'] == 'recursive' and gate is None:
            validation = []
            for seed in range(3):
                history = OUT / f'warm_row_tied_s{seed}/history.jsonl'
                last = json.loads(history.read_text().splitlines()[-1])
                rate = last['panel']['4']['clause_chain_accuracy']
                validation.append({'seed': seed, 'step': last['step'], 'L4_chain': rate})
            gate = all(row['L4_chain'] >= .95 for row in validation)
            write(OUT / 'gate.json', {'passed': gate, 'validation': validation,
                'threshold': .95, 'panel_seed': 3140001, 'all_seeds_required': True})
            if not gate:
                write(OUT / 'campaign-complete.json', {'complete': True, 'gate_passed': False,
                    'completed': completed, 'failures': [], 'planned_total': len(specs),
                    'unexecuted': [row['name'] for row in specs if row['phase'] == 'recursive'],
                    'reason': 'Prespecified warmup admission failed; no recursive seed selected'})
                print('GATE FAILED; all warmup seeds retained', flush=True)
                return
        try:
            directory = OUT / spec['name']
            if not (directory / 'complete.json').exists():
                print('START', spec['name'], flush=True)
                with (OUT / f"{spec['name']}.log").open('w') as log:
                    subprocess.run(spec['command'], cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
            if not json.loads((directory / 'complete.json').read_text()).get('complete'):
                raise RuntimeError('Run incomplete')
            completed.append(spec['name'])
            print('DONE', spec['name'], flush=True)
        except Exception as exc:
            failures.append({'run': spec['name'], 'error': str(exc)})
            write(OUT / 'campaign-progress.json', {'completed': completed, 'failures': failures,
                'total': len(specs), 'complete': False, 'seconds': time.monotonic() - start})
            raise
        write(OUT / 'campaign-progress.json', {'completed': completed, 'failures': failures,
            'total': len(specs), 'complete': len(completed) == len(specs),
            'seconds': time.monotonic() - start})
    write(OUT / 'campaign-complete.json', {'completed': completed, 'failures': failures,
        'total': len(specs), 'complete': True, 'gate_passed': gate,
        'seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
