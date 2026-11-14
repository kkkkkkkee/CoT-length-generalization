"""Prespecified independent-data replication and learned-routing constraints."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'runs/sparse-supervision-2026-11-09'
OUT = ROOT / 'runs/routing-round2-2026-11-09'


def plan():
    runs = []
    for seed in range(3):
        for replicate in range(3):
            parent = OLD / f'warm_L2_second_s{seed}/checkpoint-final.pt'
            common = ['--init', parent, '--seed', seed, '--replicate', replicate,
                      '--length', 4, '--samples', 4096, '--steps', 1000, '--batch', 128,
                      '--mode', 'second', '--data-seed', 2810001 + 10000 * seed + 100 * replicate,
                      '--noise-seed', 2910001 + 10000 * seed + 100 * replicate,
                      '--eval-seed', 2940001, '--validation-seed', 2840001,
                      '--eval-samples', 1024, '--rollout-samples', 128,
                      '--eval-lengths', 2, 4, 8, 16, 32, 64]
            cases = [('dense', .03, 'truth', 'none', 0.),
                     ('dense', .03, 'truth', 'target', .05),
                     ('dense', .03, 'truth', 'target', .20),
                     ('dense', .03, 'truth', 'prefix_inconsistent', .20),
                     ('dense', .03, 'truth', 'prefix_consistent', .20),
                     ('dense', .03, 'teacher', 'none', 0.),
                     ('row_tied', .03, 'truth', 'none', 0.),
                     ('row_tied', .03, 'teacher', 'none', 0.)]
            if replicate == 0:
                cases += [('dense', .003, 'truth', kind, rate) for kind, rate in
                          [('none', 0.), ('target', .05), ('target', .20)]]
                cases += [('diagonal', .03, source, 'none', 0.) for source in ['truth', 'teacher']]
            for routing, lr, source, noise, rate in cases:
                condition = source if noise == 'none' else f'{noise}{round(rate * 100):02d}'
                rate_tag = '003' if lr == .003 else '03'
                name = f'{routing}_lr{rate_tag}_{condition}_s{seed}_r{replicate}'
                args = common + ['--routing-mode', routing, '--lr', lr,
                    '--label-source', source, '--noise-kind', noise, '--noise-rate', rate]
                if source == 'teacher':
                    args += ['--teacher', parent]
                runs.append({'name': name, 'seed': seed, 'replicate': replicate,
                    'routing_mode': routing, 'condition': condition, 'lr': lr,
                    'command': [sys.executable, 'src/routing_experiment.py', '--output',
                                str(OUT / name), *map(str, args)]})
    return runs


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = plan()
    write(OUT / 'campaign-plan.json', {'runs': specs, 'total': len(specs),
        'primary_endpoint': 'L8 exact full-clause chain; report pretraining projection and subsequent changes separately',
        'replication': '3 local-model seeds x 3 independently drawn training/noise datasets; lower-lr and diagonal controls use replicate 0',
        'source_constraints': 'row_tied and diagonal explicitly encode variable equality, unlike arbitrary dense Q'})
    start = time.monotonic()
    completed, failures = [], []
    for spec in specs:
        directory = OUT / spec['name']
        try:
            if not (directory / 'complete.json').exists():
                print('START', spec['name'], flush=True)
                with (OUT / f"{spec['name']}.log").open('w') as log:
                    subprocess.run(spec['command'], cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
            if not json.loads((directory / 'complete.json').read_text()).get('complete'):
                raise RuntimeError('Incomplete experiment')
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
          'total': len(specs), 'complete': True, 'seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
