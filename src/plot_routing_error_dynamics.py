"""Plot three-seed, paired learning-rate diagnostics for corrupted y2 labels.

The observations are selected-query teacher-forced single-step quantities at
training length L4. They are not L8 rollout or whole-chain accuracies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(record, split, subset):
    return next(row for row in record['splits'][split]['groups'] if row['subset'] == subset)


def collect(runs, seeds=(0, 1, 2), replicate=0):
    documents = {}
    sources = []
    for lr_name, lr in [('03', .03), ('003', .003)]:
        for seed in seeds:
            run = f'dense_lr{lr_name}_target20_s{seed}_r{replicate}'
            path = runs / run / 'error-dynamics/error-dynamics.json'
            document = json.loads(path.read_text())
            config = document['config']
            if not document.get('complete'):
                raise ValueError(f'Incomplete dynamics: {run}')
            expected = {'length': 4, 'mode': 'second', 'label_source': 'truth',
                        'noise_kind': 'target', 'noise_rate': .2,
                        'routing_mode': 'dense', 'seed': seed, 'replicate': replicate}
            if any(config.get(key) != value for key, value in expected.items()) or abs(config['lr'] - lr) > 1e-12:
                raise ValueError(f'Unexpected design in {run}')
            documents[(lr, seed)] = document
            sources.append({'run': run, 'path': str(path.resolve()), 'sha256': file_hash(path),
                            'seed': seed, 'lr': lr, 'config': config})
    steps = [record['step'] for record in next(iter(documents.values()))['checkpoints']]
    checks = []
    for seed in seeds:
        high, low = documents[(.03, seed)], documents[(.003, seed)]
        for split in ['training_audit', 'validation_audit']:
            for key in ['source_clause_sha256', 'inputs_sha256', 'targets_sha256', 'mask_sha256']:
                passed = high[split][key] == low[split][key]
                checks.append({'seed': seed, 'check': f'paired_{split}_{key}', 'passed': passed})
        high_initial = high['checkpoints'][0]['splits']['training']['model_state_sha256_before']
        low_initial = low['checkpoints'][0]['splits']['training']['model_state_sha256_before']
        checks.append({'seed': seed, 'check': 'identical_initial_model', 'passed': high_initial == low_initial})
    for (lr, seed), document in documents.items():
        if [record['step'] for record in document['checkpoints']] != steps:
            raise ValueError('Checkpoint grids differ')
        for checkpoint in document['checkpoints']:
            for split, panel in checkpoint['splits'].items():
                checks.append({'seed': seed, 'lr': lr, 'step': checkpoint['step'],
                    'check': f'{split}_model_state_unchanged',
                    'passed': panel['model_state_sha256_before'] == panel['model_state_sha256_after']})
    reference = documents[(.03, seeds[0])]['validation_audit']
    for (lr, seed), document in documents.items():
        for key in ['source_clause_sha256', 'inputs_sha256', 'targets_sha256', 'mask_sha256']:
            checks.append({'seed': seed, 'lr': lr, 'check': f'common_validation_{key}',
                           'passed': reference[key] == document['validation_audit'][key]})
    if not all(check['passed'] for check in checks):
        raise ValueError(f'Paired audit failed: {[check for check in checks if not check["passed"]]}')
    return documents, sources, checks, steps


def summarize(documents, seeds, steps):
    rows = []
    for lr in [.03, .003]:
        for index, step in enumerate(steps):
            for split in ['training', 'validation']:
                for subset in ['all', 'corrupted_examples', 'uncorrupted_examples']:
                    groups = [group(documents[(lr, seed)]['checkpoints'][index], split, subset) for seed in seeds]
                    names = list(groups[0]['metrics'])
                    metrics = {}
                    for name in names:
                        values = [value['metrics'][name] for value in groups]
                        if any(value is None for value in values):
                            metrics[name] = {'values': values, 'mean': None, 'min': None, 'max': None}
                        else:
                            metrics[name] = {'values': values, 'mean': statistics.mean(values),
                                             'min': min(values), 'max': max(values)}
                    rows.append({'lr': lr, 'step': step, 'split': split, 'subset': subset,
                        'seeds': list(seeds), 'selected_labels_per_seed': [value['selected_labels'] for value in groups],
                        'metrics': metrics})
    return rows


def draw(ax, rows, lr, split, subset, metric, scale=1., label=None, style='-'):
    selected = sorted([row for row in rows if row['lr'] == lr and row['split'] == split
                       and row['subset'] == subset], key=lambda row: row['step'])
    x = [row['step'] for row in selected]
    mean = [row['metrics'][metric]['mean'] * scale for row in selected]
    low = [row['metrics'][metric]['min'] * scale for row in selected]
    high = [row['metrics'][metric]['max'] * scale for row in selected]
    color = '#D55E00' if lr == .03 else '#0072B2'
    ax.plot(x, mean, marker='o', markersize=4, linewidth=1.9, linestyle=style, color=color,
            label=label or f'LR = {lr:g}')
    ax.fill_between(x, low, high, color=color, alpha=.11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs/routing-round2-2026-11-09')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/routing-round2-2026-11-09')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--replicate', type=int, default=0)
    args = parser.parse_args()
    documents, sources, checks, steps = collect(args.runs, args.seeds, args.replicate)
    rows = summarize(documents, args.seeds, steps)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    for lr in [.03, .003]:
        draw(axes[0], rows, lr, 'training', 'corrupted_examples', 'target_label_accuracy', 100)
        draw(axes[1], rows, lr, 'validation', 'all', 'original_truth_value_accuracy', 100)
        draw(axes[2], rows, lr, 'training', 'corrupted_examples', 'target_rule_mass',
             label=f'LR {lr:g}: corrupted training')
        draw(axes[2], rows, lr, 'validation', 'all', 'target_rule_mass',
             label=f'LR {lr:g}: held-out', style='--')
    titles = ['Fit to corrupted training labels', 'Independent validation: original true y2',
              'Attention on the correct rule']
    ylabels = ['Wrong target fitted (%)', 'True next-state accuracy (%)', 'Mean attention mass']
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set(title=title, ylabel=ylabel, xlabel='Optimizer updates')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylim(-2, 102)
    axes[1].set_ylim(-2, 102)
    axes[2].set_ylim(0, .55)
    fig.suptitle('L4, selected second step: teacher-forced single-step diagnostics under 20% target corruption', fontsize=12)
    fig.text(.5, .012, 'Three model seeds; replicate 0 only. Lines: means; bands: observed seed range, not confidence intervals. No L8 chain metric.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .05, 1, .94])
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(args.output / f'error_dynamics.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    result = {'complete': True, 'seeds': args.seeds, 'replicate': args.replicate,
        'steps': steps, 'sources': sources, 'paired_checks': checks,
        'aggregate': rows, 'definitions': {
            'metric_scope': 'L4 selected second-step teacher-forced metrics; not full-chain or L8 metrics',
            'panels': 'Each run uses complete saved training data and the same independent 4096-problem validation panel',
            'replication': 'Three model seeds, one attention-data/noise replicate; do not treat checkpoints as independent observations',
            'bands': 'Observed minimum and maximum over model seeds, not confidence intervals',
            'target_fitting': 'On controlled corrupted training examples, matching the supplied target means fitting an incorrect original-task label',
        }, 'source_sha256': file_hash(Path(__file__))}
    (args.output / 'error_dynamics_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'complete': True, 'runs': len(sources),
                      'paired_checks': len(checks)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
