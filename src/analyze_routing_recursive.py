"""Summarize complete recursive curricula with matched-update controls."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs/routing-recursive-2026-11-09'
OUT = ROOT / 'figures/routing-recursive-2026-11-09'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RUNS / 'campaign-plan.json').read_text())
    rows, run_records, skipped = [], [], []
    for spec in manifest['runs']:
        directory = RUNS / spec['name']
        if not (directory / 'complete.json').exists():
            skipped.append(spec['name'])
            continue
        config = json.loads((directory / 'config.json').read_text())
        audit = json.loads((directory / 'data-audit.json').read_text())
        completion = json.loads((directory / 'complete.json').read_text())
        record = {k: spec[k] for k in ['name', 'routing', 'seed', 'regime', 'length']}
        record.update({'stage': spec.get('stage', 0), 'curriculum_length': spec.get('curriculum_length', 2),
            'initial_checkpoint': config['init'], 'teacher': config.get('teacher'),
            'labels_presented': completion['labels_presented'], 'steps': completion['steps'],
            'teacher_target_error': audit['target_error_rate_supervised'],
            'teacher_local_inconsistency': audit['intended_action_local_inconsistency_rate_supervised'],
            'teacher_full_trace_accuracy': audit['source_full_clause_trajectory_correct_rate']})
        run_records.append(record)
        for phase, filename in [('initial', 'initial-evaluation.json'), ('final', 'evaluation.json')]:
            doc = json.loads((directory / filename).read_text())
            for evaluation in doc['evaluations']:
                row = {**record, 'phase': phase, 'evaluation_length': evaluation['length'],
                       'panel_sha256': evaluation['panel_sha256'],
                       'checkpoint_sha256': doc['checkpoint_sha256']}
                for kind in ['teacher_forced', 'rollout']:
                    for key, value in evaluation.get(kind, {}).get('metrics', {}).items():
                        if isinstance(value, (float, int)):
                            row[f'{kind}_{key}'] = value
                row.update({f'attention_{key}': value for key, value in evaluation['attention'].items()})
                rows.append(row)
    if not rows:
        (OUT / 'summary.json').write_text(json.dumps({'complete': False, 'skipped': skipped}, indent=2))
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / 'per_seed_evaluation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        key = (row['routing'], row['regime'], row['stage'], row['phase'], row['evaluation_length'])
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for key, members in sorted(grouped.items()):
        metrics = {}
        for metric in fields:
            if metric.startswith(('teacher_forced_', 'rollout_', 'attention_')):
                values = [m[metric] for m in members if metric in m]
                if values:
                    metrics[metric] = {'mean': statistics.mean(values), 'min': min(values),
                        'max': max(values), 'values': values, 'n': len(values)}
        aggregate.append(dict(zip(['routing', 'regime', 'stage', 'phase', 'evaluation_length'], key),
                              seeds=[r['seed'] for r in members], metrics=metrics))
    gate_path = RUNS / 'gate.json'
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    (OUT / 'summary.json').write_text(json.dumps({'complete': not skipped, 'runs': run_records,
        'aggregate': aggregate, 'skipped': skipped, 'gate': gate,
        'notes': ['mean and observed model-seed range, not confidence intervals',
                  'continue_L2 matches optimizer updates and value-label presentations, not FLOPs',
                  'row_tied explicitly supplies variable-equality structure']}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, routing in zip(axes, ['dense', 'row_tied']):
        for regime, label, color in [('self', 'Own teacher', 'C0'), ('truth', 'True long-chain labels', 'C1'),
                                     ('continue_L2', 'Continue L2, matched updates', 'C2')]:
            x, means, lows, highs = [], [], [], []
            for stage in range(5):
                members = [r for r in rows if r['routing'] == routing and r['stage'] == stage
                           and r['phase'] == 'final' and r['evaluation_length'] == 64
                           and (stage == 0 or r['regime'] == regime)]
                if not members:
                    continue
                vals = [r['teacher_forced_clause_chain_accuracy'] * 100 for r in members]
                x.append(stage)
                means.append(statistics.mean(vals)); lows.append(min(vals)); highs.append(max(vals))
            ax.plot(x, means, '-o', label=label, color=color)
            ax.fill_between(x, lows, highs, alpha=.13, color=color)
        ax.set_title(routing)
        ax.set_xticks(range(5), ['L2 warm', 'Stage 1\nL4', 'Stage 2\nL8', 'Stage 3\nL16', 'Stage 4\nL32'])
        ax.set_ylim(-3, 103); ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('L64 exact full-clause chain accuracy (TF, %)')
    fig.suptitle('Fixed L64 test across recursive stages (1,024 problems/model)' + (' — PARTIAL' if skipped else ''))
    fig.text(.5, .01, 'Stage lengths describe the recursive/true-label curricula; continue-L2 always trains on L2. Bands: observed seed range.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, 1])
    for extension in ['png', 'pdf']:
        fig.savefig(OUT / f'recursive_overview.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'completed_runs': len(run_records), 'expected_runs': len(manifest['runs']),
                      'evaluation_rows': len(rows), 'output': str(OUT)}))


if __name__ == '__main__':
    main()
