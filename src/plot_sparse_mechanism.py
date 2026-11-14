"""Rebuild the paired mechanism overview from recorded probe CSV files."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs/sparse-supervision-2026-11-09/probes'
OUT = ROOT / 'figures/sparse-supervision-2026-11-09'
PATTERNS = [('Clean, second-step', 'extend_L4_N4096_truth_second'),
            ('20% wrong targets', 'noise_L4_target_r20'),
            ('20% wrong prefix / consistent target', 'noise_L4_prefix_consistent_r20')]
CONDITIONS = ['original', 'balance_only', 'concentrate_only', 'both', 'q_row_mean_background']
LABELS = ['Original', 'Balance only', 'Mass only', 'Mass + balance', 'Q row-mean']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    records = []
    for ax, (label, pattern) in zip(axes, PATTERNS):
        documents = [list(csv.DictReader((RUNS / f'{pattern}_s{s}/probe.csv').open())) for s in range(3)]
        for k, condition in enumerate(CONDITIONS):
            values = [float(next(row for row in document if int(row['length']) == 8
                      and row['condition'] == condition)['clause_chain_accuracy']) * 100 for document in documents]
            ax.scatter([k - .1, k, k + .1], values, color=f'C{k}', s=23, zorder=3)
            ax.plot([k - .23, k + .23], [statistics.mean(values)] * 2, color=f'C{k}', lw=3)
            records.append({'model_condition': pattern, 'intervention': condition, 'length': 8,
                'seeds': [0, 1, 2], 'values_percent': values, 'mean': statistics.mean(values),
                'min': min(values), 'max': max(values)})
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha='right')
        ax.set_title(label)
        ax.set_ylim(-3, 105)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('L8 teacher-forced exact full-clause chain (%)')
    fig.suptitle('Paired mechanism probes on a new panel; 512 sequences per seed', fontsize=13)
    fig.text(.5, .015, 'Dots: individual seeds; bars: means. Attention interventions use true rule positions; no free rollout run.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .05, 1, .94])
    for extension in ['png', 'pdf']:
        fig.savefig(OUT / f'mechanism_overview.{extension}', dpi=200, bbox_inches='tight')
    (OUT / 'mechanism_overview.json').write_text(json.dumps(records, indent=2))
    plt.close(fig)


if __name__ == '__main__':
    main()
