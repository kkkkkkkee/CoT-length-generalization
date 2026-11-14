"""Read saved results only; regenerate report data without training/evaluation."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sources = {}
rows = []
inventory = defaultdict(lambda: {'evaluation_files': 0, 'complete_evaluations': 0, 'run_markers': 0, 'valid_run_markers': 0})

def read(path):
    data = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)

for path in sorted((ROOT / 'runs').rglob('evaluation.json')):
    rel = path.relative_to(ROOT / 'runs')
    if 'source' in rel.parts:
        continue
    campaign = rel.parts[0]
    result = read(path)
    inventory[campaign]['evaluation_files'] += 1
    inventory[campaign]['complete_evaluations'] += int(result.get('complete', False))
    config_path = path.parent / 'config.json'
    if not config_path.exists():
        config_path = path.parent.parent / 'config.json'
    config = read(config_path) if config_path.exists() else result.get('config', {})
    for panel_key, ordering in [('evaluations', 'normal'), ('shuffled_evaluations', 'shuffled')]:
        for item in result.get(panel_key, []):
            for mode in ['teacher_forced', 'rollout']:
                if mode not in item:
                    continue
                panel = item[mode]
                row = {
                    'campaign': campaign, 'run': str(path.parent.relative_to(ROOT / 'runs')),
                    'source': str(path.relative_to(ROOT)), 'ordering': ordering, 'evaluation_mode': mode,
                    'length': item.get('length'), 'samples': panel.get('samples'),
                    'seed': config.get('seed'), 'replicate': config.get('replicate'),
                    'routing': config.get('routing_mode'), 'position': config.get('position_mode', 'none'),
                    'train_length': config.get('length'), 'train_samples': config.get('samples'),
                    'updates': config.get('steps'), 'lr': config.get('lr'),
                    'label_source': config.get('label_source'), 'supervision': config.get('mode'),
                    'noise_kind': config.get('noise_kind'), 'noise_rate': config.get('noise_rate'),
                    'row_tie_lambda': config.get('row_tie_lambda'), 'q_l2_lambda': config.get('q_l2_lambda'),
                    'rename_variables': config.get('rename_variables'), 'data_seed': config.get('data_seed'),
                    'inverse_temperature': item.get('inverse_temperature'),
                    'context_extra': item.get('context_extra'), 'distractor_kind': item.get('distractor_kind'),
                    'oracle_attention': item.get('oracle_attention', False),
                    'smoke_or_pilot': any('smoke' in s.lower() or 'pilot' in s.lower() for s in rel.parts),
                    'panel_sha256': item.get('panel_sha256'),
                }
                for key in ['clause_chain_accuracy','value_chain_accuracy','final_value_accuracy','value_accuracy','variable_accuracy']:
                    row[key] = panel.get('metrics', {}).get(key)
                rows.append(row)
for path in sorted((ROOT / 'runs').rglob('complete.json')):
    rel = path.relative_to(ROOT / 'runs')
    if 'source' in rel.parts:
        continue
    j = read(path)
    inventory[rel.parts[0]]['run_markers'] += 1
    inventory[rel.parts[0]]['valid_run_markers'] += int(j.get('complete', False))
with (OUT / 'evaluations.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# Group new, previously unreported campaigns; keep Phase C stages separate.
groups = defaultdict(list)
for row in rows:
    if row['smoke_or_pilot'] or row['evaluation_mode'] != 'teacher_forced':
        continue
    if not row['campaign'].startswith(('phase-c-', 'phase-d-', 'pair-coverage-control-', 'position-embedding')):
        continue
    label = row['run'].split('/')[-1].rsplit('_s', 1)[0]
    groups[(row['campaign'], label, row['length'])].append(row)
aggregates = []
for (campaign, label, length), items in sorted(groups.items()):
    by_seed = defaultdict(list)
    for row in items:
        by_seed[row['seed']].append(row['clause_chain_accuracy'])
    seed_means = {str(k): statistics.mean(v) for k, v in sorted(by_seed.items())}
    aggregates.append({'campaign': campaign, 'condition': label, 'length': length,
                       'n_runs': len(items), 'n_model_seeds': len(by_seed),
                       'tf_full_clause_mean': statistics.mean(seed_means.values()),
                       'seed_means': seed_means, 'seed_min': min(seed_means.values()),
                       'seed_max': max(seed_means.values())})
(OUT / 'aggregates.json').write_text(json.dumps(aggregates, indent=2) + '\n')
(OUT / 'source-audit.json').write_text(json.dumps({
    'scope': 'Read-only extraction of saved JSON; not a new checkpoint/tensor audit or rerun.',
    'accuracy_unit': 'fraction 0..1', 'evaluation_rows': len(rows),
    'inventory': dict(inventory), 'source_sha256': sources,
}, indent=2) + '\n')
print(json.dumps({'evaluation_rows': len(rows), 'source_files': len(sources), 'campaigns': len(inventory),
                  'aggregates': len(aggregates)}, indent=2))

# All 36 Phase B cells, read directly from the 288 saved formal runs.
phase_b = defaultdict(list)
for row in rows:
    if (row['campaign'] == 'large-scale-2026-11-10'
        and row['run'].split('/')[-1].startswith('B_')
        and row['evaluation_mode'] == 'teacher_forced'
        and row['length'] in [8, 16, 64]):
        key = (row['routing'], row['lr'], row['noise_kind'], row['noise_rate'], row['length'])
        phase_b[key].append(row['clause_chain_accuracy'])
assert len(phase_b) == 108, len(phase_b)
assert all(len(v) == 8 for v in phase_b.values())
lines = ['# Phase B 完整结果', '',
    '288 个正式运行；8 个 local FFN seeds，每条件 repeat 0。L4、4,096 题、Q=0、第二步 value 监督、1,000 次 Adam 更新、batch 128。', '',
    '下表为 1,024 题面板上的 TF 完整五槽整链正确率，单位 %。按八个模型 seeds 等权平均；各长度分别计算，不平均不同长度。', '',
    '| 路由 | 学习率 | 污染类型 | 请求比例 | n | L8 | L16 | L64 |',
    '|---|---:|---|---:|---:|---:|---:|---:|']
for routing, lr, kind, rate in sorted(set(k[:4] for k in phase_b)):
    values = [statistics.mean(phase_b[(routing, lr, kind, rate, length)]) * 100 for length in [8,16,64]]
    lines.append(f'| {routing} | {lr:g} | {kind} | {rate:.0%} | 8 | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
lines += ['', '`target` 且比例为 0 表示干净真值；`prefix_consistent` 对实际污染后的输入保持局部转移一致，不能当作任意错误标签鲁棒性。', '',
          '来源：[逐运行 CSV](evaluations.csv)、[主报告](实验总报告.md)。此表由 `collect_results.py` 自动生成。', '']
(OUT / 'Phase-B完整结果.md').write_text('\n'.join(lines))
