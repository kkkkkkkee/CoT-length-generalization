"""Measure causal query/key exposure and final Q values in a routing campaign.

Exposure means that an effective Q entry can appear in the task computation;
it does not assert that its numerical gradient was nonzero. Dynamic renaming
changes exposure, so its static template masks are never called training-unseen.
This script reads immutable campaign outputs and runs entirely on CPU.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import torch

from routing_model import load_routing_model
from sparse_data import tensor_hash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / 'runs/convergence-followup-2026-11-10'
BLOCKS = {'Q43': 2, 'Q44': 3}


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def pair_counts(inputs, selected_query_positions, nvars, batch_size=256):
    """Return exact counts of valid causal query/key pairs in saved inputs.

    Q43 sees key slot 2 (zero-based); Q44 sees key slot 3. Both use query
    slot 3. All key positions <= the selected query position are visible.
    Multiplicity counts selected query/key occurrences, not optimizer steps.
    """
    require(inputs.device.type == selected_query_positions.device.type == 'cpu', 'CPU tensors required')
    require(inputs.ndim == 3 and inputs.shape[-1] == 5, 'inputs must be [N,T,5]')
    require(selected_query_positions.ndim == 2 and selected_query_positions.shape[0] == inputs.shape[0],
            'query positions must be [N,J]')
    require(inputs.dtype == selected_query_positions.dtype == torch.long, 'integer token/position tensors required')
    require(nvars > 1 and batch_size > 0, 'nvars > 1 and positive batch size required')
    require(bool(((selected_query_positions >= 0) & (selected_query_positions < inputs.shape[1])).all()),
            'selected query outside inputs')
    counts = {name: torch.zeros((nvars, nvars), dtype=torch.long) for name in BLOCKS}
    invalid_queries = 0
    for offset in range(0, len(inputs), batch_size):
        x = inputs[offset:offset + batch_size]
        positions = selected_query_positions[offset:offset + batch_size]
        query = x[:, :, 3].gather(1, positions)
        query_valid = (query >= 1) & (query <= nvars)
        invalid_queries += int((~query_valid).sum())
        visible = torch.arange(x.shape[1])[None, None, :] <= positions[:, :, None]
        for name, slot in BLOCKS.items():
            keys = x[:, :, slot]
            valid = visible & query_valid[:, :, None] & ((keys >= 1) & (keys <= nvars))[:, None, :]
            pair_index = (query[:, :, None] - 1) * nvars + (keys[:, None, :] - 1)
            counts[name] += torch.bincount(pair_index[valid], minlength=nvars * nvars).reshape(nvars, nvars)
    return counts, {'examples': len(inputs), 'selected_queries': selected_query_positions.numel(),
                    'invalid_selected_queries': invalid_queries, 'visible_key_policy': 'key position <= selected query position',
                    'count_unit': 'selected query/key occurrences in saved data; not optimizer presentations'}


def coverage_statistics(counts):
    n = counts.shape[0]
    diagonal = torch.eye(n, dtype=torch.bool)
    result = {}
    for name, mask in [('diagonal', diagonal), ('offdiagonal', ~diagonal)]:
        v = counts[mask]
        result[name] = {'possible_entries': v.numel(), 'seen_entries': int((v > 0).sum()),
                        'unseen_entries': int((v == 0).sum()), 'coverage_fraction': (v > 0).double().mean().item(),
                        'pair_occurrences': int(v.sum()), 'counts_min': int(v.min()),
                        'counts_max': int(v.max()), 'counts_mean': v.double().mean().item()}
    return result


def value_statistics(values):
    v = values.detach().double().flatten()
    if not v.numel():
        return {'count': 0}
    quantiles = torch.quantile(v, torch.tensor([0., .05, .25, .5, .75, .95, 1.], dtype=torch.float64))
    return {'count': v.numel(), 'mean': v.mean().item(), 'std': v.std(unbiased=False).item(),
            'quantiles': dict(zip(['min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max'], quantiles.tolist())),
            'absolute_mean': v.abs().mean().item(), 'absolute_max': v.abs().max().item(),
            'exact_zero_count': int((v == 0).sum()), 'exact_zero_fraction': (v == 0).double().mean().item(),
            'abs_le_1e-8_fraction': (v.abs() <= 1e-8).double().mean().item()}


def matrix_statistics(q, counts):
    diagonal = torch.eye(q.shape[0], dtype=torch.bool)
    seen = counts > 0
    return {name: value_statistics(q[mask]) for name, mask in {
        'seen_offdiagonal': seen & ~diagonal, 'unseen_offdiagonal': ~seen & ~diagonal,
        'seen_diagonal': seen & diagonal, 'unseen_diagonal': ~seen & diagonal,
        'all_offdiagonal': ~diagonal}.items()}


def load_run(campaign, spec, panels):
    directory = campaign / spec['name']
    require(Path(spec['name']).name == spec['name'], 'run must be direct campaign child')
    config, complete, audit = [read(directory / f) for f in ['config.json', 'complete.json', 'data-audit.json']]
    require(complete.get('complete') is True, 'run incomplete')
    require(config['seed'] == spec['seed'] and config['replicate'] == spec['replicate'], 'manifest/config mismatch')
    require(complete['steps'] == config['steps'], 'run did not reach configured steps')
    require(complete['ff_sha256_before'] == complete['ff_sha256_after'] == config['ff_sha256_before'], 'frozen FFN hash changed')
    data_path = directory / 'training-data.pt'
    saved = torch.load(data_path, map_location='cpu', weights_only=False)
    supervision = saved['supervision']
    inputs, queries = supervision['inputs'], supervision['selected_query_positions']
    input_hash, query_hash = tensor_hash(inputs), tensor_hash(queries)
    require(input_hash == audit['inputs_sha256'] == supervision['audit']['inputs_sha256'], 'input tensor/audit hash mismatch')
    require(tensor_hash(supervision['targets']) == audit['targets_sha256'], 'target tensor/audit hash mismatch')
    require(tensor_hash(supervision['mask']) == audit['mask_sha256'], 'mask tensor/audit hash mismatch')
    require(tensor_hash(saved['sample']['clauses']) == audit['source_clause_sha256'], 'source clauses/audit hash mismatch')
    require(torch.equal(queries, supervision['query_positions'].gather(1, supervision['selected_steps'])),
            'selected query positions differ from supervision mask/steps')
    require(bool(supervision['mask'].gather(1, supervision['selected_steps']).all()), 'selected position missing from supervision mask')
    panel_key = hashlib.sha256(f'{input_hash}:{query_hash}:{config["nvars"]}'.encode()).hexdigest()
    if panel_key not in panels:
        counts, metadata = pair_counts(inputs, queries, config['nvars'])
        panels[panel_key] = {'inputs_sha256': input_hash, 'selected_query_positions_sha256': query_hash,
                            'nvars': config['nvars'], **metadata,
                            'blocks': {block: {'coverage': coverage_statistics(c), 'visible_pair_counts': c.tolist()}
                                       for block, c in counts.items()}}
    else:
        counts = {block: torch.tensor(panels[panel_key]['blocks'][block]['visible_pair_counts']) for block in BLOCKS}
    checkpoint = directory / 'checkpoint-final.pt'
    require(file_hash(checkpoint) == complete['final_checkpoint_sha256'], 'final checkpoint hash mismatch')
    model, state = load_routing_model(checkpoint, 'cpu')
    require(state['step'] == config['steps'] and model.config.nvars == config['nvars'], 'checkpoint metadata mismatch')
    rename = config.get('rename_variables', False)
    record = {key: spec[key] for key in ['name', 'treatment', 'seed', 'replicate']}
    record.update({'panel_key': panel_key, 'rename_variables': rename, 'routing_mode': config['routing_mode'],
                   'row_tie_lambda': config.get('row_tie_lambda', 0.), 'q_l2_lambda': config.get('q_l2_lambda', 0.),
                   'steps': config['steps'], 'examples_presented': complete['examples_presented'],
                   'init_sha256': config['init_sha256'], 'final_checkpoint_sha256': complete['final_checkpoint_sha256'],
                   'training_data_file_sha256': file_hash(data_path),
                   'exposure_interpretation': 'template coverage only; actual renamed presentations not reconstructed' if rename else
                       'coverage of saved fixed inputs; an entry can receive task gradient only if causally exposed, but nonzero gradient is not guaranteed',
                   'blocks': {block: {'coverage': panels[panel_key]['blocks'][block]['coverage']} for block in BLOCKS}})
    for block in BLOCKS:
        q = getattr(model, block).detach()
        if rename:
            # The static zero mask is not an unseen set for dynamic renaming.
            record['blocks'][block]['all_offdiagonal_values'] = value_statistics(q[~torch.eye(q.shape[0], dtype=torch.bool)])
        else:
            record['blocks'][block]['values_by_static_exposure'] = matrix_statistics(q, counts[block])
    return record


def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['treatment']].append(r)
    result = {}
    for treatment, members in sorted(groups.items()):
        flat = []
        for r in members:
            row = {'seed': r['seed'], 'replicate': r['replicate']}
            for block in BLOCKS:
                coverage = r['blocks'][block]['coverage']
                for kind in ['diagonal', 'offdiagonal']:
                    row[f'{block}.{kind}.coverage_fraction'] = coverage[kind]['coverage_fraction']
                for seen in ['seen_offdiagonal', 'unseen_offdiagonal']:
                    values = r['blocks'][block].get('values_by_static_exposure', {}).get(seen, {})
                    if values.get('count'):
                        for field in ['mean', 'exact_zero_fraction', 'absolute_mean']:
                            row[f'{block}.{seen}.{field}'] = values[field]
                        for field in ['p05', 'p50', 'p95']:
                            row[f'{block}.{seen}.{field}'] = values['quantiles'][field]
            flat.append(row)
        metrics = {}
        for key in sorted({k for row in flat for k in row} - {'seed', 'replicate'}):
            by_seed = defaultdict(list)
            for row in flat:
                if key in row:
                    by_seed[row['seed']].append(row[key])
            means = {str(seed): statistics.mean(vals) for seed, vals in sorted(by_seed.items())}
            metrics[key] = {'mean': statistics.mean(means.values()), 'seed_means': means,
                            'seed_mean_range': [min(means.values()), max(means.values())]}
        result[treatment] = {'n_runs': len(members), 'n_model_seeds': len({r['seed'] for r in members}),
                             'rename_variables': members[0]['rename_variables'], 'metrics': metrics}
    return result


def report(path, document):
    groups = document['aggregate']
    def val(t, key, pct=False):
        stat = groups[t]['metrics'].get(key)
        return f'{stat["mean"] * (100 if pct else 1):.3f}' if stat else '—'
    lines = ['# 有限数据的 query–key pair 覆盖诊断（2026-11-10）', '',
             f'读取 {document["completed_runs"]}/{document["expected_runs"]} 个 main run；状态：{"完整" if document["complete"] else "部分结果"}。', '',
             '每个被监督的 query，枚举其因果可见位置（key position ≤ query position）。query 使用第 4 槽变量；Q43 的 key 来自第 3 槽，Q44 来自第 4 槽。只统计 token ID 在 1..nvars 内的有效变量配对。', '',
             '这里的“已见”表示该有效 Q 矩阵元素曾有机会进入任务计算，不保证数值梯度非零；“未见”表示固定输入中没有这样的直接计算路径。计数单位是静态数据中的 query/key 出现次数，不是 optimizer step。硬 tying 的共享参数以及软正则可以影响未见的有效矩阵元素。', '',
             '## 原始模板的矩阵覆盖率（%）', '',
             '| 处理 | Q43 对角 | Q43 非对角 | Q44 对角 | Q44 非对角 |', '|---|---:|---:|---:|---:|']
    for t in groups:
        keys = [f'{b}.{kind}.coverage_fraction' for b in BLOCKS for kind in ['diagonal', 'offdiagonal']]
        lines.append('| ' + t + ' | ' + ' | '.join(val(t, k, True) for k in keys) + ' |')
    lines += ['', '重命名条件只报告这张原始模板覆盖表；每次展示的变量置换改变实际 pair 曝光，不能把模板中未覆盖的项称为训练未见。本脚本没有重放 augmentation RNG。', '',
              '## 非重命名条件的最终 Q', '',
              '表中每个统计量先在同一模型 seed 内平均两个 data replicate，再在模型 seed 间等权平均。p05/p50/p95 是各 run 矩阵元素分位数的层级均值，不是将所有矩阵混在一起后的分位数。']
    for block in BLOCKS:
        lines += ['', f'### {block}', '',
                  '| 处理 | 非对角集合 | 均值 | p05 / p50 / p95 | 精确零比例（%） |', '|---|---|---:|---|---:|']
        for t, group in groups.items():
            if group['rename_variables']:
                continue
            for key, label in [('seen_offdiagonal', '已见'), ('unseen_offdiagonal', '未见')]:
                lines.append('| ' + t + ' | ' + label + ' | ' + val(t, f'{block}.{key}.mean') + ' | ' +
                             ' / '.join(val(t, f'{block}.{key}.{p}') for p in ['p05', 'p50', 'p95']) + ' | ' + val(t, f'{block}.{key}.exact_zero_fraction', True) + ' |')
    lines += ['', '## 解释与边界', '',
              '- 对从 Q=0 开始、无正则/无共享参数/无重命名的 dense，静态数据不曝光的独立矩阵元素没有任务损失的直接梯度路径；若训练保持这一协议，它们应保留初始化。这里用最终实际值检验，而不是预设其为零。',
              '- soft row-tie 惩罚通过同一行的非对角均值耦合矩阵项；hard row_tied 将同一行所有非匹配项表示为同一个参数。因此即便某些有效矩阵项没有静态直接曝光，其值仍可改变。',
              '- 有限均匀数据下，Q43 每题可见的非匹配规则变量多于 Q44 的非匹配历史答案变量，覆盖差异是可检验的设计结果；本报告使用实际张量计数。',
              '- 覆盖、最终背景值与泛化相关只支持候选机制。要作因果归因，仍需固定/操纵 pair 覆盖并控制监督信息、参数化、优化预算以及正确匹配 margin；本分析未证明覆盖是唯一原因。',
              '- 精确零比例使用 Q==0；JSON 另保留 |Q|≤1e−8 比例、均值、标准差、全套分位数、每 seed 汇总和逐 run 结果。',
              '- 审计实际重算 saved inputs、targets、mask 和 source clauses 的 tensor hash，并验证 final checkpoint 文件 hash；原始 pair count 矩阵按相同数据去重保存。', '',
              '完整机器可读结果：`' + document['output'] + '`。']
    if document['skipped']:
        lines += ['', '未纳入：'] + [f'- {s["name"]}: {s["reason"]}' for s in document['skipped']]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path, default=ROOT / 'notes/convergence-pair-coverage-2026-11-10.md')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    plan = read(args.campaign / 'campaign-plan.json')
    specs = [s for s in plan['runs'] if s.get('phase', 'main') == 'main']
    require(bool(specs), 'manifest contains no main runs')
    keys = [(s['treatment'], s['seed'], s['replicate']) for s in specs]
    require(len(set(keys)) == len(keys), 'duplicate manifest cells')
    records, panels, skipped = [], {}, []
    for spec in specs:
        try:
            records.append(load_run(args.campaign, spec, panels))
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            skipped.append({'name': spec['name'], 'reason': str(exc)})
    output = args.output or args.campaign / 'analysis/pair-coverage.json'
    document = {'complete': len(records) == len(specs) and not skipped,
                'expected_runs': len(specs), 'completed_runs': len(records), 'skipped': skipped,
                'campaign': str(args.campaign.resolve()), 'output': str(output.resolve()),
                'pair_indexing': 'matrix[row=query token ID-1][column=key token ID-1]',
                'runs': records, 'coverage_panels': panels, 'aggregate': aggregate(records),
                'caveats': ['exposure is a structural opportunity for task gradient, not measured nonzero gradient',
                            'renamed conditions have only static template coverage; no static seen/unseen Q split',
                            'hard tying and soft regularization can change effective matrix entries without direct static exposure']}
    write_json(output, document)
    report(args.report, document)
    print(json.dumps({'complete': document['complete'], 'completed_runs': len(records), 'expected_runs': len(specs),
                      'unique_static_panels': len(panels), 'output': str(output), 'report': str(args.report)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
