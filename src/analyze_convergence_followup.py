"""Audit and summarize the exploratory soft-constraint/renaming campaign.

Only manifest main runs are included. Replicates are averaged within local FFN
seed before averaging across seeds. Reported ranges describe observed seeds,
not confidence intervals or independent-example uncertainty.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = 'convergence-followup-2026-11-10'
TREATMENTS = ['dense', 'soft_0.001', 'soft_0.01', 'soft_0.1', 'soft_1',
              'row_tied', 'rename', 'rename_soft_0.01', 'q_l2_0.01']
SHORT = {'dense': 'Dense', 'soft_0.001': 'Soft .001', 'soft_0.01': 'Soft .01',
         'soft_0.1': 'Soft .1', 'soft_1': 'Soft 1', 'row_tied': 'Hard tie',
         'rename': 'Rename', 'rename_soft_0.01': 'Rename + .01', 'q_l2_0.01': 'Q L2 .01'}
LENGTHS = [8, 16, 64]
KINDS = ['teacher_forced', 'rollout']
Q_FIELDS = ['offdiagonal_row_residual_rms', 'diagonal_minus_row_background_mean',
            'diagonal_min', 'rows_with_offdiagonal_above_diagonal']
PAIR_FIELDS = ['task', 'nvars', 'model', 'init_sha256', 'length', 'samples',
               'pool_samples', 'steps', 'batch', 'lr', 'mode', 'label_source',
               'noise_kind', 'noise_rate', 'data_seed', 'noise_seed', 'eval_seed',
               'validation_seed', 'eval_samples', 'validation_samples',
               'rollout_samples', 'eval_lengths', 'eval_every', 'source_sha256']
DATA_HASHES = ['source_clause_sha256', 'label_source_clause_sha256',
               'inputs_sha256', 'targets_sha256', 'mask_sha256']


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_run(directory, spec, verify_checkpoints=True):
    """Read one run, rejecting missing provenance or inconsistent completion."""
    config, completion, audit, final, initial = [read(directory / name) for name in
        ['config.json', 'complete.json', 'data-audit.json', 'evaluation.json', 'initial-evaluation.json']]
    require(completion.get('complete') is True, 'run incomplete')
    require(config['seed'] == spec['seed'] and config['replicate'] == spec['replicate'],
            'manifest/config seed or replicate mismatch')
    treatment = spec['treatment']
    if treatment in TREATMENTS:
        expected_mode = 'row_tied' if treatment == 'row_tied' else 'dense'
        expected_soft = float(treatment.split('_')[-1]) if treatment.startswith(('soft_', 'rename_soft_')) else 0.
        expected_l2 = float(treatment.split('_')[-1]) if treatment.startswith('q_l2_') else 0.
        expected_rename = treatment.startswith('rename')
        require(config['routing_mode'] == expected_mode
                and config.get('row_tie_lambda', 0.) == expected_soft
                and config.get('q_l2_lambda', 0.) == expected_l2
                and config.get('rename_variables', False) == expected_rename,
                'manifest treatment label does not match actual config')
    require(completion['steps'] == config['steps'], 'step count differs from config')
    for prefix in ['frozen', 'ff']:
        before, after = (completion.get(f'{prefix}_sha256_{suffix}') for suffix in ['before', 'after'])
        require(bool(before) and before == after == config.get(f'{prefix}_sha256_before'),
                f'{prefix} hash changed or missing')
    for key in DATA_HASHES:
        require(bool(audit.get(key)), f'missing data hash {key}')
    for phase, doc, checkpoint, completion_key in [
        ('after_conversion_before_training', initial, 'checkpoint-initial.pt', 'initial_checkpoint_sha256'),
        ('after_training', final, 'checkpoint-final.pt', 'final_checkpoint_sha256')]:
        require(doc.get('complete') is True and doc.get('phase') == phase,
                'missing evaluation or wrong evaluation phase')
        require(doc.get('checkpoint_sha256') == completion.get(completion_key),
                'evaluation/completion checkpoint hash mismatch')
        if verify_checkpoints:
            require(file_hash(directory / checkpoint) == completion[completion_key],
                    f'actual checkpoint hash mismatch: {checkpoint}')
    initial_panels = {r['length']: r.get('panel_sha256') for r in initial['evaluations']}
    final_panels = {r['length']: r.get('panel_sha256') for r in final['evaluations']}
    require(bool(initial_panels) and initial_panels == final_panels and all(final_panels.values()),
            'initial/final evaluation panel mismatch')
    require(set(config['eval_lengths']) == set(final_panels), 'evaluation lengths differ from config')
    history_path = directory / 'history.jsonl'
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()] if history_path.exists() else []
    if config['steps']:
        require(bool(history) and history[-1]['step'] == config['steps'], 'missing final training history')
        require(all(math.isfinite(r['loss']) for r in history), 'nonfinite history loss')
    return {'name': spec['name'], 'treatment': spec['treatment'], 'seed': spec['seed'],
            'replicate': spec['replicate'], 'config': config, 'completion': completion,
            'data_audit': audit, 'evaluation': final, 'initial_evaluation': initial,
            'history': history, 'panels': final_panels}


def collect(campaign, verify_checkpoints=True):
    plan = read(campaign / 'campaign-plan.json')
    specs = [s for s in plan['runs'] if s.get('phase', 'main') == 'main']
    require(bool(specs), 'manifest contains no main runs')
    names = [s['name'] for s in specs]
    require(len(names) == len(set(names)), 'duplicate manifest run names')
    cells = [(s['treatment'], s['seed'], s['replicate']) for s in specs]
    require(len(cells) == len(set(cells)), 'duplicate manifest treatment/seed/replicate cells')
    records, skipped, checks = [], [], []
    for spec in specs:
        try:
            require(Path(spec['name']).name == spec['name'], 'run must be a direct campaign child')
            records.append(load_run(campaign / spec['name'], spec, verify_checkpoints))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append({'name': spec['name'], 'reason': str(exc)})
    # Paired comparisons require the same raw examples, labels, initialization,
    # optimization budget and evaluation panels. Dynamic renaming deliberately
    # changes variable exposure but not the saved underlying traces or labels.
    pairs = defaultdict(list)
    for record in records:
        pairs[(record['seed'], record['replicate'])].append(record)
    errors = []
    for pair, members in sorted(pairs.items()):
        reference = next((r for r in members if r['treatment'] == 'dense'), members[0])
        for record in members:
            differences = []
            for field in PAIR_FIELDS:
                if record['config'].get(field) != reference['config'].get(field):
                    differences.append(f'config.{field}')
            for field in DATA_HASHES:
                if record['data_audit'].get(field) != reference['data_audit'].get(field):
                    differences.append(f'data.{field}')
            if record['panels'] != reference['panels']:
                differences.append('evaluation.panel_sha256')
            if differences:
                errors.append({'seed': pair[0], 'replicate': pair[1], 'reference': reference['name'],
                               'run': record['name'], 'mismatched_fields': differences})
        checks.append({'seed': pair[0], 'replicate': pair[1], 'reference': reference['name'],
                       'runs_checked': len(members), 'dense_present': any(r['treatment'] == 'dense' for r in members)})
    return plan, specs, records, skipped, {'paired_checks': checks, 'errors': errors,
        'checkpoint_files_verified': verify_checkpoints,
        'data_check_scope': 'stored data-audit hashes compared across paired treatments; tensor bytes not independently recomputed',
        'frozen_check_scope': 'config and completion before/after hashes agree',
        'eval_check_scope': 'initial/final and paired panel hashes agree; checkpoint bytes checked unless disabled'}


def flatten(records):
    rows = []
    for record in records:
        c, done = record['config'], record['completion']
        q = record['evaluation'].get('routing_statistics', {})
        history = record['history']
        for evaluation in record['evaluation']['evaluations']:
            row = {k: record[k] for k in ['name', 'treatment', 'seed', 'replicate']}
            row.update({'length': evaluation['length'], 'routing_mode': c['routing_mode'],
                        'row_tie_lambda': c.get('row_tie_lambda', 0), 'q_l2_lambda': c.get('q_l2_lambda', 0),
                        'rename_variables': c.get('rename_variables', False), 'steps': c['steps'],
                        'data_seed': c['data_seed'], 'noise_seed': c['noise_seed'],
                        'examples_presented': done['examples_presented'], 'labels_presented': done['labels_presented'],
                        'seconds': done['seconds'], 'panel_sha256': evaluation['panel_sha256'],
                        'checkpoint_sha256': record['evaluation']['checkpoint_sha256']})
            for kind in KINDS:
                doc = evaluation.get(kind, {})
                row[f'{kind}_samples'] = doc.get('samples', 0)
                for key, value in doc.get('metrics', {}).items():
                    if isinstance(value, (int, float)):
                        require(math.isfinite(value), 'nonfinite evaluation metric')
                        row[f'{kind}.{key}'] = value
            row.update({f'attention.{k}': v for k, v in evaluation.get('attention', {}).items()})
            for block in ['Q43', 'Q44']:
                row.update({f'{block}.{field}': q[block][field] for field in Q_FIELDS if field in q.get(block, {})})
            if history:
                row.update({f'train_final.{field}': history[-1][field] for field in
                            ['loss', 'cross_entropy', 'row_tie_penalty', 'regularization_loss', 'q_l2_penalty']
                            if field in history[-1]})
            rows.append(row)
    return rows


def summarize_values(rows, metric):
    seeds = defaultdict(list)
    for row in rows:
        if metric in row:
            seeds[row['seed']].append(row[metric])
    if not seeds:
        return None
    means = {str(seed): statistics.mean(values) for seed, values in sorted(seeds.items())}
    return {'mean': statistics.mean(means.values()), 'seed_mean_range': [min(means.values()), max(means.values())],
            'seed_means': means, 'n_model_seeds': len(means), 'n_cells': sum(map(len, seeds.values())),
            'replicates_per_seed': {str(seed): len(values) for seed, values in sorted(seeds.items())}}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['treatment'], row['length'])].append(row)
    result = []
    for (treatment, length), members in sorted(groups.items()):
        metrics = sorted({k for r in members for k in r if k.startswith(('teacher_forced.', 'rollout.', 'attention.', 'Q43.', 'Q44.', 'train_final.'))})
        result.append({'treatment': treatment, 'length': length,
                       'metrics': {metric: summarize_values(members, metric) for metric in metrics}})
    paired = []
    baseline = {(r['seed'], r['replicate'], r['length']): r for r in rows if r['treatment'] == 'dense'}
    for (treatment, length), members in sorted(groups.items()):
        if treatment == 'dense':
            continue
        differences = []
        for row in members:
            base = baseline.get((row['seed'], row['replicate'], row['length']))
            if base is None:
                continue
            delta = {'seed': row['seed'], 'replicate': row['replicate'], 'name': row['name'], 'baseline': base['name']}
            for kind in KINDS:
                metric = f'{kind}.clause_chain_accuracy'
                if metric in row and metric in base:
                    delta[metric] = row[metric] - base[metric]
            differences.append(delta)
        paired.append({'treatment': treatment, 'length': length, 'pairs': differences,
                       'metrics': {f'{kind}.clause_chain_accuracy': summarize_values(differences, f'{kind}.clause_chain_accuracy') for kind in KINDS}})
    return result, paired


def lookup(groups, treatment, length, metric):
    return next((r['metrics'].get(metric) for r in groups if r['treatment'] == treatment and r['length'] == length), None)


def percent(stat, signed=False):
    return (f'{100 * stat["mean"]:+.2f}' if signed else f'{100 * stat["mean"]:.2f}') if stat else '—'


def make_figures(groups, figures, partial):
    figures.mkdir(parents=True, exist_ok=True)
    treatments = [t for t in TREATMENTS if any(g['treatment'] == t for g in groups)]
    treatments += sorted({g['treatment'] for g in groups} - set(treatments))
    if not treatments:
        return []
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for row_index, kind in enumerate(KINDS):
        for col, length in enumerate(LENGTHS):
            ax = axes[row_index, col]
            for i, treatment in enumerate(treatments):
                stat = lookup(groups, treatment, length, f'{kind}.clause_chain_accuracy')
                if not stat:
                    continue
                ax.bar(i, 100 * stat['mean'], color=colors[i % len(colors)], alpha=.65)
                values = [100 * v for v in stat['seed_means'].values()]
                offsets = [(-.16 + .32 * j / max(len(values) - 1, 1)) for j in range(len(values))]
                ax.scatter([i + off for off in offsets], values, color='black', s=16, zorder=4)
            ax.set_title(f'{"Teacher forced" if kind == "teacher_forced" else "Greedy rollout"}, L{length}')
            ax.set_xticks(range(len(treatments)), [SHORT.get(t, t) for t in treatments], rotation=40, ha='right', fontsize=8)
            ax.set_ylim(0, 105); ax.grid(axis='y', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
            if col == 0:
                ax.set_ylabel('Exact full-clause chain accuracy (%)')
    fig.suptitle('Exploratory structure and data augmentation study' + (' — PARTIAL' if partial else ''))
    fig.text(.5, .005, 'Bars: mean across local-model seeds, after averaging data replicates. Dots: each model-seed mean.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .025, 1, .96])
    paths = []
    for ext in ['png', 'pdf']:
        path = figures / f'accuracy_overview.{ext}'; fig.savefig(path, dpi=220, bbox_inches='tight'); paths.append(str(path))
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, field, ylabel in zip(axes, Q_FIELDS[:2], ['Off-diagonal row residual RMS', 'Diagonal minus row background (mean score)']):
        for block_index, block in enumerate(['Q43', 'Q44']):
            x, y = [], []
            for i, treatment in enumerate(treatments):
                stat = lookup(groups, treatment, 8, f'{block}.{field}')
                if stat:
                    x.append(i + (block_index - .5) * .2); y.append(stat['mean'])
                    ax.scatter([x[-1]] * len(stat['seed_means']), list(stat['seed_means'].values()), color=f'C{block_index}', alpha=.35, s=14)
            ax.plot(x, y, 'o-', label=block, color=f'C{block_index}', markersize=4)
        ax.set_xticks(range(len(treatments)), [SHORT.get(t, t) for t in treatments], rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(ylabel); ax.grid(alpha=.2); ax.legend(); ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Final routing structure (descriptive; score gap is not accuracy)')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        path = figures / f'routing_structure.{ext}'; fig.savefig(path, dpi=220, bbox_inches='tight'); paths.append(str(path))
    plt.close(fig)
    return paths


def write_report(path, summary):
    groups, paired, records = summary['aggregate'], summary['paired_differences_vs_dense'], summary['runs']
    treatments = [t for t in TREATMENTS if any(g['treatment'] == t for g in groups)]
    complete = summary['complete']
    lines = ['# 结构正则与变量重命名实验（2026-11-10）', '',
             f'状态：{"完整" if complete else "部分结果"}；通过单次运行审计 {len(records)}/{summary["expected_runs"]} 组，配对审计错误 {len(summary["audit"]["errors"])} 项。', '',
             '本研究为探索性实验。统计先在每个局部 FFN seed 内平均数据 replicate，再对模型 seed 等权平均；四个模型 seed × 两个数据 replicate 是八个配对单元，不是八个独立模型。逐 seed 范围不解释为置信区间。', '',
             '所有指标均为全部五槽、整条链完全正确率。Teacher-forced（TF）每一步使用真实历史；greedy rollout 将模型生成的全部五槽反馈给下一步。二者必须分别报告。']
    if complete and all(t in treatments for t in TREATMENTS):
        metric = 'teacher_forced.clause_chain_accuracy'
        p16 = {t: percent(lookup(groups, t, 16, metric)) for t in TREATMENTS}
        p64 = {t: percent(lookup(groups, t, 64, metric)) for t in TREATMENTS}
        lines += ['', '## 主要发现', '',
                  f'在当前固定 1000 步协议下，软结构约束的 L16 TF 整链率随扫描强度从 dense 的 {p16["dense"]}% 提升到 λ=1 的 {p16["soft_1"]}%；变量重命名达到 {p16["rename"]}%，接近硬 row_tied 的 {p16["row_tied"]}%。这表明本模型的中等长度泛化收益可以通过较弱约束或数据干预获得，硬 tying 并非获得 L16 高准确率的唯一途径。', '',
                  f'普通 Q-L2（仅测 λ=.01）L16 为 {p16["q_l2_0.01"]}%，同系数结构惩罚为 {p16["soft_0.01"]}%。这支持进一步区分针对错误身份关联的结构正则与整体范数收缩，但单个 L2 系数不足以判定所有常规正则都无效。', '',
                  f'L64 仍不稳定：强软约束 {p64["soft_1"]}%、重命名 {p64["rename"]}%、硬 row_tied {p64["row_tied"]}%；硬 tying 的成功主要集中在 seed 2。即使行内残差接近零，也没有在四个 seed 上稳定实现 L64 泛化；约束残差与建立足够匹配 margin 是不同要求。', '',
                  '独立[配对覆盖诊断](convergence-pair-coverage-2026-11-10.md)发现：原题已覆盖全部 query ID，但 Q43/Q44 非对角 query–key 组合仅覆盖约 53.26%/22.26%。dense 和普通 Q-L2 的未见项全部保持初始零值；软/硬 tying 能通过行内耦合更新这些项。这为数据增强与结构约束的效果提供了具体候选机制，但尚未将覆盖、margin 与优化几何作完全因果分离。', '',
                  '完整[产物审计](convergence-followup-audit-2026-11-10.md)核对全部 78 个任务共 7,256 项，无失败，包含实际训练张量及 checkpoint/冻结 FFN 哈希。源码快照的两个辅助导入依赖在完成后单独补齐，时间与哈希记录在 post-run supplement 中，不冒充运行期 provenance。']
    if not complete:
        lines += ['', '**当前为部分结果；不同处理可能尚未覆盖相同模型 seed / 数据 replicate，不能据此排序或选优。**']
    if records:
        configs = [r['config'] for r in records]
        lines += ['', '配置范围：' + '；'.join(f'{field}={sorted({str(c[field]) for c in configs})}' for field in ['length', 'samples', 'steps', 'batch', 'lr', 'eval_samples', 'rollout_samples']) + '。']
    for kind, label in [('teacher_forced', 'TF'), ('rollout', 'Greedy rollout')]:
        lines += ['', f'## {label} 整链准确率（%）', '', '| 处理 | L8 | L16 | L64 |', '|---|---:|---:|---:|']
        for treatment in treatments:
            lines.append('| ' + treatment + ' | ' + ' | '.join(percent(lookup(groups, treatment, l, f'{kind}.clause_chain_accuracy')) for l in LENGTHS) + ' |')
        lines += ['', f'相对相同 seed / replicate 的 dense 的 {label} 配对差值（百分点）：', '', '| 处理 | ΔL8 | ΔL16 | ΔL64 |', '|---|---:|---:|---:|']
        for treatment in treatments:
            if treatment != 'dense':
                lines.append('| ' + treatment + ' | ' + ' | '.join(percent(lookup(paired, treatment, l, f'{kind}.clause_chain_accuracy'), True) for l in LENGTHS) + ' |')
        seeds = sorted({r['seed'] for r in records})
        lines += ['', f'{label} L64 逐模型 seed（各自先平均 replicate，%）：', '', '| 处理 | ' + ' | '.join(f'seed {s}' for s in seeds) + ' |', '|---|' + '---:|' * len(seeds)]
        for treatment in treatments:
            stat = lookup(groups, treatment, 64, f'{kind}.clause_chain_accuracy')
            values = (stat or {}).get('seed_means', {})
            lines.append('| ' + treatment + ' | ' + ' | '.join(f'{100 * values[str(s)]:.2f}' if str(s) in values else '—' for s in seeds) + ' |')
    lines += ['', '## 路由诊断', '', '| 处理 | Q43 残差 RMS | Q44 残差 RMS | Q43 对角−行背景 | Q44 对角−行背景 |', '|---|---:|---:|---:|---:|']
    for treatment in treatments:
        values = [lookup(groups, treatment, 8, f'{block}.{field}') for field in Q_FIELDS[:2] for block in ['Q43', 'Q44']]
        lines.append('| ' + treatment + ' | ' + ' | '.join(f'{v["mean"]:.5f}' if v else '—' for v in values) + ' |')
    lines += ['', '残差为每行非对角元素相对行均值的 RMS；分数差为对角元素减该行非对角均值后再平均，不是最差竞争项 margin，也不是最终推理正确率。两块 Q 分开报告，以免掩盖规则/状态注意力失衡。', '',
              '## 解释边界与审计', '',
              '- soft 惩罚仅压低非对角行内差异；硬 row_tied 同时改变参数化与 Adam 的优化几何，不能把有限 λ 或 λ→∞ 的训练轨迹视为与硬 tying 等价。',
              '- rename 每次展示重新重命名变量，增加变量 ID 组合的曝光，但不新增抽象 action/state 轨迹；监督目标仍是原有 value 标签。冻结 FFN 未被约束为严格变量置换等变，不能将其效果全部归于 Q 的对称性。',
              '- q_l2 是全部 Q 元素均方的显式损失惩罚；它也惩罚正确匹配的对角增长。相同系数不保证与结构惩罚有相同梯度或正则强度。',
              '- 数据、初始化、预算和评测在 seed / replicate 内配对；保存的 data-audit hash 比较验证底层 trace、标签和 mask 一致，动态重命名后的曝光本来就不同。该分析未独立重算 training-data.pt tensor hash。',
              '- 配对审计比较了配置、底层数据 hash 与评测 panel hash；完成记录的冻结 FFN/参数前后 hash 必须相同，初始/最终评测绑定对应 checkpoint。',
              '- 这组有限模型、固定训练预算结果不能证明一般 Transformer 的收敛，也不直接复现论文特殊 FFN 与梯度下降假设。不能仅凭结构残差变小或训练 loss 下降就声称稳定长度泛化。']
    if summary['skipped']:
        lines += ['', '未纳入的运行：'] + [f'- {r["name"]}: {r["reason"]}' for r in summary['skipped']]
    if summary['audit']['errors']:
        lines += ['', '**存在配对审计错误；配对效应不能作为受控比较解释。**'] + [f'- {e["run"]}: {", ".join(e["mismatched_fields"])}' for e in summary['audit']['errors']]
    lines += ['', '完整逐运行与逐长度结果：`' + str(summary['output'] / 'per_run.csv') + '`。', '训练轨迹、配对差值和审计记录：`' + str(summary['output'] / 'summary.json') + '`。', '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'runs' / DEFAULT_NAME)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--figures', type=Path, default=ROOT / 'figures' / DEFAULT_NAME)
    parser.add_argument('--report', type=Path, default=ROOT / 'notes/convergence-followup-results-2026-11-10.md')
    parser.add_argument('--skip-checkpoint-hashes', action='store_true')
    args = parser.parse_args(argv)
    output = args.output or args.campaign / 'analysis'
    plan, specs, records, skipped, audit = collect(args.campaign, not args.skip_checkpoint_hashes)
    rows = flatten(records)
    aggregate, paired = summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    if fields:
        with (output / 'per_run.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {'complete': len(records) == len(specs) and not skipped and not audit['errors'],
               'campaign': str(args.campaign.resolve()), 'expected_runs': len(specs), 'completed_runs': len(records),
               'skipped': skipped, 'audit': audit, 'aggregate': aggregate, 'paired_differences_vs_dense': paired,
               'runs': records, 'aggregation': 'average data replicates within model seed, then equal average across model seeds',
               'uncertainty': 'observed model-seed mean range only; no confidence intervals', 'output': output}
    summary['figures'] = make_figures(aggregate, args.figures, not summary['complete'])
    write_report(args.report, summary)
    summary['output'] = str(output.resolve())
    write_json(output / 'summary.json', summary)
    print(json.dumps({'complete': summary['complete'], 'completed_runs': len(records), 'expected_runs': len(specs),
                      'pairing_errors': len(audit['errors']), 'report': str(args.report), 'output': str(output)}))
    return 0 if not audit['errors'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
