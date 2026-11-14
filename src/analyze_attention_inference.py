"""Report validation-locked inference policies, including greedy and permutation."""
from pathlib import Path
from collections import defaultdict
import json,statistics,csv
ROOT=Path(__file__).resolve().parents[1]
def main():
    base=ROOT/'runs/attention-inference-2026-11-10'
    p=json.loads((base/'campaign-progress.json').read_text());assert p['complete']
    plan=json.loads((base/'campaign-plan.json').read_text());selection=json.loads((base/'selection.json').read_text())
    groups=defaultdict(list);flat=[]
    for job in plan['jobs']:
        doc=json.loads((base/job['name']/'result.json').read_text());assert doc['model_hash_before']==doc['model_hash_after']
        for row in doc['rows']:
            for e in row['evaluations']:
                tf=e['teacher_forced']['metrics'];roll=e['rollout']['metrics'];att=e['attention']
                d=dict(family=doc['family'],seed=doc['seed'],kind=row['method']['kind'],beta=row['method']['beta'],
                       shuffled=row['shuffled'],length=e['length'],tf=tf['clause_chain_accuracy'],
                       greedy=roll['clause_chain_accuracy'],final_value=roll['final_value_accuracy'],
                       relevant_mass=att['relevant_mass'],rule_top1=att['rule_strict_top1_among_predicates'],
                       target_rule_mass=att['target_rule_mass'],current_state_mass=att['current_state_mass'])
                groups[d['family'],d['kind'],d['shuffled'],d['length']].append(d);flat.append(d)
    summary={}
    for (family,kind,shuffled,length),rows in groups.items():
        s=dict(n=len(rows),beta=rows[0]['beta'],**{k:statistics.mean(r[k] for r in rows) for k in ['tf','greedy','final_value','relevant_mass','rule_top1','target_rule_mass','current_state_mass']})
        s['tf_range']=[min(r['tf'] for r in rows),max(r['tf'] for r in rows)]
        summary.setdefault(family,{}).setdefault(kind,{}).setdefault('shuffled' if shuffled else 'normal',{})[str(length)]=s
    (base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (base/'per-model.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['# 不提供正确规则位置的推理干预结果','','28个固定checkpoint，7组×4seed，replicate0。无训练。参数先用L4/L8正常和打乱验证面板选择，锁定后才测试新L4/L16/L64面板。每模型每面板256题TF、前32题真正greedy自由生成。完整clause整链为主要指标；greedy最终value单列，不混为整链。','','## L64正常顺序','','|模型组|原始TF|锐化TF|分组TF|原始greedy|锐化greedy|分组greedy|','|---|---:|---:|---:|---:|---:|---:|']
    for family,s in summary.items():
        vals=[100*s[k]['normal']['64'][metric] for metric in ['tf','greedy'] for k in ['original','sharpen','grouped']]
        lines.append('|'+family+'|'+'|'.join(f'{v:.2f}%' for v in vals)+'|')
    lines+=['','## 验证选择与L64规则打乱','','|模型组|锐化β|分组β|打乱原始TF|打乱锐化TF|打乱分组TF|','|---|---:|---:|---:|---:|---:|']
    for family,s in summary.items():
        vals=[100*s[k]['shuffled']['64']['tf'] for k in ['original','sharpen','grouped']]
        lines.append('|'+family+f'|{s["sharpen"]["normal"]["64"]["beta"]}|{s["grouped"]["normal"]["64"]["beta"]}|'+ '|'.join(f'{v:.2f}%' for v in vals)+'|')
    lines+=['','## 解释边界','','分组给所有predicate分配总量1/2，并按模型自身注意力选择内部比例；另1/2给当前answer。不读取真实规则位置或变量匹配答案。它使用任务已知的子句类型和当前状态结构先验，不是无结构的普遍方法。温度幂操作含位置偏置；不能与旧接口只乘身份Q logits的temperature直接混为同一干预。两次前向未做等算力比较。','','每个结果跨4个FFN seed平均，范围见summary.json，固定32题greedy样本较小。诊断统计的真值规则索引仅用于事后attention测量；禁止传入推理wrapper。前次oracle结果用不同测试面板，不能逐题配对比较。']
    report=ROOT/'notes/attention-inference-results-2026-11-10.md';report.write_text('\n'.join(lines)+'\n');print(report)
    for f,s in summary.items():print(f,{k:round(100*s[k]['normal']['64']['tf'],2) for k in ['original','sharpen','grouped']})
if __name__=='__main__':main()
