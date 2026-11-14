"""Aggregate paired oracle-position diagnostics across four fixed model seeds."""
from pathlib import Path
from collections import defaultdict
import json
import csv
import statistics
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,default=ROOT/'runs/attention-causal-2026-11-10');args=parser.parse_args()
    base=args.campaign;plan=json.loads((base/'campaign-plan.json').read_text());progress=json.loads((base/'campaign-progress.json').read_text())
    if not progress['complete']:raise RuntimeError('Campaign incomplete')
    groups=defaultdict(list);flat=[];maxerr=0.;sham=0.
    for job in plan['jobs']:
        result=json.loads((base/job['name']/'result.json').read_text())
        assert result['checkpoint_sha256']==job['checkpoint_sha256']
        assert result['parameter_sha256_before']==result['parameter_sha256_after']
        rows={}
        for r in result['rows']:
            key=(r['length'],r['condition']);rows[key]=r
            g=f"{job['family']}/{job['treatment']}"
            groups[g,r['length'],r['condition']].append(r)
            flat.append(dict(group=g,seed=job['seed'],**{k:r[k] for k in ('length','condition')},
                             **r['metrics'],**r['diagnostics'],**r['paired']))
            maxerr=max(maxerr,r['random_max_l1_error']);sham=max(sham,r['sham_max_logit_delta'])
        # Average the three randomized directions WITHIN each model seed first.
        for length in plan['lengths']:
            for spec in result['conditions'][2:11]:
                controls=[rows[length,f'random_{spec["name"]}_s{s}'] for s in range(3)]
                avg=dict(metrics={k:statistics.mean(x['metrics'][k] for x in controls) for k in controls[0]['metrics']},
                         diagnostics={k:statistics.mean(x['diagnostics'][k] for x in controls) for k in controls[0]['diagnostics']},
                         paired={k:statistics.mean(x['paired'][k] for x in controls) for k in controls[0]['paired']})
                groups[g,length,'random_mean_'+spec['name']].append(avg)
    summary={}
    for (g,l,c),rows in sorted(groups.items()):
        values=[r['metrics']['clause_chain'] for r in rows]
        summary.setdefault(g,{}).setdefault(str(l),{})[c]=dict(n_seeds=len(rows),
            mean=statistics.mean(values),min=min(values),max=max(values),
            paired_delta=statistics.mean(r['paired']['delta'] for r in rows),
            rescued=statistics.mean(r['paired']['rescued'] for r in rows),
            harmed=statistics.mean(r['paired']['harmed'] for r in rows),
            diagnostics={k:statistics.mean(r['diagnostics'][k] for r in rows) for k in rows[0]['diagnostics']})
    (base/'summary.json').write_text(json.dumps(dict(groups=summary,sham_max_logit_difference=sham,random_max_l1_error=maxerr),indent=2)+'\n')
    with (base/'per-model.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['# 注意力集中与比例干预结果','','固定28个checkpoint、7个条件组，每组4个模型seed（固定replicate0）。每模型每长度256道相同新测试题。所有结果为使用oracle规则位置的teacher-forced完整clause整链准确率，不是无需额外信息的推理方法。',
           f'','审计：模型参数hash不变；sham最大logit差 %.3g；随机对照逐query L1最大误差 %.3g。'%(sham,maxerr),'',
           '|模型组（L64）|原始|仅集中到1|仅平衡(.5)|集中≥.99且平衡|m=1,r=.5|规则泄漏50%|',
           '|---|---:|---:|---:|---:|---:|---:|']
    keys=['original','boost_1.0','ratio_0.5','boost_balance','grid_m1.0_r0.5','leak_rule_0.5']
    for g,ls in summary.items():
        row=ls['64'];lines.append('|'+g+'|'+'|'.join(f'{100*row[k]["mean"]:.2f}%' for k in keys)+'|')
    lines+=['','## L64 等幅随机对照','','随机方向先在每个模型内平均，再跨4个seed平均。L1相同不代表方向、支持大小或m/r改变量相同；随机控制也可能损害真正需要的attention。','','|模型组|规则泄漏50%|等幅随机|集中+平衡|等幅随机|','|---|---:|---:|---:|---:|']
    for g,ls in summary.items():
        row=ls['64'];lines.append('|'+g+'|'+'|'.join(f'{100*row[k]["mean"]:.2f}%' for k in ['leak_rule_0.5','random_mean_leak_rule_0.5','boost_balance','random_mean_boost_balance'])+'|')
    lines+=['','## 四seed范围与配对变化','','|模型组|原始L64范围|集中且平衡范围|集中且平衡配对提升（百分点）|','|---|---:|---:|---:|']
    for g,ls in summary.items():
        b=ls['64']['original'];v=ls['64']['boost_balance'];lines.append(f'|{g}|{100*b["min"]:.2f}–{100*b["max"]:.2f}%|{100*v["min"]:.2f}–{100*v["max"]:.2f}%|{100*v["paired_delta"]:+.2f}|')
    lines+=['','集中程度定义为两个正确位置的联合质量m，不等于熵低。比例r为其中正确规则所占份额。比例1/2是预先指定诊断点，不宣称为所有FFN的普遍最优值。干预同时改变FFN混合输入；只能说明给定模型/样本下这些权重变化的因果效果，不证明训练为何形成该注意力。正常顺序、真实前缀；未测偏离真实轨迹后的恢复。无测试后挑选checkpoint或最优m/r。所有50个条件及配对outcomes保存在campaign中。']
    report=ROOT/'notes/attention-causal-results-2026-11-10.md';report.write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for ax,(g,ls) in zip(axs.flat,summary.items()):
        v=np.array([[100*ls['64'][f'grid_m{m}_r{r}']['mean'] for r in (.25,.5,.75)] for m in (.5,.9,.99,1.)])
        im=ax.imshow(v,vmin=0,vmax=100,aspect='auto',cmap='viridis');ax.set_title(g,fontsize=10)
        ax.set_xticks(range(3),['.25','.50','.75']);ax.set_yticks(range(4),['.50','.90','.99','1.00'])
        ax.set_xlabel('rule share r');ax.set_ylabel('relevant mass m')
        for i in range(4):
            for j in range(3):ax.text(j,i,f'{v[i,j]:.1f}',ha='center',va='center',color='white' if v[i,j]<50 else 'black',fontsize=9)
    axs.flat[-1].axis('off');fig.colorbar(im,ax=list(axs.flat[:-1]),label='L64 full-clause exact-chain accuracy (%)',shrink=.8)
    fig.suptitle('Oracle-position intervention: concentration × rule/state ratio')
    fig.savefig(base/'mass-ratio-L64.png',dpi=160);fig.savefig(base/'mass-ratio-L64.pdf');plt.close(fig)
    print(report)
    for g,ls in summary.items():print(g,{k:round(100*ls['64'][k]['mean'],2) for k in keys})
if __name__=='__main__':main()
