"""Create four briefing figures from saved per-run results; no model execution."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'brief-figures';OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':12,'axes.titlesize':14,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'pdf.fonttype':42,'savefig.dpi':180,'legend.frameon':False})
paths=[HERE/'evaluations.csv',ROOT/'runs/attention-causal-2026-11-10/per-model.csv',ROOT/'runs/attention-inference-2026-11-10/per-model.csv']
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
raw,causal,infer=map(read,paths)
manifest=[]
colors=['#6B7684','#D07B35','#376DA0','#39856C']
def select(rows,**kwargs):return [r for r in rows if all(r[k]==str(v) for k,v in kwargs.items())]
def seed_values(rows,metric):
 assert rows
 by=defaultdict(list)
 for r in rows:by[r['seed']].append(float(r[metric])*100)
 return [statistics.mean(by[s]) for s in sorted(by)]
def new(n=1,w=10,h=4.4):
 fig,ax=plt.subplots(1,n,figsize=(w,h),layout='constrained',squeeze=False)
 for a in ax[0]:
  a.set_ylim(-2,110);a.set_yticks([0,25,50,75,100]);a.set_ylabel('完整整链正确率（%）');a.grid(axis='y',color='#E5E9EE');a.set_axisbelow(True)
 return fig,list(ax[0])
def bars(ax,groups,labels,palette=colors):
 means=[]
 for i,v in enumerate(groups):
  m=statistics.mean(v);means.append(m)
  ax.bar(i,m,color=palette[i%len(palette)],alpha=.5,width=.6)
  ax.scatter(i+np.linspace(-.13,.13,len(v)),v,color=palette[i%len(palette)],s=26,zorder=3,edgecolors='white',linewidth=.7)
  ax.text(i,min(m+4,104),f'{m:.2f}%',ha='center',fontsize=11)
 ax.set_xticks(range(len(labels)),labels)
 return means
def save(fig,name,data):
 fig.savefig(OUT/f'{name}.png',bbox_inches='tight');fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
 manifest.append({'name':name,'data':data,'png_sha256':hashlib.sha256((OUT/f'{name}.png').read_bytes()).hexdigest()})
# 1: Distinct campaigns, separate panels; no cross-campaign matched effect claim.
fig,ax=new(2,12,4.7)
groups=[]
for c in ['dense','soft_0.1','soft_1','row_tied']:
 rr=[r for r in select(raw,campaign='phase-d-structural-2026-11-10',length=64,evaluation_mode='teacher_forced') if r['run'].split('/')[-1].startswith(f'main_{c}_s')]
 assert len(rr)==8
 groups.append(seed_values(rr,'clause_chain_accuracy'))
a=bars(ax[0],groups,['dense','soft .1','soft 1','hard']);ax[0].set_title('结构约束：Phase D · L64',loc='left')
other=[]
for c in ['self','truth','continue_L2']:
 rr=[r for r in select(raw,campaign='phase-c-length-curriculum-2026-11-10',length=64,evaluation_mode='teacher_forced') if r['run'].split('/')[-1].startswith(f'row_tied_{c}_stage4_s')]
 assert len(rr)==8
 other.append(seed_values(rr,'clause_chain_accuracy'))
b=bars(ax[1],other,['自教师课程','真值课程','持续 L2'],[colors[2],colors[1],colors[3]]);ax[1].set_title('训练课程：Phase C · hard · L64',loc='left')
save(fig,'01-structure-curriculum',{'D_seed_means':groups,'C_seed_values':other,'D_means':a,'C_means':b})
# 2: Order sensitivity after position-aware local training.
fig,ax=new();a=ax[0];data={}
for j,order in enumerate(['normal','shuffled']):
 vals=[]
 for mode in ['none','learned_abs','sinusoidal','relative']:
  rr=select(raw,campaign='position-pretrained-2026-11-10',routing='row_tied',position=mode,ordering=order,length=4,evaluation_mode='teacher_forced')
  assert len(rr)==8
  vals.append(statistics.mean(seed_values(rr,'clause_chain_accuracy')))
 a.bar(np.arange(4)+(j-.5)*.33,vals,width=.3,label=['正常顺序','打乱规则顺序'][j],color=[colors[2],colors[1]][j])
 for i,v in enumerate(vals):a.text(i+(j-.5)*.33,v+2.5,f'{v:.1f}',ha='center',fontsize=10)
 data[order]=vals
a.set_xticks(range(4),['无位置','可学习绝对位置','固定正弦','相对位置']);a.legend(loc='upper left',bbox_to_anchor=(0,-.15),ncol=2);a.set_title('新 FFN 预训练：L4 规则顺序对照',loc='left')
save(fig,'02-position-order',data)
# 3: Oracle causal test, only relative group; includes exact-amplitude random control.
fig,ax=new(w=10,h=4.5);a=ax[0];groups=[]
for condition in ['original','boost_1.0','ratio_0.5','boost_balance','random']:
 rr=select(causal,group='pretrained/relative_row_tied',length=64)
 rr=[r for r in rr if r['condition'].startswith('random_boost_balance_s')] if condition=='random' else select(rr,condition=condition)
 assert len(rr)==(12 if condition=='random' else 4)
 groups.append(seed_values(rr,'clause_chain'))
means=bars(a,groups,['原始','只集中','只平衡','集中 + 平衡','等幅随机'],[colors[0],colors[2],colors[1],colors[3],colors[0]])
a.set_title('相对位置模型：L64 注意力因果干预',loc='left')
save(fig,'03-causal',{'means':means,'seed_means':groups})
# 4: All tested families and both methods; avoid showing only successful families.
families=['phase_d/dense','phase_d/soft_1','phase_d/row_tied','pretrained/none_row_tied','pretrained/learned_abs_row_tied','pretrained/sinusoidal_row_tied','pretrained/relative_row_tied']
labels=['D dense','D soft1','D hard','新 FFN\n无位置','新 FFN\n绝对位置','新 FFN\n正弦','新 FFN\n相对位置']
fig,ax=new(w=12,h=4.7);a=ax[0];data={}
for j,kind in enumerate(['original','sharpen','grouped']):
 vals=[]
 for family in families:
  rr=select(infer,family=family,length=64,shuffled=False,kind=kind);assert len(rr)==4
  vals.append(statistics.mean(seed_values(rr,'tf')))
 a.bar(np.arange(7)+(j-1)*.24,vals,width=.22,color=[colors[0],colors[2],colors[3]][j],label=['原始','锐化','结构分组'][j])
 data[kind]=vals
a.set_xticks(range(7),labels);a.legend(loc='upper left',bbox_to_anchor=(0,-.2),ncol=3);a.set_title('无需正确规则位置：验证集选参后的 L64 测试',loc='left')
save(fig,'04-non-oracle',data)
(OUT/'manifest.json').write_text(json.dumps({'inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'figures':manifest},ensure_ascii=False,indent=2)+'\n')
print('Built four briefing figures from saved CSVs, with PNG/PDF and data manifest.')
