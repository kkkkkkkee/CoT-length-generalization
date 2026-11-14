"""Build publication/export figures and a visual Markdown report from saved results.
No training, evaluation or changes to historical run artifacts.
"""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, re, statistics, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / 'visuals'
FIG.mkdir(exist_ok=True)
SOURCE = HERE / '实验总报告-数值详版.md'
original = SOURCE.read_text()
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.titlesize':12,
    'axes.titleweight':'bold','axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.unicode_minus':False,'figure.dpi':120,'savefig.dpi':180,'pdf.fonttype':42,
    'figure.facecolor':'white','axes.facecolor':'white','legend.frameon':False})
COLORS=['#376DA0','#D07B35','#39856C','#9565A9','#C45559','#6B7684']
ROUTE={'dense':'#6B7684','soft .1':'#D07B35','soft 1':'#376DA0','row_tied':'#39856C'}
blocks=list(re.finditer(r'(?m)^\|.*\n(?:\|.*\n)+',original))
tables=[]
for m in blocks:
    lines=m.group().strip().splitlines()
    cells=[[v.strip() for v in line.strip('|').split('|')] for line in lines]
    titles=re.findall(r'^#{2,3} (.+)$',original[:m.start()],re.M)
    tables.append({'context':titles[-1], 'header':cells[0], 'rows':cells[2:], 'markdown':m.group()})
assert len(tables)==38
with (HERE/'evaluations.csv').open() as f: raw=list(csv.DictReader(f))
raw=[r for r in raw if r['smoke_or_pilot']=='False']
with (ROOT/'runs/attention-inference-2026-11-10/per-model.csv').open() as f: inference=list(csv.DictReader(f))
manifest=[]
replacements={}

def num(s): return float(s.strip().replace('%','').replace('+',''))
def matrix(i,start=1,end=None):return np.array([[num(x) for x in r[start:end]] for r in tables[i]['rows']])
def labels(i):return [r[0] for r in tables[i]['rows']]
def axes(n=1,w=11,h=4.5):
    fig,aa=plt.subplots(1,n,figsize=(w,h),layout='constrained',squeeze=False)
    return fig,list(aa[0])
def accuracy(ax,title=None):
    ax.set_ylim(-2,105);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('完整整链正确率（%）')
    ax.grid(axis='y',color='#E5E9EE',linewidth=.7);ax.set_axisbelow(True)
    if title:ax.set_title(title,loc='left',pad=12)
def save(fig,name,title,caption,table_ids=(),data=None):
    fig.suptitle(title,fontsize=16,fontweight='bold')
    fig.savefig(FIG/f'{name}.png',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    manifest.append({'file':f'visuals/{name}.png','pdf':f'visuals/{name}.pdf','title':title,
        'caption':caption,'table_indices':list(table_ids),'data':data,
        'data_precision':'Tables use audited rounded percentages; raw CSV panels retain original fractions.'})
    return f'![{title}](visuals/{name}.png)\n\n{caption}\n'
def bind(ids,fig,name,title,caption,**kwargs):
    text=save(fig,name,title,caption,ids,**kwargs)
    for j,i in enumerate(ids):
        link=f'[完整数值：附表 {i+1:02d}](数值附录.md#表-{i+1:02d})。\n'
        replacements[i]=(text+'\n'+link) if j==0 else link

def heat(ax,data,rowlabels,collabels,title='',vmin=0,vmax=100):
    im=ax.imshow(data,cmap='Blues',vmin=vmin,vmax=vmax,aspect='auto')
    ax.set_xticks(range(len(collabels)),collabels)
    ax.set_yticks(range(len(rowlabels)),rowlabels)
    ax.tick_params(length=0,pad=8)
    for (i,j),v in np.ndenumerate(data):
        if np.isfinite(v):
            txt=f'{v:.1f}' if 0<v<100 else f'{v:.0f}'
            if 0<v<.1:txt=f'{v:.2f}'
            ax.text(j,i,txt,ha='center',va='center',color='white' if v>60 else '#26394B',fontsize=9)
        else:ax.text(j,i,'未设置',ha='center',va='center',fontsize=9,color='#6B7684')
    for sp in ax.spines.values():sp.set_visible(False)
    ax.set_title(title,loc='left',pad=15)
    cb=ax.figure.colorbar(im,ax=ax,fraction=.035,pad=.035,ticks=[0,50,100])
    cb.set_label('完整整链正确率（%）',fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return im

def grouped(ax,data,rowlabels,series,title='',colors=None):
    data=np.asarray(data);y=np.arange(len(rowlabels));n=data.shape[1];width=.74/n
    for j in range(n):
        offset=(j-(n-1)/2)*width
        ax.barh(y+offset,data[:,j],height=width*.9,label=series[j],color=(colors or COLORS)[j],alpha=.94)
        for k,v in enumerate(data[:,j]):
            ax.text(v+1,y[k]+offset,f'{v:.1f}',va='center',fontsize=8,color='#344454')
    ax.set_yticks(y,rowlabels);ax.invert_yaxis();ax.set_xlim(0,114)
    ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('完整整链正确率（%）')
    ax.grid(axis='x',color='#E5E9EE',linewidth=.7);ax.set_axisbelow(True)
    ax.set_title(title,loc='left',pad=12)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.17),ncol=len(series),fontsize=9)

def box(ax,x,y,w,h,text,color='#EAF0F7',size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.025',
        facecolor=color,edgecolor='#CAD4DF',linewidth=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.6)
def arrow(ax,x,y,xx,yy):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=15,color='#65798E',linewidth=1.5))

# Conceptual figures: no invented empirical measurements.
fig,aa=axes(w=13,h=6);ax=aa[0];ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.text(.01,.95,'端到端复现：依据原文附录 A 的双头实验',fontsize=13,fontweight='bold')
for x,t in [(.02,'五槽输入\n可训练 embedding'),(.35,'单层双头注意力\n残差 + LayerNorm'),(.68,'GELU FFN\n全部参数联合训练')]:box(ax,x,.65,.28,.20,t)
arrow(ax,.30,.75,.35,.75);arrow(ax,.63,.75,.68,.75)
ax.text(.01,.50,'后续机制模型：沿用原文理论骨架，采用有限模型实现',fontsize=13,fontweight='bold')
for x,t in [(.02,'五槽输入\n固定 one-hot'),(.35,'单头路由\nQ43：规则 · Q44：状态'),(.68,'共享 GELU FFN\n先局部预训练，再冻结')]:box(ax,x,.20,.28,.20,t,color='#EAF3EF')
arrow(ax,.30,.30,.35,.30);arrow(ax,.63,.30,.68,.30)
ax.text(.02,.04,'原文理论中的两个块内部仍为 dense；soft / row_tied 为本项目新增约束。\n原文特殊 FFN、clipping 和梯度条件未被完整复现。',fontsize=10,color='#52677B')
replacements[1]=save(fig,'01-models','两类模型与原文的对应关系','结构图用于说明架构与训练流程；完整维度和配置见[附表 02](数值附录.md#表-02)。',[1])
fig,aa=axes(3,w=12,h=4)
rng=np.random.default_rng(10);q=rng.normal(0,.45,(6,6));np.fill_diagonal(q,1.5)
soft=q.copy()
for i in range(6):
 mask=np.arange(6)!=i;soft[i,mask]=-.45+.15*q[i,mask]
hard=np.repeat(np.linspace(-.8,-.2,6)[:,None],6,axis=1);np.fill_diagonal(hard,np.linspace(1.1,1.5,6))
for ax,z,title in zip(aa,[q,soft,hard],['dense：非匹配项可各不相同','soft：惩罚非匹配项的行内差异','hard row_tied：行内非匹配项共享']):
 ax.imshow(z,vmin=-1,vmax=1.5,cmap='RdBu_r');ax.set_title(title,fontsize=11,pad=12)
 ax.set_xlabel('key 变量');ax.set_ylabel('query 变量');ax.set_xticks([]);ax.set_yticks([])
replacements[2]=save(fig,'02-routing','路由约束究竟约束了什么？','示意矩阵，不是训练得到的 Q；三个面板使用相同色阶。hard 同一行的非对角项同色，不同行可以有不同参数。dense / soft / hard 的两块 Q 参数数分别为 32,768 / 32,768 / 512；diagonal 为 256。rename 改变变量曝光，Q-L2 惩罚全部 Q 元素。完整定义见[附表 03](数值附录.md#表-03)。',[2])
fig,aa=axes(w=12,h=6);ax=aa[0];ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
stages=[('复现与初步现象','12 基线 + 9 自训练阶段\n温度 · 上下文 · 固定面板动力学'),('监督与有限数据','81 稀疏监督任务 + 18 模型诊断\n第二轮 87 + 78 + 3 任务'),('结构与优化控制','A/B：502 任务；正则：72 + 6\n覆盖：32 + 4；C：220；D：32 + 4'),('位置机制','v1/v2/v3：48 / 48 / 64 运行\n新 FFN：16 local + 64 routing'),('oracle 机制干预','28 checkpoint × 3 长度 × 50 条件\n集中 · 平衡 · 泄漏 · 等幅随机'),('无 oracle 推理','28 模型先验证、后测试\n56 任务；短链选参后测长链')]
for idx,(title,body) in enumerate(stages):
 row=idx//3;col=idx%3 if row==0 else 2-idx%3;x=.02+col*.33;y=.57 if row==0 else .12
 box(ax,x,y,.29,.28,title+'\n'+body,size=10,color='#EAF0F7' if row==0 else '#EAF3EF')
 if idx<2:arrow(ax,x+.29,y+.14,x+.33,y+.14)
 if idx in [3,4]:arrow(ax,x,y+.14,x-.04,y+.14)
arrow(ax,.825,.57,.825,.40)
replacements[4]=save(fig,'03-roadmap','实验路线：从复现到机制干预','各框是研究问题的推进关系，面积不代表任务量。训练阶段、评测任务和 smoke 不属于同一统计单位；v2/v3 重合条件不能算独立复制。完整完成清单见[附表 05](数值附录.md#表-05)。',[4])

fig,aa=axes();ax=aa[0]
for name,v,c in zip(labels(5),matrix(5),COLORS):ax.plot([.5,1,2,4],v,'o-',label=name,color=c,lw=2)
ax.set_xscale('log',base=2);ax.set_xticks([.5,1,2,4],['0.5','1','2','4']);ax.set_xlabel('推理 logits 乘数 α');accuracy(ax);ax.legend(ncol=4,loc='upper left')
bind([5],fig,'04-temperature','更尖的注意力并非总是更好','L160，3 seeds 的均值。α=2 后继续增大至 4，多个任务退化；这是测试面板上的探索性扫描。')
fig,aa=axes();ax=aa[0]
for name,v,c in zip(labels(6),matrix(6),COLORS):ax.plot([0,20,160],v,'o-',label=name,color=c,lw=2)
ax.set_xlabel('额外无关规则条数');ax.set_xticks([0,20,80,160]);accuracy(ax);ax.legend(ncol=4)
bind([6],fig,'05-context','推理深度不变，上下文增加也会导致失败','L5 固定，α=1，随机干扰；显示报告中的 0/20/160 三个条件。干扰包含重复离路径变量，属于明确的 OOD。')
fig,aa=axes(2,w=12,h=4.5)
for j,(name,c) in enumerate(zip(tables[7]['header'][1:],COLORS)):
 aa[0].plot(range(4),matrix(7)[:,j],'o-',label=name,color=c)
aa[0].set_xticks(range(4),['L5 基线','至 L10','至 L20','至 L40']);accuracy(aa[0],'S5 递归自训练');aa[0].set_ylabel('正确率（%，指标见图例）');aa[0].legend(fontsize=8)
for j,name in enumerate(tables[8]['header'][1:]):aa[1].plot([50,100,200,300],matrix(8)[:,j],'o-',label=name,color=COLORS[j])
aa[1].set_xlabel('训练 epoch');accuracy(aa[1],'固定独立面板上的训练动力学');aa[1].legend()
bind([7,8],fig,'06-training','短链习得、继续训练与长链泛化具有不同时间尺度','左图区分完整整链与最终 value，三种子均值；早期自训练尚无等预算短链控制。右图 L5/L40 各 256 题，跨 checkpoint 使用相同题目。')
fig,aa=axes(2,w=13,h=4.7)
for a,idx,title in [(aa[0],[0,2,4],'真值监督'),(aa[1],[1,3,5],'教师监督')]:
 vals=matrix(9)[idx]
 for j,(name,c) in enumerate(zip(['second','random','all'],COLORS)):
  a.bar(np.arange(3)+(j-1)*.23,vals[:,j],width=.21,color=c,label=name)
 a.set_xticks(range(3),['4,096 题\n等题数','匹配 value\n标签预算','512 题\n等题数']);accuracy(a,title);a.legend()
bind([9],fig,'07-supervision','全步骤监督的优势取决于比较预算','L8，3 seeds；三组柱分别表示三个独立预算协议。等标签条件下 all 的优势基本消失。')
fig,aa=axes();ax=aa[0]
for j,(name,c) in enumerate(zip(['错误目标','错误前缀 + 原目标','错误前缀 + 一致目标'],COLORS)):
 ax.plot([0,5,20,40],matrix(10)[:,j],'o-',color=c,label=name,lw=2.5)
ax.set_xlabel('请求污染率（%）');ax.set_xticks([0,5,20,40]);accuracy(ax);ax.legend()
bind([10],fig,'08-noise','局部一致性比全局“错误标签率”更能解释退化','L8，固定 second 监督。局部一致组对实际输入没有错误转移标签，因此不能解释为可承受 40% 任意标签噪声。')
fig,aa=axes(w=12,h=4.8)
heat(aa[0],matrix(11),labels(11),['原始','只平衡','只集中','两者一起','Q 行均值投影'])
bind([11],fig,'09-early-probe','早期探针：污染模型同时受集中程度与比例限制','L8 完整整链，18 个模型中的主要条件组；使用 oracle 正确规则位置。旧随机对照有幅度截断，不用于特异性归因。')
fig,aa=axes(2,w=13,h=5.5)
grouped(aa[0],matrix(12),labels(12),['lr=.03'],'独立数据重复：L8')
grouped(aa[1],matrix(13),labels(13),['lr=.03','lr=.003'],'配对学习率：仅 repeat 0')
bind([12,13],fig,'10-replication','从已学路由继续训练时，较低学习率改善外推','左图 3 FFN seeds × 3 数据 repeats；右图只用每 seed 的 repeat 0。相同更新次数不代表相同优化进程。')
fig,aa=axes(w=12,h=4.2)
grouped(aa[0],matrix(14),labels(14),['lr=.03','lr=.003'])
aa[0].set_xlabel('比例 / 准确率（%）')
bind([14],fig,'11-memorization','拟合污染目标时，正确规则检索也遭到破坏','L4 第二步诊断，20% target 污染，1,000 步；三种子均值。第一行越高意味着错误标签记忆越强，并非模型更好。')
fig,aa=axes(2,w=13,h=5.4)
for j,ax in enumerate(aa):
 for k,row in enumerate(tables[15]['rows']):
  before,after=map(float,row[j+1].split('→'));ax.plot([before,after],[k,k],color='#A4B4C4',lw=3)
  ax.scatter(before,k,color=COLORS[5],s=35,label='转换后' if k==0 else None)
  ax.scatter(after,k,color=COLORS[2],s=45,label='训练后' if k==0 else None)
 ax.set_yticks(range(6),labels(15));ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xlabel('完整整链正确率（%）');ax.set_title(['L8','L16'][j]);ax.legend(loc='upper left',bbox_to_anchor=(0,-.14),ncol=2);ax.grid(axis='x',alpha=.2)
bind([15],fig,'12-projection','结构转换的即时收益与继续训练收益分开看','每段线从转换后连接到训练后；dense/row_tied 为 3×3 重复，diagonal 仅每 seed 的 repeat 0。')
fig,aa=axes();ax=aa[0]
grouped(ax,matrix(16).T,['L2 warm','自教师至 L32','真值至 L32','持续 L2'],['dense','row_tied'],colors=[ROUTE['dense'],ROUTE['row_tied']])
bind([16],fig,'13-recursive','早期三种子：持续短题也能达到相同 L64 终点','各组柱对应不同训练方案。三种子均值，终点比较匹配更新与 value 标签呈现，不匹配 FLOPs。')
fig,aa=axes(2,w=13,h=5.3)
data=matrix(17,2)
heat(aa[0],data[:4],[r[1] for r in tables[17]['rows'][:4]],['L8','L16','L64'],'dense')
heat(aa[1],data[4:],[r[1] for r in tables[17]['rows'][4:]],['L8','L16','L64'],'row_tied')
bind([17],fig,'14-phase-a','Phase A：结构约束改善中等长度，但 L64 仍不稳定','8 FFN seeds × 3 数据 repeats；学生从 Q=0 开始。各格为完整整链百分比，不能把 24 配对当成 24 个独立 FFN。')
fig,aa=axes(w=11,h=3.8);ax=aa[0]
for k,r in enumerate(tables[18]['rows']):
 mean=num(r[1]);lo,hi=[num(x) for x in r[2].split('～')]
 ax.errorbar(mean,k,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color=COLORS[2],ms=7)
ax.set_yticks(range(4),labels(18));ax.invert_yaxis();ax.set_xlim(0,52);ax.axvline(0,color='#7F8B97',ls='--');ax.grid(axis='x',alpha=.2);ax.set_xlabel('row_tied − dense：L8 提升（百分点）')
bind([18],fig,'15-phase-a-ci','Phase A：结构约束的配对收益','点为平均差值，误差线为 95% 分层 bootstrap 区间；先抽 FFN seeds，再抽 seed 内数据 repeats。')
fig,aa=axes(2,w=13,h=4.8)
for ax,route in zip(aa,['dense','row_tied']):
 for j,lr in enumerate([.003,.01,.03]):
  selected=[r for r in tables[37]['rows'] if r[0]==route and float(r[1])==lr and r[2]=='target']
  selected.sort(key=lambda r:num(r[3]));ax.plot([num(r[3]) for r in selected],[num(r[5]) for r in selected],'o-',label=f'lr={lr:g}',color=COLORS[j],lw=2)
 ax.set_xlabel('请求 target 污染率（%）');ax.set_xticks([0,5,20,40]);accuracy(ax,route);ax.legend()
bind([19],fig,'16-phase-b','Phase B：从 Q=0 开始，低学习率在固定预算下训练不足','L8，8 seeds，1,000 更新。图中为 target 剂量曲线；前缀一致/不一致对照和 L16/L64 全部数字保留在附录。')
fig,aa=axes(2,w=14,h=6)
grouped(aa[0],matrix(20)[:,[1]],labels(20),['L16'],'中等长度')
grouped(aa[1],matrix(20)[:,[2]],labels(20),['L64'],'更长链')
bind([20],fig,'17-regularization','软约束与重命名可改善 L16，但 1,000 步仍不足以稳定 L64','4 FFN seeds × 2 数据 repeats；两图使用相同 0–100% 横轴。hard row_tied 的 L64 成功主要集中于少数 seed。')
fig,aa=axes(2,w=13,h=4.8)
for col,ax in enumerate(aa):
 values=np.array([[float(v.strip()) for v in r[col+1].split('/')] for r in tables[21]['rows']])
 for j,lab in enumerate(['Q43','Q44']):ax.plot(np.arange(len(values))+(j-.5)*.12,values[:,j],'o',label=lab,color=COLORS[j],ms=6)
 ax.set_xticks(range(len(values)),labels(21),rotation=25,ha='right');ax.set_title(['非对角行内残差 RMS','对角减平均行背景'][col]);ax.set_ylabel(['RMS','分数差'][col]);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.legend()
bind([21],fig,'18-q-geometry','压低行内残差与建立匹配分数差是不同要求','4 seeds 内先平均 repeats，再跨 seeds 平均；各处理是离散协议。平均分数差不等于最差竞争规则 margin。')
fig,aa=axes(2,w=12,h=4.5)
for ax,j,title in [(aa[0],1,'L16'),(aa[1],2,'L64')]:
 vals=matrix(22)[:,j];grouped(ax,np.array([[vals[0],vals[1]],[vals[2],vals[3]]]),['dense','row_tied'],['uniform','balanced'],title)
bind([22],fig,'19-coverage','平衡配对曝光没有救回 dense','4 seeds × 2 repeats；配对曝光方案改变变量链分布，不表示全部非匹配变量对都被覆盖。')

# Phase C trajectory and Phase D individual seeds are computed from raw CSV.
fig,aa=axes(2,w=13,h=4.8)
for ax,route in zip(aa,['dense','row_tied']):
 for j,regime in enumerate(['self','truth','continue_L2']):
  ys=[]
  for stage in range(5):
   prefix=f'warm_{route}_s' if stage==0 else f'{route}_{regime}_stage{stage}_s'
   rr=[r for r in raw if r['campaign']=='phase-c-length-curriculum-2026-11-10' and r['run'].split('/')[-1].startswith(prefix) and r['length']=='64' and r['evaluation_mode']=='teacher_forced']
   assert len(rr)==8
   ys.append(100*statistics.mean(float(r['clause_chain_accuracy']) for r in rr))
  ax.plot(range(5),ys,'o-',label=['自教师','真值课程','持续 L2'][j],color=COLORS[j],lw=2)
 ax.set_xticks(range(5),['warm','阶段 1','阶段 2','阶段 3','阶段 4']);accuracy(ax,route);ax.legend()
bind([23],fig,'20-phase-c','Phase C：长链成功不要求训练长度不断增长','纵轴始终为独立 L64 测试成绩；8 seeds，原始 CSV 重算。课程分支依次训练 L4/8/16/32，持续 L2 分支每阶段始终训练 L2。终点只取阶段 4，不平均中间阶段。')
fig,aa=axes(2,w=13,h=4.8)
ddata=[]
for j,(cond,lab) in enumerate([('dense','dense'),('soft_0.1','soft .1'),('soft_1','soft 1'),('row_tied','row_tied')]):
 means=[];end=[]
 for length in [4,8,16,32,64]:
  rr=[r for r in raw if r['campaign']=='phase-d-structural-2026-11-10' and r['run'].split('/')[-1].startswith(f'main_{cond}_s') and r['length']==str(length) and r['evaluation_mode']=='teacher_forced']
  assert len(rr)==8
  byseed=[100*statistics.mean(float(r['clause_chain_accuracy']) for r in rr if r['seed']==str(s)) for s in range(4)]
  means.append(statistics.mean(byseed))
  if length==64:end=byseed
 aa[0].plot([4,8,16,32,64],means,'o-',label=lab,color=ROUTE[lab],lw=2)
 aa[1].bar(j,statistics.mean(end),width=.62,color=ROUTE[lab],alpha=.3)
 aa[1].scatter(j+np.linspace(-.15,.15,4),end,s=42,color=ROUTE[lab],edgecolors='white',zorder=3)
 aa[1].text(j,statistics.mean(end)+4,f'{statistics.mean(end):.2f}%',ha='center',fontsize=10)
 ddata.append({'condition':cond,'lengths':[4,8,16,32,64],'means':means,'L64_seed_means':end})
aa[0].set_xscale('log',base=2);aa[0].set_xticks([4,8,16,32,64],['4','8','16','32','64']);aa[0].set_xlabel('评测链长 L');accuracy(aa[0],'长度曲线');aa[0].legend(fontsize=9)
accuracy(aa[1],'L64：均值与四个 FFN seeds');aa[1].set_xticks(range(4),['dense','soft .1','soft 1','row_tied'])
bind([24,25],fig,'21-phase-d','Phase D：硬约束的收益体现在长链与跨种子稳定性','4 seeds × 2 repeats，2,000 更新。右图每个点先平均该 seed 的两个数据 repeats；点是独立 FFN seed，不是置信区间。数据从原始 CSV 重算。',data=ddata)

fig,aa=axes(w=11,h=4.2)
zz=np.array([[np.nan if x=='未设置' else num(x) for x in r[1:]] for r in tables[27]['rows']])
heat(aa[0],zz,labels(27),['v1','v2','v3'])
bind([27],fig,'22-position-history','旧 FFN 的位置实验：固定正弦并未普遍破坏泛化','row_tied，L64 完整整链（%）。v1 数据重复不独立；v2/v3 重合条件不能当作独立复制。各版本 dense 与 row_tied 的可学习位置参数冻结策略不同。')
fig,aa=axes(2,w=13,h=4.7)
for k,ax in enumerate(aa):heat(ax,matrix(28,2)[k*4:(k+1)*4],[r[0] for r in tables[28]['rows'][k*4:(k+1)*4]],['L4','L8','L16','L64'],['dense：可学习位置项被冻住','row_tied：可学习位置项可训练'][k])
bind([28],fig,'23-position-v3','v3 结果必须连同冻结策略一起解读','旧 FFN，4 seeds × 2 repeats。此图保留历史证据，不用于无条件比较两种路由的 PE 效应；当时没有保存规则打乱面板。')
fig,aa=axes(2,w=13,h=4.9)
for k,route in enumerate(['dense','row_tied']):
 rr=[r for r in tables[29]['rows'] if r[1]==route]
 heat(aa[k],np.array([[num(x) for x in r[2:]] for r in rr]),[r[0] for r in rr],['L4\n正常','L4\n打乱','L8\n正常','L16\n正常','L64\n正常'],route)
bind([29],fig,'24-position-pretrain','新 FFN 预训练：位置依赖与无位置基线下降必须分开解释','16 local + 64 routing 全部完成。可学习位置项在两种路由中统一训练；每条件 4 seeds × 2 repeats。无位置 L64 也为 0%，不能把全部失败归因于 PE。')

# Mechanism diagram and intervention heatmaps.
fig,aa=axes(3,w=13,h=3.6)
for ax,name,vals in zip(aa,['原始：定义 m 与 r','仅集中：增加 m，保持 r','仅平衡：保持 m，调整 r'],[[.2,.1,.7],[.6,.3,.1],[.15,.15,.7]]):
 ax.pie(vals,colors=[COLORS[0],COLORS[1],'#DEE4EB'],startangle=90,wedgeprops={'width':.38,'edgecolor':'white'})
 ax.set_title(name,pad=10,fontsize=11)
 ax.text(0,0,f'm = {sum(vals[:2]):.2f}\nr = {vals[0]/sum(vals[:2]):.2f}',ha='center',va='center',fontsize=11)
fig.legend(['正确规则','当前状态','无关背景'],loc='lower center',ncol=3)
replacements[30]=save(fig,'25-mass-balance','注意力“集中”与“平衡”是两个不同操作','数值为操作示意，不是模型测量值。m 为两个正确位置的总质量，r 为其中规则占比。正式实验另含泄漏、二维剂量网格和逐 query 等 L1 随机对照；使用 oracle 位置。完整设计见[附表 31](数值附录.md#表-31)。',[30])
fig,aa=axes(w=12,h=5.4)
heat(aa[0],matrix(31),labels(31),['原始','只集中\nm=1','只平衡\nr=.5','集中 + 平衡','规则泄漏\n50%'])
bind([31],fig,'26-causal','相对位置模型：只集中不够，集中与平衡一起恢复','L64，7 组 × 4 seeds，repeat 0，256 题。所有预训练组均为 row_tied；此为 oracle 正确位置干预，不能视为无需额外信息的推理算法。')
fig,aa=axes(2,w=13,h=4.8)
short=['Phase D dense','Phase D soft1','Phase D row_tied','预训练各组（均相同）']
grouped(aa[0],matrix(32)[:,:2],short,['规则泄漏 50%','等幅随机'],'破坏侧')
grouped(aa[1],matrix(32)[:,2:],short,['集中 + 平衡','等幅随机'],'恢复侧')
bind([32],fig,'27-random','等幅随机对照：恢复侧比破坏侧更有区分力','L64；随机方向先在模型内平均，再跨 4 seeds。预训练四组在这里数值均相同，合并展示；随机控制只匹配 L1 幅度，不匹配修改位置数。')

families=['phase_d/dense','phase_d/soft_1','phase_d/row_tied','pretrained/none_row_tied','pretrained/learned_abs_row_tied','pretrained/sinusoidal_row_tied','pretrained/relative_row_tied']
short=['Phase D dense','Phase D soft1','Phase D row_tied','预训练 none','预训练 abs','预训练 sinusoidal','预训练 relative']
fig,aa=axes(2,w=14,h=6.8)
for ax,metric,title in [(aa[0],'tf','TF：256 题 / seed'),(aa[1],'greedy','Greedy：32 题 / seed')]:
 for j,(kind,c) in enumerate(zip(['original','sharpen','grouped'],[COLORS[5],COLORS[0],COLORS[2]])):
  for i,family in enumerate(families):
   rr=[r for r in inference if r['family']==family and r['length']=='64' and r['shuffled']=='False' and r['kind']==kind]
   assert len(rr)==4,(family,kind)
   vals=np.array([float(r[metric])*100 for r in rr]);y=i+(j-1)*.23
   ax.plot([min(vals),max(vals)],[y,y],color=c,alpha=.4,lw=2)
   ax.scatter(vals,np.full(4,y),color=c,s=15,alpha=.5)
   ax.scatter([np.mean(vals)],[y],marker='D',s=35,color=c,label=['原始','锐化','分组'][j] if i==0 else None,zorder=5)
 ax.set_yticks(range(7),short);ax.invert_yaxis();ax.set_xlim(-3,105);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('完整整链正确率（%）');ax.set_title(title,loc='left',pad=15);ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right')
bind([33],fig,'28-inference','不提供正确规则位置，也能恢复部分模型的 L64 泛化','先在 L4/L8 验证集选 β，全部锁定后才测试 L64。菱形为 4 seeds 均值，小点为各 seed，横线为最小–最大范围，均不是置信区间；数据从逐模型 CSV 重算。')
fig,aa=axes(w=12,h=5.3);ax=aa[0]
for j,(kind,c) in enumerate(zip(['sharpen','grouped'],[COLORS[0],COLORS[2]])):
 for i,f in enumerate(families):
  means=[]
  for shuffled in ['False','True']:
   vals=[float(r['tf'])*100 for r in inference if r['family']==f and r['kind']==kind and r['shuffled']==shuffled and r['length']=='64'];assert len(vals)==4
   means.append(statistics.mean(vals))
  y=i+(j-.5)*.27;ax.plot(means,[y,y],color=c,lw=2)
  ax.scatter(means[0],y,marker='o',s=38,facecolors='white',edgecolors=c,zorder=3)
  ax.scatter(means[1],y,marker='x',s=40,color=c,zorder=3,label=['锐化','分组'][j] if i==0 else None)
ax.set_yticks(range(7),short);ax.invert_yaxis();ax.set_xlim(-3,105);ax.set_xlabel('L64 完整整链正确率（%）');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right')
bind([34],fig,'29-order','规则顺序改变后，哪些恢复仍能保留？','空心圆为正常顺序，叉号为打乱顺序；同色横线连接同一方法的两种顺序，重合时只显一个符号。四 seed 均值。sinusoidal 的分组恢复从 54.10% 降至 5.57%；选择的 β 见附表。')

# Convert remaining administrative tables into short prose/links, keep only 3 reference tables.
replacements[35]='各阶段已完成测试与冻结/哈希审计；检查数量、覆盖范围和历史记录见[质量控制明细（附表 36）](数值附录.md#表-36)。本次只重建图表与文档，不重新训练模型。\n'
replacements[36]='\n'.join('- '+r[0]+'：'+r[1] for r in tables[36]['rows'])+'\n\n- [数值附录](数值附录.md)：全部 38 张原始表格。\n- [图表重建脚本](visualize_report.py)：生成全部新增 PNG/PDF 图表与本报告。\n- [图表来源清单](visuals/manifest.json)：图表说明、来源表格、文件哈希与主要逐 seed 数据。\n'
replacements[37]='Phase B 全部 36 条件及三个长度的数值见[完整结果表](Phase-B完整结果.md)，亦保存在[数值附录，表 38](数值附录.md#表-38)。正文第 7 节展示噪声剂量曲线。\n'
# Write numerical appendix with stable anchors, preserving every original table.
appendix=['# 数值附录','', '本文保存可视化报告中的全部原始数值与设置表。数值单位及统计口径以各节说明为准；不同阶段的数据不能直接合并。', '']
for i,t in enumerate(tables):appendix.extend([f'## 表 {i+1:02d}','',t['context'],'',t['markdown'].strip(),''])
(HERE/'数值附录.md').write_text('\n'.join(appendix))
updated=original
for i in reversed(range(len(blocks))):
 if i in replacements:updated=updated[:blocks[i].start()]+replacements[i].rstrip()+'\n'+updated[blocks[i].end():]
# Remove redundant old figures now superseded by fuller, Chinese-labelled plots.
updated=re.sub(r'!\[无正确规则位置的 L64 推理干预\]\([^\n]+\)\n\n图中范围是四 seed 的最小/最大值，不是 95% 置信区间。\n','',updated)
updated=updated.replace('本报告汇总当前工作区中已经完成的实验，包括未达到预期的结果、后续修正和工程试跑。','本报告以可视化展示当前工作区中已经完成的实验，包括未达到预期的结果、后续修正和工程试跑。完整数值与设置保留在[数值附录](数值附录.md)，所有新增图提供 PNG 和可导出的 PDF。')
updated=updated.replace('下面是 α=1、随机干扰的完整整链率：','下图展示 α=1、随机干扰的完整整链率：')
updated=updated.replace('下表为 TF 完整整链：','下图为 TF 完整整链：')
updated=updated.replace('v3 的完整 TF 结果：','v3 各模式的 TF 结果如下：')
updated=updated.replace('所以表中均值不必相同','所以图中均值不必相同').replace('表格单位均为','图表单位均为')
updated=updated.replace('### 2.2 两类模型不能直接混合比较','### 2.2 两类模型不能直接混合比较\n\n原文理论部分采用第二类的基本骨架；附录 A 的实证模型明确为双头。我们第二类使用普通 GELU FFN 与 Adam，soft / row_tied 则是额外拓展，不能等同于原定理全部条件的严格复现。')
# Reading aid, grounded explicitly in sources and plot semantics.
intro='''\n**读图约定：**除另注外，纵轴或颜色表示 TF 完整五槽整链正确率（%），越高越好。各图在原章节保留训练与评测设置。只有 Phase A 配对收益图使用 95% bootstrap 区间；Phase D 和无 oracle 推理图展示逐 FFN seed 的实际分散程度。示意图明确标注不属于实测数据。主阶段曲线与 seed 点从保存的 CSV 重算，其余图由已核对、保留两位小数的汇总表绘制。\n'''
updated=updated.replace('## 目录\n',intro+'\n## 目录\n')
updated=updated.split('## 附录 A：Phase B 全部条件')[0]+'''## 附录 A：完整数值与导出图表

全部设置表和数值表见[数值附录](数值附录.md)，Phase B 的完整条件另见[专项附表](Phase-B完整结果.md)。每张新增图的同名 PDF 位于 `visuals/`，可直接用于演示或论文排版；[图表索引](图表索引.md)提供全部 PNG/PDF 链接。

重建本报告图表与正文：

```bash
python reports/experiment-report-2026-11-11/visualize_report.py
```
'''
# Copy the existing measured mass-ratio grid to keep every displayed image local.
grid=ROOT/'runs/attention-causal-2026-11-10/mass-ratio-L64.png'
shutil.copy2(grid,FIG/'30-mass-ratio-L64.png')
shutil.copy2(grid.with_suffix('.pdf'),FIG/'30-mass-ratio-L64.pdf')
updated=updated.replace('../../runs/attention-causal-2026-11-10/mass-ratio-L64.png','visuals/30-mass-ratio-L64.png')
# Store all graph data for review; preserve rounded table data exactly.
for item in manifest:
 item['source_tables']=[{'index':i,'context':tables[i]['context'],'header':tables[i]['header'],'rows':tables[i]['rows']} for i in item['table_indices']]
 item['png_sha256']=hashlib.sha256((HERE/item['file']).read_bytes()).hexdigest()
 item['pdf_sha256']=hashlib.sha256((HERE/item['pdf']).read_bytes()).hexdigest()
manifest.append({'file':'visuals/30-mass-ratio-L64.png','pdf':'visuals/30-mass-ratio-L64.pdf',
    'title':'L64 注意力质量与规则比例的二维干预网格','existing_artifact':str(grid.relative_to(ROOT)),
    'png_sha256':hashlib.sha256(grid.read_bytes()).hexdigest(),
    'pdf_sha256':hashlib.sha256(grid.with_suffix('.pdf').read_bytes()).hexdigest()})
inputs=[SOURCE,HERE/'evaluations.csv',ROOT/'runs/attention-inference-2026-11-10/per-model.csv',Path(__file__)]
(FIG/'manifest.json').write_text(json.dumps({'source_report_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    'figures':manifest},ensure_ascii=False,indent=2)+'\n')
index=['# 图表索引','','所有图均可单独导出。PNG 用于阅读，PDF 为矢量版本。完整实验设计及解释见[实验总报告](实验总报告.md)。','']
for i,item in enumerate(manifest,1):
    index.extend([f"{i}. **{item['title']}**：[PNG]({item['file']}) · [PDF]({item['pdf']})",''])
(HERE/'图表索引.md').write_text('\n'.join(index))
(HERE/'实验总报告.md').write_text(updated)
print(json.dumps({'new_figures':len(manifest),'tables_moved':len(replacements),'remaining_reference_tables':len(tables)-len(replacements),'report_characters':len(updated)},ensure_ascii=False))
