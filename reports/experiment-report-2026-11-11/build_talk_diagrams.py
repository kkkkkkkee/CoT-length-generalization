"""Explanatory diagrams for the 30-minute talk; no empirical data are synthesized."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
HERE=Path(__file__).resolve().parent;OUT=HERE/'talk-figures';OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':13,'pdf.fonttype':42,'axes.unicode_minus':False,'savefig.dpi':180})
manifest=[]
def canvas(title,h=5):
 fig,ax=plt.subplots(figsize=(12,h));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');fig.suptitle(title,fontsize=19,fontweight='bold');return fig,ax

def box(ax,x,y,w,h,text,color='#EAF0F7',fs=13):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.018',facecolor=color,edgecolor='#C5D2DD'))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,linespacing=1.65)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,color='#60788A',lw=1.6))
def save(fig,name,description):
 fig.savefig(OUT/f'{name}.png',bbox_inches='tight');fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
 manifest.append({'name':name,'kind':'explanatory diagram, not a measured result','description':description,'png_sha256':hashlib.sha256((OUT/f'{name}.png').read_bytes()).hexdigest()})
fig,ax=canvas('任务：变量决定读哪条规则，动作决定状态如何更新',5.3)
box(ax,.02,.64,.29,.21,'当前答案\n变量 B，状态 2',color='#FFF0DE')
box(ax,.37,.64,.29,.21,'找到 source = B 的规则\nC ← g2(B)')
box(ax,.72,.64,.26,.21,'执行局部转移\ng2(2) = 4',color='#E6F2ED')
arrow(ax,(.31,.745),(.37,.745));arrow(ax,(.66,.745),(.72,.745))
box(ax,.02,.23,.42,.22,'输入规则池（位置可打乱）\nB ← g1(A)　　D ← g3(C)　　C ← g2(B)',fs=12)
box(ax,.57,.23,.41,.22,'下一条答案\n变量 C，状态 4 → 继续下一步',color='#E6F2ED')
arrow(ax,(.51,.64),(.51,.51));arrow(ax,(.51,.51),(.78,.51));arrow(ax,(.78,.51),(.78,.45))
ax.text(.02,.05,'示意例：g2 取一个满足 g2(2)=4 的合法置换。规则以五槽编码，答案也包含变量与状态；完整整链指标要求所有槽、所有步都正确。',fontsize=10,color='#52677B')
save(fig,'01-task','Illustrative legal S5 transition; not a sampled test record.')
fig,ax=canvas('共同训练骨架：先学局部计算，再学检索',4.7)
box(ax,.02,.47,.27,.34,'局部预训练\nL1 · Q 固定为 0\n训练普通 GELU FFN')
box(ax,.365,.47,.27,.34,'路由训练\n冻结 FFN\n训练 Q43 / Q44',color='#E6F2ED')
box(ax,.71,.47,.27,.34,'独立评测\n完整整链 · L2…L64\nTF + 自由 greedy',color='#FFF0DE')
arrow(ax,(.29,.64),(.365,.64));arrow(ax,(.635,.64),(.71,.64))
ax.text(.03,.25,'Q43：当前变量匹配规则的 source 槽      Q44：当前变量匹配答案的 variable 槽',fontsize=13)
ax.text(.03,.10,'固定 one-hot · 单头 · FFN 宽 512 · 无残差 / LayerNorm / 可训练 V/O 投影',fontsize=12,color='#52677B')
save(fig,'02-model','Shared mechanism-model training pipeline; position-aware training adds learnable position parameters in both stages.')
fig,ax=canvas('Phase C：同一起点，三种后续训练方案',5.5)
box(ax,.02,.41,.2,.22,'同一 local FFN\nQ = 0 → L2 warm',fs=12)
ys=[.73,.43,.13];titles=['自教师课程','真值课程','持续短题控制'];sub=['自己生成标签，不筛选或修正','任务生成器提供正确标签','每阶段使用新的 L2 真值题']
for y,title,desc in zip(ys,titles,sub):
 arrow(ax,(.22,.52),(.32,y+.1));box(ax,.32,y,.19,.20,title+'\n'+desc,fs=10,color='#EAF0F7')
for row,y in enumerate(ys):
 vals=['L4','L8','L16','L32'] if row<2 else ['L2']*4
 for i,v in enumerate(vals):
  x=.56+i*.105;box(ax,x,y+.035,.075,.13,v,color=['#EAF0F7','#FFF0DE','#E6F2ED'][row],fs=12)
  if i<3:arrow(ax,(x+.075,y+.1),(x+.105,y+.1))
ax.text(.32,.015,'每阶段：4,096 题 · 1,000 更新 · 128,000 次 value 标签呈现；匹配更新和标签呈现，不匹配 FLOPs。',fontsize=10,color='#52677B')
save(fig,'03-curriculum','Each routing mode has its own warmup; all seeds and branches retained, total routing budget 5000 updates including warmup.')
fig,ax=canvas('位置实验：先对齐预训练，再检验规则顺序依赖',5.4)
box(ax,.02,.62,.27,.26,'4 种位置模式\n无位置 / 绝对 / 正弦 / 相对\n每种 4 个 local seeds',fs=12)
box(ax,.365,.62,.27,.26,'16 个局部模型\nL1 · 固定 16,000 步\nFFN + 可学习位置项',color='#E6F2ED',fs=12)
box(ax,.71,.62,.27,.26,'64 个路由运行\ndense / hard × 2 数据重复\n冻结 FFN，继续训练位置项',color='#FFF0DE',fs=11)
arrow(ax,(.29,.75),(.365,.75));arrow(ax,(.635,.75),(.71,.75))
box(ax,.10,.16,.32,.24,'正常规则顺序\n长度 L2 / 4 / 8 / 16 / 32 / 64')
box(ax,.58,.16,.32,.24,'随机打乱规则顺序\n同一逻辑轨迹、同一正确答案')
arrow(ax,(.85,.62),(.26,.40));arrow(ax,(.85,.62),(.74,.40))
ax.text(.04,.035,'位置只加入 attention 分数：绝对 / 正弦为 p(q)·p(k)/√32，相对为 b(q−k)；位置向量不直接加到 value 或 FFN。',fontsize=10,color='#52677B')
save(fig,'04-position-design','Score-only biases; same learnable-position policy in dense and tied routing; fixed sinusoidal table never trained.')
fig,ax=canvas('推理干预：先短链验证选参，再锁定长链测试',4.6)
for x,text,color in [(.02,'固定 28 个 checkpoint\n原始 / 锐化 / 结构分组\n不更新任何模型参数','#EAF0F7'),(.365,'短链验证\nL4 / L8 × 正常 / 打乱\n每组模型选择各方法的 β','#E6F2ED'),(.71,'锁定 β 后测试\n新 L4 / L16 / L64 题目\n256 TF + 32 greedy / 面板','#FFF0DE')]:box(ax,x,.43,.27,.36,text,color,11)
arrow(ax,(.29,.61),(.365,.61));arrow(ax,(.635,.61),(.71,.61))
ax.text(.04,.23,'锐化：对完整注意力取 β 次幂，再归一化。',fontsize=13)
ax.text(.04,.09,'结构分组：所有规则共占 1/2，当前状态占 1/2；组内仍按模型注意力分配，不读取正确规则位置。',fontsize=12)
save(fig,'05-validation','Global validation/test phase barrier; beta selection per model family and method, averaged over seeds, lengths and orders.')
(OUT/'manifest.json').write_text(json.dumps({'figures':manifest},ensure_ascii=False,indent=2)+'\n')
print('Created five explanatory PNG/PDF diagrams.')
