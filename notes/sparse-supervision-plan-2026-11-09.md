# 稀疏步骤监督：第一轮实验计划（2026-11-09）

## 研究问题

局部状态更新已经学会时，固定第二步监督、每题随机单步监督、全部步骤监督，如何随有限数据和教师错误影响长度扩展？以一次共同教师的扩展作为主比较，递归课程作为后续实验。错误的全局答案与违反局部转移的标签分开。

## 模型及准入

S5，128个变量、120个动作、5个状态；固定正交one-hot，单头，无残差、LayerNorm、V/O projection。只保留变量槽之间的两个dense Q43/Q44块，初始为零。普通共享512宽GELU MLP完成五槽输出，Adam优化。这是理论启发的有限模型，缺少原定理的特殊FFN、初始化尺度、clipping与渐近条件，不是严格定理复现。

先L1全五槽训练FFN，固定Q=0。训练数据32768条，batch256，lr0.001。最初4000-step工程试跑仅到约76%局部准确率且仍快速上升，保留为local_s0；正式预训练允许最多16000步，固定间隔检查并早停。必须同时通过独立L1面板、600个action/state组合×16组变量对，以及长上下文oracle正确规则/当前状态各0.5的检验（value和variable单步准确率均>=99.95%）。失败不能启动主实验。

attention阶段冻结全部FFN参数，仅训练Q43/Q44；冻结参数前后做hash核验。三个独立初始化seed=0/1/2。同一seed的监督分支共用局部模型；同一扩展阶段共用教师和题目。所有attention组只对第5槽state/value计算loss，完整五槽仍用于输入和自由生成。

## 阶段顺序与固定设置

1. 工程pilot：局部准入后，用L2、4096题、1000 updates、batch128、Adam lr0.03，检查second/all/random三种干净监督。若全部无法形成可用检索，先记录失败并诊断，不直接铺噪声大网格。
2. 第一阶段共同教师使用L2 second checkpoint，全部L4学生分支也从该seed的同一个checkpoint初始化；教师并不按测试集挑选，若其扩展能力不足则如实保留为低质量教师条件。主面板L4，各512/4096题、second/random/all × truth/teacher，3seeds。固定1000 updates、batch128，按选中value平均CE。
3. 等标签控制：L4 all组1024题、batch32，与second/random组4096题、batch128配对，1000 updates；训练集监督位置数和累计value-label呈现数一致，但题数、前缀信息与计算量不同，分别披露。
4. 噪声主面板：L4、4096题、second、真值为起点，噪声0.05/0.20/0.40，target/prefix_inconsistent/prefix_consistent，三个seed。同seed采用同一噪声随机数与替换，rate嵌套，变量槽保持真值。0噪声复用干净second。该面板用于识别错误机制，不当作真实teacher分布。

random每题固定选一个1..L位置，跨epoch不重抽。训练样本由共同16384题pool取前缀，保证不同N嵌套；局部预训练单独使用32768。三seed的局部预训练数据独立，attention阶段训练数据共同固定，属于独立初始化/局部预训练重复，不是三个独立attention数据集重复。记录监督query变量覆盖、action/state覆盖、前缀信息量、教师生成完整轨迹成本、optimizer steps及累计标签呈现。

## 评测与解释

最终独立面板默认每长度1024条TF、128条真实五槽greedy，L=1/2/4/8/16/32/64；训练过程中使用另一个固定面板。L4阶段的主要终点预先指定为L8整条clause成功率（一次翻倍），其他长度为外推诊断。报告逐步value/variable、全clause整链、自由生成最终值和首错分布。整条TF全clause正确事件与确定性greedy整轨迹全对等价；TF value-only指标不能代替它。

注意力记录目标规则质量、当前状态质量、二者总和、绝对差、严格规则top1及并列率；Q=0时并列不算成功检索。所有统计保留逐seed，并报告均值与范围；不将checkpoint当独立重复。初轮3seed及单个有限架构仅支持探索性结论。

若只有错误前缀但目标保持原题正确值，监督要求纠错或绕过前缀；若目标同步变为g2(污染y1)，它保持局部一致但全局轨迹错误。这两组不能混称相同的标签噪声。

## 产物

代码：src/sparse_model.py、sparse_data.py、sparse_experiment.py。实验目录：runs/sparse-supervision-2026-11-09。原始复现数据不改动。每run保存配置/源码hash、训练数据、审计、检查点、独立评测、预算和冻结核验。阶段失败与pilot都保留并独立标注。
