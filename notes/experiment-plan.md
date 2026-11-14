# 第一轮实验预案

本文件写于完成 C6 seed0 30-epoch 工程试跑后、正式四任务多种子实验之前。

## 原文复现范围

附录A给定：C6/S5，五token组成一个clause，单层decoder，双头，NoPE，Adam 1e-4，300 epochs，L5训练，评估至L160；S5按5→10→20→40自生成伪标签递归训练。

公开文献/源码未提供可运行代码与完整超参。下面为独立补全，故结果只能称作独立实现的定性复现检验，不能称逐数值精确复现：nvars=256，token embedding=64，可训练共享embedding，五槽拼接width320，双头，pre-LN残差，GELU FFN宽1280，无dropout，Adam默认beta，无weight decay，每epoch固定8192条训练数据一遍，batch128，5槽平均交叉熵。所有缺失设置记录于config.json。

正式基线：C6/S5/C5/aliasC5 × seed0/1/2，300 epochs，共12次训练。C5与aliasC5是拓展对照。aliasC5每种cyclic动作有24个不同token，共120个动作token，与S5完全匹配词表和状态数，但只有5种不同函数。

## 评测与主要假设

1. 分开报告teacher-forced value/variable/full-clause逐步准确率、全部clause精确率，以及无约束五槽greedy自由生成最终value与完整value链。TF全clause精确率等于确定性greedy全clause精确率；TF仅value全对无此保证，因variable可能错误。
2. 固定评测种子，长度5/10/20/40/80/160；独立评测集，不用于训练。记录1024条TF与至少128条rollout，给binomial Wilson区间和跨训练种子范围。
3. 保存训练过程，区分局部操作学会、正确规则检索形成和长链泛化三个时间尺度。穷尽L1的(action,state)对诊断局部能力；它不等同于证明任意注意力输入下FFN正确。
4. 推理逆温度alpha=0.5/1/2/4，仅改变attention logits；比较长度泛化变化与注意力目标质量。残差允许直接读取当前state，因此self-answer attention低不代表state丢失。
5. 固定推理深度L5增加离路径predicate，测试上下文体积与推理深度的影响；干扰分布有重复变量，是明确的OOD拓展。
6. S5自训练必须使用固定上轮模型真实自由生成，五个token均反馈；记录伪标签错误，不使用真值筛选。GT仅作事后audit。失败也保留，不把teacher-forced真值喂入当自训练。

## 解释边界

CoT训练格式和中间状态被提供，所以研究的是受监督算法习得动力学，不能宣称自然语言CoT自发涌现。理论仅证明某些充分保证，不给一般模型泛化不可能的上界。若原文C6/S5差异不出现，优先报告缺失超参/架构敏感性，并进行明确标记的架构对照，不能只展示成功版本。候选新现象必须至少多种子重复并排除局部能力、词表曝光、自由生成误差和有限统计样本的解释。
