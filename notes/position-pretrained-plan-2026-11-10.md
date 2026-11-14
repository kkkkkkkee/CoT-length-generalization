# 带位置偏置的 FFN 预训练与冻结路由实验

本次实现延续现有 score-only position bias：绝对编码为 p_q·p_k/√32，正弦表固定，相对项为 b(q-k)。位置向量不直接加到 value/FFN 输入；FFN 通过带位置权重的 attention 混合输入学习。不能称为完整标准 Transformer。

16 个局部模型：4 种模式（none / learned_abs / sinusoidal / relative）× seed 0–3，从头初始化。L1，32768 个样本，batch256，lr .001，固定16000步；训练 FFN 与可学习位置项，Q保持0。以最终独立 L1 面板与遍历 action/state 的16个变量对检查达到99.95%作为准入。oracle L64不用于准入，避免长链结果选择偏差。失败seed记下并阻止其下游，不替换seed。

64 个路由模型：每个局部模型 × dense/row_tied × 2 数据重复。继承全部位置状态，冻结 FFN，路由和可学习位置项均可训练；固定正弦不训练。L4第二步真值，4096题，2000步，batch128，lr .03。按seed/重复在各处理间配对数据。评测L2/4/8/16/32/64，同时保存正常和打乱predicate的TF全clause整链、value整链、greedy指标和attention诊断。打乱前后动作/状态/标签不变，检索目标索引同步重映射。

局部阶段预算固定，4个seed × 2个数据重复在seed内先求均值，不把8个run当成8个独立FFN。失败局部模型和下游阻止数必须连同结果汇报。局部预训练仅覆盖L1的两个可见位置，位置OOD仍然存在。

实现修正：v2/v3的dense位置项在attention阶段被冻结，row_tied则可训练，因此此前两者的PE效应比较存在混杂。本轮统一策略并用参数实际更新、FFN冻结与checkpoint重载测试验证。旧产物保留且不改写。

目录 runs/position-pretrained-2026-11-10，固定manifest和源码快照。每任务写入开始状态、失败原因、完成记录；已完成任务可跳过；不覆盖未完成目录。独立进程后台运行。
