# 独立方法审查

本审查依据初版 `src/data.py`、`src/model.py`、`src/train.py`，以及论文正文和 Appendix A。后续实现变更应以实际配置和源代码哈希为准。

## 已核查

- 数据符合五槽 LEGO 编码；变量无放回、初态与动作均匀，状态按真实动作表递推。
- 训练输入删掉最后 answer，再使用索引 `L:2L` 的输出预测 `L+1:` 的 answer；因果 attention mask 阻止读取未来真实答案，切片没有发现一步错位。
- 模型没有位置 embedding，attention 在 clause 层进行。predicate 排列控制需同步使用 `predicate_positions` 解释热图；原始数据中的目标 predicate 是第 `t` 条，而 query 是第 `L+t` 条。
- 模型文件明确记录工程选择并未归于论文：可训练 64 维 token embeddings、两头、GELU、pre-LN 和残差。训练配置记录参数、数据哈希与源文件哈希，有助于审查实际运行。

## 指标语义

`value_step` 是真实完整前缀下单步 value 准确率。`variable_step` 同理。`value_chain` 是所有步骤 value 在真实前缀下都正确的比例；若变量预测错误，此数值不能代表完整 greedy rollout 的成功率。

`clause_chain` 要求每步五个槽都正确。对确定性 greedy 生成，这与逐步自由生成的整条 clause 轨迹完全正确事件等价，可以批量计算。此等价不适用于最终答案正确率，因为最终答案可以在早先有错误的情况下碰巧正确。

随后审查 `evaluate.py`：实际 greedy rollout 将五槽全部预测反馈，不使用 oracle 变量或 blank；teacher-forced 与 rollout 的指标分开命名；聚合按样本数加权；首次错误的最后一个 bin 表示未出错删失，而不是实际在 L+1 步出错。所有正确 token 概率的乘积是概率诊断，不应称为 greedy accuracy。

训练损失对样本、步骤和五槽取均值，而理论 Eq. 6 使用求和，最小值不变但梯度尺度不同。三个 blank 槽很容易学会，会让平均 CE 看起来下降较快；应另看 value CE/variable CE。长期 value 平台不能由总体 CE 小直接排除。

## 架构对机制解释的影响

默认残差将当前 answer 的状态向量直接传给后续 FFN。因此当前-answer attention 小不意味着当前状态信息消失；论文理论模型直接将 attention 输出送 FFN，没有这条通路。不能在默认常规模型上用 attention 的两目标质量直接验证理论中的全部平衡条件。

标准 Q/K 分解、value/output projections、可训练 embeddings、双头、LayerNorm 和 FFN 也改变了优化动力学。双头可能分别检索不同内容；平均两头 attention 可能掩盖 head specialization。建议按头报告 predicate/answer 质量、目标 top-1 率、score margin，并配合推理时温度与 oracle 检索的因果干预。

对当前这一层 NoPE 架构，answer query 对全部 predicate 位置置换的不变性基本由架构保证：它看到完整 predicate 集合，keys/values 直接来自各原始 clause，尚无前层给 predicate 注入不同历史。因此 permutation check 主要验证实现和无位置编码性质；仅凭置换不变性不能证明模型学会按变量检索。需要目标 predicate 权重或反事实改变变量连接的干预来支持后者。

`evaluate.py` 由 log attention 比值估计 score margin，正常数值范围内成立；极端小 float32 softmax 权重下溢到 0 后会发生截断，返回的有限 margin 不再是未截断原始 score 的精确差值。温度扫描出现极端值时应直接记录原始 logits 复核。

无残差、无 LayerNorm 变体更接近理论计算图，但仍不自动满足固定正交 embeddings、稀疏 Q、特殊 sReLU、clipping 和分阶段冻结等假设。应将“理论启发的消融”与“严格定理设定”区分。

## 数据与预算混淆

300 epochs 的含义取决于训练集大小；论文没有说明这一点。当前固定 8192 条 L5 数据意味着 40960 次 action/state 更新样本，随后多轮重用。S5 有 600 个不同 action/state 组合，C6 有 36 个，平均单组合曝光量相差约 16.7 倍。因此 S5 学习慢首先可能是局部运算数据需求和表征容量问题，而非单纯检索失败。

`aliasC5` 与 S5 都有 120 个 action token、5 个状态，能控制词表规模和单 token 曝光。它是动作别名任务：只有 5 个不同的实际置换。可以视为 C5×C24 的非忠实作用；不能称为 120 元 simply-transitive action。它与 S5 每个状态映射 fiber 的 token 数相同，但跨状态的 fiber 交叠结构不同。

对任意均匀的传递群作用，给定状态和目标值，随机 action 映射至该目标的概率都为 `1 / nstates`。固定状态数时，不能将 S5 与 C5 的差别简单归结为随机输出碰撞概率。不同的是实现同一映射的 action 身份数及其跨状态结构。

建议先比较训练与独立 ID value 精度、局部 action/state 组合覆盖情况，再解释 OOD 差距。如果固定数据上明显过拟合，在线新样本训练是有信息量的控制。

## 局部运算诊断的用途与限制

`src/local_rule.py` 对 L1 的全部 `(action, initial_state)` 组合做穷尽评测，每个组合使用相同的一组随机变量对。该诊断检验模型在只有一个 predicate 与当前 answer 的短上下文中能否完成局部更新，并给出 action×state 错误矩阵。

**L1 比训练长度 L5 更短，也是上下文分布变化。** 此诊断不是从模型内部直接读出的 FFN truth table，不能把 L1 失败单独解释为 FFN 没学会；可能是注意力比例、LayerNorm、残差或其他输入分布变化。反之，L1 全对也不能证明长上下文 attention 混合后的 FFN 决策区域正确。应结合 L5 表现、oracle attention、真实/反事实当前状态干预来判断。

变量对仅采样 4 或 8 组时可作为快速诊断；这不是对所有变量对的证明。矩阵条目共享变量对，不能将所有条目当成独立同分布重复计算置信区间。

## 稳健架构对照的优先次序

1. 首先在完整默认架构上完成任务与别名控制，避免同时改变大量因素。
2. 再对代表性任务比较有/无残差；这能判断当前-state 路径是否改变 attention 分工与温度响应。
3. 若出现局部运算能力不足，先增加训练样本/每动作曝光或适度增加 FFN，而不是将检索指标继续细分后强行归因。
4. 若局部运算已可靠但 OOD 崩塌，检验 attention 温度、目标 top-1 准确率及 oracle 检索。温度提高只在检索排序正确时才可能可靠改善。
5. 未来单独搭建定理中的冻结 FFN + block-sparse Q 机制模型，观测对角 score 的增长、平衡与长度尺度；不要把该模型结果和常规双头模型合并成同一架构曲线。

所有架构对照保持数据量、训练预算与种子尽可能一致，并报告参数量。单次初步运行只支持提出候选现象，稳定的跨 seed 复现后再给机制性结论。

## 固定面板训练动力学

`src/dynamics.py` 在同一组独立测试样本上读取多个 checkpoint，默认 epoch 为 0/5/10/15/20/25/50/100/200/300，长度为 5/40/160，样本数分别 256/256/128，inverse temperature 固定为 1。不同长度使用不同面板，同一长度跨 checkpoint 完全相同；JSON 保存面板哈希、checkpoint 哈希与分析源码哈希。

该面板可诊断 ID 准确率进入平台后，OOD 准确率与目标检索 margin 是否仍继续上升。由于固定面板在训练后用于观测，数据不会进入优化；但若后续根据这些曲线选择超参数，最终结论仍需另一组独立测试样本。跨 checkpoint 的点高度相关，不可当作独立重复计算显著性。margin 与 OOD 同步变化只是观察性证据，因果解释需温度等干预支持。
