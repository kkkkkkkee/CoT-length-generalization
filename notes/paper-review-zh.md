# 文献精读：CoT 状态追踪、注意力集中与长度泛化

文献：Yu Huang, Zixin Wen, Aarti Singh, Yuejie Chi, Yuxin Chen, *Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization*, NeurIPS 2025。本文依据用户提供的 151 页会议 PDF；以下页码均为 PDF 页码，与文中印刷页码一致。作者在第 1 页另推荐 arXiv:2511.07378 全版。本笔记没有核对全版是否有修订。

## 1. 研究问题及最小模型

论文研究的不是自然语言中无监督“自发产生思维链”，而是**显式监督中间状态后，模型如何学会局部更新与相关上下文检索，以及该机制能否外推到更长的链**。因此最直接的复现对象是 CoT 训练下的算法形成动力学，而非 CoT 格式自身的涌现。

任务是 LEGO 状态追踪（第 4–6 页）：给定互不重复的变量 \(x_0,\ldots,x_L\)，变换 \(g_1,\ldots,g_L\) 和初值 \(y_0\)，按 \(y_\ell=g_\ell(y_{\ell-1})\) 递推。变量从有限词表无放回均匀采样；动作有放回均匀采样；初态均匀采样。

每个 clause 被编码为一个五槽向量：

- predicate：\((x_\ell,g_\ell,x_{\ell-1},\mathrm{blank},\mathrm{blank})\)；
- answer：\((\mathrm{blank},\mathrm{blank},\mathrm{blank},x_\ell,y_\ell)\)。

输入先放全部 \(L\) 条 predicate，再放初态及已知 answer；模型从当前 answer 预测下一条 answer。这里一个 attention 位置对应一个 clause，而不是每个原始字符或每个槽。五槽的拼接已经提供了局部结构，不能将其等同于没有局部信息的纯 token NoPE 模型。第 27 页将其类比为局部聚合层的输出。

核心计算可拆成两个部分（第 9 页图 3）：attention 检索下一条 predicate 与当前 answer；FFN 将动作和当前状态合成为下一状态。这给出了可操作的动力学观测：局部运算何时学会、检索何时形成、两者与长度泛化何时同步或分离。

## 2. 理论模型、保证及适用范围

### 2.1 理论架构与训练约束

第 6–8 页及第 29 页给出的模型相当特殊：

1. 固定正交 token embeddings，blank 为零向量；五槽拼接到 \(5d\) 维。
2. 一层 attention 后直接接 FFN；正文定义没有标准 Transformer 的完整残差、LayerNorm 或可训练 value/output projection。
3. attention 用单一矩阵 \(Q\) 表示双线性分数，\(Q^{(0)}=0\)，并不分析标准 \(W_Q^\top W_K\) 分解参数的实际优化动力学。
4. Assumption C.2 强制仅 \(Q_{4,3}\) 与 \(Q_{4,4}\) 两个块可训练，其余块固定为零；这预先限制了可用的匹配槽。保留的块内部仍是完整稠密矩阵。
5. FFN 使用特定平滑 ReLU、按输出类别独立的一组神经元、固定正偏置，以及 Assumption C.1 的上侧 logit clipping。
6. 词表 \(d\to\infty\)，变量词表随之增长，动作词表至多为 polylog\((d)\)，状态数也有规定的渐近尺度；学习率和迭代数按多项式/对数尺度取值。
7. 算法 1/2（第 28 页）先在 \(L=1\) 固定均匀 attention 训练 FFN，再冻结 FFN 训练 attention；这属于人为解耦课程，不是端到端网络的普遍保证。

这些限制不使结果无意义，但意味着“常规双头 Transformer 的实证复现”和“定理中的机制复现”是两个层级，报告必须区分。

### 2.2 Simply transitive action

Assumption 4.1（第 8 页）要求对每个 \(y,y'\) 恰有一个动作 \(g\) 使 \(g(y)=y'\)。Theorem 4.1 给出：在全部假设及课程训练成立时，由 \(L=1,2\) 训练的模型可推广到 \(L=O(d^{c_*})\)；短链 attention 在目标 predicate 与当前 answer 的联合质量达到 \(1-O(d^{-c_*})\)。

机制是 \(Q_{4,3}\)、\(Q_{4,4}\) 的变量匹配对角元增长，而无关项保持小。第 33–34 页描述 attention 形成的三阶段：先产生 predicate 相对优势，再共同增长并保持受控差距，最后差距收缩、检索高度集中。第 31 页 FFN 还呈现由初始化特征强度排序的隐式学习课程及错误特征抵消。

注意第 8 页称 simply transitive action “isomorphic to the action of the cyclic group” 的表述过强：一般 simply transitive/regular action 不必来自循环群，例如非循环有限群对自身的左乘也是 simply transitive。Assumption 4.1 本身并未要求交换性或循环性，实验只测试 C6。这里应视为值得澄清的表述问题；尚未完整审计全部证明，不能据此宣称定理错误。扩展到非交换群的 regular action 是一个有理论动机的实验方向。

### 2.3 Symmetric group natural action 与自训练

第 9–10 页研究 \(S_n\) 对 \(n\) 个状态的自然置换作用。对固定 \(y,y'\)，有 \((n-1)!\) 个动作把前者映射到后者。FFN 学习的是这组动作的 fiber 共享表征（第 32 页，Eq. 18），与 simply transitive 情况中单独动作—状态配对的表征不同。

Theorem 5.1 保证对当前长度 \(L\) 训练后可继续解 \(2L\)，并用 greedy 模型自标注扩展课程；反复进行后可接近变量词表允许的最大链长。证明利用了足够准确的 teacher：下一轮伪标签与真标签一致，因此不是“在大量错误自标注下仍必然自我纠正”的定理。

**定理给的是常数倍泛化的充分保证，不是更远外推不可能的上界。** 论文经验曲线支持 S5 的直接外推较差，不能将这种结果升级为一切 S5 模型均受常数倍硬上限约束。

当 \(n\ge5\) 时与非可解群字问题及 NC1 完备性相连。超过 TC0 的表述依赖通用的 TC0 ≠ NC1 猜想，并且理论任务使用线性长度 CoT、随词表变化的模型维度等条件。不能从有限小规模实验直接实证证明复杂度类分离。

## 3. 原文实验到底做了什么

第 25–27 页 Appendix A 是实验说明的主要来源。

| 项目 | 原文可确认的内容 |
|---|---|
| 任务 | C6 与 S5，五槽 clause 编码，LEGO 分布 |
| 模型 | 一层 decoder-only Transformer，两个 attention heads，FFN；正文及讨论强调 NoPE |
| 优化 | Adam，学习率 \(10^{-4}\)，300 epochs，称训练至损失接近零 |
| 基础训练长度 | \(L=5\) |
| 外推长度 | 图 1 展示至 \(L=160\) |
| 自训练 | \(5\to10\to20\to40\)，后一阶段用前一阶段 greedy 自回归生成的标签 |
| 热图 | 图 2/4 对双头及 100 条独立样本平均 |
| 位置控制 | 图 4 随机打乱 predicate clause 的位置，检验依赖变量匹配而非绝对位置 |
| 外推检索诊断 | 图 5/6 对 \(L=10,20\) 分析 predicate 和 answer 的目标注意力 |

原文图 1a：C6 从 5 外推到 160 时接近完美，S5 明显下降。图 1b：S5 多轮自训练逐步改善外推。图 2：已训练模型注意力集中到目标 predicate 与当前 answer。图 5/6 给出一个需要保留的细节：C6 的当前-answer attention 可以随长度/位置显著下降而性能仍稳定；S5 的目标-predicate attention 衰减与性能下降相伴。因此只记录二者 attention 和，或假设每个头都必须同时平衡检索二者，可能遗漏双头实际机制。

### 未明确给出、需要自行声明的配置

PDF 中未找到以下数值或实现细节：embedding/hidden dimension、FFN width、变量词表大小、batch size、每个 epoch 的数据量、固定数据还是在线生成、训练/验证/测试样本数、精确种子和重复次数、Adam betas/epsilon、weight decay、gradient clipping、初始化、激活函数、LayerNorm 和残差配置、value/output projection 配置，以及是否应用理论中的 block sparsity 与 logit clipping。正文理论单头与附录双头的区别也不能自行抹平。未找到官方代码仓库链接。

因此独立实现可以称为**依照论文公开设置的定性复现/机制复现**，不宜声称逐超参数、逐数值精确复现。300 epochs 不能在数据量未知时转为确定的 update 数；报告实际 steps、batch size 和样本量更可审查。

## 4. 评测口径与常见误判

正文 Eq. 5（第 8 页）是给定真实 answer prefix 的平均下一 clause 准确率；Appendix A 明确表示实验选择更严格的整条 CoT 生成，并报告所有 answer value 都正确的概率。附录同时使用了 final-answer accuracy 的措辞，但随即说明为计算效率报告 all-answer-steps-correct，因此不要把这两个指标混为一谈。

建议每次报告：

1. teacher-forced 下一步 value 准确率与交叉熵；
2. teacher-forced 下一步 variable 准确率；
3. 每条序列所有步骤、所有必要 token 全部正确的 exact trajectory rate；
4. 真正 free rollout 的最终 value 准确率与首次错误位置；
5. 若只预测 value、变量路径始终提供真值，必须标明这是 value-only 跟踪，而不是完整下一 clause 自回归。

一个有用的计算等价：对确定性 greedy 生成，“真实前缀下每个步骤均预测正确”的事件，等价于“greedy rollout 整条轨迹完全正确”的事件。因此可以批量 teacher-forced 前向并取样本内 all() 来得到 exact trajectory rate。它**不等价于最终答案正确率**：较早的错误可能在后续错误下偶然恢复；若只检查 value 而放任 variable 预测错误，也不能推出完整 clause rollout 的等价。

长链 exact trajectory 随长度下降可以单纯由小的逐步错误积累产生，未必代表新的检索相变。应同时报告每步误差、整链准确率和实测误差相关性；不要默认各步独立而直接以 \(p^L\) 代替实测。

## 5. 面向 physics of AI 的最小复现与拓展

### 5.1 首先应复现的现象

- L5 训练的 C6/S5 在长度网格上的 teacher-forced 与 exact-trajectory 差异。
- 完整训练期间 FFN 局部运算、变量拷贝、目标 predicate 检索、当前 answer 检索和长度泛化的先后出现。
- 打乱 predicate 顺序后是否保持性能。
- S5 自训练课程是否改善；同时直接监督长数据和等算力继续短数据训练作为对照。

若第一轮不能复现 C6 优于 S5，应先检查隐藏超参数、容量/样本量差异、训练充分性和评价口径，再将差异解释为新现象。复现失败本身有价值，但不是对定理的反例，除非满足定理全部假设。

### 5.2 建议的观测量与动力学假设

记录每个 checkpoint 的：train/ID/OOD loss，目标 predicate/answer attention（按头分开），无关质量、attention entropy、目标与最大干扰 score margin、局部 action accuracy、变量复制 accuracy、free-rollout 首错位置，以及每个种子达到阈值的训练步数。

softmax 竞争给出可检验的长度尺度：若单个目标分数比背景高 \(\Delta\)，背景数量为 \(N\)，其质量近似 \(1/(1+N e^{-\Delta})\)。两个目标与两头要分别处理；这不是普遍精确公式。可以检查不同训练时间、任务和温度的曲线能否按 \(\log L-\Delta\) collapse，并报告拟合残差。若准确率崩塌早于目标质量崩塌，可能涉及 FFN 输入分布偏移；若 attention 很稀释仍正确，则可能是冗余的 fiber 聚合或残差通路。

### 5.3 优先的因果扩展

**A. 推理时调节 attention 温度。** 固定已训练模型，扫描 logits 乘数 \(\beta\)，同时测 ID/OOD 准确率、目标质量和 top-1 检索准确率。若仅提高 \(\beta\) 即延长可解长度，说明 softmax 稀释存在因果作用；若损害准确率，可能因“选错对象更自信”或两种必要信息失衡。应对每个头分别干预，避免把双头分工平均掉。温度选择使用独立验证集；最终测试报告完整扫描，避免只报最优测试点。

**B. oracle 检索干预。** 将 attention 只保留目标 predicate 与当前 answer，并比较固定 1/2–1/2、保留原相对比例等方案。结合 oracle 当前-state 输入，可区分检索瓶颈、组合计算瓶颈和误差传播。只改变 attention 的干预若与标准训练分布过远，应说明其局限。

**C. 固定状态数、控制 action redundancy。** 比较 C5、D5、AGL(1,5)、S5，自然作用的 stabilizer 大小分别为 1、2、4、24。原文 C6 对 S5 同时改变状态数与动作数，不是完全控制比较。更基础的对照是 C5 对 S5，再加入多个 action token 表示相同 cyclic 变换的“动作别名”控制，分离词表规模、每动作曝光量和 fiber 共享学习。

此处有一个关键数学细节：对任何均匀采样的传递作用，\(P[g(y)=y']=1/|Y|\)。所以固定状态数时，随机动作与目标产生相同 output 的**概率并不会随 stabilizer 增大**；变化的是对应该映射的不同 action 身份数和学习到的表征。不能把实验结果简单解释为“对称群随机标签碰撞概率更大”。非传递或非均匀动作分布才会进一步改变这个概率。

**D. 解耦有效计算长度与无关上下文长度。** 固定需要执行的链长，在输入加入不连接到当前链的 predicate；或固定总上下文但改变有效链长度。分别改变重复同动作干扰、同输出不同动作干扰、错误输出干扰，测试稀释与表征共享的独立贡献。每条干扰变量必须避免意外构成多条合法后继，除非明确研究歧义。

**E. CoT 的因果作用。** 对同一任务做训练和测试的 CoT 截断、当前-state 遮蔽、过去旧-state 遮蔽等。当前-state 的替换最好加入分布内反事实（同步重新计算下游真值），以避免将任意 OOD token 破坏误称为 CoT 必要性证明。最终答案监督的无-CoT baseline 需要匹配数据与训练预算，且其失败不证明所有无-CoT 架构都不能解任务。

**F. 自训练临界点与噪声。** 测每轮 teacher 生成的完整轨迹正确率；扫描长度放大倍数及 controlled label noise，寻找 teacher 质量下降后的停滞或崩塌。加入只延长训练、ground-truth curriculum、固定 teacher、按置信度筛选等对照。论文的理论 bootstrap 并不覆盖任意噪声，所以这里是拓展而非已被证明的性质。

## 6. 发现新现象的证据标准

单个种子和超参数扫描只适合提出候选现象。至少需多个独立训练种子、独立测试集、明确报告训练预算和变动项，并将每序列指标的采样不确定性与跨 seed 不确定性分开。若现象只出现于某种常规架构而非理论机制模型，这也是有意义的边界结果，但应明确归因层级。

适合优先跟进的候选包括：ID accuracy 已饱和后检索 margin 继续增长而 OOD 突然改善；attention entropy 变化先于或晚于算法准确率的阶段；输出全对但检索不集中；温度干预导致外推改善与短链下降的非单调曲线；同样 action collision probability 下不同 fiber 大小仍产生不同动力学；自训练随 teacher 误差出现阈值。任何这些都需对照和多 seed 后才能称为稳定发现，不能在运行前预设结论。
