# 无oracle位置的推理干预

问题：已知oracle attention集中/平衡可以恢复模型，是否能只利用输入与模型自己的注意力达到类似结果？

沿用上轮固定28个checkpoint，不挑选成绩：Phase D dense/soft1/row_tied各4seed，新PE预训练4种模式的row_tied各4seed，均replicate0。

方法：原始；完整attention取β次幂再归一化（β=2,4,8）；结构分组（β=1,2,4,8）：用输入五槽格式辨认predicate，将一半权重分给predicate内部按原attention取β次幂的分布，一半给当前answer。这不读取规则正确位置、不比较变量同一性、不访问动作表或标签；规则选择仍依赖模型。分组引入任务结构先验，不称为通用Transformer技巧。若无可见predicate则保持原attention，不进行真值修复。

所有模型先完成独立L4/L8验证，seed11140001，128题，正常/打乱顺序；每模型家族内跨4seed、长度、顺序平均全clause TF整链成绩，分别给sharpen/grouped选一个β（平局取最小）。全部selection锁定之后才开始L64测试。

新测试面板seed12140001，每长度L4/16/64，256题TF，前32题完整自由生成；正常/打乱顺序。主比较original与两类验证选择方法。报告TF全clause整链、greedy整链与最终value，以及实际rule/current-state attention诊断（真值位置只用于评测统计）。不得根据L64换参数。固定模型参数hash、checkpoint和源码hash。不使用测试面板调模型或训练。

尚不对推理成本做等算力比较；实现通过两次前向改attention。有限模型与小规模seed结论，不推断任意LLM。结构分组不等于oracle正确规则集中。第一轮不会同时新增训练，保持机制问题清晰。
