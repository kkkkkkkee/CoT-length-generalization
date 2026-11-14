# 大规模实验计划（2026-11-10）

## 研究问题

1. `row_tied` 在增加模型 seed 和独立数据/噪声抽样后，是否仍稳定优于 `dense`？
2. 错误伪标签的退化由错误类型、噪声率还是优化过程主导？
3. 长度课程、自教师和持续短题训练的差异，是否在更大 seed 面板上仍然存在？
4. 显式变量身份约束的收益能否由较弱的结构约束/正则化取得？

## 固定协议

- S5、128 variables、已通过局部 FFN 准入的模型；attention 阶段冻结 FFN，仅训练 Q。
- 每组 4096 训练题，1000 Adam 更新，batch 128；保存初始投影与最终 checkpoint。
- 训练标签只监督第二步 value；`truth`、`teacher` 和污染条件严格区分。
- 独立评测面板固定为 1024 个 TF 题和 128 个真实 greedy rollout，长度 L2/L4/L8/L16/L32/L64；同时保留 512 题 validation 曲线。
- 不按终点评测筛选 seed；每个 seed 和 data replicate 都完整保留。
- 主指标：L8、L16、L64 全 clause chain accuracy；辅指标为 value/variable/token accuracy、首错位置、attention mass、训练/验证 loss。

## 阶段 A：主复制面板

使用 8 个局部模型 seed（已有 0/1/2，新增 3–7）。每个模型使用 3 个独立训练/噪声 data replicate。每个配对运行：

| routing | label source / noise |
|---|---|
| dense、row_tied | truth-clean、teacher-clean、target-20%、prefix_inconsistent-20% |

共 `8 × 3 × 2 × 4 = 192` 个 attention run。`lr=.03`。主比较采用配对差值（同一 local seed、data replicate、面板）。

## 阶段 B：优化和噪声剂量

固定阶段 A 的 replicate 0，覆盖 8 个 local seed。`dense` 和 `row_tied` 各运行 `lr ∈ {0.003, 0.01, 0.03}`、`target noise ∈ {0, .05, .20, .40}`，truth 标签。共 `8 × 2 × 3 × 4 = 192` run。额外在 0.20 噪声下加入 prefix-inconsistent 和 prefix-consistent 两个局部结构对照（每 routing×lr×seed），共 48 run。

## 阶段 C：长度课程和控制

8 个 local seed，优先 `row_tied`，并保留 `dense` 对照；阶段 L2 warm 后运行 L4→L8→L16→L32，三种 regime：self-teacher、truth curriculum、continue-L2。每阶段固定 4096 题/1000 更新，并将最终模型统一评测至 L64。完整矩阵为 `8 × 2 × 3 × 4 = 192` runs；若阶段 A 已显示 dense 在所有 seed 失败，可将 dense 降为 3 个代表 seed，但不得在主结果中删除失败记录。

## 阶段 D：结构约束（探索）

在 A 的 4 个代表 seed 和 replicate 0 上比较 `dense`、`row_tied`、`diagonal` 及可实现的 row-tied 正则化版本。先做转换前后评测，再做 1000 步训练；每种结构至少覆盖 truth-clean、teacher-clean、target-20%。这部分只作为机制分析，不能与预注册主终点混合。

## 统计和停止规则

- 每阶段先运行 12 组 smoke（2 seeds × 2 routing × 3 条件），确认 checkpoint、hash、评测面板和审计记录完整后再铺开。
- 任何 run 的 `complete.json` 缺失、输入 hash 不一致、FFN 冻结 hash 改变或 validation 非有限值时暂停扩展，保留失败目录并修复后从新 campaign 目录重跑。
- 主结果按 local-model seed 和 data replicate 分层报告，不把 24 个配对当作 24 个独立模型；给出配对均值、标准差、95% bootstrap CI。
- 先完成 A，再根据 A 的 effect size 决定 C/D 是否全量；不得用中途结果挑选最优学习率或 seed。

## 资源预算

attention run 约 10 秒量级；阶段 A/B/C 合计约 576 runs，单 GPU 预计数小时。local FFN 新 seed 3–7 需先完成并通过既定 L1 独立验证准入。每 run 保存训练数据和 checkpoint，预计每 100 runs 数 GB，运行前检查共享盘余量。

## 本次实际启动

本次启动脚本为 `run_large_scale_campaign.py`，输出目录为
`runs/large-scale-2026-11-10`。先运行 12 组 smoke，再为 seed 3–7
训练并验证新的局部 FFN，随后为这些 seed 训练 L2 warmup 教师。主 campaign
固定为阶段 A 的 192 组和阶段 B 的 288 组，共 480 个 attention run；已有
seed 0–2 的合格 checkpoint 直接作为配对基线。阶段 C/D 保留在本计划中，待
A/B 的完整结果和磁盘余量审计后按预先规则启动。

为控制磁盘，主 campaign 每个 run 只在初始和最终保存 checkpoint（`eval-every=1000`），
仍保留独立评测、数据审计、源代码哈希和最终路由统计。任何不完整目录均拒绝覆盖，
campaign progress 使用原子写入，可在进程中断后从已完成 run 继续。
