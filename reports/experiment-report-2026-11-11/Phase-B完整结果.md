# Phase B 完整结果

288 个正式运行；8 个 local FFN seeds，每条件 repeat 0。L4、4,096 题、Q=0、第二步 value 监督、1,000 次 Adam 更新、batch 128。

下表为 1,024 题面板上的 TF 完整五槽整链正确率，单位 %。按八个模型 seeds 等权平均；各长度分别计算，不平均不同长度。

| 路由 | 学习率 | 污染类型 | 请求比例 | n | L8 | L16 | L64 |
|---|---:|---|---:|---:|---:|---:|---:|
| dense | 0.003 | prefix_consistent | 20% | 8 | 2.14 | 0.01 | 0.00 |
| dense | 0.003 | prefix_inconsistent | 20% | 8 | 0.44 | 0.00 | 0.00 |
| dense | 0.003 | target | 0% | 8 | 2.10 | 0.01 | 0.00 |
| dense | 0.003 | target | 5% | 8 | 1.60 | 0.01 | 0.00 |
| dense | 0.003 | target | 20% | 8 | 0.46 | 0.00 | 0.00 |
| dense | 0.003 | target | 40% | 8 | 0.01 | 0.00 | 0.00 |
| dense | 0.01 | prefix_consistent | 20% | 8 | 46.85 | 1.29 | 0.00 |
| dense | 0.01 | prefix_inconsistent | 20% | 8 | 0.46 | 0.00 | 0.00 |
| dense | 0.01 | target | 0% | 8 | 46.88 | 1.29 | 0.00 |
| dense | 0.01 | target | 5% | 8 | 19.38 | 0.13 | 0.00 |
| dense | 0.01 | target | 20% | 8 | 0.56 | 0.00 | 0.00 |
| dense | 0.01 | target | 40% | 8 | 0.00 | 0.00 | 0.00 |
| dense | 0.03 | prefix_consistent | 20% | 8 | 77.17 | 15.16 | 0.00 |
| dense | 0.03 | prefix_inconsistent | 20% | 8 | 0.81 | 0.00 | 0.00 |
| dense | 0.03 | target | 0% | 8 | 77.11 | 15.19 | 0.00 |
| dense | 0.03 | target | 5% | 8 | 26.70 | 0.21 | 0.00 |
| dense | 0.03 | target | 20% | 8 | 0.67 | 0.00 | 0.00 |
| dense | 0.03 | target | 40% | 8 | 0.00 | 0.00 | 0.00 |
| row_tied | 0.003 | prefix_consistent | 20% | 8 | 33.65 | 0.60 | 0.00 |
| row_tied | 0.003 | prefix_inconsistent | 20% | 8 | 8.25 | 0.04 | 0.00 |
| row_tied | 0.003 | target | 0% | 8 | 33.64 | 0.61 | 0.00 |
| row_tied | 0.003 | target | 5% | 8 | 25.62 | 0.42 | 0.00 |
| row_tied | 0.003 | target | 20% | 8 | 8.35 | 0.02 | 0.00 |
| row_tied | 0.003 | target | 40% | 8 | 0.78 | 0.00 | 0.00 |
| row_tied | 0.01 | prefix_consistent | 20% | 8 | 99.33 | 56.43 | 0.00 |
| row_tied | 0.01 | prefix_inconsistent | 20% | 8 | 33.29 | 0.79 | 0.00 |
| row_tied | 0.01 | target | 0% | 8 | 99.27 | 56.10 | 0.00 |
| row_tied | 0.01 | target | 5% | 8 | 88.72 | 21.70 | 0.00 |
| row_tied | 0.01 | target | 20% | 8 | 34.38 | 1.01 | 0.00 |
| row_tied | 0.01 | target | 40% | 8 | 3.74 | 0.01 | 0.00 |
| row_tied | 0.03 | prefix_consistent | 20% | 8 | 99.99 | 99.82 | 16.00 |
| row_tied | 0.03 | prefix_inconsistent | 20% | 8 | 37.59 | 1.71 | 0.00 |
| row_tied | 0.03 | target | 0% | 8 | 99.99 | 99.78 | 15.39 |
| row_tied | 0.03 | target | 5% | 8 | 92.63 | 43.82 | 0.12 |
| row_tied | 0.03 | target | 20% | 8 | 38.84 | 1.90 | 0.00 |
| row_tied | 0.03 | target | 40% | 8 | 4.85 | 0.00 | 0.00 |

`target` 且比例为 0 表示干净真值；`prefix_consistent` 对实际污染后的输入保持局部转移一致，不能当作任意错误标签鲁棒性。

来源：[逐运行 CSV](evaluations.csv)、[主报告](实验总报告.md)。此表由 `collect_results.py` 自动生成。
