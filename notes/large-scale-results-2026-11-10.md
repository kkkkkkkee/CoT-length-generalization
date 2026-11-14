# Large-scale campaign results (2026-11-10)

Campaign status: 502/502 tasks complete; no failures. Metrics below are teacher-forced full-clause chain accuracy, evaluated on 1,024 examples per length.

The completed campaign covers the smoke, local-model/warmup, Phase A, and Phase B tasks. The planned length-curriculum and structural-regularization studies (Phases C/D) have not run yet. For Phase A/B students, all routing treatments start from the admitted local FFN with Q=0; teacher labels, when used, come from the seed-matched L2 warmup model. Thus the results below test learning from the same zero-routing initialization under the stated 1,000-update budget. They are not a direct replication of the earlier experiment that initialized students from an L2-trained routing checkpoint.

## Phase A (192 runs)

| routing | labels/noise | n | L8 | L16 | L64 |
|---|---:|---:|---:|---:|---:|
| dense | teacher / clean | 24 | 0.6235 | 0.0625 | 0.0000 |
| dense | truth / clean | 24 | 0.7729 | 0.1478 | 0.0000 |
| dense | truth / prefix_inconsistent | 24 | 0.0066 | 0.0000 | 0.0000 |
| dense | truth / target | 24 | 0.0058 | 0.0000 | 0.0000 |
| row_tied | teacher / clean | 24 | 0.9649 | 0.6987 | 0.0145 |
| row_tied | truth / clean | 24 | 0.9998 | 0.9980 | 0.1605 |
| row_tied | truth / prefix_inconsistent | 24 | 0.3725 | 0.0157 | 0.0000 |
| row_tied | truth / target | 24 | 0.3778 | 0.0178 | 0.0000 |

## Phase B (288 runs)

| routing | lr | noise | rate | n | L8 | L16 | L64 |
|---|---:|---|---:|---:|---:|---:|---:|
| dense | 0.003 | prefix_consistent | 0.2 | 8 | 0.0214 | 0.0001 | 0.0000 |
| dense | 0.003 | prefix_inconsistent | 0.2 | 8 | 0.0044 | 0.0000 | 0.0000 |
| dense | 0.003 | target | 0 | 8 | 0.0210 | 0.0001 | 0.0000 |
| dense | 0.003 | target | 0.05 | 8 | 0.0160 | 0.0001 | 0.0000 |
| dense | 0.003 | target | 0.2 | 8 | 0.0046 | 0.0000 | 0.0000 |
| dense | 0.003 | target | 0.4 | 8 | 0.0001 | 0.0000 | 0.0000 |
| dense | 0.01 | prefix_consistent | 0.2 | 8 | 0.4685 | 0.0129 | 0.0000 |
| dense | 0.01 | prefix_inconsistent | 0.2 | 8 | 0.0046 | 0.0000 | 0.0000 |
| dense | 0.01 | target | 0 | 8 | 0.4688 | 0.0129 | 0.0000 |
| dense | 0.01 | target | 0.05 | 8 | 0.1938 | 0.0013 | 0.0000 |
| dense | 0.01 | target | 0.2 | 8 | 0.0056 | 0.0000 | 0.0000 |
| dense | 0.01 | target | 0.4 | 8 | 0.0000 | 0.0000 | 0.0000 |
| dense | 0.03 | prefix_consistent | 0.2 | 8 | 0.7717 | 0.1516 | 0.0000 |
| dense | 0.03 | prefix_inconsistent | 0.2 | 8 | 0.0081 | 0.0000 | 0.0000 |
| dense | 0.03 | target | 0 | 8 | 0.7711 | 0.1519 | 0.0000 |
| dense | 0.03 | target | 0.05 | 8 | 0.2670 | 0.0021 | 0.0000 |
| dense | 0.03 | target | 0.2 | 8 | 0.0067 | 0.0000 | 0.0000 |
| dense | 0.03 | target | 0.4 | 8 | 0.0000 | 0.0000 | 0.0000 |
| row_tied | 0.003 | prefix_consistent | 0.2 | 8 | 0.3365 | 0.0060 | 0.0000 |
| row_tied | 0.003 | prefix_inconsistent | 0.2 | 8 | 0.0825 | 0.0004 | 0.0000 |
| row_tied | 0.003 | target | 0 | 8 | 0.3364 | 0.0061 | 0.0000 |
| row_tied | 0.003 | target | 0.05 | 8 | 0.2562 | 0.0042 | 0.0000 |
| row_tied | 0.003 | target | 0.2 | 8 | 0.0835 | 0.0002 | 0.0000 |
| row_tied | 0.003 | target | 0.4 | 8 | 0.0078 | 0.0000 | 0.0000 |
| row_tied | 0.01 | prefix_consistent | 0.2 | 8 | 0.9933 | 0.5643 | 0.0000 |
| row_tied | 0.01 | prefix_inconsistent | 0.2 | 8 | 0.3329 | 0.0079 | 0.0000 |
| row_tied | 0.01 | target | 0 | 8 | 0.9927 | 0.5610 | 0.0000 |
| row_tied | 0.01 | target | 0.05 | 8 | 0.8872 | 0.2170 | 0.0000 |
| row_tied | 0.01 | target | 0.2 | 8 | 0.3438 | 0.0101 | 0.0000 |
| row_tied | 0.01 | target | 0.4 | 8 | 0.0374 | 0.0001 | 0.0000 |
| row_tied | 0.03 | prefix_consistent | 0.2 | 8 | 0.9999 | 0.9982 | 0.1600 |
| row_tied | 0.03 | prefix_inconsistent | 0.2 | 8 | 0.3759 | 0.0171 | 0.0000 |
| row_tied | 0.03 | target | 0 | 8 | 0.9999 | 0.9978 | 0.1539 |
| row_tied | 0.03 | target | 0.05 | 8 | 0.9263 | 0.4382 | 0.0012 |
| row_tied | 0.03 | target | 0.2 | 8 | 0.3884 | 0.0190 | 0.0000 |
| row_tied | 0.03 | target | 0.4 | 8 | 0.0485 | 0.0000 | 0.0000 |

## Interpretation

- `row_tied` is robustly better than `dense` at L8 under all Phase A paired conditions; all 24 matched seed×replicate pairs improve for row_tied.
- The paired mean L8 gains (row_tied minus dense), with a hierarchical bootstrap that resamples the eight local-model seeds and then the three data replicates within each selected seed, are: truth-clean +22.68 percentage points (95% interval +16.54 to +29.07), teacher-clean +34.14 (+27.46 to +41.21), target-20% +37.20 (+26.51 to +48.91), and prefix-inconsistent-20% +36.58 (+27.02 to +47.23). The 24 pairs share only eight model initializations, so they are not 24 independent models.
- Clean labels yield near-perfect row_tied L8/L16, but L64 remains seed-dependent and averages only 0.1605 (Phase A truth-clean).
- Lower learning rates underfit the fixed 1,000-step budget; lr=.03 is strongest in this protocol. This is an optimization result, not evidence that lr=.03 is globally optimal.
- Target noise and prefix-inconsistent noise cause severe degradation. Prefix-consistent noise is almost as easy as clean, showing that preserving local transition consistency matters.
- These results support an explicit variable-matching inductive bias, but do not establish general long-context generalization beyond L32/L64 in this finite model.

The L64 truth-clean row_tied means by local-model seed are 0.00%, 0.20%, 73.70%, 1.76%, 0.00%, 32.78%, 0.00%, and 19.95%. This concentration in a few seeds is why the aggregate L64 mean should not be presented as a reliable generalization result.
