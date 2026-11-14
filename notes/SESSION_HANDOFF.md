# Session handoff — 2026-11-10

User asked to read previous session and continue. Recovered prior last unresolved question: “文章证明了存在一个泛化解，那么我们在什么样的数据结构和正则约束下能够让模型收敛到这样的解呢”.

Prior campaign A/B is complete: 502/502; report `large-scale-results-2026-11-10.md`. No old experiments were rerun.

Current work:
- Theory memo: `convergence-conditions-2026-11-10.md`. Key correction: paper already proves GD learnability under its specific assumptions; finite Adam+GELU+second-only data are not the exact theorem setting. Row tying is not full permutation invariance.
- Added `--row-tie-lambda`, `--q-l2-lambda`, `--rename-variables`, independent augmentation RNG and exposure/regularization logs to `src/routing_experiment.py`.
- New augmentation module `src/routing_augmentation.py`; tests 36 passed across affected routing/data suites.
- Exploration plan: `convergence-followup-plan-2026-11-10.md`.
- Completed `run_convergence_followup.py` (78/78, no failures, 837 seconds), launcher PID and time in `runs/convergence-followup-2026-11-10/launcher.json`.
- All 6 smoke and 72 main tasks passed. Main panel: first four admitted S5 local FFN seeds x two dataset replicates x 9 treatments, clean L4 second-step labels, Q=0, 1000 Adam steps .03.
- Immutable manifest/provenance snapshot in campaign directory. Do not edit campaign source while it runs: runner checks SHA before each run.
- Progress: `runs/convergence-followup-2026-11-10/campaign-progress.json`; log `campaign.log`.
- Analysis script completed and executed: `src/analyze_convergence_followup.py`. Run after completion to emit report, per-run CSV, summary JSON and PNG/PDF.
- Read-only audit completed (7,256 checks, zero failures) at `notes/convergence-followup-audit-2026-11-10.md`.

Results: `convergence-followup-results-2026-11-10.md`; pair diagnostic: `convergence-pair-coverage-2026-11-10.md`; figures in `figures/convergence-followup-2026-11-10/`.

L16 TF full-chain: dense 16.91%; soft λ=.001/.01/.1/1 -> 33.28/55.19/96.08/98.67%; hard row_tied 99.85%; rename 98.96%; Q L2 .01 24.62%. L64 remains unstable: soft1 1.56%, rename .83%, hard tie 19.53% mostly seed2. Greedy confirms the same qualitative pattern. These are fixed 1000-step exploratory results, not convergence proof.

Mechanism: all 128 query IDs covered, but static nonmatching query-key pairs cover Q43 53.26%, Q44 22.26%; dense and Q L2 unseen entries remain exactly zero, soft/hard ties change them through row coupling. Need controlled coverage/margin interventions for causal attribution.

Post-run code guard: teacher+rename is now rejected before writes because invalid teacher state tokens in variable-ID range need joint remapping. Truth runs unaffected. Added regression; final 37 tests passed. Saved campaign source remains original; patch details in `runs/convergence-followup-2026-11-10/postrun-code-note.json`. Two auxiliary imports model.py/local_rule.py were absent from original source manifest; copied post-run with explicit `source-supplement-postrun.json` provenance, no original hashes overwritten.

Natural next research step: (1) query-pair coverage-controlled data vs rename; (2) equal-longer-budget soft/hard/dense training and margin tracking; (3) complete permutation-invariant routing to test held-out query IDs; (4) theorem-matched supervision and algebra tasks. No new experiments from these proposals have been launched.
