# Convergence follow-up: independent implementation audit

Audit time: 2026-11-10 12:07 Asia/Shanghai. The campaign was active; this audit did not modify any campaign source. At the final status check, 16/78 tasks were complete and no failures were reported. The artifact checks below covered the first 9 completed tasks and the fourth main task's available configuration, rather than pretending to audit results that did not yet exist.

## Finding

No issue requiring the current clean-truth campaign to stop was found. The implemented panel is an exploratory test of finite-budget generalization and solution selection. It is not a convergence theorem, and a fixed 1,000-update endpoint does not demonstrate that a treatment has reached its limiting solution.

## Pairing, random numbers, and semantic preservation

- The main panel fixes four admitted local-FFN seeds, two independently generated datasets per local seed, and nine treatments (72 runs), preceded by six engineering smoke runs. All four initialization checkpoints were loaded during this audit: each was a completed local-stage model with its admission gate passed and both Q blocks exactly zero.
- Within each seed and data replicate, the source data seed, supervision/noise seed, sample count, batch size, optimizer-step count, and learning rate are identical. The completed first main comparisons have identical source-clause, input, target, and supervision-mask hashes. The batch-order generator uses `seed + 711`; augmentation uses a separate generator with `augmentation_seed`. Turning augmentation on does not consume batch-order RNG state. The two renaming treatments also receive the same augmentation stream within a paired cell.
- Renaming preserves blank/action/state tokens and consistently permutes variable IDs throughout each presented example. Float64 random-key sorting gives a bijection for every example, with approximate uniformity due only to finite-precision key ties. The configuration's word “uniform” should be read with that numerical qualification. Each presentation is newly renamed; the stored base training tensors remain unchanged.
- The original sampled datasets already cover all 128 supervised query variables in the inspected cells. Therefore any benefit from renaming in this panel cannot be described as rescuing wholly unseen query variables. It changes exposure balance and variable-identity pairings while reusing the same abstract action/state traces and value labels. It does not hold the number of distinct tokenized inputs fixed.
- This campaign explicitly uses clean truth labels and no noise. The supervised target, prefix-state, and prefix-variable error audits are all zero in inspected runs. For the supported truth/noise modes, variable renaming leaves value corruption and local transition consistency unchanged because it does not change actions or valid state tokens.
- Generic teacher-plus-renaming use needs an additional boundary check before being studied: teacher traces are generated once on the original IDs, then renamed during student training, rather than regenerated after every renaming. Further, an invalid teacher value token that happens to be a variable ID would be renamed when it occurs in an input prefix while the separately stored value target is not renamed. This edge case is absent from the current truth-label manifest; the present audit does not certify arbitrary malformed teacher traces under augmentation.

## Objective, optimizer, and records

The row penalty is

\[
R_{\rm row}=\frac12\sum_{Q\in\{Q_{43},Q_{44}\}}
\frac{1}{n(n-1)}\sum_i\sum_{j\ne i}(Q_{ij}-\bar Q_{i,\ne i})^2.
\]

The derivative with respect to each diagonal entry is zero, and each off-diagonal row's gradients sum to zero. Its zero set contains arbitrary row-specific backgrounds and diagonals, exactly the intended soft row-tying constraint. It does not shrink matching margins directly. Numerical row-mean evaluation can leave a tiny residual for hard-tied matrices.

The Q-norm control is the mean squared entry over both full Q matrices. It is an explicit additive loss term, not AdamW weight decay. Its normalization is comparable to the row penalty's mean normalization, but equal coefficients do not imply equal gradient norms or equal effects: Q norm penalizes diagonal scores and row backgrounds as well. The single Q-norm coefficient is a limited control, not a complete norm-regularization sweep. Hard tying changes the parameterization and Adam geometry; increasing finite soft-penalty strength need not reproduce hard-tied trajectories.

The dedicated tests passed (10 tests, including CPU/CUDA augmentation checks, central finite differences, gradient geometry, zero sets, and runner integration). In inspected production records, `loss = cross_entropy + row_tie_lambda * row_tie_penalty + q_l2_lambda * q_l2_penalty` within floating-point tolerance. Unweighted penalties and the weighted regularization loss are separate fields. Query-exposure histogram totals equal labels presented; each completed main run inspected presented 128,000 labels over 1,000 updates. Frozen-state and FFN hashes are unchanged.

History loss and penalties are from the last minibatch before the optimizer update; routing statistics and validation metrics are from the updated model at that checkpoint. They should not be interpreted as a dataset-average loss measured on exactly the same parameter state as the validation score.

## Evaluation and interpretation

- Final evaluation seed 6510001 and validation seed 6610001 differ from each other and from all training-data seeds in this manifest. Evaluation uses its own explicit generators. The inspected paired runs share final panel hashes, giving paired comparisons on exactly the same examples.
- Every treatment is evaluated after conversion and before training. This avoids attributing a projection benefit to subsequent updates. In this campaign all branches begin at Q=0, so the hard-tied conversion itself does not supply a nonzero routing solution.
- Validation during training uses L4/L8 and does not select checkpoints or terminate successful runs. Final reporting uses the fixed final checkpoint, with teacher-forced and actual free-generation metrics. The free-generation sample is a subset of the final panel, not an independent test set. Repeated evaluations of the same fixed panels should not be counted as additional independent evidence.
- Four local-model seeds and two within-seed datasets are not eight independent local models. Report paired treatment effects and seed-level variability. The manifest lists L8/L16/L64 and two evaluation modes; these are multiple descriptive endpoints, not a single preregistered confirmatory hypothesis.
- Neither the task algebra (S5 versus aliasC5), query-variable coverage support, nor predicate-order structure is manipulated in this panel. Its “data” intervention is specifically fresh variable-ID exposure. Results cannot settle all the broader data-structure questions by themselves.

## Provenance and restart behavior

- All nine live source hashes and copied source-snapshot hashes matched the manifest at audit time. The inspected run configurations' source and initialization hashes also matched the campaign manifest. The scheduler checks all recorded live source hashes before each task and refuses a changed manifest on restart.
- Runs execute the live workspace source after that hash check. The source snapshot is a provenance copy, rather than the path executed by subprocesses. A change between the preflight check and module import remains a narrow race; there is no evidence of one in inspected configurations. Do not edit these source files until the campaign finishes.
- The per-campaign nonblocking file lock prevents a second scheduler from running simultaneously. A completed task is reused; any nonempty incomplete task directory is rejected. This safely preserves failed work instead of overwriting it.
- This is task-level continuation, not mid-run optimizer resume: checkpoint files do not save Adam state, the batch cursor, or augmentation RNG state. An interrupted task cannot be resumed bit-for-bit through the current launcher. Preserve its directory and create an explicitly named replacement attempt if necessary.
- Reuse of an existing completed run checks its completion marker but does not revalidate all of its source/configuration/checkpoint hashes. Current artifact checks supply that validation for the inspected prefix of the campaign. Before resuming after a failure, or before the final report, audit all reused artifacts against the fixed manifest. Initialization hashes are recorded at manifest creation and in each run, but are not checked by the scheduler before every task; retain the parent checkpoints unchanged.
- Progress elapsed time restarts when the scheduler process restarts, and therefore should not be presented as total campaign wall time across multiple invocations.

These limitations are reporting or future-extension qualifications. None changes the validity of the currently inspected clean-truth paired comparisons.


## Final full-artifact audit

Completed at 2026-11-10T12:20:08+08:00 (Asia/Shanghai). The campaign completed all 78 tasks: six smoke and 72 main runs, with no scheduler failures. This audit loaded and checked every run after completion against the saved source snapshot and immutable manifest; later changes to live runner guards are outside the campaign's recorded source version.

Result: **PASS**, 7,256 checks, 0 failures.

- Recomputed all nine source-snapshot hashes, all four parent checkpoint hashes, and all 156 initial/final checkpoint file hashes. Every run's source and initialization hashes and full parsed CLI configuration matched the manifest.
- Loaded all initial/final checkpoints and recomputed FFN and frozen-parameter hashes from tensor bytes. Every run retained its seed's actual parent FFN; all initial effective Q blocks were zero; final checkpoint steps and routing modes matched the planned treatment.
- Verified fixed budgets (main: 1,000 updates and 128,000 example/value-label presentations; smoke: 25 and 3,200), complete/history exposure histogram totals, history checkpoint schedules, weighted loss decompositions, and clean-truth data audits.
- Recomputed saved source/input/target/mask tensor hashes, then checked exact within-cell data pairing, initial/final panel identity, all paired final and validation panels, evaluation sample counts, and distinct validation/final seeds. Within each paired cell the query-exposure stream agrees among all unaugmented treatments and among both renaming treatments.
- The six smoke runs remain excluded from the 72-run scientific panel. These checks establish artifact consistency and the implemented comparisons; they do not convert the exploratory design or fixed-budget endpoints into a convergence guarantee.
- The snapshot is not fully standalone: `evaluate.py` imports `model.py` and `sparse_experiment.py` imports `local_rule.py`, neither of which was in the campaign source list. The audit imported these two auxiliary modules from the live source directory after giving the saved snapshot first precedence; their relevant functions are not called on the routing training/checkpoint audit path. Their current hashes are recorded in the JSON evidence, but their campaign-time hashes cannot be reconstructed from this manifest. Future snapshots should include transitive local imports as well.

Machine-readable evidence: [convergence-followup-final-audit-2026-11-10.json](convergence-followup-final-audit-2026-11-10.json).

Post-run source supplement: on 2026-11-10T12:20:51+08:00, byte-identical copies of the two missing auxiliary imports were added to the saved source tree without overwriting any existing file. Their origin, copy time, byte counts, and hashes are recorded in `runs/convergence-followup-2026-11-10/source-supplement-postrun.json`. This makes the snapshot importable for future checks but does not establish campaign-time provenance for those two files or alter the original manifest.
