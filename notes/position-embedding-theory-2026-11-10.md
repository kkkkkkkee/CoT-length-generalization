# Position encoding and length generalization

The paper's no-position score depends on variable identities only. With standard absolute position vectors p_t added to query/key representations, the score becomes

S(t,s)=h_t^T A h_s+h_t^T A p_s+p_t^T A h_s+p_t^T A p_s.

The first term is the identity route analyzed in the paper; the other terms are positional shortcuts. If the target key is s*, define the identity margin Delta_id and the worst positional perturbation delta_pos. Then Delta_eff >= Delta_id-2 delta_pos, and target attention is bounded by 1/(1+(N-1) exp(-Delta_eff)). For an exact length-L trajectory with failure probability at most delta, a sufficient heuristic condition is Delta_eff >= log((N-1)L/delta). Thus a bounded, sufficiently small position term can preserve the original mechanism, while an uncontrolled learned absolute term has no such guarantee.

Absolute learned embeddings are trained only on the L4 positions and can improve in-distribution short chains by providing a positional shortcut, but they have no theorem-level behavior at unseen positions and should degrade under predicate permutation. Sinusoidal embeddings are computable at new positions and have bounded structure, so degradation should be milder if their score scale is controlled; they still break arbitrary predicate-order invariance. Relative position terms preserve translation symmetry but can still encode distance shortcuts when predicate order is fixed.

If position information enters the frozen FFN input, the local map changes from f(y,g) to f(y,g,t), so the theorem's repeated position-independent composition no longer applies. This is why the experiment separates score-only position bias from full local pretraining with position information.

Predictions: dense routing should show the largest normal-order short-chain gain and the largest L64/permuted-order collapse; hard row_tied should retain more identity-route generalization but will also degrade if the positional term dominates; sinusoidal should be safer than learned absolute, and random predicate order should expose any positional shortcut.
