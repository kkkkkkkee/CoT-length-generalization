# Position-embedding experiment plan

This campaign tests whether absolute position information replaces variable matching.
It has two routes: (A) inject position information only during frozen-FFN routing,
and (B) pretrain the local FFN with the same position mechanism before routing.
Each route compares no position, learned absolute position, and fixed sinusoidal
position. Dense and row_tied routing are tested with the same seeds, data, 2000
updates, and L8/L16/L64 evaluation. A second evaluation panel randomly permutes
predicate clauses while preserving the logical trace.

The implementation must record position mode, maximum position, whether FFN was
pretrained with position, predicate permutation seed, frozen-FFN hashes, and
normal/shuffled metrics separately. Learned absolute embeddings are indexed by
clause position; sinusoidal embeddings use the standard log-spaced frequencies.
No result may mix normal and shuffled panels or compare different FFN hashes.
