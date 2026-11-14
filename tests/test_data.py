"""Tests of semantic validity and distributional support of LEGO examples."""

import unittest

import torch

from src.data import Task, permute_predicates


class TestLEGOData(unittest.TestCase):
    def test_action_families_and_matched_alias_control(self):
        for name, states, actions in [("C6", 6, 6), ("C5", 5, 5), ("S5", 5, 120), ("aliasC5", 5, 120)]:
            task = Task(name, nvars=16)
            self.assertEqual((task.nstates, task.nactions), (states, actions))
            self.assertEqual(task.vocab_size, 1 + 16 + states + actions)
            expected = torch.arange(states).expand(actions, -1)
            self.assertTrue(torch.equal(task.action_table.sort(dim=1).values, expected))
        symmetry, aliases = Task("S5"), Task("aliasC5")
        self.assertEqual(torch.unique(symmetry.action_table, dim=0).shape[0], 120)
        unique_aliases, counts = torch.unique(aliases.action_table, dim=0, return_counts=True)
        self.assertEqual(unique_aliases.shape[0], 5)
        self.assertTrue(torch.all(counts == 24))
        # The matched tasks have the same number of actions mapping any state
        # to any target, but aliases agree on *all* states within each class.
        for task in [symmetry, aliases]:
            for state in range(5):
                self.assertTrue(torch.equal(torch.bincount(task.action_table[:, state], minlength=5), torch.full((5,), 24)))

    def test_semantics_and_token_ranges(self):
        for name in ["C6", "C5", "S5", "aliasC5"]:
            with self.subTest(name=name):
                task = Task(name, nvars=17)
                sample = task.generate(16, 16, torch.Generator().manual_seed(7))
                clauses, variables, actions, states = (sample[k] for k in ["clauses", "variables", "actions", "states"])
                self.assertEqual(tuple(clauses.shape), (16, 33, 5))
                self.assertTrue(all(t.dtype == torch.long for t in sample.values()))
                self.assertGreaterEqual(clauses.min().item(), 0)
                self.assertLess(clauses.max().item(), task.vocab_size)
                self.assertTrue(torch.equal(variables.sort(dim=1).values, torch.arange(1, 18).expand(16, -1)))
                for row in range(16):
                    # Decode clauses as a symbol table and execute their action.
                    known = {int(clauses[row, 16, 3]): int(clauses[row, 16, 4]) - task.value_offset}
                    for step in range(16):
                        target, action_token, source, zero_a, zero_b = map(int, clauses[row, step])
                        self.assertEqual((zero_a, zero_b), (0, 0))
                        updated = int(task.action_table[action_token - task.action_offset, known[source]])
                        known[target] = updated
                        answer = clauses[row, 17 + step]
                        self.assertEqual(answer.tolist(), [0, 0, 0, target, updated + task.value_offset])
                        self.assertEqual(updated, states[row, step + 1].item())
                self.assertTrue(torch.equal(clauses[:, :16, 1] - task.action_offset, actions))

    def test_reproducibility(self):
        task = Task("S5")
        first = task.generate(8, 5, torch.Generator().manual_seed(123))
        second = task.generate(8, 5, torch.Generator().manual_seed(123))
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]))

    def test_permutation_preserves_trace_and_mapping(self):
        original = Task("S5").generate(8, 12, torch.Generator().manual_seed(13))
        original_copy = original["clauses"].clone()
        reordered = original
        for seed in [14, 15]:
            reordered = permute_predicates(reordered, torch.Generator().manual_seed(seed))
            recovered = reordered["clauses"].gather(1, reordered["predicate_positions"][:, :, None].expand(-1, -1, 5))
            self.assertTrue(torch.equal(recovered, original["clauses"][:, :12]))
            self.assertTrue(torch.equal(reordered["clauses"][:, 12:], original["clauses"][:, 12:]))
            for key in ["variables", "actions", "states"]:
                self.assertTrue(torch.equal(reordered[key], original[key]))
        self.assertTrue(torch.equal(original["clauses"], original_copy))
        self.assertFalse(torch.equal(reordered["clauses"], original["clauses"]))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Task("unknown")
        with self.assertRaises(ValueError):
            Task("C5", nvars=1)
        for batch, length in [(0, 1), (1, 0), (1, 5)]:
            with self.assertRaises(ValueError):
                Task("C5", nvars=5).generate(batch, length)


if __name__ == "__main__":
    unittest.main()
