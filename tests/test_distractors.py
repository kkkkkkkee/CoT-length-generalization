"""Mechanistic-intervention checks: labels and valid paths stay unchanged."""

import unittest

import torch

from src.data import Task, permute_predicates
from src.distractors import add_distractors


class TestDistractors(unittest.TestCase):
    def test_disconnected_context_and_query_mapping(self):
        task = Task("S5", nvars=16)
        original = task.generate(8, 5, torch.Generator().manual_seed(4))
        original_copy = original["clauses"].clone()
        result = add_distractors(original, task, 160, torch.Generator().manual_seed(5))
        self.assertEqual(tuple(result["clauses"].shape), (8, 171, 5))
        extras = result["clauses"][:, 5:165]
        for row in range(8):
            used = set(original["variables"][row].tolist())
            extra_vars = set(extras[row, :, [0, 2]].flatten().tolist())
            self.assertEqual(len(extra_vars), 2)
            self.assertFalse(used & extra_vars)
        self.assertTrue(torch.equal(result["clauses"][:, :5], original["clauses"][:, :5]))
        self.assertTrue(torch.equal(result["clauses"][:, 165:], original["clauses"][:, 5:]))
        self.assertTrue(torch.equal(result["states"], original["states"]))
        self.assertTrue(torch.equal(original["clauses"], original_copy))
        for row in range(8):
            for step in range(5):
                query = result["clauses"][row, result["query_positions"][row, step]]
                predicate = result["clauses"][row, result["target_predicate_positions"][row, step]]
                self.assertEqual(query[3].item(), predicate[2].item())
                self.assertEqual(query[4].item() - task.value_offset, original["states"][row, step].item())

    def test_first_step_compatibility(self):
        for name in ["C6", "C5", "S5", "aliasC5"]:
            task = Task(name)
            original = task.generate(12, 5, torch.Generator().manual_seed(9))
            for kind, expected in [("compatible", True), ("incompatible", False)]:
                result = add_distractors(original, task, 32, torch.Generator().manual_seed(10), kind=kind)
                actions = result["clauses"][:, 5:37, 1] - task.action_offset
                output = task.action_table[actions, original["states"][:, :1]]
                compatible = output.eq(original["states"][:, 1:2])
                self.assertTrue(bool(compatible.all() if expected else (~compatible).all()))

    def test_permutations_repeated_additions_and_reproducibility(self):
        task = Task("S5")
        original = task.generate(8, 5, torch.Generator().manual_seed(20))
        permuted = permute_predicates(original, torch.Generator().manual_seed(21))
        first = add_distractors(permuted, task, 3, torch.Generator().manual_seed(22))
        repeated = add_distractors(first, task, 7, torch.Generator().manual_seed(23))
        same = add_distractors(permuted, task, 3, torch.Generator().manual_seed(22))
        self.assertTrue(torch.equal(first["clauses"], same["clauses"]))
        self.assertTrue(torch.equal(repeated["target_predicate_positions"], permuted["predicate_positions"]))
        recovered = repeated["clauses"].gather(1, repeated["target_predicate_positions"][:, :, None].expand(-1, -1, 5))
        self.assertTrue(torch.equal(recovered, original["clauses"][:, :5]))
        self.assertTrue(torch.equal(repeated["query_positions"], torch.arange(15, 20).expand(8, -1)))
        self.assertTrue(torch.equal(repeated["distractor_positions"], torch.arange(5, 15).expand(8, -1)))

    def test_zero_and_invalid_inputs(self):
        task = Task("C5", nvars=6)
        sample = task.generate(2, 5, torch.Generator().manual_seed(1))
        zero = add_distractors(sample, task, 0)
        self.assertTrue(torch.equal(zero["clauses"], sample["clauses"]))
        self.assertEqual(tuple(zero["distractor_positions"].shape), (2, 0))
        with self.assertRaises(ValueError):
            add_distractors(sample, task, 1)
        with self.assertRaises(ValueError):
            add_distractors(sample, task, -1)
        with self.assertRaises(ValueError):
            add_distractors(sample, task, 0, kind="unrecognized")


if __name__ == "__main__":
    unittest.main()
