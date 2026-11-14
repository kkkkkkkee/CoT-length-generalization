"""Causal data separation and semantic noise controls for sparse supervision."""
import unittest

import torch

from src.data import Task
from src.sparse_data import build_supervision, masked_value_loss, select_supervised


class TestSparseData(unittest.TestCase):
    def setUp(self):
        self.task = Task("S5", nvars=32)
        self.sample = self.task.generate(128, 8, torch.Generator().manual_seed(912))

    def build(self, **kwargs):
        return build_supervision(self.sample, self.task, seed=17, **kwargs)

    def test_masks_queries_and_fixed_random_positions(self):
        second = self.build()
        self.assertEqual(second["mask"].sum().item(), 128)
        self.assertTrue(second["mask"][:, 1].all())
        self.assertTrue((second["selected_query_positions"] == 9).all())
        self.assertTrue(torch.equal(second["selected_targets"][:, 0], self.sample["clauses"][:, 10, 4]))
        all_steps = self.build(mode="all")
        self.assertTrue(all_steps["mask"].all())
        a, b = self.build(mode="random"), self.build(mode="random")
        self.assertTrue(torch.equal(a["selected_steps"], b["selected_steps"]))
        self.assertTrue((a["mask"].sum(1) == 1).all())
        self.assertEqual(set(a["selected_steps"].flatten().tolist()), set(range(8)))
        selected = select_supervised(a, torch.tensor([2, 9, 17]))
        self.assertTrue(torch.equal(selected["query_positions"], 8 + selected["step_indices"]))
        self.assertTrue(torch.equal(selected["targets"], a["targets"][[2, 9, 17]].gather(1, selected["step_indices"])))

    def test_target_corruption_cannot_change_prefix_or_source(self):
        before = self.sample["clauses"].clone()
        clean = self.build()
        noisy = self.build(noise_kind="target", noise_rate=1)
        self.assertTrue(torch.equal(noisy["inputs"], clean["inputs"]))
        changed = noisy["targets"].ne(clean["targets"])
        self.assertTrue(torch.equal(changed, clean["mask"]))
        self.assertTrue(torch.equal(self.sample["clauses"], before))
        self.assertEqual(noisy["audit"]["target_error_rate_supervised"], 1)
        self.assertEqual(noisy["audit"]["prefix_state_error_rate_supervised"], 0)
        noisy["inputs"].zero_()
        self.assertTrue(torch.equal(self.sample["clauses"], before))
        self.assertFalse((noisy["targets"] == 0).any())

    def test_prefix_noise_changes_only_y1_not_target_or_variable(self):
        clean = self.build()
        noisy = self.build(noise_kind="prefix_inconsistent", noise_rate=1)
        changed = noisy["inputs"].ne(clean["inputs"])
        expected = torch.zeros_like(changed)
        expected[:, 9, 4] = True
        self.assertTrue(torch.equal(changed, expected))
        self.assertTrue(torch.equal(noisy["targets"], clean["targets"]))
        self.assertEqual(noisy["audit"]["prefix_variable_error_rate_supervised"], 0)
        self.assertEqual(noisy["audit"]["intended_action_local_inconsistency_rate_supervised"], 1)

    def test_consistent_noise_preserves_local_transition_not_true_trajectory(self):
        inconsistent = self.build(noise_kind="prefix_inconsistent", noise_rate=1)
        consistent = self.build(noise_kind="prefix_consistent", noise_rate=1)
        self.assertTrue(torch.equal(inconsistent["inputs"], consistent["inputs"]))
        states = consistent["inputs"][:, 9, 4] - self.task.value_offset
        expected = self.task.action_table[self.sample["actions"][:, 1], states] + self.task.value_offset
        self.assertTrue(torch.equal(consistent["selected_targets"][:, 0], expected))
        self.assertEqual(consistent["audit"]["intended_action_local_inconsistency_rate_supervised"], 0)
        self.assertEqual(consistent["audit"]["target_error_rate_supervised"], 1)
        self.assertEqual(consistent["audit"]["prefix_state_error_rate_supervised"], 1)

    def test_corruption_rates_are_nested_with_same_replacements(self):
        low = self.build(noise_kind="prefix_consistent", noise_rate=0.2)
        high = self.build(noise_kind="prefix_consistent", noise_rate=0.6)
        self.assertFalse((low["corrupted"] & ~high["corrupted"]).any())
        self.assertTrue(torch.equal(low["inputs"][low["corrupted"]], high["inputs"][low["corrupted"]]))
        self.assertTrue(torch.equal(low["targets"][low["corrupted"]], high["targets"][low["corrupted"]]))
        self.assertTrue(torch.equal(low["noise_scores"], high["noise_scores"]))

    def test_teacher_history_and_target_are_verbatim_without_truth_repair(self):
        teacher = self.sample["clauses"].clone()
        teacher[:, 9:, 3:] = 0  # Deliberately invalid generated states and variables.
        before = teacher.clone()
        data = self.build(label_source="teacher", teacher_clauses=teacher)
        self.assertTrue(torch.equal(data["inputs"], teacher[:, :-1]))
        self.assertTrue(torch.equal(data["targets"], teacher[:, 9:, 4]))
        self.assertEqual(data["audit"]["invalid_prefix_state_rate_supervised"], 1)
        self.assertEqual(data["audit"]["prefix_variable_error_rate_supervised"], 1)
        self.assertEqual(data["audit"]["valid_prefix_state_supervised_labels"], 0)
        self.assertIsNone(data["audit"]["local_inconsistency_rate_given_valid_prefix_state"])
        data["targets"].fill_(19)
        self.assertTrue(torch.equal(teacher, before))
        self.assertTrue((data["inputs"][:, 9:, 4] == 0).all())
        teacher[:, 0, 0] = 0
        with self.assertRaises(ValueError):
            self.build(label_source="teacher", teacher_clauses=teacher)

    def test_loss_has_no_gradient_for_unselected_steps_or_other_slots(self):
        data = self.build()
        logits = torch.randn(128, 8, 5, self.task.vocab_size, requires_grad=True)
        loss = masked_value_loss(logits, data["targets"], data["mask"])
        expected = torch.nn.functional.cross_entropy(logits[:, 1, 4], data["targets"][:, 1])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        allowed[:, 1, 4] = True
        self.assertTrue((logits.grad[~allowed] == 0).all())
        self.assertGreater(logits.grad[allowed].abs().sum().item(), 0)

    def test_query_and_action_state_coverage_counts_actual_selected_inputs(self):
        teacher = self.sample["clauses"].clone()
        # Query for second answer is first generated answer at position 9.
        teacher[:3, 9, 3] = torch.tensor([0, self.task.action_offset, 7])
        teacher[:3, 9, 4] = torch.tensor([0, self.task.value_offset + 2, self.task.value_offset + 1])
        data = self.build(label_source="teacher", teacher_clauses=teacher)
        audit = data["audit"]
        expected_variables = [0] * self.task.nvars
        expected_pairs = [[0] * self.task.nstates for _ in range(self.task.nactions)]
        expected_invalid_states = [0] * self.task.nactions
        for i in range(128):
            variable = int(teacher[i, 9, 3])
            action = int(self.sample["actions"][i, 1])
            state = int(teacher[i, 9, 4]) - self.task.value_offset
            if 1 <= variable <= self.task.nvars:
                expected_variables[variable - 1] += 1
            if 0 <= state < self.task.nstates:
                expected_pairs[action][state] += 1
            else:
                expected_invalid_states[action] += 1
        self.assertEqual(audit["selected_query_variable_counts"], expected_variables)
        self.assertEqual(audit["selected_query_invalid_variable_count"], 2)
        self.assertEqual(audit["selected_query_variables_unseen"], expected_variables.count(0))
        self.assertEqual(audit["selected_true_action_input_state_counts"], expected_pairs)
        self.assertEqual(audit["selected_true_action_input_state_pairs_unseen"], sum(row.count(0) for row in expected_pairs))
        self.assertEqual(audit["selected_invalid_input_state_count"], 1)
        self.assertEqual(audit["selected_invalid_input_state_counts_by_true_action"], expected_invalid_states)
        self.assertEqual(sum(map(sum, expected_pairs)), 127)

    def test_prefix_information_counts_are_distinct_from_loss_labels(self):
        second = self.build()["audit"]
        all_steps = self.build(mode="all")["audit"]
        random = self.build(mode="random")
        self.assertEqual(second["supervised_value_labels"], 128)
        self.assertEqual(second["input_information_accounting"]["distinct_intermediate_prefix_answers_total"], 128)
        self.assertEqual(all_steps["supervised_value_labels"], 128 * 8)
        self.assertEqual(all_steps["input_information_accounting"]["distinct_intermediate_prefix_answers_total"], 128 * 7)
        self.assertEqual(all_steps["input_information_accounting"]["intermediate_prefix_answer_occurrences_across_selected_queries"], 128 * sum(range(8)))
        self.assertEqual(random["audit"]["input_information_accounting"]["distinct_intermediate_prefix_answers_total"], int(random["selected_steps"].sum()))

    def test_ambiguous_conditions_are_rejected(self):
        for kwargs in [dict(noise_rate=0.2), dict(noise_kind="target", mode="all"),
                       dict(noise_kind="target", label_source="teacher"), dict(noise_rate=float("nan"))]:
            with self.assertRaises(ValueError):
                self.build(**kwargs)
        short = self.task.generate(2, 1, torch.Generator().manual_seed(0))
        with self.assertRaises(ValueError):
            build_supervision(short, self.task)


if __name__ == "__main__":
    unittest.main()
