"""Verify corruption semantics and weighted diagnostic summaries on CPU."""
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import Task
from sparse_data import build_supervision
from sparse_model import SparseClauseTransformer, SparseModelConfig
from routing_error_dynamics import evaluate_subsets, make_validation, run_diagnostics


class LocalOracle(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.task = task

    def forward_queries(self, inputs, queries, return_attention=False):
        b, t, _ = inputs.shape
        j = queries.shape[1]
        logits = torch.full((b, j, 5, self.task.vocab_size), -10.)
        attention = torch.zeros(b, 1, j, t)
        for row in range(b):
            for column in range(j):
                position = int(queries[row, column])
                current = inputs[row, position]
                state = int(current[4]) - self.task.value_offset
                candidates = inputs[row, :position + 1]
                rule = candidates[candidates[:, 2].eq(current[3])][0]
                action = int(rule[1]) - self.task.action_offset
                answer = [0, 0, 0, int(rule[0]), self.task.value_offset + int(self.task.action_table[action, state])]
                for slot, token in enumerate(answer):
                    logits[row, column, slot, token] = 10.
                attention[row, 0, column, :position + 1] = 1 / (position + 1)
        return logits, attention


def group(result, name):
    return next(row for row in result['groups'] if row['subset'] == name)


def test_target_noise_and_locally_consistent_prefix_are_distinguished():
    task = Task('S5', 16)
    sample = task.generate(11, 4, torch.Generator().manual_seed(100))
    target = build_supervision(sample, task, noise_kind='target', noise_rate=1., seed=7)
    consistent = build_supervision(sample, task, noise_kind='prefix_consistent', noise_rate=1., seed=7)
    bad = evaluate_subsets(LocalOracle(task), task, sample, target, batch_size=4)
    good = evaluate_subsets(LocalOracle(task), task, sample, consistent, batch_size=4)
    assert group(bad, 'all')['metrics']['target_label_accuracy'] == 0.
    assert group(bad, 'all')['metrics']['original_truth_value_accuracy'] == 1.
    assert group(bad, 'corrupted_locally_inconsistent')['selected_labels'] == 11
    assert group(good, 'all')['metrics']['target_label_accuracy'] == 1.
    assert group(good, 'all')['metrics']['original_truth_value_accuracy'] == 0.
    assert group(good, 'corrupted_locally_consistent')['selected_labels'] == 11
    assert group(good, 'all')['metrics']['target_rule_mass'] == pytest.approx(1 / 6)
    assert group(good, 'all')['metrics']['current_state_mass'] == pytest.approx(1 / 6)
    assert group(good, 'all')['metrics']['rule_strict_top1_among_predicates'] == 0.
    assert group(good, 'all')['metrics']['rule_top_tie_rate'] == 1.
    assert group(good, 'uncorrupted_examples')['selected_labels'] == 0
    assert all(value is None for value in group(good, 'uncorrupted_examples')['metrics'].values())


def test_batch_weighting_and_invalid_local_state_handling():
    task = Task('C5', 16)
    sample = task.generate(11, 4, torch.Generator().manual_seed(101))
    supervision = build_supervision(sample, task, noise_kind='target', noise_rate=.5, seed=9)
    torch.manual_seed(8)
    model = SparseClauseTransformer(SparseModelConfig(task.vocab_size, task.nvars, hidden=12))
    full = evaluate_subsets(model, task, sample, supervision, batch_size=11)
    uneven = evaluate_subsets(model, task, sample, supervision, batch_size=4)
    for first, second in zip(full['groups'], uneven['groups']):
        assert first['selected_labels'] == second['selected_labels']
        for key, value in first['metrics'].items():
            if value is None:
                assert second['metrics'][key] is None
            else:
                assert second['metrics'][key] == pytest.approx(value, abs=1e-6)
    supervision['inputs'][:, 5, 4] = 0  # y1 is the selected prefix; invalid state token.
    invalid = evaluate_subsets(model, task, sample, supervision, batch_size=4)
    assert group(invalid, 'all')['invalid_prefix_state_labels'] == 11
    assert group(invalid, 'all')['valid_local_transition_labels'] == 0
    assert group(invalid, 'all')['metrics']['local_transition_accuracy'] is None
    assert invalid['model_state_sha256_before'] == invalid['model_state_sha256_after']


def test_validation_data_and_noise_draws_are_independent():
    task = Task('C5', 16)
    config = {'length': 4, 'mode': 'second', 'label_source': 'truth',
              'noise_kind': 'prefix_consistent', 'noise_rate': .5}
    a = make_validation(task, config, 64, 101, 202)
    b = make_validation(task, config, 64, 101, 203)
    c = make_validation(task, config, 64, 102, 202)
    assert torch.equal(a['sample']['clauses'], b['sample']['clauses'])
    assert not torch.equal(a['supervision']['corrupted'], b['supervision']['corrupted'])
    assert torch.equal(a['supervision']['corrupted'], c['supervision']['corrupted'])
    assert not torch.equal(a['sample']['clauses'], c['sample']['clauses'])
    assert a['supervision']['audit']['intended_action_local_inconsistency_rate_supervised'] == 0.


def test_end_to_end_outputs_preserve_hashes_and_checkpoint_steps(tmp_path, capsys):
    run = tmp_path / 'run'
    run.mkdir()
    task = Task('C5', 16)
    config = {'task': task.name, 'nvars': task.nvars, 'length': 4, 'samples': 11,
              'mode': 'second', 'label_source': 'truth', 'noise_kind': 'target',
              'noise_rate': .5, 'data_seed': 91, 'eval_seed': 92, 'validation_seed': 93}
    (run / 'config.json').write_text(json.dumps(config))
    sample = task.generate(11, 4, torch.Generator().manual_seed(config['data_seed']))
    supervision = build_supervision(sample, task, noise_kind='target', noise_rate=.5, seed=94)
    torch.save({'sample': sample, 'supervision': supervision}, run / 'training-data.pt')
    (run / 'data-audit.json').write_text(json.dumps(supervision['audit']))
    model = SparseClauseTransformer(SparseModelConfig(task.vocab_size, task.nvars, hidden=12))
    for step in [0, 250]:
        name = 'checkpoint-initial.pt' if step == 0 else f'checkpoint-{step:05d}.pt'
        torch.save({'model': model.state_dict(), 'model_config': model.config_dict(),
                    'stage': 'attention', 'args': config, 'step': step}, run / name)
    output = tmp_path / 'diagnostics'
    result = run_diagnostics(run, output, validation_seed=95, validation_noise_seed=96,
                             validation_samples=13, steps=[0, 250], batch=4)
    assert result['complete'] is True
    assert [row['step'] for row in result['checkpoints']] == [0, 250]
    assert result['checkpoints'][0]['splits'] == result['checkpoints'][1]['splits']
    assert (output / 'error-dynamics.csv').exists()
    assert (output / 'validation-data.pt').exists()
    assert result['training_audit']['source_clause_sha256'] != result['validation_audit']['source_clause_sha256']
    with pytest.raises(ValueError, match='Missing requested checkpoints'):
        run_diagnostics(run, tmp_path / 'missing', validation_seed=95, steps=[0, 500])
