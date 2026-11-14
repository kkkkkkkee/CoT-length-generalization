"""Check that the soft row-tying objective constrains only row residuals."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import Task
from routing_experiment import parse_args, row_tie_penalty, run_experiment
from routing_model import convert_model
from sparse_model import SparseClauseTransformer, SparseModelConfig


class Matrices(nn.Module):
    def __init__(self, q43, q44):
        super().__init__()
        self.config = SimpleNamespace(nvars=q43.shape[0])
        self.Q43 = nn.Parameter(q43.clone())
        self.Q44 = nn.Parameter(q44.clone())


def test_penalty_is_mean_row_variance_with_correct_gradient_geometry():
    generator = torch.Generator().manual_seed(64)
    model = Matrices(*(torch.randn(5, 5, generator=generator, dtype=torch.float64)
                       for _ in range(2)))
    penalty = row_tie_penalty(model)
    expected = torch.stack([
        torch.stack([torch.cat((q[i, :i], q[i, i + 1:])).var(unbiased=False)
                     for i in range(5)]).mean()
        for q in (model.Q43, model.Q44)
    ]).mean()
    torch.testing.assert_close(penalty, expected, atol=1e-14, rtol=1e-14)
    penalty.backward()
    for q in (model.Q43, model.Q44):
        assert torch.equal(q.grad.diagonal(), torch.zeros(5, dtype=torch.float64))
        torch.testing.assert_close(q.grad.sum(-1), torch.zeros(5, dtype=torch.float64),
                                   atol=1e-15, rtol=0)


def test_penalty_gradient_agrees_with_central_finite_differences():
    generator = torch.Generator().manual_seed(7)
    model = Matrices(*(torch.randn(4, 4, generator=generator, dtype=torch.float64)
                       for _ in range(2)))
    row_tie_penalty(model).backward()
    epsilon = 1e-6
    for q in (model.Q43, model.Q44):
        for i, j in ((0, 0), (0, 1), (2, 3), (3, 1)):
            with torch.no_grad():
                original = q[i, j].item()
                q[i, j] = original + epsilon
                plus = row_tie_penalty(model).item()
                q[i, j] = original - epsilon
                minus = row_tie_penalty(model).item()
                q[i, j] = original
            assert (plus - minus) / (2 * epsilon) == pytest.approx(q.grad[i, j].item(), abs=2e-10)


def test_penalty_zero_set_and_invariance_to_diagonal_or_row_offsets():
    diagonal = torch.arange(4, dtype=torch.float64)
    background = torch.tensor([-2., -.5, 1., 3.], dtype=torch.float64)
    q = background[:, None].expand(4, 4).clone()
    q.diagonal().copy_(diagonal)
    model = Matrices(q, 2 * q)
    assert row_tie_penalty(model).item() == 0
    with torch.no_grad():
        model.Q43[0, 1] += 1
    positive = row_tie_penalty(model).item()
    assert positive > 0
    with torch.no_grad():
        model.Q43.diagonal().add_(123)
        model.Q44.diagonal().sub_(47)
        model.Q43.add_(torch.arange(4, dtype=torch.float64)[:, None])
    assert row_tie_penalty(model).item() == pytest.approx(positive, abs=1e-15)
    task = Task('C5', nvars=4)
    dense = SparseClauseTransformer(SparseModelConfig(task.vocab_size, task.nvars, hidden=8))
    with torch.no_grad():
        dense.Q43.copy_(q)
        dense.Q44.copy_(2 * q)
    assert row_tie_penalty(convert_model(dense, 'row_tied')).item() == 0


def test_runner_records_unweighted_penalties_and_weighted_total_loss(tmp_path, capsys):
    task = Task('C5', nvars=8)
    torch.manual_seed(921)
    model = SparseClauseTransformer(SparseModelConfig(task.vocab_size, task.nvars, hidden=12))
    with torch.no_grad():
        model.Q43.normal_(0, .2)
        model.Q44.normal_(0, .2)
    model.configure_stage('attention')
    directory = tmp_path / 'foundation'
    directory.mkdir()
    checkpoint = directory / 'checkpoint-final.pt'
    torch.save({'model': model.state_dict(), 'model_config': model.config_dict(),
                'stage': 'attention', 'args': {'task': 'C5', 'nvars': task.nvars}, 'step': 1},
               checkpoint)
    (directory / 'complete.json').write_text(json.dumps({'complete': True}))
    args = parse_args(['--output', str(tmp_path / 'run'), '--init', str(checkpoint),
        '--task', 'C5', '--nvars', '8', '--length', '2', '--samples', '7',
        '--pool-samples', '7', '--batch', '3', '--steps', '1', '--eval-every', '1',
        '--eval-samples', '3', '--validation-samples', '3', '--rollout-samples', '0',
        '--eval-lengths', '2', '4', '--eval-batch', '3', '--device', 'cpu',
        '--row-tie-lambda', '.7', '--q-l2-lambda', '.2'])
    expected_penalty = row_tie_penalty(model).item()
    expected_q_l2 = ((model.Q43.square().mean() + model.Q44.square().mean()) / 2).item()
    run_experiment(args)
    record = json.loads((args.output / 'history.jsonl').read_text().splitlines()[0])
    assert record['row_tie_lambda'] == .7
    assert record['row_tie_penalty'] == pytest.approx(expected_penalty, rel=1e-6)
    assert record['q_l2_lambda'] == .2
    assert record['q_l2_penalty'] == pytest.approx(expected_q_l2, rel=1e-6)
    assert record['regularization_loss'] == pytest.approx(.7 * expected_penalty + .2 * expected_q_l2, rel=1e-6)
    assert record['loss'] == pytest.approx(record['cross_entropy'] + record['regularization_loss'], rel=1e-6)
    config = json.loads((args.output / 'config.json').read_text())
    assert config['row_tie_lambda'] == .7


def test_teacher_renaming_rejected_before_loading_unmapped_targets(tmp_path):
    args = parse_args(['--output', str(tmp_path / 'run'), '--init', str(tmp_path / 'init.pt'),
                      '--teacher', str(tmp_path / 'teacher.pt'), '--label-source', 'teacher',
                      '--rename-variables', '--device', 'cpu'])
    with pytest.raises(ValueError, match='requires truth labels'):
        run_experiment(args)
    assert not args.output.exists()
