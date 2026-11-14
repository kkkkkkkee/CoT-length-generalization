"""Small CPU integrations checking provenance, frozen FFN and projection baselines."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import Task
from evaluate import greedy_rollout
from routing_experiment import parse_args, parameter_hash, q_statistics, run_experiment, validate_checkpoint
from routing_model import convert_model, load_routing_model
from sparse_model import SparseClauseTransformer, SparseModelConfig


def foundation(tmp_path, task_name='C5', stage='attention', gate=True):
    task = Task(task_name, nvars=8)
    torch.manual_seed(817)
    model = SparseClauseTransformer(SparseModelConfig(task.vocab_size, task.nvars, hidden=12))
    with torch.no_grad():
        model.Q43.normal_(0, .2)
        model.Q44.normal_(0, .2)
    model.configure_stage(stage)
    directory = tmp_path / 'foundation'
    directory.mkdir()
    state = {'model': model.state_dict(), 'model_config': model.config_dict(),
             'stage': stage, 'args': {'task': task_name, 'nvars': task.nvars}, 'step': 1}
    path = directory / 'checkpoint-final.pt'
    torch.save(state, path)
    (directory / 'complete.json').write_text(json.dumps({'complete': True, 'local_gate_passed': gate}))
    return task, model, state, path


def arguments(tmp_path, checkpoint, *extra):
    return parse_args(['--output', str(tmp_path / 'experiment'), '--init', str(checkpoint),
        '--task', 'C5', '--nvars', '8', '--length', '2', '--samples', '7',
        '--pool-samples', '13', '--batch', '3', '--steps', '3', '--eval-every', '1',
        '--eval-samples', '5', '--validation-samples', '5', '--rollout-samples', '3',
        '--eval-lengths', '2', '4', '--eval-batch', '3', '--device', 'cpu', *extra])


def test_failed_local_gate_and_equal_vocab_wrong_task_rejected(tmp_path):
    task, model, state, path = foundation(tmp_path, task_name='S5', stage='local', gate=False)
    with pytest.raises(ValueError, match='admission gate'):
        validate_checkpoint(model, state, task, path, admission=True)
    alias = Task('aliasC5', nvars=8)
    assert alias.vocab_size == task.vocab_size
    with pytest.raises(ValueError, match='task or vocabulary'):
        validate_checkpoint(model, state, alias, path)


def test_zero_training_records_conversion_gain_separately(tmp_path, capsys):
    task, source, _, checkpoint = foundation(tmp_path)
    args = arguments(tmp_path, checkpoint, '--routing-mode', 'row_tied', '--steps', '0')
    result = run_experiment(args)
    expected = convert_model(source, 'row_tied')
    initial, initial_state = load_routing_model(args.output / 'checkpoint-initial.pt', 'cpu')
    final, final_state = load_routing_model(args.output / 'checkpoint-final.pt', 'cpu')
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(initial.state_dict()[name], value, atol=0, rtol=0)
        torch.testing.assert_close(final.state_dict()[name], value, atol=0, rtol=0)
    before = json.loads((args.output / 'initial-evaluation.json').read_text())
    after = json.loads((args.output / 'evaluation.json').read_text())
    assert before['phase'] == 'after_conversion_before_training'
    assert before['evaluations'] == after['evaluations']
    assert initial_state['routing_mode'] == final_state['routing_mode'] == 'row_tied'
    assert result['steps'] == result['labels_presented'] == 0
    assert result['ff_sha256_after'] == parameter_hash(source.ff.named_parameters())


def test_training_freezes_ffn_and_counts_incomplete_batches(tmp_path, capsys):
    _, source, _, checkpoint = foundation(tmp_path)
    args = arguments(tmp_path, checkpoint, '--routing-mode', 'row_tied', '--mode', 'all',
                     '--data-seed', '2810001', '--noise-seed', '2820001', '--replicate', '2')
    result = run_experiment(args)
    initial, _ = load_routing_model(args.output / 'checkpoint-initial.pt', 'cpu')
    final, state = load_routing_model(args.output / 'checkpoint-final.pt', 'cpu')
    assert result['examples_presented'] == 7
    assert result['labels_presented'] == result['unique_labels'] == 14
    assert result['ff_sha256_before'] == result['ff_sha256_after']
    for name, value in source.ff.state_dict().items():
        torch.testing.assert_close(final.ff.state_dict()[name], value, atol=0, rtol=0)
    assert not torch.equal(initial.Q43, final.Q43) or not torch.equal(initial.Q44, final.Q44)
    assert q_statistics(final)['Q43']['offdiagonal_row_residual_rms'] < 1e-7
    assert state['routing_mode'] == 'row_tied'
    records = [json.loads(line) for line in (args.output / 'history.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records] == [1, 2, 3]
    for record in records:
        assert record['validation_seed'] == args.validation_seed
        assert (args.output / f'checkpoint-{record["step"]:05d}.pt').exists()
    audit = json.loads((args.output / 'data-audit.json').read_text())
    assert audit['data_seed'] == 2810001 and audit['noise_seed'] == 2820001
    assert audit['seed'] == 2820001 and audit['replicate'] == 2
    assert audit['teacher_generation_answers'] == 0
    before = json.loads((args.output / 'initial-evaluation.json').read_text())
    after = json.loads((args.output / 'evaluation.json').read_text())
    assert [x['panel_sha256'] for x in before['evaluations']] == [x['panel_sha256'] for x in after['evaluations']]


def test_teacher_trace_is_retained_without_truth_correction(tmp_path, capsys):
    _, teacher, _, checkpoint = foundation(tmp_path)
    args = arguments(tmp_path, checkpoint, '--routing-mode', 'diagonal', '--teacher', str(checkpoint),
                     '--label-source', 'teacher', '--steps', '0')
    run_experiment(args)
    saved = torch.load(args.output / 'training-data.pt', weights_only=False)
    sample = saved['sample']
    expected = sample['clauses'].clone()
    teacher.eval()
    for start in range(0, args.samples, args.batch):
        clauses = sample['clauses'][start:start + args.batch]
        expected[start:start + args.batch, args.length + 1:] = greedy_rollout(
            teacher, clauses[:, :args.length + 1], args.length)
    supervision = saved['supervision']
    assert torch.equal(supervision['inputs'], expected[:, :-1])
    assert torch.equal(supervision['targets'], expected[:, args.length + 1:, 4])
    assert supervision['audit']['ground_truth_usage'] == 'audit only; no filtering or correction'
    assert supervision['audit']['teacher_generation_answers'] == args.samples * args.length
    assert supervision['audit']['teacher_routing_mode'] == 'dense'
