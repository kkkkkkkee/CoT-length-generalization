"""Diagnose fitting of corrupted targets and routing during frozen-FFN training.

This script never trains. It evaluates the exact saved training inputs/targets
and a new independently sampled validation panel with the same corruption
mechanism. Subsets are descriptive, including whether targets are locally
consistent with the actual input state; they are not independent experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import Task
from evaluate import greedy_rollout
from routing_model import load_routing_model
from sparse_data import build_supervision, select_supervised, tensor_hash


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}:{tuple(value.shape)}:{value.dtype}'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def subset_masks(sample, supervision, task):
    """Masks use selected labels; corruption is the saved per-example draw."""
    selected = select_supervised(supervision)
    steps = selected['step_indices']
    queries = selected['query_positions']
    inputs = selected['inputs']
    previous = inputs[:, :, 4].gather(1, queries) - task.value_offset
    valid_state = (previous >= 0) & (previous < task.nstates)
    actions = sample['actions'].gather(1, steps)
    local_target = task.action_table[actions, previous.clamp(0, task.nstates - 1)] + task.value_offset
    local_consistent = valid_state & selected['targets'].eq(local_target)
    truth_previous_variables = sample['variables'].gather(1, steps)
    query_variables = inputs[:, :, 3].gather(1, queries)
    route_correct = query_variables.eq(truth_previous_variables)
    corrupt = supervision['corrupted'][:, None].expand_as(steps)
    all_labels = torch.ones_like(steps, dtype=torch.bool)
    masks = {'all': all_labels, 'corrupted_examples': corrupt,
             'uncorrupted_examples': ~corrupt,
             'locally_consistent': local_consistent,
             'locally_inconsistent': ~local_consistent,
             'valid_state_correct_routing': valid_state & route_correct,
             'corrupted_locally_consistent': corrupt & local_consistent,
             'corrupted_locally_inconsistent': corrupt & ~local_consistent,
             'uncorrupted_locally_consistent': ~corrupt & local_consistent,
             'uncorrupted_locally_inconsistent': ~corrupt & ~local_consistent}
    length = sample['actions'].shape[1]
    truth_targets = sample['clauses'][:, length + 1:, 4].gather(1, steps)
    return masks, {'truth_targets': truth_targets, 'local_targets': local_target,
                   'valid_prefix_state': valid_state, 'correct_query_variable': route_correct}


@torch.inference_mode()
def evaluate_subsets(model, task, sample, supervision, batch_size=128):
    """Exact sample-weighted selected-query diagnostics, including empty groups."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    model.eval()
    before = state_hash(model)
    device = next(model.parameters()).device
    selected = select_supervised(supervision)
    masks, metadata = subset_masks(sample, supervision, task)
    length = sample['actions'].shape[1]
    quantities = {}
    n = len(selected['inputs'])
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        inputs = selected['inputs'][start:stop].to(device)
        queries = selected['query_positions'][start:stop].to(device)
        steps = selected['step_indices'][start:stop].to(device)
        targets = selected['targets'][start:stop].to(device)
        truth = metadata['truth_targets'][start:stop].to(device)
        local_targets = metadata['local_targets'][start:stop].to(device)
        logits, attention = model.forward_queries(inputs, queries, return_attention=True)
        value_logits = logits[:, :, 4]
        prediction = value_logits.argmax(-1)
        target_ce = F.cross_entropy(value_logits.flatten(0, 1), targets.flatten(), reduction='none').reshape_as(targets)
        truth_ce = F.cross_entropy(value_logits.flatten(0, 1), truth.flatten(), reduction='none').reshape_as(truth)
        a = attention[:, 0]
        rule = a.gather(-1, steps[..., None]).squeeze(-1)
        state = a.gather(-1, queries[..., None]).squeeze(-1)
        other_rules = a[:, :, :length].clone()
        other_rules.scatter_(-1, steps[..., None], -torch.inf)
        best_other = other_rules.amax(-1)
        mass = rule + state
        share = torch.where(mass > 0, rule / mass.clamp_min(torch.finfo(a.dtype).tiny), .5)
        batch_values = {
            'target_label_ce': target_ce, 'original_truth_value_ce': truth_ce,
            'target_label_accuracy': prediction.eq(targets),
            'original_truth_value_accuracy': prediction.eq(truth),
            'local_transition_accuracy': prediction.eq(local_targets),
            'target_rule_mass': rule, 'current_state_mass': state,
            'relevant_mass': mass, 'rule_share_within_relevant': share,
            'absolute_rule_state_gap': (rule - state).abs(),
            'rule_strict_top1_among_predicates': rule > best_other,
            'rule_top_tie_rate': rule == best_other,
        }
        for key, value in batch_values.items():
            quantities.setdefault(key, []).append(value.cpu().double())
    values = {key: torch.cat(parts) for key, parts in quantities.items()}
    groups = []
    for name, mask in masks.items():
        count = int(mask.sum())
        valid_local = mask & metadata['valid_prefix_state']
        metrics = {key: values[key][mask].mean().item() if count else None for key in values}
        # Invalid state tokens have no defined local transition; exclude them
        # instead of treating the clamped table lookup as a valid target.
        metrics['local_transition_accuracy'] = (values['local_transition_accuracy'][valid_local].mean().item()
                                               if valid_local.any() else None)
        groups.append({'subset': name, 'selected_labels': count,
            'examples': int(mask.any(-1).sum()), 'valid_local_transition_labels': int(valid_local.sum()),
            'invalid_prefix_state_labels': int((mask & ~metadata['valid_prefix_state']).sum()),
            'incorrect_query_variable_labels': int((mask & ~metadata['correct_query_variable']).sum()),
            'metrics': metrics})
    after = state_hash(model)
    if before != after:
        raise RuntimeError('Post-hoc diagnostics changed model state')
    return {'examples': n, 'selected_labels': int(masks['all'].sum()), 'groups': groups,
            'model_state_sha256_before': before, 'model_state_sha256_after': after}


@torch.inference_mode()
def make_validation(task, config, samples, data_seed, noise_seed, device='cpu', batch=128):
    """Generate one frozen diagnostic panel, shared by all checkpoints."""
    sample = task.generate(samples, config['length'], torch.Generator().manual_seed(data_seed))
    teacher_clauses = None
    teacher_hash = None
    if config['label_source'] == 'teacher':
        path = Path(config['teacher'])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        teacher, state = load_routing_model(path, device)
        if state['args']['task'] != task.name or teacher.config.nvars != task.nvars:
            raise ValueError('Validation teacher does not match task')
        teacher_hash = file_hash(path)
        teacher_clauses = sample['clauses'].clone()
        teacher.eval()
        length = config['length']
        for start in range(0, samples, batch):
            clauses = sample['clauses'][start:start + batch].to(device)
            teacher_clauses[start:start + batch, length + 1:] = greedy_rollout(
                teacher, clauses[:, :length + 1], length).cpu()
        del teacher
    supervision = build_supervision(sample, task, config['mode'], config['label_source'],
        teacher_clauses, config['noise_kind'], config['noise_rate'], seed=noise_seed)
    return {'sample': sample, 'supervision': supervision,
            'validation_metadata': {'data_seed': data_seed, 'noise_seed': noise_seed,
                'teacher_sha256': teacher_hash, 'samples': samples,
                'source': 'new independently sampled validation data; no optimization or model selection'}}


def write_outputs(output, result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'error-dynamics.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temp.replace(target)
    rows = []
    for record in result['checkpoints']:
        for split, panel in record['splits'].items():
            for group in panel['groups']:
                rows.append({'step': record['step'], 'split': split, 'subset': group['subset'],
                    'selected_labels': group['selected_labels'], 'examples': group['examples'],
                    'valid_local_transition_labels': group['valid_local_transition_labels'],
                    'invalid_prefix_state_labels': group['invalid_prefix_state_labels'],
                    'incorrect_query_variable_labels': group['incorrect_query_variable_labels'],
                    **group['metrics'], 'checkpoint_sha256': record['checkpoint_sha256']})
    if rows:
        with (output / 'error-dynamics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_diagnostics(run, output, device='cpu', validation_seed=3040001,
                    validation_noise_seed=None, validation_samples=None,
                    steps=(0, 250, 500, 750, 1000), batch=128, allow_incomplete=False):
    run, output = Path(run), Path(output)
    if (output / 'error-dynamics.json').exists():
        raise RuntimeError('Refusing to overwrite error-dynamics.json')
    config = json.loads((run / 'config.json').read_text())
    if validation_seed in {config['data_seed'], config['eval_seed'], config.get('validation_seed')}:
        raise ValueError('Diagnostic validation data seed must be distinct from training and previous evaluation seeds')
    if validation_samples is None:
        validation_samples = config['samples']
    if validation_noise_seed is None:
        validation_noise_seed = validation_seed + 73
    if validation_samples < 1 or batch < 1:
        raise ValueError('validation_samples and batch must be positive')
    task = Task(config['task'], config['nvars'])
    saved = torch.load(run / 'training-data.pt', map_location='cpu', weights_only=False)
    audit = json.loads((run / 'data-audit.json').read_text())
    for field in ['inputs', 'targets', 'mask']:
        if tensor_hash(saved['supervision'][field]) != audit[f'{field}_sha256']:
            raise ValueError(f'Saved training {field} disagrees with source audit')
    if tensor_hash(saved['sample']['clauses']) != audit['source_clause_sha256']:
        raise ValueError('Saved training sample disagrees with source audit')
    paths = {step: run / ('checkpoint-initial.pt' if step == 0 else f'checkpoint-{step:05d}.pt') for step in steps}
    missing = [step for step, path in paths.items() if not path.exists()]
    if missing and not allow_incomplete:
        raise ValueError(f'Missing requested checkpoints: {missing}')
    validation = make_validation(task, config, validation_samples, validation_seed,
                                 validation_noise_seed, device, batch)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(validation, output / 'validation-data.pt')
    records = []
    for step, path in paths.items():
        if not path.exists():
            continue
        model, state = load_routing_model(path, device)
        if state['args']['task'] != task.name or model.config.nvars != task.nvars:
            raise ValueError('Checkpoint task differs from run config')
        if state.get('step') != step:
            raise ValueError('Checkpoint step differs from requested step')
        splits = {name: evaluate_subsets(model, task, data['sample'], data['supervision'], batch)
                  for name, data in [('training', saved), ('validation', validation)]}
        records.append({'step': step, 'checkpoint': str(path.resolve()),
                        'checkpoint_sha256': file_hash(path), 'splits': splits})
        print(json.dumps({'run': run.name, 'step': step, 'complete': True}), flush=True)
        del model
    result = {'schema_version': 1, 'experiment': 'routing_corruption_error_dynamics',
        'complete': not missing and len(records) == len(steps),
        'missing_checkpoint_steps': missing, 'run': str(run.resolve()), 'config': config,
        'checkpoints': records, 'training_data_sha256': file_hash(run / 'training-data.pt'),
        'source_config_sha256': file_hash(run / 'config.json'),
        'validation_data_sha256': file_hash(output / 'validation-data.pt'),
        'training_audit': audit, 'validation_audit': validation['supervision']['audit'],
        'validation_metadata': validation['validation_metadata'],
        'definitions': {
            'scope': 'Post-hoc selected-query teacher-forced diagnostics; no training, no free-rollout result',
            'target_label': 'The saved training or independently corrupted validation supervision label',
            'original_truth': 'The original uncorrupted task state, used only to measure diagnostic accuracy and CE',
            'corrupted_examples': 'Per-example controlled corruption draw; teacher-generated errors are not relabeled as synthetic corruption',
            'local_consistency': 'Target equals the intended true action applied to the actual input state; invalid input states count inconsistent',
            'local_transition_accuracy': 'Accuracy against intended action applied to actual prefix; undefined input states excluded and denominator reported',
            'routing': 'Attention is measured at the intended rule position even if a generated query variable is incorrect',
            'subsets': 'Groups overlap and are descriptive; they are not independent replicates',
            'noise_rng': 'Validation uses the same corruption mechanism and requested probability, with independent draws; it does not reuse training corruption IDs',
            'empty_subsets': 'Zero labels reported with null metrics, never assigned zero accuracy',
        },
        'source_sha256': {name: file_hash(Path(__file__).parent / name) for name in
            ['routing_error_dynamics.py', 'routing_model.py', 'sparse_model.py', 'sparse_data.py', 'data.py', 'evaluate.py']}}
    write_outputs(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--validation-seed', type=int, default=3040001)
    parser.add_argument('--validation-noise-seed', type=int)
    parser.add_argument('--validation-samples', type=int)
    parser.add_argument('--steps', type=int, nargs='+', default=[0, 250, 500, 750, 1000])
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = run_diagnostics(args.run, args.output, args.device, args.validation_seed,
        args.validation_noise_seed, args.validation_samples, args.steps, args.batch, args.allow_incomplete)
    print(json.dumps({'output': str(args.output), 'complete': result['complete'],
                      'checkpoints': len(result['checkpoints'])}), flush=True)


if __name__ == '__main__':
    main()
