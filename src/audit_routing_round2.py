"""Read-only CPU audit for routing replications and recursive curricula.

Only audit.json and audit-failures.json are written. Missing runs never count
as a pass, except stages explicitly omitted by the prespecified all-seed gate.
Teacher decoding is not rerun: saved inputs/targets, provenance, pairing and
local consistency are checked; the unavailable final clause's nonvalue slots
cannot independently reconstruct the recorded full teacher-trace hash.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import sys

import torch

from audit_sparse_campaign import Audit, canonical, sha256
from data import Task
from routing_experiment import parameter_hash, parse_args
from routing_model import load_routing_model, routing_parameter_counts
from sparse_data import build_supervision, tensor_hash


def same_values(audit, values, check, subject):
    audit.check(len(set(values.values())) <= 1, check, str(subject), values=values)


def audit_metrics(audit, value, subject, samples, length):
    audit.check(value['samples'] == samples and value['length'] == length,
                'evaluation_size', subject)
    counts, metrics = value['counts'], value['metrics']
    for stem in ['value', 'variable', 'clause']:
        count = counts[stem + '_chain_correct']
        audit.check(0 <= count <= samples and
                    abs(metrics[stem + '_chain_accuracy'] - count / samples) < 1e-12,
                    'chain_metric_from_counts', subject, metric=stem)
    for stem in ['value', 'clause']:
        hist = counts['first_' + stem + '_error_histogram']
        audit.check(len(hist) == length + 1 and sum(hist) == samples
                    and hist[-1] == counts[stem + '_chain_correct'],
                    'first_error_histogram_accounting', subject, metric=stem)


def audit_campaign(root):
    root = Path(root).resolve()
    plan_path = root / 'campaign-plan.json'
    plan = json.loads(plan_path.read_text())
    specs = plan['runs']
    by_name = {spec['name']: spec for spec in specs}
    recursive = any(spec.get('phase') == 'recursive' for spec in specs)
    audit = Audit()
    audit.check(len(by_name) == len(specs), 'unique_plan_names', 'manifest')
    audit.check(plan.get('total') == len(specs), 'declared_plan_size', 'manifest')
    configs, records, completed, missing = {}, {}, {}, []
    panel_groups = defaultdict(dict)
    checked = []

    @lru_cache(maxsize=16)
    def checkpoint(path):
        return load_routing_model(Path(path).resolve(), 'cpu')

    @lru_cache(maxsize=32)
    def source_sample(task_name, nvars, data_seed, pool_samples, samples, length):
        pool = Task(task_name, nvars).generate(max(samples, pool_samples), length,
                torch.Generator().manual_seed(data_seed))
        return {key: value[:samples].clone() for key, value in pool.items()}

    @lru_cache(maxsize=32)
    def panel_hash(task_name, nvars, seed, samples, length):
        sample = Task(task_name, nvars).generate(samples, length,
                torch.Generator().manual_seed(seed + length))
        return tensor_hash(sample['clauses'])

    def check_eval(document, c, name, phase, path):
        subject = name + ':' + phase
        audit.check(document.get('complete') is True, 'evaluation_complete', subject)
        audit.check(document['checkpoint_sha256'] == sha256(path),
                    'evaluation_checkpoint_hash', subject)
        audit.check(document['phase'] == phase, 'evaluation_phase', subject)
        audit.check(document['routing_mode'] == c['routing_mode'],
                    'evaluation_routing_mode', subject)
        audit.check(document['panel_seed'] == c['eval_seed'], 'evaluation_seed', subject)
        lengths = [row['length'] for row in document['evaluations']]
        audit.check(lengths == c['eval_lengths'], 'evaluation_lengths', subject)
        result = {}
        for row in document['evaluations']:
            length = row['length']
            key = (c['task'], c['nvars'], c['eval_seed'], c['eval_samples'], length)
            result[length] = row['panel_sha256']
            audit.check(row['panel_sha256'] == panel_hash(*key),
                        'evaluation_panel_reconstructed', subject, length=length)
            audit.check(row['oracle_attention'] is False, 'nonoracle_evaluation', subject,
                        length=length)
            panel_groups[key][subject] = row['panel_sha256']
            audit_metrics(audit, row['teacher_forced'], subject + f':L{length}:TF',
                          c['eval_samples'], length)
            rollout_n = min(c['eval_samples'], c['rollout_samples'])
            if rollout_n:
                audit_metrics(audit, row['rollout'], subject + f':L{length}:rollout',
                              rollout_n, length)
        return result

    for spec in specs:
        name = spec['name']
        directory = root / name
        complete_path = directory / 'complete.json'
        if not complete_path.exists():
            missing.append(name)
            continue
        start_checks, start_failed = audit.check_count, len(audit.failed)
        try:
            complete = json.loads(complete_path.read_text())
            if not complete.get('complete'):
                missing.append(name)
                continue
            c = json.loads((directory / 'config.json').read_text())
            expected_args = vars(parse_args(spec['command'][2:]))
            for key, expected in canonical(expected_args).items():
                audit.check(c.get(key) == expected, 'config_matches_manifest_command',
                            name, key=key, expected=expected, actual=c.get(key))
            initial, initial_state = checkpoint(str(directory / 'checkpoint-initial.pt'))
            final, final_state = checkpoint(str(directory / 'checkpoint-final.pt'))
            parent, parent_state = checkpoint(c['init'])
            task = Task(c['task'], c['nvars'])
            audit.check(c['stage'] == initial_state['stage'] == final_state['stage'] == 'attention',
                        'attention_stage', name)
            audit.check(c['model'] == initial.config_dict() == final.config_dict() == parent.config_dict(),
                        'model_configuration_preserved', name)
            audit.check(task.vocab_size == final.config.vocab_size and task.nvars == final.config.nvars,
                        'task_vocabulary', name)
            audit.check(parent_state['args']['task'] == c['task'], 'parent_task_identity', name)
            audit.check(c['initialization_routing_mode'] == getattr(parent, 'routing_mode', 'dense'),
                        'parent_routing_metadata', name)
            audit.check(initial.routing_mode == final.routing_mode == c['routing_mode']
                        == complete['routing_mode'], 'routing_mode_consistency', name)
            for key, value in canonical(final_state['args']).items():
                audit.check(c.get(key) == value, 'checkpoint_arg_matches_config', name, key=key)
            audit.check(initial_state['step'] == 0 and final_state['step'] == c['steps']
                        == complete['steps'], 'checkpoint_steps', name)
            audit.check(sha256(c['init']) == c['init_sha256'], 'parent_checkpoint_hash', name)
            for phase, path, field in [('initial', directory / 'checkpoint-initial.pt',
                                       'initial_checkpoint_sha256'),
                                      ('final', directory / 'checkpoint-final.pt',
                                       'final_checkpoint_sha256')]:
                audit.check(sha256(path) == complete[field], 'completion_checkpoint_hash', name,
                            phase=phase)
            ff_names = set(parent.ff.state_dict())
            audit.check(ff_names == set(initial.ff.state_dict()) == set(final.ff.state_dict()),
                        'ff_tensor_names', name)
            for key in ff_names:
                audit.check(torch.equal(parent.ff.state_dict()[key], initial.ff.state_dict()[key])
                            and torch.equal(parent.ff.state_dict()[key], final.ff.state_dict()[key]),
                            'ff_tensor_bitwise_frozen_against_parent', name, parameter=key)
            ff_hash = parameter_hash(final.ff.named_parameters())
            audit.check(ff_hash == c['ff_sha256_before'] == complete['ff_sha256_before']
                        == complete['ff_sha256_after'], 'ff_actual_hash', name)
            audit.check(initial.frozen_state_hash() == final.frozen_state_hash()
                        == c['frozen_sha256_before'] == complete['frozen_sha256_before']
                        == complete['frozen_sha256_after'], 'actual_frozen_state_hashes', name)
            coefficients = routing_parameter_counts(final)
            recorded_count = c.get('routing_parameter_coefficients', c.get('routing_degrees_of_freedom'))
            audit.check(coefficients['independent_coefficients'] == recorded_count,
                        'routing_coefficient_count', name)
            audit.check(c['trainable_parameters'] == [key for key, p in final.named_parameters()
                                                     if p.requires_grad], 'trainable_parameter_names', name)
            for block in ['Q43', 'Q44']:
                original = getattr(parent, block).detach()
                before = getattr(initial, block).detach()
                after = getattr(final, block).detach()
                mask = ~torch.eye(task.nvars, dtype=torch.bool)
                audit.check(torch.equal(before.diag(), original.diag()),
                            'initial_diagonal_preserved', name, block=block)
                if c['routing_mode'] == 'dense':
                    audit.check(torch.equal(before, original), 'dense_initial_matrix_exact', name,
                                block=block)
                elif c['routing_mode'] == 'row_tied':
                    background = original.masked_fill(~mask, 0).sum(-1) / (task.nvars - 1)
                    expected = background[:, None].expand_as(original)[mask]
                    # GPU versus CPU reduction order can differ by a few ulps.
                    audit.check(torch.allclose(before[mask], expected, atol=1e-6, rtol=1e-6),
                                'initial_background_is_parent_row_mean', name, block=block,
                                max_error=(before[mask] - expected).abs().max().item())
                for phase, matrix in [('initial', before), ('final', after)]:
                    audit.check(bool(torch.isfinite(matrix).all()), 'finite_routing_matrix', name,
                                phase=phase, block=block)
                    if c['routing_mode'] == 'diagonal':
                        audit.check(matrix[mask].count_nonzero().item() == 0,
                                    'zero_offdiagonal_constraint', name, phase=phase, block=block)
                    elif c['routing_mode'] == 'row_tied':
                        off = matrix[mask].reshape(task.nvars, task.nvars - 1)
                        audit.check(torch.equal(off, off[:, :1].expand_as(off)),
                                    'exact_row_tying_constraint', name, phase=phase, block=block)
            if spec.get('phase') == 'warmup':
                admission = json.loads((Path(c['init']).parent / 'complete.json').read_text())
                audit.check(parent_state['stage'] == 'local' and admission['local_gate_passed'],
                            'warmup_admitted_local_parent', name)
                audit.check(parent.Q43.count_nonzero().item() == parent.Q44.count_nonzero().item() == 0,
                            'warmup_starts_from_zero_q', name)
            teacher_mode = None
            if c['label_source'] == 'teacher':
                teacher, teacher_state = checkpoint(c['teacher'])
                audit.check(sha256(c['teacher']) == c['teacher_sha256'], 'teacher_checkpoint_hash', name)
                audit.check(teacher_state['args']['task'] == c['task']
                            and teacher.config_dict() == final.config_dict(), 'teacher_task_identity', name)
                teacher_mode = teacher.routing_mode
                audit.check(c['teacher_used'] is True, 'teacher_usage_metadata', name)
            else:
                audit.check(c['teacher_used'] is False, 'truth_source_metadata', name)

            training = torch.load(directory / 'training-data.pt', map_location='cpu', weights_only=False)
            sample, supervision = training['sample'], training['supervision']
            data_audit = json.loads((directory / 'data-audit.json').read_text())
            audit.check(supervision['audit'] == data_audit, 'saved_data_audits_identical', name)
            expected = source_sample(c['task'], c['nvars'], c['data_seed'], c['pool_samples'],
                                     c['samples'], c['length'])
            for key, value in expected.items():
                audit.check(torch.equal(sample[key], value), 'training_source_reconstructed', name, field=key)
            source_hash = tensor_hash(sample['clauses'])
            audit.check(source_hash == data_audit['source_clause_sha256'], 'training_source_hash', name)
            for field in ['inputs', 'targets', 'mask']:
                audit.check(tensor_hash(supervision[field]) == data_audit[field + '_sha256'],
                            'supervision_tensor_hash', name, field=field)
            teacher_trace = None
            if c['label_source'] == 'teacher':
                length = c['length']
                audit.check(torch.equal(supervision['inputs'][:, :length + 1],
                                        sample['clauses'][:, :length + 1]),
                            'teacher_prompt_preserved', name)
                audit.check(torch.equal(supervision['inputs'][:, length + 1:, 4],
                                        supervision['targets'][:, :-1]),
                            'teacher_target_prefix_value_agreement', name)
                # The training artifact omits nonvalue slots of the very last
                # generated clause; they are irrelevant to every loss query.
                teacher_trace = torch.cat((supervision['inputs'], sample['clauses'][:, -1:].clone()), dim=1)
                teacher_trace[:, -1, 4] = supervision['targets'][:, -1]
                audit.check(data_audit['ground_truth_usage'] == 'audit only; no filtering or correction',
                            'teacher_ground_truth_usage_record', name)
            expected_supervision = build_supervision(sample, task, c['mode'], c['label_source'],
                teacher_trace, c['noise_kind'], c['noise_rate'], seed=c['noise_seed'])
            for field, value in expected_supervision.items():
                if isinstance(value, torch.Tensor):
                    audit.check(torch.equal(supervision[field], value),
                                'supervision_reconstructed', name, field=field)
            # Recompute semantic audit fields without pretending to recover the
            # missing final teacher clause's nonvalue slots.
            unchecked_teacher_fields = {'label_source_clause_sha256', 'source_full_clause_trajectory_correct_rate'}
            for field, value in expected_supervision['audit'].items():
                if c['label_source'] == 'teacher' and field in unchecked_teacher_fields:
                    continue
                audit.check(data_audit.get(field) == value, 'semantic_data_audit_recomputed', name, field=field)
            for field in ['data_seed', 'noise_seed', 'replicate']:
                audit.check(data_audit[field] == c[field], 'sampling_seed_metadata', name, field=field)
            audit.check(data_audit['teacher_routing_mode'] == teacher_mode,
                        'teacher_routing_metadata', name)
            expected_generation = c['samples'] * c['length'] if teacher_mode else 0
            audit.check(data_audit['teacher_generation_answers'] == expected_generation,
                        'teacher_generation_budget', name)
            labels_per_example = supervision['selected_targets'].shape[1]
            labels = int(supervision['mask'].sum())
            audit.check(labels == complete['unique_labels'] == data_audit['supervised_value_labels']
                        == c['samples'] * labels_per_example, 'unique_value_label_budget', name)
            epoch_steps = (c['samples'] + c['batch'] - 1) // c['batch']
            epochs, remainder = divmod(c['steps'], epoch_steps)
            presentations = epochs * c['samples'] + min(remainder * c['batch'], c['samples'])
            audit.check(complete['examples_presented'] == presentations
                        and complete['labels_presented'] == presentations * labels_per_example,
                        'presentation_budget', name)
            targets = supervision['targets'][supervision['mask']]
            invalid_targets = (targets < task.value_offset) | (targets >= task.value_offset + task.nstates)
            supplemental = {
                'supervised_value_labels': targets.numel(),
                'invalid_supervised_target_count': int(invalid_targets.sum()),
                'invalid_supervised_target_rate': invalid_targets.double().mean().item(),
                'definition': 'supervised target token is outside the legal task-state token range',
                'target_error_rate_supervised': data_audit['target_error_rate_supervised'],
                'local_inconsistency_rate_supervised': data_audit['intended_action_local_inconsistency_rate_supervised'],
                'prefix_variable_error_rate_supervised': data_audit['prefix_variable_error_rate_supervised'],
                'invalid_prefix_state_rate_supervised': data_audit['invalid_prefix_state_rate_supervised'],
            }
            before_eval = json.loads((directory / 'initial-evaluation.json').read_text())
            after_eval = json.loads((directory / 'evaluation.json').read_text())
            before_panels = check_eval(before_eval, c, name, 'after_conversion_before_training',
                                      directory / 'checkpoint-initial.pt')
            after_panels = check_eval(after_eval, c, name, 'after_training', directory / 'checkpoint-final.pt')
            audit.check(before_panels == after_panels, 'initial_final_panel_pairing', name)
            audit.check(c['validation_seed'] != c['eval_seed'], 'validation_test_separation', name)
            history_path = directory / 'history.jsonl'
            if c['steps']:
                history = [json.loads(line) for line in history_path.read_text().splitlines()]
                expected_steps = list(range(c['eval_every'], c['steps'] + 1, c['eval_every']))
                if not expected_steps or expected_steps[-1] != c['steps']:
                    expected_steps.append(c['steps'])
                audit.check([row['step'] for row in history] == expected_steps,
                            'validation_checkpoints_complete', name)
                for row in history:
                    audit.check(row['validation_seed'] == c['validation_seed'],
                                'validation_seed_metadata', name, step=row['step'])
                    for panel in row['validation_evaluations']:
                        key = (c['task'], c['nvars'], c['validation_seed'], c['validation_samples'], panel['length'])
                        audit.check(panel['panel_sha256'] == panel_hash(*key),
                                    'validation_panel_reconstructed', name, step=row['step'], length=panel['length'])
            configs[name], completed[name] = c, complete
            records[name] = {'source_hash': source_hash,
                'label_source_hash': data_audit['label_source_clause_sha256'],
                'inputs_hash': tensor_hash(supervision['inputs']),
                'targets_hash': tensor_hash(supervision['targets']),
                'noise_scores_hash': tensor_hash(supervision['noise_scores']),
                'corruption_mask_hash': tensor_hash(supervision['corrupted']),
                'supplemental_target_audit': supplemental,
                'routing_coefficient_counts': coefficients}
            checked.append(name)
        except Exception as exc:
            audit.check(False, 'run_artifacts_readable', name,
                        exception=type(exc).__name__, error=str(exc))
        audit.per_run[name] = {'checks': audit.check_count - start_checks,
                               'failed_checks': len(audit.failed) - start_failed,
                               **records.get(name, {})}

    data_groups, teacher_groups, noise_groups, budget_groups = (defaultdict(dict) for _ in range(4))
    design_pairs = defaultdict(dict)
    for name in checked:
        c, record = configs[name], records[name]
        data_key = (c['task'], c['nvars'], c['data_seed'], c['length'], c['samples'], c['pool_samples'])
        data_groups[data_key][name] = record['source_hash']
        if c['label_source'] == 'teacher':
            teacher_groups[(*data_key, c['teacher_sha256'])][name] = record
        noise_groups[(*data_key, c['noise_seed'])][name] = record
        spec = by_name[name]
        budget_key = (c['seed'], spec.get('stage', 0)) if recursive else (c['seed'], c['replicate'])
        budget_groups[budget_key][name] = completed[name]
        design_pairs[budget_key][name] = c
        if spec.get('phase') == 'recursive':
            stage, routing, regime = spec['stage'], spec['routing'], spec['regime']
            expected_parent = (f'warm_{routing}_s{c["seed"]}' if stage == 1 else
                               f'{routing}_{regime}_stage{stage - 1}_s{c["seed"]}')
            expected_path = root / expected_parent / 'checkpoint-final.pt'
            audit.check(Path(c['init']).resolve() == expected_path,
                        'recursive_parent_lineage', name)
            audit.check(c['routing_mode'] == routing, 'recursive_routing_matches_manifest', name)
            audit.check(c['label_source'] == ('teacher' if regime == 'self' else 'truth'),
                        'recursive_regime_source', name)
            audit.check(c['length'] == (2 if regime == 'continue_L2' else spec['curriculum_length']),
                        'recursive_training_length', name)
            if regime == 'self':
                audit.check(Path(c['teacher']).resolve() == expected_path,
                            'recursive_teacher_is_own_previous_model', name)
    for key, values in data_groups.items():
        same_values(audit, values, 'paired_training_source', key)
    for key, values in teacher_groups.items():
        for field in ['inputs_hash', 'targets_hash', 'label_source_hash']:
            same_values(audit, {name: row[field] for name, row in values.items()},
                        'same_teacher_same_source_labels', (key, field))
    for key, values in noise_groups.items():
        same_values(audit, {name: row['noise_scores_hash'] for name, row in values.items()},
                    'paired_noise_random_draws', key)
        by_rate, prefix_by_rate = defaultdict(dict), defaultdict(dict)
        for name, row in values.items():
            c = configs[name]
            by_rate[c['noise_rate']][name] = row['corruption_mask_hash']
            if c['noise_kind'] in {'prefix_inconsistent', 'prefix_consistent'}:
                prefix_by_rate[c['noise_rate']][name] = row['inputs_hash']
        for rate, masks in by_rate.items():
            same_values(audit, masks, 'paired_noise_corruption_masks', (key, rate))
        for rate, inputs in prefix_by_rate.items():
            same_values(audit, inputs, 'paired_corrupted_prefix_inputs', (key, rate))
    for key, values in panel_groups.items():
        same_values(audit, values, 'common_evaluation_panels', key)
    for key, values in budget_groups.items():
        for field in ['steps', 'unique_labels', 'labels_presented', 'examples_presented']:
            same_values(audit, {name: row[field] for name, row in values.items()},
                        'paired_training_budget', (key, field))
    design_seeds = {}
    for key, values in design_pairs.items():
        for field in (['data_seed', 'noise_seed'] if recursive else
                      ['init_sha256', 'data_seed', 'noise_seed']):
            same_values(audit, {name: row[field] for name, row in values.items()},
                        'shared_sampling_design', (key, field))
        first = next(iter(values.values()))
        design_seeds[key] = (first['data_seed'], first['noise_seed'])
    for index, field in enumerate(['data_seed', 'noise_seed']):
        actual = [pair[index] for pair in design_seeds.values()]
        audit.check(len(actual) == len(set(actual)), 'independent_replicate_or_stage_seeds', field)

    expected_unexecuted = []
    gate_result = None
    if recursive:
        gate_path = root / 'gate.json'
        if gate_path.exists():
            gate_result = json.loads(gate_path.read_text())
            validation = []
            for seed in range(3):
                name = f'warm_row_tied_s{seed}'
                history = [json.loads(line) for line in (root / name / 'history.jsonl').read_text().splitlines()]
                row = history[-1]
                c = configs[name]
                audit.check(row['step'] == c['steps'], 'gate_uses_final_fixed_step', name)
                audit.check(row['validation_seed'] == gate_result['panel_seed'] != c['eval_seed'],
                            'gate_uses_independent_validation', name)
                validation.append({'seed': seed, 'step': row['step'],
                                   'L4_chain': row['panel']['4']['clause_chain_accuracy']})
            audit.check(gate_result['threshold'] == .95 and gate_result['all_seeds_required'] is True,
                        'prespecified_gate_rule', 'recursive_gate')
            expected_pass = all(row['L4_chain'] >= .95 for row in validation)
            audit.check(gate_result['validation'] == validation and gate_result['passed'] == expected_pass,
                        'gate_reconstructed_from_all_seeds', 'recursive_gate')
            if not expected_pass:
                expected_unexecuted = [spec['name'] for spec in specs if spec.get('phase') == 'recursive']
                audit.check(not any(name in checked for name in expected_unexecuted),
                            'no_selected_seed_after_failed_gate', 'recursive_gate')
        elif any(by_name[name].get('phase') == 'recursive' for name in checked):
            audit.check(False, 'gate_exists_before_recursion', 'recursive_gate')
    final_record_path = root / 'campaign-complete.json'
    final_record = json.loads(final_record_path.read_text()) if final_record_path.exists() else None
    if expected_unexecuted:
        audit.check(final_record is not None and final_record.get('gate_passed') is False
                    and sorted(final_record.get('unexecuted', [])) == sorted(expected_unexecuted),
                    'gated_stop_completion_record', 'campaign')
    required_names = set(by_name) - set(expected_unexecuted)
    campaign_complete = required_names == set(checked)
    if campaign_complete:
        audit.check(final_record is not None and final_record.get('complete') is True,
                    'campaign_completion_record', 'campaign')
        if final_record:
            audit.check(set(final_record.get('completed', [])) == set(checked)
                        and not final_record.get('failures'), 'campaign_completed_run_set', 'campaign')
    passed = campaign_complete and not audit.failed
    return {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'campaign': str(root), 'plan_sha256': sha256(plan_path), 'audit_source_sha256': sha256(__file__),
        'status': 'failed' if audit.failed else ('passed' if passed else 'incomplete'),
        'passed': passed, 'campaign_complete': campaign_complete,
        'checks_passed_on_available_runs': not audit.failed, 'planned_runs': len(specs),
        'checked_runs': len(checked), 'checked_run_names': checked,
        'missing_runs': [name for name in missing if name not in expected_unexecuted],
        'expected_unexecuted_after_gate': expected_unexecuted,
        'check_count': audit.check_count, 'failed_checks': audit.failed,
        'per_run': audit.per_run, 'recursive_gate': gate_result,
        'notes': ['CPU-only; source checkpoints and data audits are never modified',
            'Routing counts are actual optimized coefficients, not identifiable statistical degrees of freedom',
            'FFN tensors compared bitwise with saved initial and source parent parameters',
            'Initial projection checked separately from subsequent training',
            'Complete training samples and controlled noise regenerated from recorded independent RNG seeds',
            'Teacher greedy decoding not independently rerun; saved prefixes and targets and paired provenance checked',
            'The saved training artifact omits the final generated clause nonvalue slots, preventing independent reconstruction of its full teacher-trace hash',
            'All frozen and checkpoint hashes use actual local artifacts, not only recorded strings',
            'Source-code versions are retained per run; historical source hashes need not equal current files']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', '--root', dest='runs', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'runs/routing-round2-2026-11-09')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = audit_campaign(args.runs)
    output = args.output or args.runs / 'audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, value in [(output, result), (output.with_name(output.stem + '-failures.json'), result['failed_checks'])]:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(path)
    print(json.dumps({'output': str(output), **{key: result[key] for key in
        ['status', 'passed', 'planned_runs', 'checked_runs', 'check_count']},
        'failed_checks': result['failed_checks'], 'missing_runs': result['missing_runs']}, ensure_ascii=False))
    if result['failed_checks']:
        sys.exit(1)
    if not result['passed'] and not args.allow_incomplete:
        sys.exit(2)


if __name__ == '__main__':
    main()
