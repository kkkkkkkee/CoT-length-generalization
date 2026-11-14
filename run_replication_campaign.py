"""Freeze sources, run the bounded replication campaign, and retain all results."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src'))
from replication_experiment import (write_json, train_run, evaluate_checkpoint,
                                    select_alpha, make_pseudo, _hash_file, configure)

SOURCES = ['run_replication_campaign.py', 'src/replication_experiment.py',
           'src/model.py', 'src/data.py', 'src/evaluate.py', 'src/generate_pseudo.py']


def protocol(smoke=False):
    return {
        'version': 1, 'smoke': smoke, 'tasks': ['C6', 'S5'],
        'seeds': [0] if smoke else [0, 1, 2], 'policies': ['fixed', 'fresh'],
        'base_epochs': 2 if smoke else 900, 'milestones': [1, 2] if smoke else [300, 900],
        'stage_epochs': 1 if smoke else 300, 'stage_lengths': [10, 20, 40],
        'branches': ['self', 'truth', 'continue_L5'],
        'samples': 16 if smoke else 8192, 'batch': 8 if smoke else 128,
        'token_dim': 4 if smoke else 64, 'nvars': 256, 'lr': 1e-4,
        'validation_lengths': [5, 10, 20], 'validation_samples': 4 if smoke else 256,
        'validation_seed': 121200001, 'test_seed': 121900001,
        'test_lengths': [5, 10, 20, 40, 80, 160], 'test_samples': 4 if smoke else 512,
        'alphas': [0.5, 1., 2., 4.],
        'base_training_count': 4 if smoke else 12,
        'curriculum_training_count': 9 if smoke else 27,
        'primary_metric': 'greedy rollout value_chain_accuracy, all five slots fed back',
        'base_comparison': 'paired initialization and epoch-one data; checkpoints 300/900; fixed versus fresh-per-epoch',
        'parent_selection': 'one S5 data policy across seeds at final base epoch; mean validation L5/10/20 greedy value-chain at alpha=1; exact ties choose fixed',
        'temperature_selection': 'one alpha per condition/checkpoint across seeds on validation L5/10/20; default alpha=1 always reported separately; never used for teachers',
        'stage_data': 'fixed 8192 per stage in ALL branches, paired self/truth prompts; short branch new L5 truth',
        'budget_caveat': 'curriculum branches match optimizer updates and sequence presentations, not value-label counts or FLOPs',
        'pseudo_policy': 'all full greedy traces retained; truth audit only; no filtering/correction; no teacher-quality stop gate',
        'stopping': 'fixed preregistered budget, no early stopping or test-based checkpoint selection; failed conditions retained',
        'selection_caveat': 'selected S5 policy is a validation-selected setting, not a new independent seed replication',
        'operational_high_success': {'metric': 'default alpha=1 L160 greedy value-chain',
                                     'mean_at_least': 0.99, 'each_seed_at_least': 0.95},
    }


def locked_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise RuntimeError(f'Frozen decision changed: {path}')
    else:
        write_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    design = protocol(args.smoke)
    if not args.frozen:
        source = out / 'source'
        hashes = {name: _hash_file(HERE / name) for name in SOURCES}
        plan = {**design, 'source_sha256': hashes}
        locked_json(out / 'campaign-plan.json', plan)
        for name in SOURCES:
            dest = source / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if _hash_file(dest) != hashes[name]:
                    raise RuntimeError(f'Snapshot changed: {dest}')
            else:
                shutil.copyfile(HERE / name, dest)
        if args.plan_only:
            print(json.dumps({'plan': str(out / 'campaign-plan.json'), 'training_runs': design['base_training_count'] + design['curriculum_training_count']}))
            return
        command = [sys.executable, str(source / 'run_replication_campaign.py'), '--output', str(out), '--frozen']
        if args.smoke:
            command.append('--smoke')
        os.execv(sys.executable, command)
    plan = json.loads((out / 'campaign-plan.json').read_text())
    if {k: plan[k] for k in design} != design:
        raise RuntimeError('Protocol changed')
    def check_sources():
        if any(_hash_file(HERE / name) != digest for name, digest in plan['source_sha256'].items()):
            raise RuntimeError('Frozen source hash mismatch')
    check_sources()
    lock = (out / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(out / 'process.json', {'pid': os.getpid(), 'started_unix': time.time(), 'source': str(HERE), 'python': sys.executable})
    configure(0)
    progress = {'complete': False, 'pid': os.getpid(), 'phase': 'starting', 'completed_training': []}
    def update(phase, name):
        check_sources()
        if shutil.disk_usage(out).free < 2 * 2**30:
            raise RuntimeError('Less than 2 GiB free; preserving artifacts and stopping')
        progress.update(phase=phase, current=name, updated_unix=time.time())
        write_json(out / 'campaign-progress.json', progress)
        print(json.dumps({'phase': phase, 'current': name}), flush=True)
    def spec(task, seed, length=5, policy='fixed', epochs=None, init=None, data_seed=None, pseudo=None):
        value = {'task': task, 'seed': seed, 'length': length, 'data_policy': policy,
                 'epochs': epochs or design['base_epochs'], 'samples': design['samples'],
                 'batch': design['batch'], 'lr': design['lr'], 'nvars': design['nvars'],
                 'token_dim': design['token_dim'], 'data_seed': data_seed if data_seed is not None else 10000 + seed,
                 'probe_samples': 4 if args.smoke else 256,
                 'validation_lengths': design['validation_lengths'], 'validation_seed': design['validation_seed'],
                 'eval_every': 1 if args.smoke else 50, 'save_every': 1 if args.smoke else 50,
                 'milestones': [epochs] if epochs else design['milestones']}
        if init:
            value.update(init=str(init), init_sha256=_hash_file(init))
        if pseudo:
            value.update(pseudo_data=str(pseudo), pseudo_sha256=_hash_file(pseudo))
        return value
    def train(name, config):
        update('training', name)
        result = train_run(out / name, config)
        progress['completed_training'].append(name)
        write_json(out / 'campaign-progress.json', progress)
        return result
    def validate(name, checkpoint):
        update('validation', name)
        return evaluate_checkpoint(checkpoint, out / 'validation' / f'{name}.json',
                    design['validation_lengths'], design['alphas'], design['validation_samples'],
                    design['validation_seed'], diagnostics=False)
    def test(name, checkpoint, alpha):
        update('test', name)
        return evaluate_checkpoint(checkpoint, out / 'test' / f'{name}.json',
                    design['test_lengths'], sorted({1., alpha}), design['test_samples'],
                    design['test_seed'])
    try:
        bases, validations, selections = {}, {}, {}
        for task in design['tasks']:
            for policy in design['policies']:
                for seed in design['seeds']:
                    name = f'base_{task}_{policy}_s{seed}'
                    train(name, spec(task, seed, policy=policy))
                    for epoch in design['milestones']:
                        key = f'{name}_e{epoch}'
                        checkpoint = out / name / f'checkpoint-{epoch:04d}.pt'
                        bases[key] = checkpoint
                        validations[key] = validate(key, checkpoint)
                for epoch in design['milestones']:
                    group = f'{task}_{policy}_e{epoch}'
                    selections[group] = select_alpha([validations[f'base_{task}_{policy}_s{s}_e{epoch}'] for s in design['seeds']])
        scores = {}
        for policy in design['policies']:
            values = [row['rollout']['metrics']['value_chain_accuracy'] for seed in design['seeds']
                      for row in validations[f'base_S5_{policy}_s{seed}_e{design["base_epochs"]}']['evaluations']
                      if row['inverse_temperature'] == 1.]
            scores[policy] = sum(values) / len(values)
        selected_policy = max(design['policies'], key=lambda p: scores[p])
        locked_json(out / 'base-selection.json', {'temperature': selections, 'S5_parent_policy': selected_policy,
                     'S5_parent_scores': scores, 'test_not_used': True,
                     'validation_sha256': {key: _hash_file(out / 'validation' / f'{key}.json') for key in validations}})
        # All base selection decisions are durable before any new long test runs.
        for task in design['tasks']:
            for policy in design['policies']:
                for epoch in design['milestones']:
                    for seed in design['seeds']:
                        name = f'base_{task}_{policy}_s{seed}_e{epoch}'
                        test(name, bases[name], selections[f'{task}_{policy}_e{epoch}']['alpha'])
        parents = {(branch, seed): out / f'base_S5_{selected_policy}_s{seed}' / f'checkpoint-{design["base_epochs"]:04d}.pt'
                   for branch in design['branches'] for seed in design['seeds']}
        for stage, length in enumerate(design['stage_lengths'], 1):
            stage_checks, stage_valid, stage_select = {}, {}, {}
            for branch in design['branches']:
                for seed in design['seeds']:
                    name = f'S5_{branch}_stage{stage}_s{seed}'
                    parent = parents[branch, seed]
                    data_seed = 122000001 + 10000 * seed + stage
                    pseudo = None
                    if branch == 'self':
                        update('pseudo_generation', name)
                        pseudo = out / 'pseudo' / f'{name}.pt'
                        make_pseudo(parent, pseudo, length, design['samples'], data_seed)
                    config = spec('S5', seed, length=5 if branch == 'continue_L5' else length,
                                  epochs=design['stage_epochs'], init=parent, data_seed=data_seed, pseudo=pseudo)
                    checkpoint = train(name, config)
                    parents[branch, seed] = checkpoint
                    stage_checks[name] = checkpoint
                    stage_valid[name] = validate(name, checkpoint)
                stage_select[branch] = select_alpha([stage_valid[f'S5_{branch}_stage{stage}_s{s}'] for s in design['seeds']])
            locked_json(out / f'stage{stage}-selection.json', {'temperature': stage_select, 'test_not_used': True,
                        'validation_sha256': {name: _hash_file(out / 'validation' / f'{name}.json') for name in stage_valid}})
            for branch in design['branches']:
                for seed in design['seeds']:
                    name = f'S5_{branch}_stage{stage}_s{seed}'
                    test(name, stage_checks[name], stage_select[branch]['alpha'])
        progress.update(complete=True, phase='complete', current=None, updated_unix=time.time())
        write_json(out / 'campaign-progress.json', progress)
        write_json(out / 'complete.json', progress)
    except Exception:
        progress.update(phase='failed', traceback=traceback.format_exc(), updated_unix=time.time())
        write_json(out / 'campaign-progress.json', progress)
        raise


if __name__ == '__main__':
    main()
