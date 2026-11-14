"""Bounded end-to-end replication: paired data controls and resumable training.

This module deliberately reuses the existing two-head implementation. It does
not substitute the block-sparse mechanism model or tune on the long test set.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from data import Task
from model import ClauseTransformer, ModelConfig, load_model
from evaluate import (aggregate_batches, greedy_rollout, rollout_metrics,
                      teacher_forced_metrics, _hash_file)
from generate_pseudo import generate_pseudo_dataset


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def save_tensor(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def configure(seed):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')


def epoch_data_seed(seed, epoch, policy):
    # Epoch one exactly matches the original baseline's generated dataset.
    return 10000 + seed if policy == 'fixed' or epoch == 1 else 120000000 + 100000 * seed + epoch


@torch.no_grad()
def probe(model, clauses, length, batch=64):
    model.eval()
    device = next(model.parameters()).device
    losses = torch.zeros(5, dtype=torch.float64, device=device)
    correct = torch.zeros(3, dtype=torch.long, device=device)
    for offset in range(0, len(clauses), batch):
        data = clauses[offset:offset + batch].to(device)
        logits = model(data[:, :-1])[:, length:2 * length]
        target = data[:, length + 1:]
        losses += F.cross_entropy(logits.flatten(0, 2), target.flatten(), reduction='none').reshape(-1, 5).double().sum(0)
        same = logits.argmax(-1).eq(target)
        correct += torch.stack((same[:, :, 4].all(1).sum(), same.all(2).all(1).sum(), same[:, :, 3].all(1).sum()))
    return {'samples': len(clauses), 'length': length,
            'slot_ce': (losses / (len(clauses) * length)).tolist(),
            'value_chain': correct[0].item() / len(clauses),
            'clause_chain': correct[1].item() / len(clauses),
            'variable_chain': correct[2].item() / len(clauses)}


def train_run(out, spec, device='cuda', stop_after=None):
    """Resume only this exact run, including Adam and sampling RNG state.

    init is a new stage (Adam resets); latest.pt is an interrupted same stage
    (Adam does not reset). stop_after is an engineering interruption hook.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != spec:
            raise RuntimeError(f'Configuration changed: {out}')
    else:
        if any(out.iterdir()):
            raise RuntimeError(f'Unrecognized nonempty training directory: {out}')
        write_json(config_path, spec)
    if (out / 'complete.json').exists():
        done = json.loads((out / 'complete.json').read_text())
        if _hash_file(done['checkpoint']) != done['checkpoint_sha256']:
            raise RuntimeError(f'Completed checkpoint changed: {out}')
        return Path(done['checkpoint'])
    configure(spec['seed'])
    task = Task(spec['task'], spec['nvars'])
    if spec.get('init'):
        if _hash_file(spec['init']) != spec['init_sha256']:
            raise RuntimeError('Parent checkpoint changed')
        model, _ = load_model(spec['init'], device)
        if model.config.vocab_size != task.vocab_size:
            raise ValueError('Task vocabulary mismatch')
    else:
        model = ClauseTransformer(ModelConfig(task.vocab_size, token_dim=spec['token_dim'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec['lr'])
    fixed = None
    if spec.get('pseudo_data'):
        if _hash_file(spec['pseudo_data']) != spec['pseudo_sha256']:
            raise RuntimeError('Pseudo dataset changed')
        fixed = torch.load(spec['pseudo_data'], weights_only=False, map_location='cpu')['clauses'].to(device)
        if fixed.shape != (spec['samples'], 2 * spec['length'] + 1, 5):
            raise ValueError('Pseudo dataset shape mismatch')
    elif spec['data_policy'] == 'fixed':
        fixed = task.generate(spec['samples'], spec['length'], torch.Generator().manual_seed(spec['data_seed']), device=device)['clauses']
    validation = {length: task.generate(spec['probe_samples'], length,
                  torch.Generator().manual_seed(spec['validation_seed'] + length))['clauses']
                  for length in spec['validation_lengths']}
    latest = out / 'latest.pt'
    start_epoch, elapsed, data_hashes = 0, 0., []
    if latest.exists():
        state = torch.load(latest, weights_only=False, map_location=device)
        if state['spec'] != spec:
            raise RuntimeError('Resume specification mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'].cpu())
        if device != 'cpu':
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        start_epoch, elapsed = state['epoch'], state['elapsed_seconds']
        data_hashes = state['data_hashes']
        # Remove only uncommitted observations after the durable checkpoint.
        history = out / 'history.jsonl'
        if history.exists():
            rows = [json.loads(line) for line in history.read_text().splitlines()]
            history.write_text(''.join(json.dumps(row) + '\n' for row in rows if row['epoch'] <= start_epoch))
    start = time.monotonic()
    def save(epoch, named=False):
        payload = {'model': model.state_dict(), 'model_config': model.config_dict(),
                   'args': {'task': task.name, 'nvars': task.nvars}, 'spec': spec,
                   'epoch': epoch, 'optimizer': optimizer.state_dict(),
                   'cpu_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all() if device != 'cpu' else [],
                   'elapsed_seconds': elapsed + time.monotonic() - start,
                   'data_hashes': data_hashes}
        save_tensor(latest, payload)
        if named:
            # Named artifacts omit optimizer; latest.pt retains resumable state.
            save_tensor(out / f'checkpoint-{epoch:04d}.pt',
                        {k: payload[k] for k in ['model', 'model_config', 'args', 'spec', 'epoch', 'data_hashes']})
    if not latest.exists():
        save(0, named=True)
    last = start_epoch
    for epoch in range(start_epoch + 1, spec['epochs'] + 1):
        model.train()
        if fixed is None:
            data_seed = epoch_data_seed(spec['seed'], epoch, 'fresh')
            data = task.generate(spec['samples'], spec['length'], torch.Generator().manual_seed(data_seed), device=device)['clauses']
        else:
            data_seed, data = spec['data_seed'], fixed
        data_hashes.append({'epoch': epoch, 'seed': data_seed, 'sha256': tensor_hash(data)})
        permutation = torch.randperm(len(data), device=device)
        epoch_loss = torch.zeros(5, device=device)
        for ids in permutation.split(spec['batch']):
            batch = data[ids]
            logits = model(batch[:, :-1])[:, spec['length']:2 * spec['length']]
            target = batch[:, spec['length'] + 1:]
            slot = F.cross_entropy(logits.reshape(-1, task.vocab_size), target.reshape(-1), reduction='none').reshape(-1, 5).mean(0)
            loss = slot.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: {out}, epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += slot.detach() * len(ids)
        last = epoch
        if epoch == 1 or epoch % spec['eval_every'] == 0 or epoch == spec['epochs']:
            row = {'epoch': epoch, 'optimizer_updates': epoch * math.ceil(spec['samples'] / spec['batch']),
                   'sequence_presentations': epoch * spec['samples'],
                   'value_label_presentations': epoch * spec['samples'] * spec['length'],
                   'slot_ce_during_epoch': (epoch_loss / len(data)).tolist(),
                   'train_probe': probe(model, data[:spec['probe_samples']], spec['length']),
                   'validation': {str(length): probe(model, panel, length) for length, panel in validation.items()},
                   'elapsed_seconds': elapsed + time.monotonic() - start}
            with (out / 'history.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            write_json(out / 'progress.json', row)
            print(json.dumps({'run': out.name, **row}), flush=True)
        if epoch % spec['save_every'] == 0 or epoch in spec['milestones'] or epoch == spec['epochs'] or epoch == stop_after:
            save(epoch, named=epoch in spec['milestones'] or epoch == spec['epochs'])
        if epoch == stop_after:
            return None
    checkpoint = out / f'checkpoint-{last:04d}.pt'
    write_json(out / 'data-manifest.json', {'epochs': data_hashes, 'policy': spec['data_policy'],
               'unique_dataset_hashes': len({row['sha256'] for row in data_hashes})})
    write_json(out / 'complete.json', {'complete': True, 'checkpoint': str(checkpoint.resolve()),
               'checkpoint_sha256': _hash_file(checkpoint), 'epoch': last,
               'elapsed_seconds': elapsed + time.monotonic() - start})
    return checkpoint


@torch.inference_mode()
def evaluate_checkpoint(checkpoint, output, lengths, alphas, samples, panel_seed, device='cuda', diagnostics=True):
    """Both TF and actual rollout use the same full panel at every alpha."""
    checkpoint, output = Path(checkpoint), Path(output)
    config = {'checkpoint_sha256': _hash_file(checkpoint), 'lengths': list(lengths),
              'alphas': list(alphas), 'samples': samples, 'panel_seed': panel_seed,
              'diagnostics': diagnostics}
    if output.exists():
        old = json.loads(output.read_text())
        if old['config'] != config:
            raise RuntimeError(f'Evaluation specification changed: {output}')
        if old.get('complete'):
            return old
    model, state = load_model(checkpoint, device)
    model.eval()
    task = Task(state['args']['task'], state['args']['nvars'])
    result = {'config': config, 'checkpoint': str(checkpoint.resolve()), 'evaluations': [],
              'primary_metric': 'unconstrained greedy all-answer-values-correct',
              'complete': False}
    for length in lengths:
        sample = task.generate(samples, length, torch.Generator().manual_seed(panel_seed + length))
        for alpha in alphas:
            tf, roll = [], []
            for offset in range(0, samples, 16):
                chunk = {key: value[offset:offset + 16].to(device) for key, value in sample.items()}
                clauses = chunk['clauses']
                if diagnostics:
                    tf.append(teacher_forced_metrics(model, chunk, alpha))
                generated = greedy_rollout(model, clauses[:, :length + 1], length, alpha)
                roll.append(rollout_metrics(generated, clauses[:, length + 1:]))
            item = {'length': length, 'inverse_temperature': alpha,
                    'panel_sha256': tensor_hash(sample['clauses']), 'rollout': aggregate_batches(roll)}
            if tf:
                item['teacher_forced'] = aggregate_batches(tf)
            result['evaluations'].append(item)
            write_json(output, result)
    result['complete'] = True
    write_json(output, result)
    return result


def select_alpha(validations):
    """One alpha for all seeds in a condition; ties prefer unchanged alpha=1."""
    scores = {}
    for result in validations:
        for row in result['evaluations']:
            scores.setdefault(row['inverse_temperature'], []).append(row['rollout']['metrics']['value_chain_accuracy'])
    means = {alpha: sum(values) / len(values) for alpha, values in scores.items()}
    best = min(means, key=lambda a: (-means[a], abs(math.log(a)), a))
    return {'alpha': best, 'validation_means': means, 'tie_rule': 'closest log-distance to 1, then smaller alpha'}


def make_pseudo(checkpoint, path, length, samples, seed, device='cuda'):
    path = Path(path)
    if path.exists():
        data = torch.load(path, weights_only=False, map_location='cpu')
        meta = data['metadata']
        if any(meta[k] != v for k, v in {'checkpoint_sha256': _hash_file(checkpoint),
                                       'length': length, 'samples': samples, 'seed': seed}.items()):
            raise RuntimeError('Existing pseudo dataset differs')
        return
    model, _ = load_model(checkpoint, device)
    model.eval()
    payload = generate_pseudo_dataset(model, Task('S5'), length, samples, batch=32,
                                      seed=seed, temperature=1., device=device)
    payload['metadata']['checkpoint_sha256'] = _hash_file(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_tensor(path, payload)
    write_json(path.with_suffix('.audit.json'), {k: payload[k] for k in ['metadata', 'audit']})
