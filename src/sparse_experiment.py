"""Finite-data sparse supervision experiments with audited frozen computation.

Independent of the original end-to-end baseline. All stages and deviations
from the paper are explicit in configs; existing baseline files are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from data import Task, permute_predicates
from evaluate import _correctness, aggregate_batches, greedy_rollout, rollout_metrics
from local_rule import evaluate_local_rule
from sparse_model import SparseModelConfig, SparseClauseTransformer, load_model
from sparse_data import build_supervision, select_supervised, tensor_hash


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def evaluate(model, task, lengths, samples=512, rollout_samples=64,
             seed=1940001, batch_size=64, oracle=False, predicate_shuffle=False):
    """Paired, independent panels; all five slots evaluated and fed back."""
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for length in lengths:
        sample = task.generate(samples, length, torch.Generator().manual_seed(seed + length))
        if predicate_shuffle:
            sample = permute_predicates(sample, torch.Generator().manual_seed(seed + 1000000 + length))
        batches = []
        attention_totals = torch.zeros(7, dtype=torch.float64)
        nsteps = 0
        for start in range(0, samples, batch_size):
            c = sample['clauses'][start:start + batch_size].to(device)
            b = len(c)
            queries = torch.arange(length, 2 * length, device=device).expand(b, -1)
            pidx = (sample["predicate_positions"][start:start + batch_size].to(device)[..., None]
                    if predicate_shuffle else torch.arange(length, device=device)[None, :, None].expand(b, -1, -1))
            override = None
            if oracle:
                override = torch.zeros(b, 1, length, 2 * length, device=device)
                override[:, 0].scatter_(-1, pidx, .5)
                override[:, 0].scatter_(-1, queries[..., None], .5)
            logits, attention = model.forward_queries(c[:, :-1], queries,
                return_attention=True, attention_override=override)
            batches.append(_correctness(logits.argmax(-1), c[:, length + 1:], 'teacher_forced'))
            a = attention[:, 0]
            ap = a.gather(-1, pidx).squeeze(-1)
            ay = a.gather(-1, queries[..., None]).squeeze(-1)
            rest = a.clone()
            rest.scatter_(-1, pidx, 0)
            rest.scatter_(-1, queries[..., None], 0)
            other_rules = a[:, :, :length].clone()
            other_rules.scatter_(-1, pidx, -torch.inf)
            best_other = other_rules.amax(-1)
            values = [ap, ay, ap + ay, (ap - ay).abs(),
                      (ap > best_other).float(), rest.amax(-1), (ap == best_other).float()]
            attention_totals += torch.tensor([v.double().sum().item() for v in values])
            nsteps += b * length
        row = {'length': length, 'teacher_forced': aggregate_batches(batches),
               'attention': dict(zip(['target_rule_mass', 'current_state_mass',
                   'relevant_mass', 'absolute_rule_state_gap', 'rule_strict_top1_among_predicates',
                   'largest_background_mass', 'rule_top_tie_rate'], (attention_totals / nsteps).tolist())),
               'panel_sha256': tensor_hash(sample['clauses']), 'oracle_attention': oracle,
               'predicate_shuffle': predicate_shuffle,
               'permutation_seed': seed + 1000000 + length if predicate_shuffle else None}
        if rollout_samples and not oracle:
            roll = []
            for start in range(0, min(samples, rollout_samples), batch_size):
                c = sample['clauses'][start:min(start + batch_size, rollout_samples, samples)].to(device)
                pred = greedy_rollout(model, c[:, :length + 1], length)
                roll.append(rollout_metrics(pred, c[:, length + 1:]))
            row['rollout'] = aggregate_batches(roll)
        rows.append(row)
    return rows


def save_checkpoint(path, model, args, step):
    torch.save({'model': model.state_dict(), 'model_config': model.config_dict(),
                'stage': model.training_stage, 'args': vars(args), 'step': step}, path)


def compact(rows):
    return {str(r['length']): {k: r['teacher_forced']['metrics'][k] for k in
            ['value_accuracy', 'variable_accuracy', 'clause_chain_accuracy']} for r in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['local', 'attention'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--init', type=Path)
    p.add_argument('--teacher', type=Path)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--task', default='S5')
    p.add_argument('--nvars', type=int, default=128)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--position-mode', choices=['none','learned_abs','sinusoidal','relative'], default='none')
    p.add_argument('--local-gate-protocol', choices=['legacy_oracle','position_local'], default='legacy_oracle')
    p.add_argument('--length', type=int, default=2)
    p.add_argument('--samples', type=int, default=4096)
    p.add_argument('--pool-samples', type=int, default=16384)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=.03)
    p.add_argument('--mode', choices=['second', 'random', 'all'], default='second')
    p.add_argument('--label-source', choices=['truth', 'teacher'], default='truth')
    p.add_argument('--noise-kind', default='none')
    p.add_argument('--noise-rate', type=float, default=0.)
    p.add_argument('--data-seed', type=int, default=1820001)
    p.add_argument('--eval-seed', type=int, default=1940001)
    p.add_argument('--eval-samples', type=int, default=1024)
    p.add_argument('--rollout-samples', type=int, default=128)
    p.add_argument('--eval-lengths', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--local-gate', type=float, default=.9995)
    p.add_argument('--local-min-steps', type=int, default=500)
    p.add_argument('--budget-label', default='equal_examples_and_optimizer_steps')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if (args.output / 'config.json').exists():
        raise RuntimeError(f'Refusing to overwrite an existing experiment: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    task = Task(args.task, args.nvars)
    if args.stage == 'local':
        if args.init:
            raise ValueError('Local pretraining starts from a new model with Q=0')
        args.length = 1
        model = SparseClauseTransformer(SparseModelConfig(task.vocab_size, task.nvars, args.hidden, position_mode=args.position_mode)).to(args.device)
    else:
        if not args.init:
            raise ValueError('Attention training requires a local or prior-stage checkpoint')
        model, prior = load_model(args.init, args.device)
        if model.config.vocab_size != task.vocab_size or model.config.nvars != task.nvars or prior['args']['task'] != args.task:
            raise ValueError('Checkpoint and task differ')
        parent_complete = json.loads((args.init.parent / 'complete.json').read_text())
        if not parent_complete.get('complete'):
            raise ValueError('Initialization run is incomplete')
        if prior.get('stage') == 'local' and not parent_complete.get('local_gate_passed'):
            raise ValueError('Local computation failed its admission gate')
    model.configure_stage(args.stage)
    frozen_before = model.frozen_state_hash()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'model': model.config_dict(), 'trainable_parameters': [n for n, v in model.named_parameters() if v.requires_grad],
                   'frozen_sha256_before': frozen_before,
                   'init_sha256': file_hash(args.init) if args.init else None,
                   'teacher_sha256': file_hash(args.teacher) if args.teacher else None,
                   'source_sha256': {name: file_hash(Path(__file__).parent / name) for name in
                       ['sparse_experiment.py', 'sparse_model.py', 'sparse_data.py', 'data.py', 'evaluate.py', 'local_rule.py']},
                   'limitations': ['ordinary shared GELU MLP and Adam; not exact theorem architecture',
                                   'fixed finite dataset; random supervision position fixed per example',
                                   'only state/value loss during attention stage; all five slots during L1 pretraining']})
    write_json(args.output / 'config.json', config)
    pool = task.generate(max(args.samples, args.pool_samples), args.length,
                        torch.Generator().manual_seed(args.data_seed))
    sample = {k: v[:args.samples].clone() for k, v in pool.items()}
    del pool
    start = time.monotonic()
    teacher_seconds = 0.
    if args.stage == 'attention':
        teacher_clauses = None
        if args.label_source == 'teacher':
            if args.teacher is None:
                raise ValueError('Teacher labels require --teacher')
            teacher, teacher_state = load_model(args.teacher, args.device)
            if teacher_state['args']['task'] != args.task or teacher.config.nvars != task.nvars:
                raise ValueError('Teacher checkpoint and task differ')
            teacher.eval()
            teacher_clauses = sample['clauses'].clone()
            t0 = time.monotonic()
            for offset in range(0, args.samples, args.batch):
                c = sample['clauses'][offset:offset + args.batch].to(args.device)
                teacher_clauses[offset:offset + args.batch, args.length + 1:] = greedy_rollout(
                    teacher, c[:, :args.length + 1], args.length).cpu()
            teacher_seconds = time.monotonic() - t0
            del teacher
        supervision = build_supervision(sample, task, args.mode, args.label_source,
            teacher_clauses, args.noise_kind, args.noise_rate, seed=args.data_seed + 71)
        selected = select_supervised(supervision)
        query_vars = selected['inputs'][:, :, 3].gather(1, selected['query_positions'])
        histogram = torch.bincount(query_vars.flatten(), minlength=task.vocab_size)
        supervision['audit']['supervised_query_variable_histogram'] = histogram[1:task.nvars + 1].tolist()
        supervision['audit']['uncovered_query_variables'] = int(histogram[1:task.nvars + 1].eq(0).sum())
        supervision['audit']['invalid_query_variables'] = int(((query_vars < 1) | (query_vars > task.nvars)).sum())
        supervision['audit']['teacher_generation_seconds'] = teacher_seconds
        write_json(args.output / 'data-audit.json', supervision['audit'])
        torch.save({'supervision': supervision, 'sample': sample}, args.output / 'training-data.pt')
        inputs = selected['inputs'].to(args.device)
        queries = selected['query_positions'].to(args.device)
        targets = selected['targets'].to(args.device)
        labels_per_example = targets.shape[1]
    else:
        c = sample['clauses'].to(args.device)
        inputs = c[:, :2]
        queries = torch.ones(args.samples, 1, dtype=torch.long, device=args.device)
        targets = c[:, 2:]
        labels_per_example = 5
        torch.save(sample, args.output / 'training-data.pt')
        write_json(args.output / 'data-audit.json', {'examples': args.samples, 'labels_per_example': 5,
            'supervised_labels': args.samples * 5, 'source_sha256': tensor_hash(sample['clauses'])})
    optim = torch.optim.Adam([v for v in model.parameters() if v.requires_grad], lr=args.lr)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 711)
    permutation = torch.randperm(args.samples, generator=generator, device=args.device)
    cursor = 0
    labels_presented = 0
    examples_presented = 0
    gate_passed = False
    local_audit = None
    save_checkpoint(args.output / 'checkpoint-initial.pt', model, args, 0)
    for step in range(1, args.steps + 1):
        if cursor >= args.samples:
            permutation = torch.randperm(args.samples, generator=generator, device=args.device)
            cursor = 0
        ids = permutation[cursor:cursor + args.batch]
        cursor += len(ids)
        model.train()
        logits = model.forward_queries(inputs[ids], queries[ids])
        used_logits = logits if args.stage == 'local' else logits[:, :, 4]
        loss = F.cross_entropy(used_logits.reshape(-1, task.vocab_size), targets[ids].reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        labels_presented += len(ids) * labels_per_example
        examples_presented += len(ids)
        if step % args.eval_every == 0 or step == args.steps:
            check_lengths = [1] if args.stage == 'local' else [args.length, 2 * args.length]
            panel = evaluate(model, task, check_lengths, samples=512, rollout_samples=0,
                             seed=args.eval_seed - 100000)
            record = {'step': step, 'loss': loss.item(), 'elapsed_seconds': time.monotonic() - start,
                      'panel': compact(panel), 'labels_presented': labels_presented,
                      'Q43_diagonal_mean': model.Q43.diag().mean().item(),
                      'Q44_diagonal_mean': model.Q44.diag().mean().item()}
            if args.stage == 'local' and step >= args.local_min_steps:
                metrics = panel[0]['teacher_forced']['metrics']
                if min(metrics['value_accuracy'], metrics['variable_accuracy']) >= args.local_gate:
                    local_audit = evaluate_local_rule(model, task, n_variable_pairs=16,
                        seed=args.eval_seed + 101, batch_size=256)
                    gate_passed = min(local_audit['value_accuracy'], local_audit['variable_accuracy']) >= args.local_gate
                    record['local_gate_passed'] = gate_passed
                    record['local_value_accuracy'] = local_audit['value_accuracy']
                    record['local_variable_accuracy'] = local_audit['variable_accuracy']
            with (args.output / 'history.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            write_json(args.output / 'progress.json', record)
            print(json.dumps(record), flush=True)
            save_checkpoint(args.output / f'checkpoint-{step:05d}.pt', model, args, step)
            if args.stage == 'local' and gate_passed and args.local_gate_protocol == 'legacy_oracle':
                break
    frozen_after = model.frozen_state_hash()
    if frozen_after != frozen_before:
        raise RuntimeError('Frozen parameters changed')
    save_checkpoint(args.output / 'checkpoint-final.pt', model, args, step)
    if args.stage == 'local':
        if local_audit is None or args.local_gate_protocol == 'position_local':
            local_audit = evaluate_local_rule(model, task, n_variable_pairs=16,
                seed=args.eval_seed + 101, batch_size=256)
        write_json(args.output / 'local-rule.json', local_audit)
        rows = evaluate(model, task, [1, 4, 16, 64], args.eval_samples, 0, args.eval_seed, oracle=True)
        oracle_ok = all(min(r['teacher_forced']['metrics']['value_accuracy'],
            r['teacher_forced']['metrics']['variable_accuracy']) >= args.local_gate for r in rows)
        if args.local_gate_protocol == "legacy_oracle":
            gate_passed = gate_passed and oracle_ok
        else:
            # Admission uses the final local L1 panel + exhaustive action/state check.
            # Long oracle accuracy is diagnostic only: no OOD-based seed selection.
            metrics = panel[0]["teacher_forced"]["metrics"]
            gate_passed = min(metrics["value_accuracy"], metrics["variable_accuracy"],
                              local_audit["value_accuracy"], local_audit["variable_accuracy"]) >= args.local_gate
    else:
        rows = evaluate(model, task, args.eval_lengths, args.eval_samples, args.rollout_samples, args.eval_seed)
    write_json(args.output / 'evaluation.json', {'complete': True, 'evaluations': rows,
        'checkpoint_sha256': file_hash(args.output / 'checkpoint-final.pt'),
        'panel_seed': args.eval_seed, 'local_gate_passed': gate_passed if args.stage == 'local' else None})
    write_json(args.output / 'complete.json', {'complete': True, 'stage': args.stage, 'steps': step,
        'seconds': time.monotonic() - start, 'examples_presented': examples_presented,
        'labels_presented': labels_presented, 'unique_labels': args.samples * labels_per_example,
        'frozen_sha256_before': frozen_before, 'frozen_sha256_after': frozen_after,
        'local_gate_passed': gate_passed if args.stage == 'local' else None,
        'checkpoint': str((args.output / 'checkpoint-final.pt').resolve())})
    print(json.dumps({'complete': str(args.output), 'results': compact(rows), 'local_gate_passed': gate_passed}), flush=True)


if __name__ == '__main__':
    main()
