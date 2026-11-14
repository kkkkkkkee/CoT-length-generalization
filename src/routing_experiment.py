"""Attention-only finite-data experiments with explicit routing constraints.

Every treatment is evaluated after conversion and before training, preventing
an initialization/projection gain from being attributed to subsequent learning.
The frozen local computation and the original round-one runner are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from data import Task
from evaluate import greedy_rollout
from routing_model import convert_model, load_routing_model
from routing_augmentation import rename_variables
from sparse_data import build_supervision, select_supervised, tensor_hash
from sparse_experiment import compact, evaluate, file_hash, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['attention'], default='attention')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--routing-mode', choices=['dense', 'row_tied', 'diagonal'], default='dense')
    parser.add_argument('--position-mode', choices=['none', 'learned_abs', 'sinusoidal', 'relative'], default='none')
    parser.add_argument('--eval-predicate-shuffle', action='store_true')
    parser.add_argument('--replicate', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', default='S5')
    parser.add_argument('--nvars', type=int, default=128)
    parser.add_argument('--length', type=int, default=4)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--pool-samples', type=int, default=16384)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.03)
    parser.add_argument('--row-tie-lambda', type=float, default=0.,
                        help='L2 penalty on off-diagonal deviations from each Q row mean')
    parser.add_argument('--q-l2-lambda', type=float, default=0.,
                        help='Matched scale control: mean squared entries of the two Q blocks')
    parser.add_argument('--rename-variables', action='store_true',
                        help='Draw an independent variable-ID permutation per example presentation')
    parser.add_argument('--augmentation-seed', type=int, default=6120001)
    parser.add_argument('--mode', choices=['second', 'random', 'all'], default='second')
    parser.add_argument('--label-source', choices=['truth', 'teacher'], default='truth')
    parser.add_argument('--noise-kind', choices=['none', 'target', 'prefix_inconsistent', 'prefix_consistent'], default='none')
    parser.add_argument('--noise-rate', type=float, default=0.)
    parser.add_argument('--data-seed', type=int, default=2120001)
    parser.add_argument('--noise-seed', type=int, default=2220001,
                        help='Independent supervision RNG; fixes random positions and corruption draws')
    parser.add_argument('--eval-seed', type=int, default=2340001)
    parser.add_argument('--validation-seed', type=int, default=2240001)
    parser.add_argument('--eval-samples', type=int, default=1024)
    parser.add_argument('--validation-samples', type=int, default=512)
    parser.add_argument('--rollout-samples', type=int, default=128)
    parser.add_argument('--eval-lengths', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--eval-batch', type=int, default=64)
    parser.add_argument('--budget-label', default='equal_examples_and_optimizer_steps')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args(argv)


def parameter_hash(parameters):
    digest = hashlib.sha256()
    for name, value in parameters:
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}:{tuple(value.shape)}:{value.dtype}'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def q_statistics(model):
    result = {}
    for name in ('Q43', 'Q44'):
        q = getattr(model, name).detach().double()
        n = q.shape[0]
        diagonal = q.diagonal()
        mask = ~torch.eye(n, device=q.device, dtype=torch.bool)
        off = q[mask]
        row_off = q.masked_fill(~mask, -torch.inf).amax(-1)
        row_mean = (q.sum(-1) - diagonal) / max(n - 1, 1)
        row_residual = (q - row_mean[:, None])[mask]
        result[name] = {
            'diagonal_mean': diagonal.mean().item(),
            'diagonal_std': diagonal.std(unbiased=False).item(),
            'diagonal_min': diagonal.min().item(),
            'diagonal_max': diagonal.max().item(),
            'offdiagonal_mean': off.mean().item() if off.numel() else 0.,
            'offdiagonal_std': off.std(unbiased=False).item() if off.numel() else 0.,
            'offdiagonal_row_residual_rms': row_residual.square().mean().sqrt().item() if off.numel() else 0.,
            'rows_with_offdiagonal_above_diagonal': int((row_off > diagonal).sum()) if n > 1 else 0,
            'diagonal_minus_row_background_mean': (diagonal - row_mean).mean().item(),
        }
    return result


def row_tie_penalty(model):
    """Return the mean squared off-diagonal row residual over Q43 and Q44.

    This is a soft version of ``row_tied``.  It is evaluated on the effective
    matrices, so its mathematical zero set includes row-tied matrices.  Finite
    precision can leave a tiny residual when computing the row mean.
    """
    n = model.config.nvars
    if n < 2:
        return next(model.parameters()).new_zeros(())
    mask = ~torch.eye(n, device=next(model.parameters()).device, dtype=torch.bool)
    penalties = []
    for name in ('Q43', 'Q44'):
        q = getattr(model, name)
        off = q.masked_fill(~mask, 0.)
        row_mean = off.sum(-1) / (n - 1)
        residual = (q - row_mean[:, None]).masked_select(mask)
        penalties.append(residual.square().mean())
    return torch.stack(penalties).mean()


def validate_checkpoint(model, state, task, checkpoint, admission=False):
    saved_args = state.get('args', {})
    if (saved_args.get('task') != task.name or model.config.nvars != task.nvars
            or model.config.vocab_size != task.vocab_size):
        raise ValueError('Checkpoint task or vocabulary does not match the requested task')
    if admission:
        complete_path = Path(checkpoint).parent / 'complete.json'
        if not complete_path.exists():
            raise ValueError('Initialization run has no completion record')
        complete = json.loads(complete_path.read_text())
        if not complete.get('complete'):
            raise ValueError('Initialization run is incomplete')
        if state.get('stage') not in {'local', 'attention'}:
            raise ValueError('Initialization stage must be local or attention')
        if state.get('stage') == 'local' and not complete.get('local_gate_passed'):
            raise ValueError('Local initialization failed its admission gate')


def save_checkpoint(path, model, args, step):
    torch.save({'model': model.state_dict(), 'model_config': model.config_dict(),
                'routing_mode': args.routing_mode, 'stage': 'attention',
                'args': vars(args), 'step': step}, path)


def evaluation_document(model, task, args, checkpoint, initial=False):
    rows = evaluate(model, task, args.eval_lengths, args.eval_samples,
                    args.rollout_samples, args.eval_seed, batch_size=args.eval_batch)
    shuffled = (evaluate(model, task, args.eval_lengths, args.eval_samples,
                         args.rollout_samples, args.eval_seed, batch_size=args.eval_batch,
                         predicate_shuffle=True) if args.eval_predicate_shuffle else None)
    return {'complete': True, 'evaluations': rows, 'shuffled_evaluations': shuffled,
            'checkpoint_sha256': file_hash(checkpoint), 'checkpoint': str(checkpoint.resolve()),
            'panel_seed': args.eval_seed, 'routing_mode': args.routing_mode,
            'phase': 'after_conversion_before_training' if initial else 'after_training',
            'routing_statistics': q_statistics(model)}


def run_experiment(args):
    if args.steps < 0 or min(args.samples, args.batch, args.pool_samples,
            args.eval_samples, args.eval_batch, args.eval_every, args.validation_samples) < 1:
        raise ValueError('Positive dataset/evaluation sizes and nonnegative steps required')
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Learning rate must be positive and finite')
    if not math.isfinite(args.row_tie_lambda) or args.row_tie_lambda < 0:
        raise ValueError('row_tie_lambda must be finite and nonnegative')
    if not math.isfinite(args.q_l2_lambda) or args.q_l2_lambda < 0:
        raise ValueError('q_l2_lambda must be finite and nonnegative')
    if args.rollout_samples < 0 or args.length < 2 or 2 * args.length >= args.nvars:
        raise ValueError('Invalid rollout size or training/doubled length')
    if any(length < 1 or length >= args.nvars for length in args.eval_lengths):
        raise ValueError('Evaluation lengths must lie in [1,nvars)')
    if args.eval_seed == args.validation_seed:
        raise ValueError('Validation and final evaluation panels must be distinct')
    if args.label_source == 'teacher' and args.teacher is None:
        raise ValueError('Teacher labels require --teacher')
    if args.rename_variables and args.label_source != 'truth':
        raise ValueError('Variable renaming currently requires truth labels; malformed teacher tokens need joint input/target remapping')
    if (args.output / 'config.json').exists():
        raise RuntimeError(f'Refusing to overwrite experiment {args.output}')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    task = Task(args.task, args.nvars)
    args.task = task.name
    source, prior = load_routing_model(args.init, args.device)
    validate_checkpoint(source, prior, task, args.init, admission=True)
    source_ff_hash = parameter_hash(source.ff.named_parameters())
    model = convert_model(source, args.routing_mode, args.position_mode).to(args.device)
    model.configure_stage('attention')
    if parameter_hash(model.ff.named_parameters()) != source_ff_hash:
        raise RuntimeError('Routing conversion changed the pretrained FFN')
    frozen_before = model.frozen_state_hash()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    config.update({'model': model.config_dict(), 'stage': 'attention',
        'initialization_routing_mode': prior.get('routing_mode', 'dense'),
        'trainable_parameters': [name for name, _ in trainable],
        'routing_degrees_of_freedom': sum(value.numel() for _, value in trainable),
        'frozen_sha256_before': frozen_before, 'ff_sha256_before': source_ff_hash,
        'init_sha256': file_hash(args.init),
        'teacher_sha256': file_hash(args.teacher) if args.teacher else None,
        'teacher_used': args.label_source == 'teacher',
        'source_sha256': {name: file_hash(Path(__file__).parent / name) for name in
            ['routing_experiment.py', 'routing_model.py', 'routing_augmentation.py', 'sparse_experiment.py',
             'sparse_model.py', 'sparse_data.py', 'data.py', 'evaluate.py']},
        'limitations': ['routing constraints inject a known variable-matching inductive bias',
            'ordinary shared GELU MLP and Adam; not exact theorem architecture',
            'random supervision positions fixed per example',
            'initial evaluation explicitly measures conversion/projection before training',
            'noise_seed also controls fixed random supervision positions, independently of data_seed']})
    config['regularization_definition'] = 'lambda * mean_over_two_Q_blocks(mean_over_offdiagonal_entries((Q_ij - row_offdiagonal_mean_i)^2))'
    config['q_l2_definition'] = 'q_l2_lambda * mean_over_two_Q_blocks(mean_over_all_entries(Q_ij^2)); explicit loss penalty, not AdamW decay'
    config['augmentation_policy'] = ('fresh independent variable permutation per example presentation, approximately uniform via float64 random-key sorting; action/state tokens and value labels unchanged; adds variable-ID exposures without new abstract action/state traces'
                                     if args.rename_variables else 'none')
    write_json(args.output / 'config.json', config)
    start = time.monotonic()
    initial_checkpoint = args.output / 'checkpoint-initial.pt'
    save_checkpoint(initial_checkpoint, model, args, 0)
    initial_result = evaluation_document(model, task, args, initial_checkpoint, initial=True)
    write_json(args.output / 'initial-evaluation.json', initial_result)
    print(json.dumps({'phase': 'initial', 'output': str(args.output),
                      'results': compact(initial_result['evaluations'])}), flush=True)
    pool = task.generate(max(args.samples, args.pool_samples), args.length,
                        torch.Generator().manual_seed(args.data_seed))
    sample = {key: value[:args.samples].clone() for key, value in pool.items()}
    del pool
    teacher_clauses = None
    teacher_seconds = 0.
    teacher_mode = None
    if args.label_source == 'teacher':
        teacher, teacher_state = load_routing_model(args.teacher, args.device)
        validate_checkpoint(teacher, teacher_state, task, args.teacher)
        teacher_mode = teacher_state.get('routing_mode', 'dense')
        teacher.eval()
        teacher_clauses = sample['clauses'].clone()
        t0 = time.monotonic()
        for offset in range(0, args.samples, args.batch):
            clauses = sample['clauses'][offset:offset + args.batch].to(args.device)
            generated = greedy_rollout(teacher, clauses[:, :args.length + 1], args.length)
            teacher_clauses[offset:offset + args.batch, args.length + 1:] = generated.cpu()
        teacher_seconds = time.monotonic() - t0
        del teacher
    supervision = build_supervision(sample, task, args.mode, args.label_source,
        teacher_clauses, args.noise_kind, args.noise_rate, seed=args.noise_seed)
    supervision['audit'].update({'data_seed': args.data_seed, 'noise_seed': args.noise_seed,
        'replicate': args.replicate, 'teacher_generation_seconds': teacher_seconds,
        'teacher_routing_mode': teacher_mode,
        'teacher_generation_answers': args.samples * args.length if teacher_clauses is not None else 0,
        'teacher_generation_batches': math.ceil(args.samples / args.batch) if teacher_clauses is not None else 0})
    write_json(args.output / 'data-audit.json', supervision['audit'])
    torch.save({'supervision': supervision, 'sample': sample}, args.output / 'training-data.pt')
    selected = select_supervised(supervision)
    inputs, queries, targets = (selected[name].to(args.device) for name in
                                ['inputs', 'query_positions', 'targets'])
    labels_per_example = targets.shape[1]
    optim = torch.optim.Adam([value for _, value in trainable], lr=args.lr)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 711)
    augmentation_generator = torch.Generator(device=args.device).manual_seed(args.augmentation_seed)
    presented_query_counts = torch.zeros(task.nvars, dtype=torch.long, device=args.device)
    permutation = torch.randperm(args.samples, generator=generator, device=args.device)
    cursor = 0
    labels_presented = 0
    examples_presented = 0
    last_step = 0
    training_start = time.monotonic()
    for step in range(1, args.steps + 1):
        last_step = step
        if cursor >= args.samples:
            permutation = torch.randperm(args.samples, generator=generator, device=args.device)
            cursor = 0
        ids = permutation[cursor:cursor + args.batch]
        cursor += len(ids)
        model.train()
        batch_inputs = inputs[ids]
        if args.rename_variables:
            batch_inputs = rename_variables(batch_inputs, task.nvars, augmentation_generator)
        presented_queries = batch_inputs[:, :, 3].gather(1, queries[ids])
        valid_queries = presented_queries[(presented_queries >= 1) & (presented_queries <= task.nvars)]
        presented_query_counts += torch.bincount(valid_queries - 1, minlength=task.nvars)
        logits = model.forward_queries(batch_inputs, queries[ids])[:, :, 4]
        cross_entropy = F.cross_entropy(logits.reshape(-1, task.vocab_size), targets[ids].reshape(-1))
        penalty = row_tie_penalty(model)
        q_l2 = (model.Q43.square().mean() + model.Q44.square().mean()) / 2
        regularization = args.row_tie_lambda * penalty + args.q_l2_lambda * q_l2
        loss = cross_entropy + regularization
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        labels_presented += len(ids) * labels_per_example
        examples_presented += len(ids)
        if step % args.eval_every == 0 or step == args.steps:
            panel = evaluate(model, task, [args.length, 2 * args.length],
                samples=args.validation_samples, rollout_samples=0,
                seed=args.validation_seed, batch_size=args.eval_batch)
            record = {'step': step, 'loss': loss.item(),
                'cross_entropy': cross_entropy.item(),
                'row_tie_penalty': penalty.item(),
                'regularization_loss': regularization.item(),
                'row_tie_lambda': args.row_tie_lambda,
                'q_l2_penalty': q_l2.item(),
                'q_l2_lambda': args.q_l2_lambda,
                'elapsed_seconds': time.monotonic() - start,
                'panel': compact(panel), 'validation_evaluations': panel,
                'validation_seed': args.validation_seed,
                'labels_presented': labels_presented,
                'examples_presented': examples_presented,
                'presented_query_variable_counts': presented_query_counts.cpu().tolist(),
                'routing_statistics': q_statistics(model)}
            with (args.output / 'history.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            write_json(args.output / 'progress.json', record)
            print(json.dumps({key: record[key] for key in
                ['step', 'loss', 'elapsed_seconds', 'panel', 'labels_presented']}), flush=True)
            save_checkpoint(args.output / f'checkpoint-{step:05d}.pt', model, args, step)
    training_seconds = time.monotonic() - training_start
    frozen_after = model.frozen_state_hash()
    ff_after = parameter_hash(model.ff.named_parameters())
    if frozen_after != frozen_before or ff_after != source_ff_hash:
        raise RuntimeError('Frozen local computation changed')
    final_checkpoint = args.output / 'checkpoint-final.pt'
    save_checkpoint(final_checkpoint, model, args, last_step)
    final_result = evaluation_document(model, task, args, final_checkpoint)
    write_json(args.output / 'evaluation.json', final_result)
    complete = {'complete': True, 'stage': 'attention', 'routing_mode': args.routing_mode,
        'steps': last_step, 'seconds': time.monotonic() - start,
        'training_and_validation_seconds': training_seconds,
        'teacher_generation_seconds': teacher_seconds,
        'examples_presented': examples_presented, 'labels_presented': labels_presented,
        'presented_query_variable_counts': presented_query_counts.cpu().tolist(),
        'rename_variables': args.rename_variables,
        'augmentation_seed': args.augmentation_seed,
        'unique_labels': args.samples * labels_per_example,
        'unique_labels_definition': 'distinct training example/position value labels, not distinct semantic values',
        'frozen_sha256_before': frozen_before, 'frozen_sha256_after': frozen_after,
        'ff_sha256_before': source_ff_hash, 'ff_sha256_after': ff_after,
        'checkpoint': str(final_checkpoint.resolve()),
        'initial_checkpoint_sha256': file_hash(initial_checkpoint),
        'final_checkpoint_sha256': file_hash(final_checkpoint)}
    write_json(args.output / 'complete.json', complete)
    print(json.dumps({'complete': str(args.output),
                      'results': compact(final_result['evaluations'])}), flush=True)
    return complete


def main():
    run_experiment(parse_args())


if __name__ == '__main__':
    main()
