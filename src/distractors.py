"""Controlled context-volume interventions for LEGO reasoning.

These examples are an out-of-distribution diagnostic, not the paper's LEGO
distribution: all added predicates reuse two variables outside the solution
trace. Neither variable receives an initial value, so the distractor component
is disconnected from every valid reasoning step.

``compatible`` and ``incompatible`` condition actions on the TRUE first-step
transition. They are oracle-designed mechanistic interventions, not label-free
evaluation or training data. Compatibility is only defined for query zero.
"""

from typing import Optional

import torch

try:
    from .data import Sample, Task, _sampling_device
except ImportError:  # Support ``python src/<entrypoint>.py``.
    from data import Sample, Task, _sampling_device


def add_distractors(
    sample: Sample,
    task: Task,
    count: int,
    generator: Optional[torch.Generator] = None,
    kind: str = "random",
) -> Sample:
    """Append ``count`` disconnected predicates before the answer block.

    The logical CoT length and true states are unchanged. Returns a new sample
    including ``query_positions[B,L]`` for predicting the L next answers and
    ``target_predicate_positions[B,L]`` for their relevant predicate clauses.
    ``distractor_positions[B,M]`` records all added predicates, including any
    from earlier calls. Existing predicate permutations remain valid.

    At least two variable IDs must be unused by the trace when count > 0.
    Reusing these two off-path variables permits arbitrarily many distractors;
    this deliberately relaxes the original distribution's distinct-variable
    requirement. Uniform random actions are used unless ``kind`` is set to
    ``compatible``/``incompatible`` relative to true ``states[:,0:2]``.
    The input sample is never modified; unchanged tensors are shared.
    """
    if not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if kind not in {"random", "compatible", "incompatible"}:
        raise ValueError("kind must be random, compatible, or incompatible")
    clauses = sample["clauses"]
    batch, length = sample["actions"].shape
    predicate_count = clauses.shape[1] - length - 1
    if predicate_count < length or clauses.shape[0] != batch or clauses.shape[-1] != 5:
        raise ValueError("sample must contain all predicates followed by L+1 answer clauses")
    device = clauses.device
    sampling_device = _sampling_device(generator, device)
    target_positions = sample.get("target_predicate_positions", sample.get("predicate_positions"))
    if target_positions is None:
        target_positions = torch.arange(length, device=device).expand(batch, -1)
    query_positions = torch.arange(predicate_count + count, predicate_count + count + length, device=device).expand(batch, -1)

    if count:
        if task.nvars - (length + 1) < 2:
            raise ValueError("two variable IDs outside the logical trace are required")
        # Randomly choose two of the variables outside the connected trace.
        keys = torch.rand((batch, task.nvars), generator=generator, device=sampling_device, dtype=torch.float64)
        keys.scatter_(1, sample["variables"].to(sampling_device) - 1, float("inf"))
        unused = keys.argsort(dim=1)[:, :2] + 1
        if kind == "random":
            actions = torch.randint(task.nactions, (batch, count), generator=generator, device=sampling_device)
        else:
            table = task.action_table.to(sampling_device)
            initial, target = sample["states"][:, 0].to(sampling_device), sample["states"][:, 1].to(sampling_device)
            compatible = table[:, initial].transpose(0, 1).eq(target[:, None])
            eligible = compatible if kind == "compatible" else ~compatible
            if not bool(eligible.any(dim=1).all()):
                raise ValueError(f"no {kind} actions exist for at least one sample")
            actions = torch.multinomial(eligible.float(), count, replacement=True, generator=generator)
        extra = torch.zeros((batch, count, 5), dtype=torch.long, device=sampling_device)
        extra[:, :, 0] = unused[:, 1, None]
        extra[:, :, 1] = actions + task.action_offset
        extra[:, :, 2] = unused[:, 0, None]
        extended = torch.cat((clauses[:, :predicate_count], extra.to(device), clauses[:, predicate_count:]), dim=1)
    else:
        extended = clauses

    new_positions = torch.arange(predicate_count, predicate_count + count, device=device).expand(batch, -1)
    old_positions = sample.get("distractor_positions")
    all_positions = torch.cat((old_positions, new_positions), dim=1) if old_positions is not None else new_positions
    return {**sample, "clauses": extended, "query_positions": query_positions,
            "target_predicate_positions": target_positions, "distractor_positions": all_positions}
