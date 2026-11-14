"""Synthetic LEGO state-tracking data with clause-level tokenization.

Each predicate is ``(target_variable, action, source_variable, 0, 0)``;
each answer is ``(0, 0, 0, variable, value)``.  Variables are sampled
without replacement, and action indices and the initial state are uniform.
The returned action/state arrays use zero-based indices, whereas ``clauses``
and ``variables`` contain actual vocabulary token IDs.
"""

from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, Optional, Union

import torch


Device = Union[str, torch.device]
Sample = Dict[str, torch.Tensor]


def _sampling_device(generator: Optional[torch.Generator], device: Device) -> torch.device:
    """Sample beside the generator, then transfer the finished batch if needed."""
    return torch.device(generator.device if generator is not None else device)


@dataclass
class Task:
    """Task specification and reproducible data generator.

    ``C5``/``C6`` use cyclic shifts. ``S5`` has the 120 permutations in
    lexicographic order. ``aliasC5`` has 24 distinct action tokens per shift:
    action ``a`` acts as shift ``a // 24``; ``a % 24`` is an inert alias.
    Thus ``aliasC5`` and ``S5`` both have 120 actions and five states.
    """

    name: str
    nvars: int = 256
    nstates: int = field(init=False)
    nactions: int = field(init=False)
    vocab_size: int = field(init=False)
    action_offset: int = field(init=False)
    value_offset: int = field(init=False)
    action_table: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        canonical = {"c5": "C5", "c6": "C6", "s5": "S5", "aliasc5": "aliasC5"}
        if self.name.lower() not in canonical:
            raise ValueError(f"Unknown task {self.name!r}; choose C5, C6, S5, or aliasC5")
        self.name = canonical[self.name.lower()]
        if not isinstance(self.nvars, int) or self.nvars < 2:
            raise ValueError("nvars must be an integer >= 2")
        self.nstates = 6 if self.name == "C6" else 5
        if self.name == "S5":
            self.action_table = torch.tensor(list(permutations(range(5))), dtype=torch.long)
        else:
            shifts = torch.arange(self.nstates)
            if self.name == "aliasC5":
                shifts = shifts.repeat_interleave(24)
            self.action_table = (shifts[:, None] + torch.arange(self.nstates)[None, :]) % self.nstates
        self.nactions = self.action_table.shape[0]
        self.action_offset = self.nvars + 1
        self.value_offset = self.action_offset + self.nactions
        self.vocab_size = self.value_offset + self.nstates

    def generate(
        self,
        batch: int,
        length: int,
        generator: Optional[torch.Generator] = None,
        device: Device = "cpu",
    ) -> Sample:
        """Return a batch of complete, semantically valid LEGO traces.

        Shapes: clauses ``(B, 2L+1, 5)``, variables ``(B, L+1)``,
        actions ``(B, L)``, and states ``(B, L+1)``. All tensors are
        ``torch.long`` on ``device``. A CPU generator can also be used when
        requesting a CUDA output, making the sampled examples device-independent.
        """
        if not isinstance(batch, int) or batch < 1:
            raise ValueError("batch must be a positive integer")
        if not isinstance(length, int) or not 1 <= length < self.nvars:
            raise ValueError(f"length must satisfy 1 <= length < nvars ({self.nvars})")
        sample_device = _sampling_device(generator, device)
        # Sorting continuous random keys gives a uniformly random permutation.
        # Float64 avoids a material rate of ties for larger variable vocabularies.
        keys = torch.rand((batch, self.nvars), generator=generator, device=sample_device, dtype=torch.float64)
        variables = keys.argsort(dim=1)[:, :length + 1] + 1
        actions = torch.randint(self.nactions, (batch, length), generator=generator, device=sample_device)
        states = torch.empty((batch, length + 1), dtype=torch.long, device=sample_device)
        states[:, 0] = torch.randint(self.nstates, (batch,), generator=generator, device=sample_device)
        table = self.action_table.to(sample_device)
        for step in range(length):
            states[:, step + 1] = table[actions[:, step], states[:, step]]

        clauses = torch.zeros((batch, 2 * length + 1, 5), dtype=torch.long, device=sample_device)
        clauses[:, :length, 0] = variables[:, 1:]
        clauses[:, :length, 1] = actions + self.action_offset
        clauses[:, :length, 2] = variables[:, :-1]
        clauses[:, length:, 3] = variables
        clauses[:, length:, 4] = states + self.value_offset
        return {key: value.to(device) for key, value in {
            "clauses": clauses, "variables": variables, "actions": actions, "states": states,
        }.items()}


def permute_predicates(sample: Sample, generator: Optional[torch.Generator] = None) -> Sample:
    """Randomly reorder predicates, leaving the logical trace unchanged.

    Returns a new dictionary with a new clauses tensor. ``predicate_positions``
    has shape ``(B,L)``: entry ``[b,t]`` is the zero-based clause position of
    the predicate needed for logical transition ``t -> t+1``. This mapping
    remains correct if an already-permuted sample is passed in again.
    Other tensors are shared with the input and are never modified.
    """
    clauses = sample["clauses"]
    batch, length = sample["actions"].shape
    if tuple(clauses.shape) != (batch, 2 * length + 1, 5):
        raise ValueError("clauses shape must be (batch, 2*length+1, 5)")
    sample_device = _sampling_device(generator, clauses.device)
    order = torch.rand((batch, length), generator=generator, device=sample_device, dtype=torch.float64).argsort(dim=1)
    order = order.to(clauses.device)
    reordered = clauses.clone()
    reordered[:, :length] = clauses[:, :length].gather(1, order[:, :, None].expand(-1, -1, 5))
    old_positions = sample.get("predicate_positions")
    if old_positions is None:
        old_positions = torch.arange(length, device=clauses.device).expand(batch, -1)
    positions = order.argsort(dim=1).gather(1, old_positions)
    return {**sample, "clauses": reordered, "predicate_positions": positions}
