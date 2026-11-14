"""Variable-identity augmentation without changing a LEGO trace's semantics."""
from __future__ import annotations

import torch


def rename_variables(inputs: torch.Tensor, nvars: int,
                     generator: torch.Generator) -> torch.Tensor:
    """Apply one independently sampled variable permutation to each example.

    The leading dimension indexes examples; all remaining dimensions are token
    positions. Each token in ``1..nvars`` is replaced consistently within its
    example. Zero and tokens outside that interval are copied unchanged. The
    input is never modified. ``generator`` must use the input tensor's device.
    Batched float64 random keys make the permutation approximately uniform:
    the only departure from exact uniformity is the negligible probability of
    equal finite-precision keys, as in ``Task.generate``. Every row is still an
    exact bijection, including when keys tie.
    """
    if inputs.ndim < 1:
        raise ValueError('inputs must have a leading example dimension')
    if inputs.dtype not in (torch.uint8, torch.int8, torch.int16,
                            torch.int32, torch.int64):
        raise TypeError('inputs must contain integer token IDs')
    if not isinstance(nvars, int) or isinstance(nvars, bool) or nvars < 1:
        raise ValueError('nvars must be a positive integer')
    if nvars > torch.iinfo(inputs.dtype).max:
        raise ValueError('nvars cannot be represented by the input dtype')
    if inputs.shape[0] == 0 or inputs.numel() == 0:
        return inputs.clone()
    keys = torch.rand((inputs.shape[0], nvars), generator=generator,
                      device=inputs.device, dtype=torch.float64)
    permutations = keys.argsort(dim=-1) + 1
    flattened = inputs.reshape(inputs.shape[0], -1)
    variable_mask = (flattened >= 1) & (flattened <= nvars)
    indices = flattened.clamp(1, nvars).to(torch.long) - 1
    renamed = permutations.gather(1, indices).to(inputs.dtype)
    return torch.where(variable_mask, renamed, flattened).reshape_as(inputs)
