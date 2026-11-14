"""Low-dimensional routing controls with the same frozen local computation.

``row_tied`` retains an independent matching score d_i and a common nonmatching
score b_i for each query variable in each of Q43/Q44. ``diagonal`` retains only
d_i and fixes nonmatching scores to zero. These are structural priors about
variable identity, not the dense-matrix hypothesis class in the paper.

The matrices exposed as Q43/Q44 are effective tensors, not parameters that may
be edited in-place. Optimizers operate on the underlying vectors directly;
there is no post-update projection or projected Adam state.
"""
from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

try:
    from .sparse_model import SparseClauseTransformer, SparseModelConfig
except ImportError:
    from sparse_model import SparseClauseTransformer, SparseModelConfig


ROUTING_MODES = ("dense", "row_tied", "diagonal")


class RoutingClauseTransformer(SparseClauseTransformer):
    """Reuse causal computation and decoding with genuinely tied Q parameters."""

    def __init__(self, config: SparseModelConfig, routing_mode="row_tied"):
        if routing_mode not in {"row_tied", "diagonal"}:
            raise ValueError("RoutingClauseTransformer requires row_tied or diagonal; use convert_model for dense")
        # Do not call SparseClauseTransformer.__init__: it registers Q43/Q44 as
        # parameters, whereas this subclass intentionally exposes properties.
        nn.Module.__init__(self)
        self.config = config
        self.routing_mode = routing_mode
        self.width = 5 * config.vocab_size
        for suffix in ("43", "44"):
            self.register_parameter(f"diagonal{suffix}", nn.Parameter(torch.zeros(config.nvars)))
            background = (nn.Parameter(torch.zeros(config.nvars))
                          if routing_mode == "row_tied" and config.nvars > 1 else None)
            self.register_parameter(f"background{suffix}", background)
        self.register_buffer("_diagonal_mask", torch.eye(config.nvars, dtype=torch.bool), persistent=False)
        self.ff = nn.Sequential(
            nn.Linear(self.width, config.hidden),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Linear(config.hidden, self.width),
        )
        self._init_position()
        self.configure_stage("attention")

    def _matrix(self, suffix):
        diagonal = getattr(self, f"diagonal{suffix}")
        background = getattr(self, f"background{suffix}")
        if background is None:
            return torch.diag(diagonal)
        # where preserves the diagonal exactly and sends no diagonal gradient
        # into the background parameter (unlike inexact subtract/add formulas).
        return torch.where(self._diagonal_mask, diagonal[:, None], background[:, None])

    @property
    def Q43(self):
        return self._matrix("43")

    @property
    def Q44(self):
        return self._matrix("44")

    def routing_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("ff."):
                yield parameter

    def configure_stage(self, stage="attention"):
        if stage not in {"local", "attention"}:
            raise ValueError("stage must be local or attention")
        for parameter in self.routing_parameters():
            parameter.requires_grad_(stage == "attention")
        for parameter in self.ff.parameters():
            parameter.requires_grad_(stage == "local")
        if isinstance(self.position_embedding, nn.Parameter):
            parameter_trainable = True
            self.position_embedding.requires_grad_(parameter_trainable)
        for parameter in self.parameters():
            parameter.grad = None
        self.training_stage = stage
        return self

    def routing_parameter_count(self):
        return sum(p.numel() for p in self.routing_parameters())

    def effective_routing_parameters(self):
        return routing_parameter_counts(self)


def routing_parameter_counts(model):
    """Parameter coefficients versus matrix entries, not identifiable DOF.

    In well-formed contexts softmax row-shift symmetries can make some
    coefficient combinations observationally equivalent. Counts here describe
    the actual optimized parameterization, without quotienting those gauges.
    """
    mode = getattr(model, "routing_mode", "dense")
    if mode not in ROUTING_MODES:
        raise ValueError("Unknown routing mode")
    n = model.config.nvars
    independent = 2 * n * n if mode == "dense" else (4 * n if mode == "row_tied" and n > 1 else 2 * n)
    actual = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("ff.") and name != "position_embedding")
    if actual != independent:
        raise ValueError(f"Routing parameter count mismatch: {actual} != {independent}")
    return {"mode": mode, "independent_coefficients": independent,
            "effective_matrix_entries": 2 * n * n,
            "trainable_coefficients": sum(p.numel() for name, p in model.named_parameters()
                                            if not name.startswith("ff.") and name != "position_embedding" and p.requires_grad),
            "position_coefficients": sum(p.numel() for name, p in model.named_parameters()
                                         if name == "position_embedding")}


def routing_parameter_count(model):
    return routing_parameter_counts(model)["independent_coefficients"]


def effective_routing_parameters(model):
    return routing_parameter_counts(model)


@torch.no_grad()
def convert_model(model, mode="row_tied", position_mode=None):
    """Copy local computation and initialize the desired routing hypothesis class.

    Dense preserves both full Q matrices. Row-tied retains each diagonal and
    replaces each row's nonmatching entries with their mean. Diagonal retains
    only the diagonal. The source is never mutated; returned FFN is frozen and
    routing is trainable. No oracle rule positions or output labels are used.
    """
    if mode not in ROUTING_MODES:
        raise ValueError(f"mode must be one of {ROUTING_MODES}")
    config = SparseModelConfig(**asdict(model.config))
    if position_mode is not None:
        config.position_mode = position_mode
    reference = next(model.parameters())
    # Constructor initialization will be overwritten; keep caller RNG intact.
    with torch.random.fork_rng(devices=[]):
        target = (SparseClauseTransformer(config) if mode == "dense"
                  else RoutingClauseTransformer(config, mode))
    target = target.to(device=reference.device, dtype=reference.dtype)
    target.routing_mode = mode
    target.ff.load_state_dict(model.ff.state_dict())
    if target.position_embedding is not None and model.position_embedding is not None:
        if target.config.position_mode != model.config.position_mode:
            raise ValueError("Cannot silently convert between position mechanisms")
        target.position_embedding.copy_(model.position_embedding)
    for suffix in ("43", "44"):
        matrix = getattr(model, f"Q{suffix}").detach()
        if mode == "dense":
            getattr(target, f"Q{suffix}").copy_(matrix)
        else:
            diagonal = matrix.diagonal()
            getattr(target, f"diagonal{suffix}").copy_(diagonal)
            background = getattr(target, f"background{suffix}")
            if background is not None:
                # Mask, rather than subtract the diagonal from a row sum, to
                # avoid cancellation when the matching score is very large.
                offdiagonal = matrix.masked_fill(target._diagonal_mask, 0)
                background.copy_(offdiagonal.sum(-1) / (config.nvars - 1))
    target.configure_stage("attention")
    target.train(model.training)
    return target


def load_routing_model(path, device="cpu"):
    """Load old dense checkpoints or new explicitly typed routing checkpoints."""
    state = torch.load(path, map_location=device, weights_only=False)
    mode = state.get("routing_mode", state.get("args", {}).get("routing_mode", "dense"))
    if mode not in ROUTING_MODES:
        raise ValueError(f"Unknown checkpoint routing_mode: {mode}")
    config = SparseModelConfig(**state.get("model_config", state.get("config")))
    reference = next(value for value in state["model"].values() if value.is_floating_point())
    model = (SparseClauseTransformer(config) if mode == "dense"
             else RoutingClauseTransformer(config, mode)).to(device=device, dtype=reference.dtype)
    model.routing_mode = mode
    model.load_state_dict(state["model"], strict=True)
    model.configure_stage(state.get("stage", "attention"))
    routing_parameter_counts(model)
    return model, state
