"""A finite, theory-inspired model for sparse-step supervision experiments.

Nonblank token embeddings are fixed orthogonal one-hot vectors; a clause is
their five-slot concatenation. Attention has one head and only the variable
matching blocks Q43 and Q44 are trainable. There are no residual connections,
LayerNorm, value/output projections, or output masks. Optional position-score
biases are configured explicitly; the historical default has no position term.

The local computation is an ordinary shared MLP. Its GELU, learned biases and
shared hidden units differ from the theorem's per-output smoothed-ReLU network,
fixed biases and logit clipping. This is NOT an exact implementation of all the
theorem's assumptions. Direct Q matrices also avoid the zero-gradient problem
of initializing both factors of a Q/K parameterization to zero.
"""

from dataclasses import asdict, dataclass
import hashlib
import math

import torch
from torch import nn


@dataclass
class SparseModelConfig:
    vocab_size: int
    nvars: int
    hidden: int = 512
    activation: str = "gelu"
    position_mode: str = "none"
    max_position: int = 257
    position_dim: int = 32

    def __post_init__(self):
        if self.nvars < 1 or self.vocab_size <= self.nvars:
            raise ValueError("nvars must be positive and vocabulary must include blank and variables")
        if self.hidden < 1:
            raise ValueError("hidden must be positive")
        if self.activation not in {"gelu", "relu"}:
            raise ValueError("activation must be 'gelu' or 'relu'")
        if self.max_position < 1 or self.position_dim < 2:
            raise ValueError("invalid position dimensions")
        if self.position_mode not in {"none", "learned_abs", "sinusoidal", "relative"}:
            raise ValueError("position_mode must be none, learned_abs, sinusoidal, or relative")


class SparseClauseTransformer(nn.Module):
    """Single-head block-sparse attention followed by a learned local MLP.

Variable tokens have IDs 1..nvars, as in ``data.Task``. A nonvariable token
predicted in a variable slot has zero overlap with the trainable Q blocks;
it is not corrected or replaced by an oracle variable.

``forward_queries`` computes only the requested rows and is useful when just
one value per training problem is supervised. The full ``forward`` retains
the evaluator's [B,T,5,V] API, including causal outputs at predicate positions.
"""

    def __init__(self, config: SparseModelConfig):
        super().__init__()
        self.config = config
        self.width = 5 * config.vocab_size
        self.Q43 = nn.Parameter(torch.zeros(config.nvars, config.nvars))
        self.Q44 = nn.Parameter(torch.zeros(config.nvars, config.nvars))
        self.ff = nn.Sequential(
            nn.Linear(self.width, config.hidden),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Linear(config.hidden, self.width),
        )
        self._init_position()
        self.configure_stage("local")

    def _init_position(self):
        mode = self.config.position_mode
        if mode == "none":
            self.position_embedding = None
            return
        t, d = self.config.max_position, self.config.position_dim
        if mode == "learned_abs":
            self.position_embedding = nn.Parameter(torch.randn(t, d) * 0.02)
            return
        if mode == "relative":
            self.position_embedding = nn.Parameter(torch.randn(2 * t - 1) * 0.02)
            return
        pos = torch.arange(t, dtype=torch.float32)[:, None]
        idx = torch.arange(0, d, 2, dtype=torch.float32)[None, :]
        div = torch.exp(-math.log(10000.0) * idx / d)
        table = torch.zeros(t, d)
        table[:, 0::2] = torch.sin(pos * div)
        table[:, 1::2] = torch.cos(pos * div[:, :table[:, 1::2].shape[1]])
        self.register_buffer("position_embedding", table, persistent=True)

    def _position_bias(self, positions, time):
        if self.position_embedding is None:
            return None
        if time > self.config.max_position or int(positions.max()) >= self.config.max_position:
            raise ValueError("sequence exceeds configured max_position")
        keys = self.position_embedding[:time]
        if self.config.position_mode == "relative":
            keypos = torch.arange(time, device=positions.device)
            rel = positions[..., None] - keypos
            return self.position_embedding[rel + self.config.max_position - 1]
        queries = self.position_embedding[positions]
        return queries.matmul(keys.transpose(0, 1)) / math.sqrt(keys.shape[-1])

    def configure_stage(self, stage="local"):
        """Freeze all except the MLP (local) or the two Q blocks (attention)."""
        if stage not in {"local", "attention"}:
            raise ValueError("stage must be 'local' or 'attention'")
        self.Q43.requires_grad_(stage == "attention")
        self.Q44.requires_grad_(stage == "attention")
        if isinstance(self.position_embedding, nn.Parameter):
            self.position_embedding.requires_grad_(True)
        for parameter in self.ff.parameters():
            parameter.requires_grad_(stage == "local")
        # Clear stale gradients when the model moves between training stages.
        for parameter in self.parameters():
            parameter.grad = None
        self.training_stage = stage
        return self

    def config_dict(self):
        return asdict(self.config)

    def frozen_state_hash(self):
        """Hash names, shapes, dtypes and bytes of currently frozen parameters."""
        digest = hashlib.sha256()
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                value = parameter.detach().cpu().contiguous()
                digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode())
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def _query_positions(self, clauses, query_positions):
        if clauses.ndim != 3 or clauses.shape[-1] != 5 or clauses.shape[1] < 1:
            raise ValueError("clauses must have shape (batch, nonempty_time, 5)")
        if clauses.dtype != torch.long:
            raise ValueError("clause token IDs must have dtype torch.long")
        positions = torch.as_tensor(query_positions, device=clauses.device)
        if positions.dtype != torch.long:
            raise ValueError("query_positions must contain integer indices")
        if positions.ndim == 1:
            positions = positions[None].expand(clauses.shape[0], -1)
        if positions.ndim != 2 or positions.shape[0] != clauses.shape[0] or positions.shape[1] < 1:
            raise ValueError("query_positions must have shape (queries,) or (batch, queries)")
        return positions

    def forward_queries(self, clauses, query_positions, temperature=1.0,
                        return_attention=False, attention_override=None):
        """Return [B,J,5,V] logits and optionally [B,1,J,T] causal attention.

        The argument ``temperature`` follows the existing evaluator: it is a
        multiplier of logits (inverse temperature). No sqrt(width) rescaling
        is applied to the direct bilinear Q scores. ``attention_override`` is
        an optional [B,1,J,T] probability tensor for causal mechanism probes;
        it does not supply targets to the MLP or mask the output vocabulary.
        """
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive scalar")
        positions = self._query_positions(clauses, query_positions)
        batch, time, _ = clauses.shape
        queries = positions.shape[1]
        query_tokens = clauses[:, :, 3].gather(1, positions)
        pred_tokens, answer_tokens = clauses[:, :, 2], clauses[:, :, 3]
        nvars = self.config.nvars

        def block_scores(matrix, key_tokens):
            query_valid = (query_tokens >= 1) & (query_tokens <= nvars)
            key_valid = (key_tokens >= 1) & (key_tokens <= nvars)
            query_index = (query_tokens - 1).clamp(0, nvars - 1)
            key_index = (key_tokens - 1).clamp(0, nvars - 1)
            scores = matrix[query_index[:, :, None], key_index[:, None, :]]
            return scores * (query_valid[:, :, None] & key_valid[:, None, :])

        scores = (block_scores(self.Q43, pred_tokens)
                  + block_scores(self.Q44, answer_tokens)) * temperature
        position_bias = self._position_bias(positions, time)
        if position_bias is not None:
            scores = scores + position_bias
        future = torch.arange(time, device=clauses.device)[None, None, :] > positions[:, :, None]
        attention = scores.masked_fill(future, -torch.inf).softmax(-1)
        if attention_override is not None:
            if tuple(attention_override.shape) != (batch, 1, queries, time):
                raise ValueError("attention_override must have shape (batch,1,queries,time)")
            attention = attention_override[:, 0].to(device=scores.device, dtype=scores.dtype)
            if not torch.isfinite(attention).all() or (attention < 0).any():
                raise ValueError("attention_override must contain finite nonnegative weights")
            if (attention.masked_select(future) != 0).any():
                raise ValueError("attention_override cannot attend to future clauses")
            if not torch.allclose(attention.sum(-1), torch.ones_like(attention[:, :, 0]), atol=1e-6, rtol=1e-6):
                raise ValueError("attention_override rows must sum to one")

        # Exact attention-weighted one-hot embeddings without constructing a
        # potentially huge [B,T,5,V] input embedding tensor. Blank contributes
        # the zero vector in every slot, including malformed generated clauses.
        indices = clauses.permute(0, 2, 1)[:, None].expand(-1, queries, -1, -1)
        weights = attention[:, :, None, :].expand(-1, -1, 5, -1)
        weights = weights * indices.ne(0)
        mixed = attention.new_zeros(batch, queries, 5, self.config.vocab_size)
        mixed.scatter_add_(-1, indices, weights)
        logits = self.ff(mixed.flatten(-2)).reshape(batch, queries, 5, self.config.vocab_size)
        return (logits, attention[:, None]) if return_attention else logits

    def forward(self, clauses, temperature=1.0, return_attention=False,
                attention_override=None):
        positions = torch.arange(clauses.shape[1], device=clauses.device)
        return self.forward_queries(clauses, positions, temperature, return_attention,
                                    attention_override)

    @torch.no_grad()
    def decode_last(self, clauses, cache=None, temperature=1.0):
        """Exact greedy-decoding interface, caching token IDs rather than K/V.

        Fixed embeddings and direct Q blocks need only the previous clause IDs.
        Each decoding step evaluates one query and one MLP output; no logits for
        preceding positions are recomputed. All five generated slots feed back.
        """
        if cache is not None:
            if clauses.shape[1] != 1:
                raise ValueError("cached decoding accepts exactly one new clause")
            clauses = torch.cat((cache, clauses), dim=1)
        position = torch.tensor([clauses.shape[1] - 1], device=clauses.device)
        logits = self.forward_queries(clauses, position, temperature)[:, 0]
        return logits, clauses


def load_model(path, device="cpu"):
    """Load a checkpoint using the same metadata convention as model.py."""
    state = torch.load(path, map_location=device, weights_only=False)
    config = state.get("model_config", state.get("config"))
    model = SparseClauseTransformer(SparseModelConfig(**config)).to(device)
    model.load_state_dict(state["model"])
    model.configure_stage(state.get("stage", "attention"))
    return model, state
