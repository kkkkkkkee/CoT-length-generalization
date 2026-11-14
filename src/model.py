"""Explicit independent implementations of the underspecified experimental model.

The default is one conventional decoder block, two heads and no position encoding.
These engineering choices are NOT attributed to the paper: trainable 64-d token
embeddings, concatenation of five slots, pre-LN residuals and a GELU FFN.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    token_dim: int = 64
    heads: int = 2
    ff_mult: int = 4
    residual: bool = True
    layer_norm: bool = True
    activation: str = "gelu"


class ClauseTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.width = 5 * config.token_dim
        assert self.width % config.heads == 0
        self.embed = nn.Embedding(config.vocab_size, config.token_dim, padding_idx=0)
        self.qkv = nn.Linear(self.width, 3 * self.width, bias=False)
        self.attn_out = nn.Linear(self.width, self.width, bias=False)
        self.norm1 = nn.LayerNorm(self.width) if config.layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(self.width) if config.layer_norm else nn.Identity()
        self.ff = nn.Sequential(nn.Linear(self.width, config.ff_mult * self.width),
                                nn.GELU() if config.activation == "gelu" else nn.ReLU(),
                                nn.Linear(config.ff_mult * self.width, self.width))
        self.readout = nn.Linear(self.width, 5 * config.vocab_size)

    def forward(self, clauses, temperature=1.0, return_attention=False):
        # temperature is an inverse temperature: >1 sharpens attention logits.
        b, t, _ = clauses.shape
        h = self.config.heads
        x = self.embed(clauses).flatten(-2)
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, h, self.width // h).permute(2, 0, 3, 1, 4).unbind(0)
        if return_attention:
            scores = (q @ k.transpose(-1, -2)) * (temperature / math.sqrt(self.width // h))
            mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = scores.softmax(-1)
            z = attn @ v
        else:
            z = F.scaled_dot_product_attention(q * temperature, k, v, is_causal=True)
        z = self.attn_out(z.transpose(1, 2).reshape(b, t, self.width))
        x = x + z if self.config.residual else z
        z = self.ff(self.norm2(x))
        x = x + z if self.config.residual else z
        logits = self.readout(x).reshape(b, t, 5, self.config.vocab_size)
        return (logits, attn) if return_attention else logits

    def config_dict(self):
        return asdict(self.config)

    @torch.no_grad()
    def decode_last(self, clauses, cache=None, temperature=1.0):
        """Exact single-layer KV caching for unconstrained greedy clause rollout.

        At prefill clauses is the whole prompt; subsequently it is just the newly
        generated clause. Return last-position logits [B,5,d] and (key,value).
        No positional embedding or contextualized input is stored in the cache.
        """
        b, t, _ = clauses.shape
        if cache is not None and t != 1:
            raise ValueError("cached decoding accepts one new clause at a time")
        h = self.config.heads
        x = self.embed(clauses).flatten(-2)
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, h, self.width // h).permute(2, 0, 3, 1, 4).unbind(0)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        z = F.scaled_dot_product_attention(q[:, :, -1:] * temperature, k, v)
        z = self.attn_out(z.transpose(1, 2).reshape(b, 1, self.width))
        x = x[:, -1:] + z if self.config.residual else z
        z = self.ff(self.norm2(x))
        x = x + z if self.config.residual else z
        logits = self.readout(x).reshape(b, 5, self.config.vocab_size)
        return logits, (k, v)


def load_model(path, device="cuda"):
    state = torch.load(path, map_location=device, weights_only=False)
    model = ClauseTransformer(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    return model, state
