"""Gemma-4 text tower for the LTX-2.5 text encoder (``gemma4_unified_text``).

Self-contained MLX port of the transformers Gemma-4 text decoder, limited to
what LTX needs: hidden states out, no lm_head, no cache, no MoE/per-layer
inputs (both disabled in the LTX finetune's config).

Faithful oddities, all confirmed against the transformers reference and the
shipped checkpoint:

- Per-layer-type geometry: sliding layers use head_dim 256 with 8 KV heads
  and theta 10k; global ("full_attention") layers use head_dim 512 with a
  single KV head (MQA) and theta 1e6.
- ``attention_k_eq_v``: global layers have no v_proj -- V is the *raw* K
  projection (before k_norm/RoPE), passed through a scale-less RMSNorm.
- ``proportional`` RoPE on global layers: only the first
  ``partial_rotary_factor * head_dim`` dimensions rotate; implemented as
  standard rotary with the remaining frequencies set to zero (cos=1, sin=0).
- Attention scaling is 1.0 (QK-norm makes the usual 1/sqrt(d) unnecessary).
- RMSNorm multiplies by ``weight`` directly (Gemma3n-style; weights ship
  as ~1.0), computed in float32.
- Each decoder layer's full output (residual included) is multiplied by a
  checkpoint-backed scalar ``layer_scalar``.
- The LTX prompt path tokenizes at max length 1024 == sliding_window, so
  sliding attention is exactly causal attention here; no windowing needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Gemma4TextConfig:
    hidden_size: int = 3840
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    intermediate_size: int = 15360
    vocab_size: int = 262144
    rms_norm_eps: float = 1e-6
    sliding_window: int = 1024
    attention_k_eq_v: bool = True
    layer_types: List[str] = field(default_factory=list)
    rope_theta_sliding: float = 10000.0
    rope_theta_global: float = 1000000.0
    global_partial_rotary_factor: float = 0.25

    @classmethod
    def from_hf(cls, text_config: dict) -> "Gemma4TextConfig":
        rope = text_config.get("rope_parameters", {})
        return cls(
            hidden_size=text_config["hidden_size"],
            num_hidden_layers=text_config["num_hidden_layers"],
            num_attention_heads=text_config["num_attention_heads"],
            num_key_value_heads=text_config["num_key_value_heads"],
            num_global_key_value_heads=text_config.get(
                "num_global_key_value_heads", 1
            ),
            head_dim=text_config["head_dim"],
            global_head_dim=text_config.get(
                "global_head_dim", text_config["head_dim"]
            ),
            intermediate_size=text_config["intermediate_size"],
            vocab_size=text_config["vocab_size"],
            rms_norm_eps=text_config.get("rms_norm_eps", 1e-6),
            sliding_window=text_config.get("sliding_window", 1024),
            attention_k_eq_v=text_config.get("attention_k_eq_v", True),
            layer_types=list(text_config.get("layer_types", [])),
            rope_theta_sliding=rope.get("sliding_attention", {}).get(
                "rope_theta", 10000.0
            ),
            rope_theta_global=rope.get("full_attention", {}).get(
                "rope_theta", 1000000.0
            ),
            global_partial_rotary_factor=rope.get("full_attention", {}).get(
                "partial_rotary_factor", 0.25
            ),
        )

    def is_global(self, layer_idx: int) -> bool:
        if self.layer_types:
            return self.layer_types[layer_idx] == "full_attention"
        # Fallback: gemma-family default of one global layer per 6.
        return layer_idx % 6 == 5


class RMSNorm(nn.Module):
    """Gemma3n-style RMSNorm: float32 norm, multiplied by weight directly."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x32 = x.astype(mx.float32)
        normed = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
        if self.with_scale:
            normed = normed * self.weight.astype(mx.float32)
        return normed.astype(dtype)


def _rope_cos_sin(
    seq_len: int, head_dim: int, theta: float, rotary_dims: int
) -> Tuple[mx.array, mx.array]:
    """cos/sin tables in the split (first-half/second-half) convention.

    ``rotary_dims`` < ``head_dim`` implements proportional RoPE: frequencies
    beyond the rotary portion are zero, so those dimensions pass through
    unrotated (cos=1, sin=0).
    """
    half = head_dim // 2
    rope_angles = rotary_dims // 2
    exponents = mx.arange(0, 2 * rope_angles, 2).astype(mx.float32) / head_dim
    inv_freq = 1.0 / (theta**exponents)
    if half > rope_angles:
        inv_freq = mx.concatenate([inv_freq, mx.zeros((half - rope_angles,))])
    positions = mx.arange(seq_len).astype(mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x: (B, S, H, D); cos/sin: (S, D) in float32."""
    x32 = x.astype(mx.float32)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return (x32 * cos + _rotate_half(x32) * sin).astype(x.dtype)


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.is_global = config.is_global(layer_idx)
        self.num_heads = config.num_attention_heads
        if self.is_global:
            self.head_dim = config.global_head_dim
            self.num_kv_heads = config.num_global_key_value_heads
            self.theta = config.rope_theta_global
            self.rotary_dims = int(
                config.global_partial_rotary_factor * config.global_head_dim
            )
            self.k_eq_v = config.attention_k_eq_v
        else:
            self.head_dim = config.head_dim
            self.num_kv_heads = config.num_key_value_heads
            self.theta = config.rope_theta_sliding
            self.rotary_dims = config.head_dim
            self.k_eq_v = False

        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        if not self.k_eq_v:
            self.v_proj = nn.Linear(
                hidden, self.num_kv_heads * self.head_dim, bias=False
            )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        batch, seq, _ = x.shape
        cos, sin = _rope_cos_sin(seq, self.head_dim, self.theta, self.rotary_dims)

        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.head_dim)
        q = _apply_rope(self.q_norm(q), cos, sin)

        k_raw = self.k_proj(x).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        # attention_k_eq_v: V is the raw K projection (pre-norm, pre-RoPE)
        # through a scale-less RMSNorm; no rotary on V.
        if self.k_eq_v:
            v = self.v_norm(k_raw)
        else:
            v = self.v_norm(
                self.v_proj(x).reshape(batch, seq, self.num_kv_heads, self.head_dim)
            )
        k = _apply_rope(self.k_norm(k_raw), cos, sin)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # QK-norm model: attention scale is exactly 1.0.
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, -1)
        return self.o_proj(out)


class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        eps = config.rms_norm_eps
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        #: Checkpoint-backed scalar applied to the layer's full output.
        self.layer_scalar = mx.ones((1,))

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        h = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), mask)
        )
        h = h + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(h))
        )
        return h * self.layer_scalar.astype(h.dtype)


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Gemma4LanguageModel(nn.Module):
    """Drop-in for the Gemma-3 ``LanguageModel`` wrapper: same call contract,
    returns (final_hidden, all_hidden_states) with the embedding-scaled input
    first and the final-normed output last (transformers convention)."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.model = Gemma4TextModel(config)

    def _mask(
        self,
        seq_len: int,
        attention_mask: Optional[mx.array],
        dtype: mx.Dtype,
    ) -> mx.array:
        causal = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
        min_val = mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
        if attention_mask is not None:
            padding = attention_mask.astype(mx.bool_)
            combined = causal[None, :, :] & padding[:, None, :]
            mask = mx.where(combined, 0.0, min_val).astype(dtype)
            return mask[:, None, :, :]
        mask = mx.where(causal, 0.0, min_val).astype(dtype)
        return mask[None, None, :, :]

    def __call__(
        self,
        inputs: mx.array,
        input_embeddings: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
        cache=None,
    ) -> Tuple[mx.array, List[mx.array]]:
        del cache  # no generation path; encode-only
        _, seq_len = inputs.shape

        h = (
            input_embeddings
            if input_embeddings is not None
            else self.model.embed_tokens(inputs)
        )
        h = h * mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(h.dtype)
        mx.eval(h)

        all_hidden_states = [h] if output_hidden_states else []

        # Prompt length is capped at 1024 == sliding_window, so sliding
        # attention is plain causal attention at every layer.
        mask = self._mask(seq_len, attention_mask, h.dtype)

        num_layers = len(self.model.layers)
        for i, layer in enumerate(self.model.layers):
            h = layer(h, mask)
            mx.eval(h)
            if output_hidden_states and i < num_layers - 1:
                all_hidden_states.append(h)

        final = self.model.norm(h)
        mx.eval(final)
        if output_hidden_states:
            all_hidden_states.append(final)
        return final, all_hidden_states

    def sanitize(self, weights: dict) -> dict:
        """Checkpoint keys are Comfy-flat: ``model.layers.N...`` etc."""
        sanitized = {}
        for key, value in weights.items():
            if not key.startswith("model."):
                continue
            new_key = key  # module tree is model.<...> too
            if value.dtype == mx.float32:
                value = value.astype(mx.bfloat16)
            sanitized[new_key] = value
        return sanitized
