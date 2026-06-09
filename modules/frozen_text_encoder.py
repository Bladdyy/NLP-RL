"""
Text-based goal encoder for 2D ant goals using Sentence-BERT.

Converts (x, y) coordinates to the prompt "Your goal is (x,y)"
and encodes it with a frozen pretrained SBERT model (all-MiniLM-L6-v2).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import variance_scaling
from transformers import AutoTokenizer, FlaxBertModel


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_TOKENIZER    = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
_SBERT_MODEL  = FlaxBertModel.from_pretrained(MODEL_NAME, dtype=jnp.float32)
_SBERT_PARAMS = _SBERT_MODEL.params

# Precomputed token ID constants

PREFIX_IDS = jnp.array(
    _TOKENIZER("Your goal is (", add_special_tokens=False).input_ids,
    dtype=jnp.int32,
)
COMMA_ID  = jnp.int32(_TOKENIZER(",", add_special_tokens=False).input_ids[0])
SUFFIX_ID = jnp.int32(_TOKENIZER(")", add_special_tokens=False).input_ids[0])
DOT_ID    = jnp.int32(_TOKENIZER(".", add_special_tokens=False).input_ids[0])
PLUS_ID   = jnp.int32(_TOKENIZER("+", add_special_tokens=False).input_ids[0])
MINUS_ID  = jnp.int32(_TOKENIZER("-", add_special_tokens=False).input_ids[0])
DIGIT_IDS = jnp.array(
    [_TOKENIZER(str(d), add_special_tokens=False).input_ids[0] for d in range(10)],
    dtype=jnp.int32,
)

def _coord_token_ids(coord: jnp.ndarray) -> jnp.ndarray:
    """
    (B,) -> (B, 6): [sign, tens, ones, dot, frac_tens, frac_ones]
    e.g. -12.34 -> [MINUS, 1, 2, DOT, 3, 4]
    """
    sign    = jnp.where(coord < 0.0, MINUS_ID, PLUS_ID)
    scaled  = jnp.clip(jnp.round(jnp.abs(coord) * 100.0), 0.0, 9999.0).astype(jnp.int32)
    integer = jnp.clip(scaled // 100, 0, 99)
    frac    = scaled % 100

    return jnp.stack(
        [
            sign,
            DIGIT_IDS[integer // 10],
            DIGIT_IDS[integer % 10],
            jnp.full(sign.shape, DOT_ID, dtype=jnp.int32),
            DIGIT_IDS[frac // 10],
            DIGIT_IDS[frac % 10],
        ],
        axis=-1,
    )  # (B, 6)


def tokenize_goal_prompt(g: jnp.ndarray) -> jnp.ndarray:
    """
    (B, 2) -> (B, prefix_len + 6 + 1 + 6 + 1)
    representing "Your goal is (±XX.XX,±XX.XX)"
    """
    assert g.ndim == 2 and g.shape[-1] == 2, "g must be (B, 2)"
    B = g.shape[0]

    prefix = jnp.tile(PREFIX_IDS[None, :], (B, 1))
    x_tok  = _coord_token_ids(g[:, 0])
    comma  = jnp.full((B, 1), COMMA_ID,  dtype=jnp.int32)
    y_tok  = _coord_token_ids(g[:, 1])
    suffix = jnp.full((B, 1), SUFFIX_ID, dtype=jnp.int32)

    return jnp.concatenate([prefix, x_tok, comma, y_tok, suffix], axis=-1)

class FrozenSentenceBertGoalEncoder(nn.Module):
    """
    Encodes a 2D goal as "Your goal is (x,y)" via frozen SBERT,
    then projects to output_dim with stop_gradient on everything.
    """
    output_dim: int = 64

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        params = self.param("sbert_params", lambda _: _SBERT_PARAMS)

        input_ids      = tokenize_goal_prompt(g)
        attention_mask = jnp.ones_like(input_ids)

        outputs = _SBERT_MODEL(
            input_ids=input_ids,
            attention_mask=attention_mask,
            params=params,
            train=False,
        )
        cls_emb = jax.lax.stop_gradient(
            outputs.last_hidden_state[:, 0]  # (B, 384)
        )

        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        x = nn.Dense(
            self.output_dim,
            kernel_init=lecun_uniform,
            bias_init=nn.initializers.zeros,
            name="proj",
        )(cls_emb)
        return jax.lax.stop_gradient(x)  # (B, output_dim)
