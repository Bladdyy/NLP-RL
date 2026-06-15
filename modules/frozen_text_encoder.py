"""
Text-based goal encoder for 2D ant goals using a frozen pretrained text model.

Converts (x, y) coordinates to the prompt "Your goal is (x,y)"
and encodes it via a HuggingFace BERT-family model. Supports multiple
model variants via a registry keyed by short names.

Two execution paths exist:
* **Precomputed** (fast) — for environments with a finite set of possible
  goal positions (e.g. ant-maze).  Embeddings are computed once at init
  using the HF tokenizer directly, then looked up via nearest-neighbour
  argmin at training time — **no model inference during the training loop**.
* **Direct** (general) — when no goal set is known, the full frozen model
  is run every forward pass.  Token IDs are constructed manually to stay
  compatible with JAX JIT compilation.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import variance_scaling
from transformers import AutoTokenizer, FlaxBertModel


# ── Model registry ────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embed_dim": 384,
    },
    "bge": {
        "hf_name": "BAAI/bge-small-en-v1.5",
        "embed_dim": 384,
    },
    "gte": {
        "hf_name": "thenlper/gte-small",
        "embed_dim": 384,
    },
    "e5": {
        "hf_name": "intfloat/e5-small-v2",
        "embed_dim": 384,
    },
}

_MODEL_CACHE: dict[str, tuple[Any, FlaxBertModel, dict]] = {}


def _load_model(model_key: str) -> tuple[Any, FlaxBertModel, dict]:
    """Lazy-load a model + tokenizer from the registry (cached)."""
    if model_key not in _MODEL_CACHE:
        entry = MODEL_REGISTRY[model_key]
        hf_name = entry["hf_name"]
        tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        try:
            model = FlaxBertModel.from_pretrained(hf_name, dtype=jnp.float32)
        except Exception:
            # Fall back to PyTorch → Flax conversion if no native Flax weights
            model = FlaxBertModel.from_pretrained(hf_name, dtype=jnp.float32, from_pt=True)
        _MODEL_CACHE[model_key] = (tokenizer, model, model.params)
    return _MODEL_CACHE[model_key]


# Pre-load "minilm" by default so that import still works instantly.
_DEFAULT_KEY = "minilm"

_TOKENIZER, _SBERT_MODEL, _SBERT_PARAMS = _load_model(_DEFAULT_KEY)


# ── Token-ID constants (identical across all four models) ─────────────────

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


# ── JAX-compatible tokenisation (works for all BERT-family tokenizers) ────

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


# ── Precomputation helper ────────────────────────────────────────────────

def _precompute_all_goal_embeddings(
    possible_goals: np.ndarray, model_key: str,
    pooling: str = "cls",
) -> jnp.ndarray:
    """
    Precompute embeddings for a fixed set of goal coordinates.

    Args:
        possible_goals: (N, 2) numpy array of goal coordinates.
        model_key: short name in MODEL_REGISTRY.
        pooling: ``"cls"``, ``"mean"``, or ``"token"``.

    Returns:
        ``(N, embed_dim)`` for cls/mean, ``(N, seq_len, embed_dim)`` for token.
    """
    _, model, params = _load_model(model_key)
    all_embs = []
    for coord in possible_goals:
        g = jnp.array([[coord[0], coord[1]]])
        input_ids = tokenize_goal_prompt(g)
        attention_mask = jnp.ones_like(input_ids)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            params=params,
            train=False,
        )
        if pooling == "cls":
            all_embs.append(outputs.last_hidden_state[:, 0])  # (1, embed_dim)
        elif pooling == "mean":
            masked = outputs.last_hidden_state * attention_mask[:, :, None]
            emb = masked.sum(axis=1) / attention_mask.sum(axis=1, keepdims=True)
            all_embs.append(emb)  # (1, embed_dim)
        else:
            all_embs.append(outputs.last_hidden_state)  # (1, seq_len, embed_dim)
    return jnp.concatenate(all_embs, axis=0)


# ── Encoder classes ──────────────────────────────────────────────────────

class PrecomputedFrozenTextGoalEncoder(nn.Module):
    """
    Fast goal encoder for environments with a finite set of goal positions.

    Precomputes text model embeddings for all possible goals once at
    initialisation, then uses nearest-neighbour look-up at training time
    — no transformer inference is performed in the training loop.

    All pooling modes project to *output_dim* so the caller always receives
    ``(B, output_dim)`` (cls/mean) or ``(B, seq_len, output_dim)`` (token).

    Args:
        output_dim: projection output dimension (default: 64).
        possible_goals: (N, 2) array of all goal coordinates that may appear.
        model_key: short name in MODEL_REGISTRY (default: ``"minilm"``).
        pooling: ``"cls"``, ``"mean"``, or ``"token"`` (default: ``"cls"``).
    """
    possible_goals: jnp.ndarray  # (N, 2)
    output_dim: int = 64
    model_key: str = "minilm"
    pooling: str = "cls"

    def setup(self):
        goals_np = np.asarray(self.possible_goals)
        self._precomputed_goals = jnp.asarray(goals_np)
        self._precomputed_embs = _precompute_all_goal_embeddings(
            goals_np, self.model_key, pooling=self.pooling,
        )
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        self.proj = nn.Dense(
            self.output_dim,
            kernel_init=lecun_uniform,
            bias_init=nn.initializers.zeros,
            name="proj",
        )

    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        # (B, 2) -> nearest precomputed goal -> its embedding
        dists = jnp.linalg.norm(
            g[:, None, :] - self._precomputed_goals[None, :, :],
            axis=-1,
        )  # (B, N)
        nearest = jnp.argmin(dists, axis=-1)  # (B,)
        embs = self._precomputed_embs[nearest]  # (B, embed_dim) or (B, seq_len, embed_dim)
        return jax.lax.stop_gradient(self.proj(embs))


class FrozenTextGoalEncoder(nn.Module):
    """
    Encodes a 2D goal as "Your goal is (x,y)" via a frozen HuggingFace
    text model, then projects to *output_dim*.

    When *possible_goals* is provided, the fast precomputed path is used
    automatically. Otherwise the full frozen model is run each forward
    pass (slower but general).

    All pooling modes project to *output_dim*.  ``"token"`` mode returns
    ``(B, seq_len, output_dim)``; cls/mean return ``(B, output_dim)``.

    Args:
        output_dim: projection output dimension (default: 64).
        model_key: short name in MODEL_REGISTRY (default: ``"minilm"``).
        possible_goals: (N, 2) array for precomputed look-up. When
            ``None`` (default) the full model is run on every call.
            When set, ``PrecomputedFrozenTextGoalEncoder`` is used.
        pooling: ``"cls"``, ``"mean"``, or ``"token"`` (default: ``"cls"``).
    """
    output_dim: int = 64
    model_key: str = "minilm"
    possible_goals: jnp.ndarray | None = None
    pooling: str = "cls"

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        if self.possible_goals is not None:
            return PrecomputedFrozenTextGoalEncoder(
                output_dim=self.output_dim,
                possible_goals=self.possible_goals,
                model_key=self.model_key,
                pooling=self.pooling,
            )(g)

        logging.warning(
            "FrozenTextGoalEncoder: possible_goals is None, falling back to the "
            "slow direct path (full BERT inference every forward pass). "
            "Provide possible_goals to use the fast precomputed path."
        )

        # --- Direct (non-precomputed) path ---
        _, model, _ = _load_model(self.model_key)
        params = self.param("model_params", lambda _: _load_model(self.model_key)[2])

        input_ids      = tokenize_goal_prompt(g)
        attention_mask = jnp.ones_like(input_ids)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            params=params,
            train=False,
        )

        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        if self.pooling == "cls":
            emb = jax.lax.stop_gradient(outputs.last_hidden_state[:, 0])
        elif self.pooling == "mean":
            masked = outputs.last_hidden_state * attention_mask[:, :, None]
            emb = masked.sum(axis=1) / attention_mask.sum(axis=1, keepdims=True)
            emb = jax.lax.stop_gradient(emb)
        else:
            # token: (B, seq_len, 384) → project to output_dim
            emb = jax.lax.stop_gradient(outputs.last_hidden_state)

        x = nn.Dense(
            self.output_dim,
            kernel_init=lecun_uniform,
            bias_init=nn.initializers.zeros,
            name="proj",
        )(emb)
        return jax.lax.stop_gradient(x)