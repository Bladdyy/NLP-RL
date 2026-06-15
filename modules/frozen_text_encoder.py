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

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import variance_scaling
from transformers import AutoTokenizer, FlaxBertModel


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Model registry  (add new models here)
# ═══════════════════════════════════════════════════════════════════════════

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
            # Some models only have PyTorch weights uploaded; convert to Flax.
            model = FlaxBertModel.from_pretrained(hf_name, dtype=jnp.float32, from_pt=True)
        _MODEL_CACHE[model_key] = (tokenizer, model, model.params)
    return _MODEL_CACHE[model_key]


# Warm the cache with the default model so module imports are instant.
_DEFAULT_KEY = "minilm"
_TOKENIZER, _SBERT_MODEL, _SBERT_PARAMS = _load_model(_DEFAULT_KEY)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Manual tokenisation  (JAX-compatible — for the direct path only)
# ═══════════════════════════════════════════════════════════════════════════

# Token IDs are identical across all four supported models (BERT WordPiece).
_PREFIX_IDS = jnp.array(
    _TOKENIZER("Your goal is (", add_special_tokens=False).input_ids,
    dtype=jnp.int32,
)
_COMMA_ID  = jnp.int32(_TOKENIZER(",", add_special_tokens=False).input_ids[0])
_SUFFIX_ID = jnp.int32(_TOKENIZER(")", add_special_tokens=False).input_ids[0])
_DOT_ID    = jnp.int32(_TOKENIZER(".", add_special_tokens=False).input_ids[0])
_PLUS_ID   = jnp.int32(_TOKENIZER("+", add_special_tokens=False).input_ids[0])
_MINUS_ID  = jnp.int32(_TOKENIZER("-", add_special_tokens=False).input_ids[0])
_DIGIT_IDS = jnp.array(
    [_TOKENIZER(str(d), add_special_tokens=False).input_ids[0] for d in range(10)],
    dtype=jnp.int32,
)


def _coord_token_ids(coord: jnp.ndarray) -> jnp.ndarray:
    """
    (B,) -> (B, 6): [sign, tens, ones, dot, frac_tens, frac_ones]
    e.g. -12.34 -> [MINUS, 1, 2, DOT, 3, 4]
    """
    sign    = jnp.where(coord < 0.0, _MINUS_ID, _PLUS_ID)
    scaled  = jnp.clip(jnp.round(jnp.abs(coord) * 100.0), 0.0, 9999.0).astype(jnp.int32)
    integer = jnp.clip(scaled // 100, 0, 99)
    frac    = scaled % 100

    return jnp.stack(
        [
            sign,
            _DIGIT_IDS[integer // 10],
            _DIGIT_IDS[integer % 10],
            jnp.full(sign.shape, _DOT_ID, dtype=jnp.int32),
            _DIGIT_IDS[frac // 10],
            _DIGIT_IDS[frac % 10],
        ],
        axis=-1,
    )


def _tokenize_goal_prompt_jax(g: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-compatible tokenisation of ``"Your goal is (±XX.XX,±XX.XX)"``.

    (B, 2) -> (B, prefix_len + 6 + 1 + 6 + 1)

    This avoids calling the Python-only HF tokenizer inside JIT.
    """
    assert g.ndim == 2 and g.shape[-1] == 2, "g must be (B, 2)"
    B = g.shape[0]

    prefix = jnp.tile(_PREFIX_IDS[None, :], (B, 1))
    x_tok  = _coord_token_ids(g[:, 0])
    comma  = jnp.full((B, 1), _COMMA_ID, dtype=jnp.int32)
    y_tok  = _coord_token_ids(g[:, 1])
    suffix = jnp.full((B, 1), _SUFFIX_ID, dtype=jnp.int32)

    return jnp.concatenate([prefix, x_tok, comma, y_tok, suffix], axis=-1)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Precomputation  (uses the HF tokenizer on strings — Python, not JIT)
# ═══════════════════════════════════════════════════════════════════════════

def _precompute_all_goal_embeddings(
    possible_goals: np.ndarray, model_key: str,
) -> jnp.ndarray:
    """
    Precompute CLS embeddings for a fixed set of goal coordinates.

    Runs the HF tokenizer and model once per goal position in plain Python
    so that **no model inference is needed during the training loop**.

    Args:
        possible_goals: (N, 2) numpy array of goal coordinates.
        model_key: short name in MODEL_REGISTRY.

    Returns:
        (N, embed_dim) JAX array of CLS embeddings.
    """
    tokenizer, model, params = _load_model(model_key)
    embed_dim = MODEL_REGISTRY[model_key]["embed_dim"]

    # Build prompt strings — use natural number formatting for better
    # alignment with what the model saw during pre-training.
    prompts = [f"Your goal is ({x:.1f},{y:.1f})" for x, y in possible_goals]

    tokens = tokenizer(
        prompts, return_tensors="np", padding=True, truncation=True,
    )

    outputs = model(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        params=params,
        train=False,
    )
    return jnp.asarray(outputs.last_hidden_state[:, 0])  # (N, embed_dim)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Encoder classes
# ═══════════════════════════════════════════════════════════════════════════

class PrecomputedFrozenTextGoalEncoder(nn.Module):
    """
    Fast goal encoder for environments with a finite set of goal positions.

    Precomputes text model embeddings for all possible goals once at
    initialisation, then uses nearest-neighbour look-up at training time
    — no transformer inference is performed in the training loop.

    Args:
        output_dim: projection output dimension (default: 64).
        possible_goals: (N, 2) array of all goal coordinates that may appear.
        model_key: short name in MODEL_REGISTRY (default: ``"minilm"``).
    """
    output_dim: int = 64
    possible_goals: jnp.ndarray  # (N, 2)
    model_key: str = "minilm"

    def setup(self):
        goals_np = np.asarray(self.possible_goals)
        self._precomputed_goals = jnp.asarray(goals_np)
        self._precomputed_embs = _precompute_all_goal_embeddings(goals_np, self.model_key)
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
        cls_emb = self._precomputed_embs[nearest]  # (B, embed_dim)
        x = self.proj(cls_emb)
        return jax.lax.stop_gradient(x)


class FrozenTextGoalEncoder(nn.Module):
    """
    Encodes a 2D goal via a frozen HuggingFace text model.

    When *possible_goals* is provided, uses the fast precomputed path
    (``PrecomputedFrozenTextGoalEncoder``).  Otherwise falls back to
    running the full frozen model every forward pass — slower, but
    does not require a known goal set.

    Args:
        output_dim: projection output dimension (default: 64).
        model_key: short name in MODEL_REGISTRY (default: ``"minilm"``).
        possible_goals: (N, 2) array for precomputed look-up.  When
            ``None`` (default) the full model is run on every call.
    """
    output_dim: int = 64
    model_key: str = "minilm"
    possible_goals: jnp.ndarray | None = None

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        if self.possible_goals is not None:
            return PrecomputedFrozenTextGoalEncoder(
                output_dim=self.output_dim,
                possible_goals=self.possible_goals,
                model_key=self.model_key,
            )(g)

        # ── Direct (non-precomputed) path ────────────────────────────────
        _, model, _ = _load_model(self.model_key)
        params = self.param("model_params", lambda _: _load_model(self.model_key)[2])

        input_ids      = _tokenize_goal_prompt_jax(g)
        attention_mask = jnp.ones_like(input_ids)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            params=params,
            train=False,
        )
        cls_emb = jax.lax.stop_gradient(outputs.last_hidden_state[:, 0])

        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        x = nn.Dense(
            self.output_dim,
            kernel_init=lecun_uniform,
            bias_init=nn.initializers.zeros,
            name="proj",
        )(cls_emb)
        return jax.lax.stop_gradient(x)