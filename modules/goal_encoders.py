import jax
import numpy as np
import flax.linen as nn
from flax.linen.initializers import variance_scaling
import jax.numpy as jnp
from modules.utils import TransformerBackbone
from modules.frozen_text_encoder import (
    FrozenTextGoalEncoder,
    PrecomputedFrozenTextGoalEncoder,
    U4_EXACT_PATH_PROMPTS,
    U4_HIGH_LEVEL_PROMPTS,
)


class SemanticTransformerGEncoderText(nn.Module):
    """
    Goal encoder using either frozen text or trainable embeddings.

    ``embed_source="frozen"`` (default):
        Converts the 2-dim goal into a text prompt
        ``"Your goal is (x,y)"`` and encodes it via
        ``PrecomputedFrozenTextGoalEncoder`` when *possible_goals* is set,
        falling back to the slower ``FrozenTextGoalEncoder`` otherwise.

    ``embed_source="trainable"``:
        Delegates to ``TrainableEmbeddingGoalEncoder`` (learned lookup
        table). Requires *possible_goals*.

    Either way the output is ``(B, output_dim)`` — ready for the critic's
    InfoNCE loss.

    Args:
        output_dim: output embedding dimension (default: 64).
        possible_goals: (N, 2) array for precomputed look-up.
            Required when *embed_source* is ``"trainable"``.
        model_key: text model short name (default: ``"minilm"``).
        embed_source: ``"frozen"`` (pretrained text) or
            ``"trainable"`` (learned lookup).
        description_type: ``"coordinates"``, ``"exact"``, or
            ``"high_level"`` (default: ``"coordinates"``).
        pooling: ``"cls"``, ``"mean"``, or ``"token"`` (default:
            ``"cls"``). Only used when *embed_source* is ``"frozen"``.
    """
    output_dim: int = 64
    possible_goals: jnp.ndarray | None = None
    model_key: str = "minilm"
    embed_source: str = "frozen"  # "frozen" or "trainable"
    description_type: str = "coordinates"
    pooling: str = "cls"
    precomputed_embs: jnp.ndarray | None = None

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        if self.embed_source == "trainable":
            if self.possible_goals is None:
                raise ValueError(
                    "embed_source='trainable' requires a finite goal set; "
                    "possible_goals must be provided."
                )
            return TrainableEmbeddingGoalEncoder(
                output_dim=self.output_dim,
                possible_goals=self.possible_goals,
                description_type=self.description_type,
            )(g)
        return FrozenTextGoalEncoder(
            output_dim=self.output_dim,
            model_key=self.model_key,
            possible_goals=self.possible_goals,
            pooling=self.pooling,
            description_type=self.description_type,
            precomputed_embs=self.precomputed_embs,
        )(g)


def _prompt_text(goal: tuple[float, float], description_type: str) -> str:
    """
    Generate the full prompt string for a goal, matching the format
    used by the frozen text encoder.
    """
    if description_type == "coordinates":
        x, y = goal
        return f"Your goal is ({x:.2f},{y:.2f})"
    if description_type == "exact":
        return U4_EXACT_PATH_PROMPTS[goal]
    if description_type == "high_level":
        return U4_HIGH_LEVEL_PROMPTS[goal]
    raise ValueError(f"Unknown description_type: {description_type}")


class TrainableEmbeddingGoalEncoder(nn.Module):
    """
    Trainable goal encoder with per-goal attention over prompt words.

    Generates the same prompt string as ``FrozenTextGoalEncoder``,
    splits it into words, and embeds each word via a shared vocabulary.
    A per-goal learned query vector then attends over the word
    embeddings, letting each goal learn which words matter most for its
    prompt.  The attention-weighted sum replaces mean pooling to avoid
    gradient dilution from shared words.

    Structurally analogous to the frozen encoder — prompt → tokenise →
    embed → pool — but with learned embeddings and learned attention
    instead of frozen BERT.

    Only applicable when the goal space is finite (``possible_goals``).

    Args:
        output_dim: embedding dimension (default: 64).
        possible_goals: (N, 2) array of all goal coordinates that may
            appear during training.
        description_type: ``"coordinates"``, ``"exact"``, or
            ``"high_level"`` (default: ``"coordinates"``).  Selects
            the prompt template used to generate the word sequence.
    """
    possible_goals: jnp.ndarray  # (N, 2)
    output_dim: int = 64
    description_type: str = "coordinates"

    def setup(self):
        goals_np = np.asarray(self.possible_goals)
        self._precomputed_goals = jnp.asarray(goals_np)
        n_goals = len(goals_np)

        # Build vocabulary from all prompts for this description_type.
        word_seqs: list[list[str]] = []
        words_set: set[str] = set()
        for coord in goals_np:
            prompt = _prompt_text(tuple(coord), self.description_type)
            words = [w.strip(".,!?;:").lower() for w in prompt.split()]
            words_set.update(words)
            word_seqs.append(words)

        # index 0 = <pad>, actual words start at 1
        vocab = sorted(words_set)
        word_to_idx = {w: i + 1 for i, w in enumerate(vocab)}
        vocab_size = len(vocab) + 1  # +1 for <pad>

        max_len = max(len(ws) for ws in word_seqs)
        indices_np = np.zeros((n_goals, max_len), dtype=np.int32)
        for i, words in enumerate(word_seqs):
            for j, w in enumerate(words):
                indices_np[i, j] = word_to_idx[w]

        self._goal_word_indices = jnp.asarray(indices_np)  # (N, seq_len)
        self.word_embed = nn.Embed(
            num_embeddings=vocab_size,
            features=self.output_dim,
            name="word_embed",
        )
        # Per-goal attention query — each goal learns which words in
        # its prompt are informative and which are filler.
        self.goal_query = nn.Embed(
            num_embeddings=n_goals,
            features=self.output_dim,
            name="goal_query",
        )

    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        # (B, 2) -> nearest goal index
        dists = jnp.linalg.norm(
            g[:, None, :] - self._precomputed_goals[None, :, :],
            axis=-1,
        )  # (B, N)
        nearest = jnp.argmin(dists, axis=-1)  # (B,)

        # (B, seq_len) word indices for this goal
        word_indices = self._goal_word_indices[nearest]
        word_embs = self.word_embed(word_indices)  # (B, seq_len, output_dim)

        # (B, output_dim) query vector for this goal
        query = self.goal_query(nearest)  # (B, output_dim)

        # Attention: each goal's query scores each word in its prompt.
        # Padding positions (index 0) are masked out before softmax.
        scores = jnp.einsum("bd,bsd->bs", query, word_embs) / jnp.sqrt(
            self.output_dim
        )  # (B, seq_len)
        mask = (word_indices > 0).astype(jnp.float32)  # (B, seq_len)
        scores = scores - 1e9 * (1.0 - mask)
        attn = nn.softmax(scores)  # (B, seq_len)

        return jnp.einsum("bs,bsd->bd", attn, word_embs)  # (B, output_dim)


class HybridGoalEncoder(nn.Module):
    """
    Goal encoder that combines raw coordinates with a text embedding.

    Internally computes a text embedding (from a frozen pretrained text
    model or a trainable lookup) and concatenates it with the raw
    coordinates before passing through either an MLP or a semantic
    transformer backbone.

    The external interface stays ``(B, 2) -> (B, output_dim)`` so no
    changes to ``loss.py`` or ``actor.py`` are needed.

    Args:
        output_dim: final embedding dimension (default: 64).
        backbone: ``"mlp"`` or ``"semantic"`` — processing backbone.
        embed_source: ``"frozen"`` (pretrained text model) or
            ``"trainable"`` (``nn.Embed`` lookup).
        possible_goals: (N, 2) array for discrete look-up.  Required when
            *embed_source* is ``"trainable"``; optional for ``"frozen"``
            (falls back to the direct non-precomputed path).
        model_key: text model to use when *embed_source* is ``"frozen"``.
        pooling: ``"cls"``, ``"mean"``, or ``"token"``. Token-mode
            outputs are mean-pooled for the MLP backbone or kept as
            separate tokens for the transformer backbone.
            Ignored when *embed_source* is ``"trainable"``.
        mlp_width: hidden width of the MLP backbone (default: 256).
        transformer_embed_dim: token embedding dimension for the
            semantic transformer backbone (default: 144).

    Raises:
        ValueError: if *embed_source* is ``"trainable"`` and
            *possible_goals* is ``None``.
    """
    output_dim: int = 64
    backbone: str = "mlp"  # "mlp" or "semantic"
    embed_source: str = "frozen"  # "frozen" or "trainable"
    possible_goals: jnp.ndarray | None = None
    model_key: str = "minilm"
    pooling: str = "cls"
    mlp_width: int = 256
    transformer_embed_dim: int = 144
    description_type: str = "coordinates"
    precomputed_embs: jnp.ndarray | None = None
    cross_neg_count: int = 32
    """Number of cross1 + cross2 negative pairs per sample in pairwise mode."""

    @nn.compact
    def __call__(
        self, g: jnp.ndarray | None = None, *, grid: bool = False, pairwise: bool = False
    ) -> jnp.ndarray:
        """Encode goals into embeddings.

        Args:
            g: (B, 2) goal coordinates. Ignored when ``grid=True``.
            grid: When True, ignore *g* and compute the ``(N, N, output_dim)``
                grid of all composite embeddings ``f(raw_a, text_b)`` over the
                ``N = len(possible_goals)`` distinct goals. Used by the cross
                InfoNCE loss to gather negatives by goal index in O(N^2)
                encoder work instead of O(B*K).
            pairwise: When True, return a ``(B, 1 + 2*K, output_dim)``
                tensor where ``K = cross_neg_count`` with diagonal + cross1 +
                cross2 blocks (legacy per-batch path; ``grid`` is preferred).

        Returns:
            ``(B, output_dim)`` normally; ``(B, 1 + 2*K, output_dim)`` when
            ``pairwise=True``; ``(N, N, output_dim)`` when ``grid=True``.
        """
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        bias = nn.initializers.zeros

        if self.embed_source == "trainable" and self.possible_goals is None:
            raise ValueError(
                "embed_source='trainable' requires a finite goal set; "
                "possible_goals must be provided."
            )

        if grid:
            if self.possible_goals is None:
                raise ValueError("grid=True requires possible_goals to be set")
            g_in = self.possible_goals          # (N, 2)
        else:
            g_in = g                            # (B, 2)

        # ── Compute shared components ─────────────────────────────────────
        if self.backbone == "mlp":
            text_output_dim = self.mlp_width
        else:
            text_output_dim = self.transformer_embed_dim

        if self.embed_source == "trainable":
            text_repr = TrainableEmbeddingGoalEncoder(
                output_dim=text_output_dim,
                possible_goals=self.possible_goals,
                description_type=self.description_type,
            )(g_in)
        else:
            text_repr = PrecomputedFrozenTextGoalEncoder(
                possible_goals=self.possible_goals,
                output_dim=text_output_dim,
                model_key=self.model_key,
                pooling=self.pooling,
                description_type=self.description_type,
                precomputed_embs=self.precomputed_embs,
            )(g_in)

        # ── MLP backbone ────────────────────────────────────────────────
        if self.backbone == "mlp":
            # MLP needs a fixed-size vector → pool to single if 3D.
            if text_repr.ndim == 3:
                text_repr = text_repr.mean(axis=1)
            # Project raw coords up so they don't get drowned by the
            # high-dim text representation.
            g_proj = nn.Dense(64, kernel_init=lecun, bias_init=bias, name="g_proj_mlp")(g_in)
            D_text = text_repr.shape[-1]

            W1 = self.param(
                "mlp_dense1_kernel", lecun, (64 + D_text, self.mlp_width),
            )
            W1_raw = W1[:64, :]
            W1_text = W1[64:, :]
            b1 = self.param(
                "mlp_dense1_bias", nn.initializers.zeros, (self.mlp_width,),
            )
            W2 = self.param(
                "mlp_dense2_kernel", lecun, (self.mlp_width, self.output_dim),
            )
            b2 = self.param(
                "mlp_dense2_bias", nn.initializers.zeros, (self.output_dim,),
            )

            if grid:
                # (N, N, mlp_width) via outer sum of the two linear halves.
                h_raw = g_proj @ W1_raw             # (N, mlp_width)
                h_text = text_repr @ W1_text         # (N, mlp_width)
                h = h_raw[:, None, :] + h_text[None, :, :] + b1
                h = nn.swish(h)
                x = h @ W2 + b2                      # (N, N, output_dim)
                return x

            if pairwise:
                B = g_in.shape[0]
                K = min(self.cross_neg_count, (B - 1) // 2)

                row_idx = jnp.arange(B)[:, None]
                col_idx = (row_idx + 1 + jnp.arange(K)[None, :]) % B

                h_raw = g_proj @ W1_raw
                h_text = text_repr @ W1_text

                diag_h = (h_raw + h_text + b1)[:, None, :]
                h_raw_bc = jnp.broadcast_to(h_raw[:, None, :], (B, K, self.mlp_width))
                x1_h = h_raw_bc + h_text[col_idx] + b1
                h_raw_c2 = h_raw[col_idx]
                h_text_bc = jnp.broadcast_to(h_text[:, None, :], (B, K, self.mlp_width))
                x2_h = h_raw_c2 + h_text_bc + b1

                h = jnp.concatenate([diag_h, x1_h, x2_h], axis=1)
                h = h.reshape(B * (1 + 2 * K), self.mlp_width)
                h = nn.swish(h)
                x = h @ W2 + b2
                return x.reshape(B, 1 + 2 * K, self.output_dim)

            x = jnp.concatenate([g_proj, text_repr], axis=-1)
            x = nn.swish(x @ W1 + b1)
            x = x @ W2 + b2
            return x

        # ── Semantic transformer backbone ───────────────────────────────
        g_token = nn.Dense(
            self.transformer_embed_dim, kernel_init=lecun, bias_init=bias, name="g_proj",
        )(g_in)

        if grid:
            if text_repr.ndim == 3:
                text_repr = text_repr.mean(axis=1)
            N = g_in.shape[0]
            raw_tok = jnp.broadcast_to(
                g_token[:, None, None, :], (N, N, 1, self.transformer_embed_dim)
            )
            text_tok = jnp.broadcast_to(
                text_repr[None, :, None, :], (N, N, 1, self.transformer_embed_dim)
            )
            all_tokens = jnp.concatenate([raw_tok, text_tok], axis=2)
            all_tokens = all_tokens.reshape(N * N, 2, self.transformer_embed_dim)
            x = nn.remat(TransformerBackbone)(
                embed_dim=self.transformer_embed_dim,
                num_layers=4, num_heads=4, mlp_ratio=4,
                num_patches=0, dropout_rate=0.0, pooling="cls",
            )(all_tokens)
            x = nn.Dense(self.output_dim, kernel_init=lecun, bias_init=bias)(x)
            return x.reshape(N, N, self.output_dim)

        if pairwise:
            B = g_in.shape[0]
            K = min(self.cross_neg_count, (B - 1) // 2)
            if text_repr.ndim == 3:
                text_repr = text_repr.mean(axis=1)

            diag_tokens = jnp.stack([g_token, text_repr], axis=1)[:, None, :, :]
            row_idx = jnp.arange(B)[:, None]
            col_idx = (row_idx + 1 + jnp.arange(K)[None, :]) % B
            raw_tok_bc = jnp.broadcast_to(
                g_token[:, None, None, :], (B, K, 1, self.transformer_embed_dim)
            )
            tokens_c1 = jnp.concatenate(
                [raw_tok_bc, text_repr[col_idx][:, :, None, :]], axis=2
            )
            raw_c2 = g_token[col_idx][:, :, None, :]
            text_tok_bc = jnp.broadcast_to(
                text_repr[:, None, None, :], (B, K, 1, self.transformer_embed_dim)
            )
            tokens_c2 = jnp.concatenate([raw_c2, text_tok_bc], axis=2)

            all_tokens = jnp.concatenate([diag_tokens, tokens_c1, tokens_c2], axis=1)
            all_tokens = all_tokens.reshape(B * (1 + 2 * K), 2, self.transformer_embed_dim)
            x = nn.remat(TransformerBackbone)(
                embed_dim=self.transformer_embed_dim,
                num_layers=4, num_heads=4, mlp_ratio=4,
                num_patches=0, dropout_rate=0.0, pooling="cls",
            )(all_tokens)
            x = nn.Dense(self.output_dim, kernel_init=lecun, bias_init=bias)(x)
            return x.reshape(B, 1 + 2 * K, self.output_dim)

        # ── Standard (non-pairwise) semantic backbone ─────────────────────
        if text_repr.ndim == 3:
            tokens = jnp.concatenate([g_token[:, None, :], text_repr], axis=1)
        else:
            tokens = jnp.stack([g_token, text_repr], axis=1)

        x = nn.remat(TransformerBackbone)(
            embed_dim=self.transformer_embed_dim,
            num_layers=4, num_heads=4, mlp_ratio=4,
            num_patches=0, dropout_rate=0.0, pooling="cls",
        )(tokens)
        x = nn.Dense(self.output_dim, kernel_init=lecun, bias_init=bias)(x)
        return x

    def goal_indices(self, g: jnp.ndarray) -> jnp.ndarray:
        """Map (B, 2) batch goals to (B,) indices into possible_goals.

        Uses nearest-neighbour matching, identical to the mapping the
        frozen-text encoder applies internally, so ``grid[idx, :]`` is
        consistent with the per-sample encoder output.
        """
        if self.possible_goals is None:
            raise ValueError("goal_indices requires possible_goals to be set")
        dists = jnp.linalg.norm(
            g[:, None, :] - self.possible_goals[None, :, :], axis=-1,
        )
        return jnp.argmin(dists, axis=-1)
