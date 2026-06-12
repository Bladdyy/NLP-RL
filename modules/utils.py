import flax.linen as nn
from flax.linen.initializers import variance_scaling
import jax
import jax.numpy as jnp

lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


def scale_transformer_projections(params, num_layers):
    """Scale the final projection weights in transformer blocks by ``1 / sqrt(2 * num_layers)``.

    This implements the DeepNet initialisation scheme (Wang et al., 2022).
    The attention output projection (named ``out`` in
    ``MultiHeadDotProductAttention``) and the FFN output projection (named
    ``ffn_proj``) are scaled so that the residual stream is initially dominated
    by the identity path, stabilising training for very deep transformers.

    Args:
        params: Flax parameter tree containing transformer blocks.
        num_layers: Total number of transformer layers (N in 1/sqrt(2N)).

    Returns:
        New parameter tree with scaled final projection kernels.
    """
    scale = 1.0 / jnp.sqrt(2.0 * num_layers)

    def _scale_fn(path, value):
        path_str = jax.tree_util.keystr(path)
        if not path_str.endswith('.kernel'):
            return value
        if '.out.' in path_str or '.ffn_proj.' in path_str:
            return value * scale
        return value

    return jax.tree_util.tree_map_with_path(_scale_fn, params)


def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x


class TransformerBlock(nn.Module):
    """A single transformer encoder block with pre-LN architecture."""
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        # Pre-LN self-attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
        )(inputs_q=x, inputs_kv=x, deterministic=deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = residual + x

        # Pre-LN FFN
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.mlp_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.embed_dim, kernel_init=lecun_unfirom, bias_init=bias_init, name='ffn_proj')(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = residual + x

        return x


class TransformerBackbone(nn.Module):
    """Transformer backbone that converts a flat vector to a sequence, processes it with
    transformer blocks, and pools back to a single vector.

    The tokenization step (_vector_to_sequence) is designed to be easily overridden
    by subclassing, so you can experiment with different ways of converting a flat
    vector to a sequence of tokens without changing anything else.

    Default tokenizer: splits the input into evenly-sized patches and linearly projects
    each patch to embed_dim.
    """
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    num_patches: int = 8
    dropout_rate: float = 0.0
    pooling: str = "cls"  # "cls", "mean", or "flatten"

    @nn.compact
    def __call__(self, x, deterministic=True):
        # 1. Flat vector -> sequence of tokens (only if input is 2D)
        # If x is already (batch, seq_len, embed_dim), skip tokenization
        if x.ndim == 2:
            x = self._vector_to_sequence(x)

        # 2. Optional CLS token
        if self.pooling == "cls":
            cls_token = self.param(
                'cls_token',
                nn.initializers.normal(stddev=0.02),
                (1, 1, self.embed_dim)
            )
            cls_token = jnp.tile(cls_token, [x.shape[0], 1, 1])
            x = jnp.concatenate([cls_token, x], axis=1)

        # 3. Positional embeddings
        num_tokens = x.shape[1]
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, num_tokens, self.embed_dim)
        )
        x = x + pos_embed

        # 4. Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.embed_dim * self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i}',
            )(x, deterministic=deterministic)

        # 5. Pool to single vector
        if self.pooling == "cls":
            x = x[:, 0]
        elif self.pooling == "mean":
            x = jnp.mean(x, axis=1)
        elif self.pooling == "flatten":
            b, s, d = x.shape
            x = x.reshape(b, s * d)
            x = nn.LayerNorm()(x)
            x = nn.Dense(d, kernel_init=nn.initializers.xavier_uniform(), name='pool_proj')(x)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return x

    def _vector_to_sequence(self, x):
        """Convert a flat vector to a sequence of patch tokens.

        Override this method in a subclass to experiment with different
        tokenization strategies (convolutional, learned grouping, random projections, etc.).

        Args:
            x: jnp.ndarray of shape [batch, input_dim]

        Returns:
            jnp.ndarray of shape [batch, num_patches, embed_dim]
        """
        input_dim = x.shape[-1]
        patch_size = (input_dim + self.num_patches - 1) // self.num_patches
        target_dim = self.num_patches * patch_size
        if target_dim > input_dim:
            x = jnp.pad(x, ((0, 0), (0, target_dim - input_dim)))
        x = x.reshape(x.shape[0], self.num_patches, patch_size)
        x = nn.Dense(self.embed_dim, name='patch_embed', kernel_init=nn.initializers.xavier_uniform())(x)
        return x
