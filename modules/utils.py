import flax.linen as nn
from flax.linen.initializers import variance_scaling
import jax.numpy as jnp

lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


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
        x = residual + x

        # Pre-LN FFN
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.swish(x)
        x = nn.Dense(self.embed_dim)(x)
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
    use_cls_token: bool = True

    @nn.compact
    def __call__(self, x, deterministic=True):
        # 1. Flat vector -> sequence of tokens (only if input is 2D)
        # If x is already (batch, seq_len, embed_dim), skip tokenization
        if x.ndim == 2:
            x = self._vector_to_sequence(x)

        # 2. Optional CLS token
        if self.use_cls_token:
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
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = jnp.mean(x, axis=1)

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
