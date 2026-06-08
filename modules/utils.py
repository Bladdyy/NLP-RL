import flax.linen as nn
from flax.linen.initializers import variance_scaling
import jax
import jax.numpy as jnp

lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros

def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x


class TransformerEncoderLayer(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    use_relu: int = 0
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray):

        activation = nn.relu if self.use_relu else nn.swish

        # ---------------- Self-Attention ----------------
        residual = x
        if self.norm_type == "layer_norm":
            x = nn.LayerNorm()(x)

        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            kernel_init=lecun_uniform,
            bias_init=bias_init,
        )(x)

        x = x + residual

        # ---------------- MLP ----------------
        residual = x
        if self.norm_type == "layer_norm":
            x = nn.LayerNorm()(x)

        x = nn.Dense(self.mlp_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = activation(x)
        x = nn.Dense(self.d_model, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        x = x + residual

        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    d_model: int
    num_heads: int
    mlp_dim: int
    use_relu: int = 0
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x):

        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                use_relu=self.use_relu,
                norm_type=self.norm_type,
            )(x)

        return x


def apply_pooling(x, pooling_type):
    if pooling_type == "cls":
        x = x[:, 0, :]
    elif pooling_type == "attention":
        attn_logits = nn.Dense(1)(x)          
        attn_weights = jax.nn.softmax(attn_logits, axis=1) 
        x = jnp.sum(x * attn_weights, axis=1)
    elif pooling_type == "mean":
        x = jnp.mean(x, axis=1)
    else:
        raise ValueError(f"Unknown pooling_type: {pooling_type}")
    return x