import flax.linen as nn
from flax.linen.initializers import variance_scaling
import jax.numpy as jnp
from modules.utils import residual_block, TransformerBackbone

class SA_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish
            
        x = jnp.concatenate([s, a], axis=-1)
        #Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        #Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        #Final layer
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class TransformerSAEncoder(nn.Module):
    """Transformer-based state-action encoder.

    Concatenates state and action, then processes through TransformerBackbone
    to produce a 64-dim representation. The tokenization strategy can be
    customized by subclassing TransformerBackbone and overriding
    _vector_to_sequence.
    """
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    num_patches: int = 8
    dropout_rate: float = 0.0
    use_cls_token: bool = True

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([s, a], axis=-1)
        x = TransformerBackbone(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_patches=self.num_patches,
            dropout_rate=self.dropout_rate,
            use_cls_token=self.use_cls_token,
        )(x)

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class G_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish
        
        x = g
        #Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        #Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        #Final layer
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class TransformerGEncoder(nn.Module):
    """Transformer-based goal encoder.

    Processes the goal vector through TransformerBackbone to produce a
    64-dim representation. The tokenization strategy can be customized
    by subclassing TransformerBackbone and overriding _vector_to_sequence.
    """
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    num_patches: int = 8
    dropout_rate: float = 0.0
    use_cls_token: bool = True

    @nn.compact
    def __call__(self, g: jnp.ndarray):
        x = TransformerBackbone(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_patches=self.num_patches,
            dropout_rate=self.dropout_rate,
            use_cls_token=self.use_cls_token,
        )(g)

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x