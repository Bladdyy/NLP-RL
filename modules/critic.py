import flax.linen as nn
from flax.linen.initializers import variance_scaling
import jax
import jax.numpy as jnp

# from modules.utils import residual_block

lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


# =========================================================
# TRANSFORMER BLOCK
# =========================================================

class TransformerEncoderLayer(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    use_relu: int = 0
    norm_type: str = "layer_norm"
    skip_connections: int = 1

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

        if self.skip_connections == 1:
            x = x + residual

        # ---------------- MLP ----------------
        residual = x
        if self.norm_type == "layer_norm":
            x = nn.LayerNorm()(x)

        x = nn.Dense(self.mlp_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = activation(x)
        x = nn.Dense(self.d_model, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        if self.skip_connections == 1:
            x = x + residual

        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    d_model: int
    num_heads: int
    mlp_dim: int
    use_relu: int = 0
    norm_type: str = "layer_norm"
    skip_connections: int = 1

    @nn.compact
    def __call__(self, x):

        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                use_relu=self.use_relu,
                norm_type=self.norm_type,
                skip_connections=self.skip_connections
            )(x)

        return x


# =========================================================
# SA ENCODER
# =========================================================

class SA_encoder(nn.Module):
    network_width: int = 256
    network_depth: int = 4
    num_heads: int = 4
    use_relu: int = 0
    norm_type = "layer_norm"
    skip_connections: int = 1

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

        d_model = self.network_width

        # Unified token projection layer
        s = jnp.expand_dims(s, -1)
        a = jnp.expand_dims(a, -1)

        token_proj = nn.Dense(d_model, kernel_init=lecun_uniform, bias_init=bias_init)
        # SHARED projection dla state i action (ważne dla alignmentu)
        state_tokens = token_proj(s)
        action_tokens = token_proj(a)

        # Token type embeddings
        state_type = self.param(
            "state_type",
            nn.initializers.normal(),
            (1, 1, d_model)
        )

        action_type = self.param(
            "action_type",
            nn.initializers.normal(),
            (1, 1, d_model)
        )

        state_tokens = state_tokens + state_type
        action_tokens = action_tokens + action_type

        # CONCAT tokens
        x = jnp.concatenate([state_tokens, action_tokens], axis=1)

        # Transformer encoder
        x = TransformerEncoder(
            num_layers=self.network_depth,
            d_model=d_model,
            num_heads=self.num_heads,
            mlp_dim=d_model * 2,
            use_relu=self.use_relu,
            norm_type=self.norm_type,
            skip_connections=self.skip_connections,
        )(x)

        # Attention pooling
        attn_logits = nn.Dense(1)(x)          
        attn_weights = jax.nn.softmax(attn_logits, axis=1) 
        x = jnp.sum(x * attn_weights, axis=1) 

        # Output head
        x = nn.Dense(64, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        return x


# =========================================================
# G ENCODER
# =========================================================

class G_encoder(nn.Module):
    network_width: int = 256
    network_depth: int = 4
    num_heads: int = 4
    use_relu: int = 0
    norm_type = "layer_norm"
    skip_connections: int = 1

    @nn.compact
    def __call__(self, g: jnp.ndarray):

        d_model = self.network_width

        # Token embedding
        g = jnp.expand_dims(g, -1)

        token_proj = nn.Dense(d_model, kernel_init=lecun_uniform, bias_init=bias_init)
        # Shared projection
        goal_tokens = token_proj(g)

        # Transformer encoder
        x = TransformerEncoder(
            num_layers=self.network_depth,
            d_model=d_model,
            num_heads=self.num_heads,
            mlp_dim=d_model * 2,
            use_relu=self.use_relu,
            norm_type=self.norm_type,
            skip_connections=self.skip_connections,
        )(goal_tokens)

        # Attention pooling
        attn_logits = nn.Dense(1)(x)     
        attn_weights = jax.nn.softmax(attn_logits, axis=1) 
        x = jnp.sum(x * attn_weights, axis=1) 

        # Output head

        x = nn.Dense(64, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        return x
    

# class SA_encoder(nn.Module):
#     norm_type = "layer_norm"
#     network_width: int = 1024
#     network_depth: int = 4
#     skip_connections: int = 0
#     use_relu: int = 0
#     @nn.compact
#     def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

#         lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
#         bias_init = nn.initializers.zeros
        
#         if self.norm_type == "layer_norm":
#             normalize = lambda x: nn.LayerNorm()(x)
#         else:
#             normalize = lambda x: x
        
#         if self.use_relu:
#             activation = nn.relu
#         else:
#             activation = nn.swish
            
#         x = jnp.concatenate([s, a], axis=-1)
#         #Initial layer
#         x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
#         x = normalize(x)
#         x = activation(x)
#         #Residual blocks
#         for i in range(self.network_depth // 4):
#             x = residual_block(x, self.network_width, normalize, activation)
#         #Final layer
#         x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
#         return x
    
# class G_encoder(nn.Module):
#     norm_type = "layer_norm"
#     network_width: int = 1024
#     network_depth: int = 4
#     skip_connections: int = 0
#     use_relu: int = 0
#     @nn.compact
#     def __call__(self, g: jnp.ndarray):

#         lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
#         bias_init = nn.initializers.zeros

#         if self.norm_type == "layer_norm":
#             normalize = lambda x: nn.LayerNorm()(x)
#         else:
#             normalize = lambda x: x
        
#         if self.use_relu:
#             activation = nn.relu
#         else:
#             activation = nn.swish
        
#         x = g
#         #Initial layer
#         x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
#         x = normalize(x)
#         x = activation(x)
#         #Residual blocks
#         for i in range(self.network_depth // 4):
#             x = residual_block(x, self.network_width, normalize, activation)
#         #Final layer
#         x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
#         return x