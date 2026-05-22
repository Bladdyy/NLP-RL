import flax.linen as nn
from flax.linen.initializers import variance_scaling
import jax
import jax.numpy as jnp

from modules.utils import residual_block, TransformerEncoder

lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


"""
Transformer encoder critics.

SA_TransformerEncoder: A critic that takes state and action as input, treats whole vectors as separate tokens (ending up with 2 tokens),
 and processes them through a transformer encoder. 
 This allows the model to learn complex interactions between state and action.

G_TransformerEncoder: A critic that takes a goal representation as input, treats each element as a token, 
 and processes it through a transformer encoder. 
 This allows the model to learn complex relationships within the goal representation.
"""

class SA_TransformerEncoder(nn.Module):
    network_width: int = 256
    network_depth: int = 4
    num_heads: int = 4
    use_relu: int = 0
    norm_type = "layer_norm"

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        # Change state and action into token embeddings.
        state_embedding = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(s)
        action_embedding = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(a)

        # Add a token dimension (treat state and action as separate tokens)
        state_token = jnp.expand_dims(state_embedding, axis=1)
        action_token = jnp.expand_dims(action_embedding, axis=1)

        # Add your token type embeddings (learned position/type signals).
        state_type = self.param("state_type", nn.initializers.normal(), (1, 1, self.network_width))
        action_type = self.param("action_type", nn.initializers.normal(), (1, 1, self.network_width))
        state_token = state_token + state_type
        action_token = action_token + action_type

        # Concatenate into a final sequence of exactly 2 tokens
        x = jnp.concatenate([state_token, action_token], axis=1)

        x = TransformerEncoder(
            num_layers=self.network_depth // 4,
            d_model=self.network_width,
            num_heads=self.num_heads,
            mlp_dim=self.network_width * 2,
            use_relu=self.use_relu,
            norm_type=self.norm_type,
        )(x)

        attn_logits = nn.Dense(1)(x)          
        attn_weights = jax.nn.softmax(attn_logits, axis=1) 
        x = jnp.sum(x * attn_weights, axis=1) 
        x = nn.Dense(64, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        return x


class G_TransformerEncoder(nn.Module):
    network_width: int = 256
    network_depth: int = 4
    num_heads: int = 4
    use_relu: int = 0
    norm_type = "layer_norm"

    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Token embedding
        g = jnp.expand_dims(g, -1)

        token_proj = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)
        goal_tokens = token_proj(g)

        # Transformer encoder
        x = TransformerEncoder(
            num_layers=self.network_depth // 4,
            d_model=self.network_width,
            num_heads=self.num_heads,
            mlp_dim=self.network_width * 2,
            use_relu=self.use_relu,
            norm_type=self.norm_type,
        )(goal_tokens)

        # Attention pooling
        attn_logits = nn.Dense(1)(x)     
        attn_weights = jax.nn.softmax(attn_logits, axis=1) 
        x = jnp.sum(x * attn_weights, axis=1) 

        # Output head

        x = nn.Dense(64, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        return x
    
"""
MLP critics.

SA_MlpEncoder: A critic that takes state and action as input, concatenates them, and processes through an MLP.

G_MlpEncoder: A critic that takes a goal representation as input and processes it through an MLP.
"""

class SA_MlpEncoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    use_relu: int = 0
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
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
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)

        #Residual blocks
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)

        #Final layer
        x = nn.Dense(64, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class G_MlpEncoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    use_relu: int = 0
    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
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
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)

        #Residual blocks
        for _ in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        
        #Final layer
        x = nn.Dense(64, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x