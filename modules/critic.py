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
    token_mode: str = "semantic_batch"  # "flatten", "two_tokens", or "semantic_batch"
    pooling_type: str = "cls"  # "attention" or "cls"

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        if self.token_mode == "flatten":
            s = jnp.expand_dims(s, axis=-1)  # (batch, state_dim, 1)
            a = jnp.expand_dims(a, axis=-1)  # (batch, action_dim, 1)
            state_token = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(s)   # (batch, state_dim, width)
            action_token = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(a)  # (batch, action_dim, width)

            state_dim = state_token.shape[1]
            action_dim = action_token.shape[1]
            state_type = self.param("state_type", nn.initializers.normal(), (1, state_dim, self.network_width))
            action_type = self.param("action_type", nn.initializers.normal(), (1, action_dim, self.network_width))
            
            state_token = state_token + state_type
            action_token = action_token + action_type
            tokens = jnp.concatenate([state_token, action_token], axis=1)
        
        elif self.token_mode == "two_tokens":
            state_embedding = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(s)
            action_embedding = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(a)
            state_token = jnp.expand_dims(state_embedding, axis=1)   # (batch, 1, width)
            action_token = jnp.expand_dims(action_embedding, axis=1)  # (batch, 1, width)

            state_type = self.param("state_type", nn.initializers.normal(), (1, 1, self.network_width))
            action_type = self.param("action_type", nn.initializers.normal(), (1, 1, self.network_width))
            
            state_token = state_token + state_type
            action_token = action_token + action_type
            tokens = jnp.concatenate([state_token, action_token], axis=1)
        
        elif self.token_mode == "semantic_batch":
            # Parse 29-dim state (ant):
            #   [0:7]   = root qpos
            #   [7:15]  = hinge qpos
            #   [15:21] = root qvel
            #   [21:29] = hinge qvel
            root_qpos = s[..., :7]
            hinge_qpos = s[..., 7:15]
            root_qvel = s[..., 15:21]
            hinge_qvel = s[..., 21:29]

            # Reorder action to match qpos/qvel hinge order
            action_reorder = jnp.array([2, 3, 4, 5, 6, 7, 0, 1])
            a = a[..., action_reorder]

            # Body token: root qpos + root qvel
            body_raw = jnp.concatenate([root_qpos, root_qvel], axis=-1)
            body_token = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init, name='body_embed')(body_raw)
            body_token = jnp.expand_dims(body_token, axis=1)

            # Build 8 hinge tokens
            hinge_raw = jnp.stack([hinge_qpos, hinge_qvel, a], axis=-1)
            hinge_tokens = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init, name='hinge_embed')(hinge_raw)

            # Assemble token sequence
            tokens = jnp.concatenate([body_token, hinge_tokens], axis=1)
        else:
            raise ValueError(f"Unknown token_mode: {self.token_mode}")

        # Add CLS token only if using cls pooling
        if self.pooling_type == "cls":
            batch_size = tokens.shape[0]
            cls_token = self.param("cls_token", nn.initializers.normal(), (1, 1, self.network_width))
            cls_token = jnp.broadcast_to(cls_token, (batch_size, 1, self.network_width))
            x = jnp.concatenate([cls_token, tokens], axis=1)
        else:
            x = tokens

        x = TransformerEncoder(
            num_layers=self.network_depth // 4,
            d_model=self.network_width,
            num_heads=self.num_heads,
            mlp_dim=self.network_width * 2,
            use_relu=self.use_relu,
            norm_type=self.norm_type,
        )(x)

        if self.pooling_type == "cls":
            x = x[:, 0, :]
        else:
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
    pooling_type: str = "cls"  # "attention" or "cls"

    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Token embedding
        g = jnp.expand_dims(g, -1)

        token_proj = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)
        goal_tokens = token_proj(g)

        # Add CLS token only if using cls pooling
        if self.pooling_type == "cls":
            batch_size = goal_tokens.shape[0]
            cls_token = self.param("cls_token", nn.initializers.normal(), (1, 1, self.network_width))
            cls_token = jnp.broadcast_to(cls_token, (batch_size, 1, self.network_width))
            goal_tokens = jnp.concatenate([cls_token, goal_tokens], axis=1)

        # Transformer encoder
        x = TransformerEncoder(
            num_layers=self.network_depth // 4,
            d_model=self.network_width,
            num_heads=self.num_heads,
            mlp_dim=self.network_width * 2,
            use_relu=self.use_relu,
            norm_type=self.norm_type,
        )(goal_tokens)

        # Pooling
        if self.pooling_type == "cls":
            x = x[:, 0, :]
        else:
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