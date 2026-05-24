import flax.linen as nn
from flax.linen.initializers import variance_scaling
import jax
import jax.numpy as jnp
from modules.utils import residual_block, TransformerBackbone
from utils import Transition

class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
            
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
    
        #Initial layer
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        #Residual blocks
        for i in range(self.network_depth // 4):
            x = residual_block(x, self.network_width, normalize, activation)
        #Final layer
        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std
    

class TransformerActor(nn.Module):
    """Transformer-based actor that outputs action mean and log_std.

    Uses TransformerBackbone internally for sequence processing.
    The tokenization strategy can be customized by subclassing TransformerBackbone
    and overriding _vector_to_sequence.
    """
    action_size: int
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    num_patches: int = 8
    dropout_rate: float = 0.0
    use_cls_token: bool = True
    pooling: str = "cls"
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        x = TransformerBackbone(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_patches=self.num_patches,
            dropout_rate=self.dropout_rate,
            use_cls_token=self.use_cls_token,
            pooling=self.pooling,
        )(x)

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std


class SemanticTransformerActor(nn.Module):
    """Actor with semantically meaningful tokens.

    Parses the 31-dim observation (root qpos, hinge qpos, root qvel, hinge qvel,
    goal) into 10 semantically meaningful tokens:
      - 1 body token: root qpos (7) + root qvel (6) = 13 dims
      - 8 hinge tokens: each (hinge qpos, hinge qvel) = 2 dims
      - 1 goal token: goal / target position (2 dims)

    Each token type uses a separate Dense embedding layer.
    """
    action_size: int
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    use_cls_token: bool = True
    pooling: str = "cls"
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Parse 31-dim observation:
        #   [0:7]   = root qpos
        #   [7:15]  = hinge qpos [hip_1, ankle_1, ..., hip_4, ankle_4]
        #   [15:21] = root qvel
        #   [21:29] = hinge qvel
        #   [29:31] = goal / target_pos
        root_qpos = x[..., :7]
        hinge_qpos = x[..., 7:15]
        root_qvel = x[..., 15:21]
        hinge_qvel = x[..., 21:29]
        goal_token = x[..., 29:31]

        # Body token: root qpos + root qvel = 13 dims
        body_raw = jnp.concatenate([root_qpos, root_qvel], axis=-1)
        body_token = nn.Dense(
            self.embed_dim,
            kernel_init=lecun_unfirom,
            bias_init=bias_init,
            name='body_embed',
        )(body_raw)  # (batch, embed_dim)

        # 8 hinge tokens, each (qpos_i, qvel_i) = 2 dims
        hinge_raw = jnp.stack([hinge_qpos, hinge_qvel], axis=-1)  # (batch, 8, 2)
        hinge_tokens = nn.Dense(
            self.embed_dim,
            kernel_init=lecun_unfirom,
            bias_init=bias_init,
            name='hinge_embed',
        )(hinge_raw)  # (batch, 8, embed_dim)

        # Goal token: 2 dims
        goal_token = nn.Dense(
            self.embed_dim,
            kernel_init=lecun_unfirom,
            bias_init=bias_init,
            name='goal_embed',
        )(goal_token)[:, jnp.newaxis, :]  # (batch, 1, embed_dim)

        # Assemble token sequence: [body, hinge_1..8, goal]
        tokens = jnp.concatenate([
            body_token[:, jnp.newaxis, :],
            hinge_tokens,
            goal_token,
        ], axis=1)  # (batch, 10, embed_dim)

        # Pass through shared transformer backbone (skips _vector_to_sequence
        # since tokens is already 3D)
        x = TransformerBackbone(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_patches=0,
            dropout_rate=self.dropout_rate,
            use_cls_token=self.use_cls_token,
            pooling=self.pooling,
        )(tokens)

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std


class PerDimTransformerActor(nn.Module):
    """Actor with one token per input dimension.

    Treats every scalar in the observation as a separate 1-dim token.
    A shared Dense layer projects each 1-dim scalar to embed_dim.
    Makes no structural assumptions about the input, so it works for
    any environment without modification.
    """
    action_size: int
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    use_cls_token: bool = True
    pooling: str = "cls"
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Each input dim becomes a separate 1-dim token
        # x shape: (batch, input_dim) -> (batch, input_dim, 1)
        tokens = x[..., jnp.newaxis]  # (batch, input_dim, 1)
        tokens = nn.Dense(
            self.embed_dim,
            kernel_init=lecun_unfirom,
            bias_init=bias_init,
            name='token_embed',
        )(tokens)  # (batch, input_dim, embed_dim)

        x = TransformerBackbone(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_patches=0,
            dropout_rate=self.dropout_rate,
            use_cls_token=self.use_cls_token,
            pooling=self.pooling,
        )(tokens)

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std


"""
It is necessary to create the step functions that way, so as to be able to @jax.jit them. 
They need actor, sa_encoder, g_encoder and args upfront as arguments, 
since these are used in the training functions and are not expected to change during training.
"""
def generate_step_functions(actor, sa_encoder, g_encoder, args):

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh( means )

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )

    def actor_step(training_state, env, env_state, key, extra_fields):
        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = nn.tanh( means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype) )

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
        
    def multi_sample_actor_step(training_state, env, env_state, key, K, extra_fields):
        # Get K sets of actions from the actor
        keys = jax.random.split(key, K)
        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        
        actions = jnp.stack([
            nn.tanh(means + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype))
            for k in keys
        ])
        
        state = env_state.obs[:, :args.obs_dim]
        goal = env_state.obs[:, args.obs_dim:]
        
        sa_reprs = jax.vmap(
            lambda a: sa_encoder.apply(
                training_state.critic_state.params["sa_encoder"], 
                state, 
                a
            )
        )(actions)
        
        g_repr = g_encoder.apply(
            training_state.critic_state.params["g_encoder"], 
            goal
        ) 

        q_values = -jnp.sqrt(
            jnp.sum((sa_reprs - g_repr) ** 2, axis=-1)
        )
        
        best_action_idx = jnp.argmax(q_values, axis=0)
        best_actions = jnp.take_along_axis(
            actions,
            best_action_idx[None, :, None],
            axis=0
        )[0]
        
        # Step environment with best actions
        nstate = env.step(env_state, best_actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        
        return nstate, Transition(
            observation=env_state.obs,
            action=best_actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
    return actor_step, deterministic_actor_step, multi_sample_actor_step