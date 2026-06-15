import flax.linen as nn
from flax.linen.initializers import variance_scaling
import jax.numpy as jnp
from modules.utils import residual_block, TransformerBackbone
from modules.frozen_text_encoder import FrozenTextGoalEncoder, PrecomputedFrozenTextGoalEncoder
from modules.frozen_text_encoder import MODEL_REGISTRY  # noqa: F401  (expose for inspection)

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
    pooling: str = "cls"

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
            pooling=self.pooling,
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
    pooling: str = "cls"

    @nn.compact
    def __call__(self, g: jnp.ndarray):
        x = TransformerBackbone(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_patches=self.num_patches,
            dropout_rate=self.dropout_rate,
            pooling=self.pooling,
        )(g)

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class SemanticTransformerSAEncoder(nn.Module):
    """SA encoder with semantically meaningful tokens.

    Parses the 29-dim ant state (root qpos, hinge qpos, root qvel, hinge qvel)
    and the 8-dim action into 9 tokens:
      - 1 body token: root qpos (7) + root qvel (6) = 13 dims
      - 8 hinge tokens: each (hinge qpos, hinge qvel, hinge action) = 3 dims

    Each token type uses a separate Dense embedding layer to account for
    the asymmetry in raw dimension sizes.
    """
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    pooling: str = "cls"

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Parse 29-dim state:
        #   [0:7]   = root qpos (tx, ty, tz, qw, qx, qy, qz)
        #   [7:15]  = hinge qpos [hip_1, ankle_1, ..., hip_4, ankle_4]
        #   [15:21] = root qvel
        #   [21:29] = hinge qvel [same order as qpos]
        root_qpos = s[..., :7]            # (batch, 7)
        hinge_qpos = s[..., 7:15]         # (batch, 8)
        root_qvel = s[..., 15:21]         # (batch, 6)
        hinge_qvel = s[..., 21:29]        # (batch, 8)

        # Reorder action to match qpos/qvel hinge order (body tree order).
        # Action from env: [hip_4, ankle_4, hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3]
        # Desired order:   [hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4]
        action_reorder = jnp.array([2, 3, 4, 5, 6, 7, 0, 1])
        a = a[..., action_reorder]        # (batch, 8)

        # Body token: root qpos + root qvel = 13 dims
        body_raw = jnp.concatenate([root_qpos, root_qvel], axis=-1)  # (batch, 13)
        body_token = nn.Dense(
            self.embed_dim,
            kernel_init=lecun_unfirom,
            bias_init=bias_init,
            name='body_embed',
        )(body_raw)  # (batch, embed_dim)
        body_token = nn.LayerNorm(name='body_ln')(body_token)

        # Build 8 hinge tokens, each (qpos_i, qvel_i, action_i) = 3 dims
        hinge_raw = jnp.stack([hinge_qpos, hinge_qvel, a], axis=-1)  # (batch, 8, 3)
        hinge_tokens = nn.Dense(
            self.embed_dim,
            kernel_init=lecun_unfirom,
            bias_init=bias_init,
            name='hinge_embed',
        )(hinge_raw)  # (batch, 8, embed_dim)
        hinge_tokens = nn.LayerNorm(name='hinge_ln')(hinge_tokens)

        # Assemble token sequence: [body, hinge_1, ..., hinge_8]
        tokens = jnp.concatenate([
            body_token[:, jnp.newaxis, :],
            hinge_tokens,
        ], axis=1)  # (batch, 9, embed_dim)

        # Pass through shared transformer backbone (skips _vector_to_sequence
        # since tokens is already 3D)
        x = TransformerBackbone(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_patches=0,  # unused when input is 3D
            dropout_rate=self.dropout_rate,
            pooling=self.pooling,
        )(tokens)

        # Final projection to 64-dim representation
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class SemanticTransformerGEncoderText(nn.Module):
    """
    Goal encoder using frozen Sentence-BERT text encoding.

    Converts the 2-dim goal (x, y) into the prompt
    "Your goal is (x,y)" and encodes it with a pretrained SBERT model.

    When ``possible_goals`` is provided, precomputes embeddings for all
    possible positions at init time and uses fast nearest-neighbour
    look-up during training instead of running the SBERT model.

    Args:
        output_dim: output embedding dimension (default: 64).
        possible_goals: (N, 2) array of all goal coordinates that may
            appear during training. If ``None``, the original SBERT
            forward pass is used (slower but general).
    """
    output_dim: int = 64
    possible_goals: jnp.ndarray | None = None
    model_key: str = "minilm"

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        return FrozenTextGoalEncoder(
            output_dim=self.output_dim,
            model_key=self.model_key,
            possible_goals=self.possible_goals,
        )(g)


class TrainableEmbeddingGoalEncoder(nn.Module):
    """
    Trainable embedding layer for discrete goal positions.

    Maps each (x, y) goal to a learned dense vector via a lookup table
    (``nn.Embed``).  Only applicable when the goal space is finite, i.e.
    when *possible_goals* is known — typically ant-maze environments.

    Unlike the frozen-text encoders, the embedding weights **are trained**
    end-to-end with the critic objective.

    Args:
        output_dim: embedding dimension (default: 64).
        possible_goals: (N, 2) array of all goal coordinates that may
            appear during training.
    """
    output_dim: int = 64
    possible_goals: jnp.ndarray  # (N, 2)

    def setup(self):
        self._precomputed_goals = jnp.asarray(self.possible_goals)
        n_goals = self.possible_goals.shape[0]
        self.embed = nn.Embed(
            num_embeddings=n_goals,
            features=self.output_dim,
            name="goal_embed",
        )

    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        # (B, 2) -> nearest goal index -> trainable embedding
        dists = jnp.linalg.norm(
            g[:, None, :] - self._precomputed_goals[None, :, :],
            axis=-1,
        )  # (B, N)
        indices = jnp.argmin(dists, axis=-1)  # (B,)
        return self.embed(indices)  # (B, output_dim)


class SemanticTransformerGEncoder(nn.Module):
    """Goal encoder with a single semantic token.

    Embeds the 2-dim goal (future ant x, y) as a single token and processes
    it through TransformerBackbone.
    """
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    pooling: str = "cls"

    @nn.compact
    def __call__(self, g: jnp.ndarray):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Single goal token: embed 2-dim goal directly to embed_dim
        token = nn.Dense(
            self.embed_dim,
            kernel_init=lecun_unfirom,
            bias_init=bias_init,
            name='goal_embed',
        )(g)  # (batch, embed_dim)
        tokens = token[:, jnp.newaxis, :]  # (batch, 1, embed_dim)

        # Pass through shared transformer backbone
        x = TransformerBackbone(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_patches=0,  # unused when input is 3D
            dropout_rate=self.dropout_rate,
            pooling=self.pooling,
        )(tokens)

        # Final projection to 64-dim representation
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class PerDimTransformerSAEncoder(nn.Module):
    """SA encoder with one token per input dimension.

    Concatenates state and action, then treats each scalar as a separate
    1-dim token projected via a shared Dense layer. Makes no structural
    assumptions about the input.
    """
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    pooling: str = "cls"

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = jnp.concatenate([s, a], axis=-1)  # (batch, 37)

        # Each input dim -> separate 1-dim token
        tokens = x[..., jnp.newaxis]  # (batch, 37, 1)
        tokens = nn.Dense(
            self.embed_dim,
            kernel_init=lecun_unfirom,
            bias_init=bias_init,
            name='token_embed',
        )(tokens)  # (batch, 37, embed_dim)

        x = TransformerBackbone(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_patches=0,
            dropout_rate=self.dropout_rate,
            pooling=self.pooling,
        )(tokens)

        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class PerDimTransformerGEncoder(nn.Module):
    """Goal encoder with one token per input dimension.

    Treats each of the 2 goal dims as a separate 1-dim token.
    """
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    pooling: str = "cls"

    @nn.compact
    def __call__(self, g: jnp.ndarray):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Each goal dim -> separate 1-dim token
        tokens = g[..., jnp.newaxis]  # (batch, 2, 1)
        tokens = nn.Dense(
            self.embed_dim,
            kernel_init=lecun_unfirom,
            bias_init=bias_init,
            name='token_embed',
        )(tokens)  # (batch, 2, embed_dim)

        x = TransformerBackbone(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_patches=0,
            dropout_rate=self.dropout_rate,
            pooling=self.pooling,
        )(tokens)

        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x