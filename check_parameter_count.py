import jax
import jax.numpy as jnp
import tyro

from config import Args
from envs.env_functions import make_env
from modules.critic import TransformerSAEncoder, TransformerGEncoder, SA_encoder, G_encoder


def print_param_count(model_name, params):
    """Flattens the parameter tree and prints the total count."""
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total parameters in {model_name}: {total_params:,}")


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Set up environment to get actual observation / action / goal dimensions
    env = make_env(args.env_id, args)
    obs_size = env.observation_size
    action_size = env.action_size
    goal_dim = args.goal_end_idx - args.goal_start_idx
    key = jax.random.PRNGKey(args.seed)

    print(f"\nEnvironment: {args.env_id}")
    print(f"obs_size: {obs_size}, action_size: {action_size}, goal_dim: {goal_dim}")
    print("\n--- Model Parameter Counts ---")

    sa_is_transformer = args.transformer_mode in ("Full", "State", "StateGoal", "StateActor")
    g_is_transformer = args.transformer_mode in ("Full", "StateGoal")
    if sa_is_transformer:
        sa_encoder = TransformerSAEncoder(
            embed_dim=args.transformer_embed_dim,
            num_layers=args.transformer_num_layers,
            num_heads=args.transformer_num_heads,
            mlp_ratio=args.transformer_mlp_ratio,
            num_patches=args.transformer_num_patches,
            dropout_rate=args.transformer_dropout,
            use_cls_token=bool(args.transformer_use_cls_token),
        )
    else:
        sa_encoder = SA_encoder(
            network_width=args.critic_network_width,
            network_depth=args.critic_depth,
            skip_connections=args.critic_skip_connections,
            use_relu=args.use_relu,
        )
    if g_is_transformer:
        g_encoder = TransformerGEncoder(
            embed_dim=args.transformer_embed_dim,
            num_layers=args.transformer_num_layers,
            num_heads=args.transformer_num_heads,
            mlp_ratio=args.transformer_mlp_ratio,
            num_patches=args.transformer_num_patches,
            dropout_rate=args.transformer_dropout,
            use_cls_token=bool(args.transformer_use_cls_token),
        )
    else:
        g_encoder = G_encoder(
            network_width=args.critic_network_width,
            network_depth=args.critic_depth,
            skip_connections=args.critic_skip_connections,
            use_relu=args.use_relu,
        )

    dummy_obs = jnp.ones([1, obs_size])
    dummy_act = jnp.ones([1, action_size])
    sa_params = sa_encoder.init(key, dummy_obs, dummy_act)["params"]
    model_name = "SA_TransformerEncoder" if sa_is_transformer else "SA_MlpEncoder"
    print_param_count(model_name, sa_params)

    dummy_goal = jnp.ones([1, goal_dim])
    g_params = g_encoder.init(key, dummy_goal)["params"]
    model_name = "G_TransformerEncoder" if g_is_transformer else "G_MlpEncoder"
    print_param_count(model_name, g_params)

    print("-------------------------------\n")