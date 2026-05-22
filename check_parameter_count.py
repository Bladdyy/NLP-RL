import jax
import jax.numpy as jnp
import tyro

from config import Args
from modules.critic import SA_TransformerEncoder, G_TransformerEncoder, SA_MlpEncoder, G_MlpEncoder

def print_param_count(model_name, params):
    """Flattens the parameter tree and prints the total count."""
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total parameters in {model_name}: {total_params:,}")

if __name__ == "__main__":
    args = tyro.cli(Args)

    batch_size = 1 
    obs_dim = 30
    action_size = 10
    goal_dim = 2
    key = jax.random.PRNGKey(0)
    
    print("\n--- Model Parameter Counts ---")

    if args.transformer_critic:
        sa_encoder = SA_TransformerEncoder(network_width=args.critic_network_width, network_depth=args.critic_depth, use_relu=args.use_relu)
        g_encoder = G_TransformerEncoder(network_width=args.critic_network_width, network_depth=args.critic_depth, use_relu=args.use_relu)
    
    else:
        sa_encoder = SA_MlpEncoder(network_width=args.critic_network_width, network_depth=args.critic_depth, use_relu=args.use_relu)
        g_encoder = G_MlpEncoder(network_width=args.critic_network_width, network_depth=args.critic_depth, use_relu=args.use_relu)
    
    dummy_obs = jnp.ones([batch_size, obs_dim])
    dummy_act = jnp.ones([batch_size, action_size])
    sa_params = sa_encoder.init(key, dummy_obs, dummy_act)['params']
    model_name = "SA_TransformerEncoder" if args.transformer_critic else "SA_MlpEncoder"
    print_param_count(model_name, sa_params)


    dummy_goal = jnp.ones([batch_size, goal_dim])
    g_params = g_encoder.init(key, dummy_goal)['params']
    model_name = "G_TransformerEncoder" if args.transformer_critic else "G_MlpEncoder"
    print_param_count(model_name, g_params)
    
    print("-------------------------------\n")