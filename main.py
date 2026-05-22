import jax
import tyro
import time
import optax
import wandb
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from brax import envs
from flax.training.train_state import TrainState

from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue
from config import Args

from envs.env_functions import make_env
from modules.actor import Actor, generate_step_functions
from modules.critic import SA_TransformerEncoder, G_TransformerEncoder, SA_MlpEncoder, G_MlpEncoder
from utils import TrainingState, Transition, save_params, jit_wrap, setup_project, save_results
from train import create_training_functions

if __name__ == "__main__":

    print("Starting training script...", flush=True)
    
    args = tyro.cli(Args)
    save_path, trigger_sync, (key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key) = setup_project(args)

    # Training environment setup ------------------------------------------------------------------------------------------------------------------------------
    env = make_env(args.env_id, args)
    env = envs.training.wrap(env, episode_length=args.episode_length,)

    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)
    print(f"obs_size: {obs_size}, action_size: {action_size}", flush=True)


    # Evaluation environment setup ----------------------------------------------------------------------------------------------------------------------------
    if not args.eval_env_id:
        args.eval_env_id = args.env_id
        
    eval_env = make_env(args.eval_env_id, args)
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=args.episode_length,
    )
    eval_env_keys = jax.random.split(eval_env_key, args.num_envs)
    eval_env_state = jax.jit(eval_env.reset)(eval_env_keys)
    eval_env.step = jax.jit(eval_env.step)


    # Actor and critic setup ----------------------------------------------------------------------------------------------------------------------------------

    # Actor
    actor = Actor(action_size=action_size, network_width=args.actor_network_width, network_depth=args.actor_depth, skip_connections=args.actor_skip_connections, use_relu=args.use_relu)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr)
    )

    # Critic
    if args.transformer_critic:
        sa_encoder = SA_TransformerEncoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
        g_encoder = G_TransformerEncoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    
    else:
        sa_encoder = SA_MlpEncoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
        g_encoder = G_MlpEncoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    
    critic_state = TrainState.create(
        apply_fn=None,
        params={
            "sa_encoder": sa_encoder_params, 
            "g_encoder": g_encoder_params
            },
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    # Entropy coefficient
    target_entropy = -args.entropy_param * action_size # action_size = 8 for ant, 17 for humanoid, etc
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )
    
    # Train state
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )


    # Replay Buffer setup -------------------------------------------------------------------------------------------------------------------------------------
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={
            "state_extras": {
                "truncation": 0.0,
                "seed": 0.0,
            }
        },
    )
    
    replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=args.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=args.batch_size,
                num_envs=args.num_envs,
                episode_length=args.episode_length,
            )
        )
    
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)
    
    key, prefill_key = jax.random.split(key, 2)
    
    actor_step, deterministic_actor_step, multi_sample_actor_step = generate_step_functions(actor, sa_encoder, g_encoder, args)

    prefill_replay_buffer, training_epoch = create_training_functions(actor, sa_encoder, g_encoder, env, args, replay_buffer, target_entropy, deterministic_actor_step, actor_step, multi_sample_actor_step)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(training_state, env_state, buffer_state, prefill_key)
    

    # Evaluator setup -----------------------------------------------------------------------------------------------------------------------------------------

    if args.eval_actor == 0:
        '''Setting up evaluator'''
        evaluator = CrlEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
        
    elif args.eval_actor == 1:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            lambda training_state, env, env_state, extra_fields: actor_step(
                training_state,
                env,
                env_state,
                eval_actor_key,
                extra_fields
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    
    elif args.eval_actor > 1:
        key, eval_actor_key = jax.random.split(key)
        evaluator = CrlEvaluator(
            # Replace deterministic_actor_step with a partial function of multi_sample_actor_step
            lambda training_state, env, env_state, extra_fields: multi_sample_actor_step(
                training_state, 
                env, 
                env_state, 
                eval_actor_key, 
                args.eval_actor,
                extra_fields
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    

    # Training loop start -------------------------------------------------------------------------------------------------------------------------------------
    training_walltime = 0
    print('Starting training....', flush=True)
    print(f"Using devices: {jax.devices()}", flush=True)

    start_time = time.time() 
    for ne in range(args.num_epochs):
        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, epoch_key)
        
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }

        metrics = evaluator.run_evaluation(training_state, metrics)

        print(f"epoch {ne} out of {args.num_epochs} complete. metrics: {metrics}", flush=True)

        if args.checkpoint:
            if ne < 5 or ne >= args.num_epochs - 5 or ne % 10 == 0:
                # Save current policy and critic params.
                params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
                path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)
        
        if args.track:
            wandb.log(metrics, step=ne)

            if args.wandb_mode == 'offline':
                trigger_sync()
        
        hours_passed = (time.time() - start_time) / 3600
        print(f"Time elapsed: {hours_passed:.3f} hours", flush=True)

    save_results(actor, args, training_state, buffer_state, save_path)