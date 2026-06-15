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
from modules.actor import Actor, TransformerActor, SemanticTransformerActor, PerDimTransformerActor, generate_step_functions
from modules.critic import SA_encoder, G_encoder, TransformerSAEncoder, TransformerGEncoder, SemanticTransformerSAEncoder, SemanticTransformerGEncoder, SemanticTransformerGEncoderText, TrainableEmbeddingGoalEncoder, HybridGoalEncoder, PerDimTransformerSAEncoder, PerDimTransformerGEncoder
from utils import TrainingState, Transition, save_params, jit_wrap, setup_project, save_results
from train import create_training_functions


def make_transformer_optimizer(learning_rate, weight_decay, max_norm, params, warmup_steps=50_000):
    """Build an optimizer with gradient clipping, linear LR warmup, and AdamW weight decay
    applied only to multi-dimensional parameters (weights), not 1D biases or norms.
    Returns (optimizer, lr_schedule) where lr_schedule can be called with a step count."""
    lr_schedule = optax.join_schedules([
        optax.linear_schedule(0.0, learning_rate, transition_steps=warmup_steps),
        optax.constant_schedule(learning_rate),
    ], boundaries=[warmup_steps])

    def wd_mask_fn(params):
        def is_weight(path, value):
            return value.ndim == 2
        return jax.tree_util.tree_map_with_path(is_weight, params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay, mask=wd_mask_fn),
    )
    return optimizer, lr_schedule

_TRANSFORMER_CLASSES = {
    "actor": {
        "semantic": SemanticTransformerActor,
        "per_dim": PerDimTransformerActor,
        "patches": TransformerActor,
    },
    "sa_encoder": {
        "semantic": SemanticTransformerSAEncoder,
        "per_dim": PerDimTransformerSAEncoder,
        "patches": TransformerSAEncoder,
    },
    "g_encoder": {
        "semantic": SemanticTransformerGEncoder,
        "per_dim": PerDimTransformerGEncoder,
        "patches": TransformerGEncoder,
    },
}


def _make_transformer(component, *, action_size=None):
    """Create a transformer network for the given component using the configured tokenization."""
    cls = _TRANSFORMER_CLASSES[component][args.tokenization]
    kwargs = dict(
        embed_dim=args.transformer_embed_dim,
        num_layers=args.transformer_num_layers,
        num_heads=args.transformer_num_heads,
        mlp_ratio=args.transformer_mlp_ratio,
        dropout_rate=args.transformer_dropout,
        pooling=args.transformer_pooling,
    )
    if action_size is not None:
        kwargs["action_size"] = action_size
    if args.tokenization == "patches":
        kwargs["num_patches"] = args.transformer_num_patches
    return cls(**kwargs)


if __name__ == "__main__":

    print("Starting training script...", flush=True)
    
    args = tyro.cli(Args)
    save_path, trigger_sync, (key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key) = setup_project(args)

    # Training environment setup ------------------------------------------------------------------------------------------------------------------------------
    env = make_env(args.env_id, args)
    # Extract possible goals before wrapping — used by the text encoder
    # for precomputed embedding look-up (avoids running SBERT every batch).
    possible_goals = getattr(env, 'possible_goals', None)
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
    # Also extract eval goals and merge with training goals so the precomputed
    # lookup table covers both (eval goals may differ from training goals).
    eval_possible_goals = getattr(eval_env, 'possible_goals', None)
    if possible_goals is not None and eval_possible_goals is not None:
        # Union of training and eval goals (deduplicated)
        all_goals = jnp.unique(
            jnp.concatenate([possible_goals, eval_possible_goals], axis=0),
            axis=0,
        )
        possible_goals = all_goals
    elif eval_possible_goals is not None:
        possible_goals = eval_possible_goals
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=args.episode_length,
    )
    eval_env_keys = jax.random.split(eval_env_key, args.num_envs)
    eval_env_state = jax.jit(eval_env.reset)(eval_env_keys)
    eval_env.step = jax.jit(eval_env.step)


    # Actor and critic setup ----------------------------------------------------------------------------------------------------------------------------------

    # Actor
    actor_is_transformer = args.transformer_mode in ("Full", "StateActor")
    if actor_is_transformer:
        actor = _make_transformer("actor", action_size=action_size)
    else:
        actor = Actor(action_size=action_size, network_width=args.actor_network_width, network_depth=args.actor_depth, skip_connections=args.actor_skip_connections, use_relu=args.use_relu)
    actor_params = actor.init(actor_key, np.ones([1, obs_size]))
    actor_lr_schedule = None
    if actor_is_transformer:
        actor_tx, actor_lr_schedule = make_transformer_optimizer(
            args.transformer_lr, args.transformer_weight_decay,
            args.grad_clip_max_norm, actor_params
        )
    else:
        actor_tx = optax.adam(learning_rate=args.actor_lr)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=actor_tx,
    )

    # Critic
    sa_is_transformer = args.transformer_mode in ("Full", "State", "StateGoal", "StateActor")
    g_is_transformer = args.transformer_mode in ("Full", "StateGoal")
    if sa_is_transformer:
        sa_encoder = _make_transformer("sa_encoder")
    else:
        sa_encoder = SA_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    if args.text_encoder:
        if args.goal_end_idx - args.goal_start_idx != 2:
            raise ValueError("text_encoder currently supports only 2D goals")
        if args.text_model not in {"minilm", "bge", "gte", "e5"}:
            raise ValueError(f"Unknown text_model '{args.text_model}'; "
                             f"choose from minilm, bge, gte, e5")
        if args.hybrid_goal_encoder and args.text_pooling not in {"cls", "mean", "token"}:
            raise ValueError(f"Unknown text_pooling '{args.text_pooling}'; "
                             f"choose from cls, mean, token")
        if args.hybrid_goal_encoder:
            g_encoder = HybridGoalEncoder(
                output_dim=64,
                backbone="semantic" if g_is_transformer else "mlp",
                embed_source="trainable" if args.trainable_embedding else "frozen",
                possible_goals=possible_goals,
                model_key=args.text_model,
                pooling=args.text_pooling,
            )
        else:
            g_encoder = SemanticTransformerGEncoderText(
                output_dim=64,
                possible_goals=possible_goals,
                model_key=args.text_model,
            )
    elif args.trainable_embedding:
        if possible_goals is None:
            raise ValueError(
                "trainable_embedding requires a finite goal set; "
                "the environment must expose possible_goals"
            )
        g_encoder = TrainableEmbeddingGoalEncoder(
            output_dim=64,
            possible_goals=possible_goals,
        )
    elif g_is_transformer:
        g_encoder = _make_transformer("g_encoder")
    else:
        g_encoder = G_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    critic_params = {
        "sa_encoder": sa_encoder_params,
        "g_encoder": g_encoder_params
    }

    critic_lr_schedule = None
    critic_any_transformer = sa_is_transformer or g_is_transformer
    if critic_any_transformer:
        critic_tx, critic_lr_schedule = make_transformer_optimizer(
            args.transformer_lr, args.transformer_weight_decay,
            args.grad_clip_max_norm, critic_params
        )
    else:
        critic_tx = optax.adam(learning_rate=args.critic_lr)
    critic_state = TrainState.create(
        apply_fn=None,
        params=critic_params,
        tx=critic_tx,
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

        # Log learning rates
        if actor_is_transformer:
            step = training_state.gradient_steps.item()
            metrics["training/learning_rate"] = actor_lr_schedule(step)
        else:
            metrics["training/learning_rate_actor"] = args.actor_lr
            metrics["training/learning_rate_critic"] = args.critic_lr

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