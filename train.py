import jax
import jax.numpy as jnp

from buffer import TrajectoryUniformSamplingQueue

from modules.actor import generate_step_functions
from loss import create_loss_functions


"""
It is necessary to create the training functions that way, so as to be able to @jax.jit them. 
They need actor, sa_encoder, g_encoder, env, args and replay_buffer upfront as arguments, 
since these are used in the training functions and are not expected to change during training.
"""
def create_training_functions(actor, sa_encoder, g_encoder, env, args, replay_buffer, target_entropy):
    update_actor_and_alpha, update_critic = create_loss_functions(actor, sa_encoder, g_encoder, args, target_entropy)
    deterministic_actor_step, actor_step, multi_sample_actor_step = generate_step_functions(actor, sa_encoder, g_encoder, args)
    
    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t): #conducts a single actor step in environment
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            if args.expl_actor == 1:
                env_state, transition = actor_step(training_state, env, env_state, current_key, extra_fields=("truncation", "seed"))
            elif args.expl_actor == 0:
                env_state, transition = deterministic_actor_step(training_state, env, env_state, extra_fields=("truncation", "seed"))
            else:
                env_state, transition = multi_sample_actor_step(training_state, env, env_state, current_key, args.expl_actor, extra_fields=("truncation", "seed"))
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)

        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state


    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key, = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)

        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)

        training_state = training_state.replace(gradient_steps = training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)

        return (training_state, key,), metrics


    @jax.jit
    def training_step(training_state, env_state, buffer_state, key, t):
        experience_key1, experience_key2, sampling_key, training_key, sgd_batches_key = jax.random.split(key, 5)
        
        # update buffer
        env_state, buffer_state = get_experience(
            training_state,
            env_state,
            buffer_state,
            experience_key1,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )
            
        transitions_list = []
        for _ in range(args.num_episodes_per_env):
            buffer_state, new_transitions = replay_buffer.sample(buffer_state)
            transitions_list.append(new_transitions)

        # Concatenate all sampled transitions
        transitions = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0),
            *transitions_list
        )

        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys
        )
        
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        
        # I added this code, so as to ensure len(transitions.observation) is divisible by batch_size
        num_full_batches = len(transitions.observation) // args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:num_full_batches * args.batch_size], transitions)
        
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )
        
        if args.use_all_batches == 0:
            num_total_batches = transitions.observation.shape[0]
            selected_indices = jax.random.permutation(
                sgd_batches_key, 
                num_total_batches
            )[:args.num_sgd_batches_per_training_step]
            transitions = jax.tree_util.tree_map(
                lambda x: x[selected_indices], 
                transitions
            )        
        

        # take actor-step worth of training-step
        (training_state, _,), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        return (training_state, env_state, buffer_state,), metrics


    # Main training functions ---------------------------------------------------------------------------------------------------------------------------------

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(
                training_state,
                env_state,
                buffer_state,
                key,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_prefill_actor_steps)[0]

    @jax.jit
    def training_epoch(training_state, env_state, buffer_state, key):  
        @jax.jit
        def f(carry, t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs,), metrics = training_step(ts, es, bs, train_key, t)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), jnp.arange(args.num_training_steps_per_epoch * args.training_steps_multiplier))

        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    return prefill_replay_buffer, training_epoch