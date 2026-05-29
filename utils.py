import flax
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import pickle
from etils import epath
import wandb
import os
from wandb_osh.hooks import TriggerWandbSyncHook
import wandb_osh
from envs.env_functions import make_env
from brax.io import html
import flax.linen as nn
import random
import numpy as np

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState

class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()

def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)

def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))

def jit_wrap(buffer):
    buffer.insert_internal = jax.jit(buffer.insert_internal)
    buffer.sample_internal = jax.jit(buffer.sample_internal)
    return buffer

def setup_project(args):
    save_path = None
    trigger_sync = None
    
    # Print every arg
    print("Arguments:", flush=True)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}", flush=True)
    print("\n", flush=True)

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    print(f"env_steps_per_actor_step: {args.env_steps_per_actor_step}", flush=True)

    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    print(f"num_prefill_env_steps: {args.num_prefill_env_steps}", flush=True)

    args.num_prefill_actor_steps = int(np.ceil(args.min_replay_size / args.unroll_length))
    print(f"num_prefill_actor_steps: {args.num_prefill_actor_steps}", flush=True)

    args.num_training_steps_per_epoch = (args.total_env_steps - args.num_prefill_env_steps) // (args.num_epochs * args.env_steps_per_actor_step)
    print(f"num_training_steps_per_epoch: {args.num_training_steps_per_epoch}", flush=True)
    
    run_name = (
    f"{args.env_id}"
    f"{'_' + args.eval_env_id if args.eval_env_id else ''}"
    f"_batchsize:{args.batch_size}"
    f"_envsteps:{args.total_env_steps}"
    f"_nenvs:{args.num_envs}"
    f"_criticwidth:{args.critic_network_width}"
    f"_actorwidth:{args.actor_network_width}"
    f"_criticdepth:{args.critic_depth}"
    f"_actordepth:{args.actor_depth}"
    f"_seed:{args.seed}"
    f"_transformercritic:{args.transformer_critic}"
    f"{'_tokenmode:' + str(args.token_mode) + '_pooling:' + str(args.pooling_type) if args.transformer_critic else ''}"
    )
    
    print("-----------------------------------------------------")
    print(f"run_name: {run_name}", flush=True)
    
    if args.track:

        if args.wandb_group ==  '.':
            args.wandb_group = None
            
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()
        
    if args.checkpoint:
        from pathlib import Path
        from datetime import datetime
        short_run_name = f"runs/{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_path = Path(args.wandb_dir) / Path(short_run_name)
        os.mkdir(path=save_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    return save_path, trigger_sync, (jax.random.split(key, 7))


def save_results(actor, args, training_state, buffer_state, save_path):
    if args.checkpoint:
        # Save current policy and critic params.
        params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
        path = f"{save_path}/final.pkl"
        save_params(path, params)
        
    # After training is complete, render the final policy
    if args.capture_vis:
        def render_policy(training_state, save_path):
            """Renders the policy and saves it as an HTML file."""
            @jax.jit
            def policy_step(env_state, actor_params):
                means, _ = actor.apply(actor_params, env_state.obs)
                actions = nn.tanh(means)
                next_state = env.step(env_state, actions)
                return next_state, env_state 
            
            rollout_states = []
            for i in range(args.num_render):
                env = make_env(args.eval_env_id, args)
                
                rng = jax.random.PRNGKey(seed=i+1)
                env_state = jax.jit(env.reset)(rng)
                
                for _ in range(args.vis_length):
                    env_state, current_state = policy_step(env_state, training_state.actor_state.params)
                    rollout_states.append(current_state.pipeline_state)
            
            # Render and save
            html_string = html.render(env.sys, rollout_states)
            render_path = f"{save_path}/vis.html"
            with open(render_path, "w") as f:
                f.write(html_string)
            wandb.log({"vis": wandb.Html(html_string)})
            
        print("Rendering final policy...", flush=True)
        try:
            render_policy(training_state, save_path)
        except Exception as e:
            print(f"Error rendering final policy: {e}", flush=True)
        
    #After training is complete, save the Args
    if args.checkpoint:
        with open(f"{save_path}/args.pkl", 'wb') as f:
            pickle.dump(args, f)
        print(f"Saved args to {save_path}/args.pkl", flush=True)
        
    #After training is complete, save the replay buffer (if save_buffer is 1, this takes a lot of memory)
    if args.checkpoint:
        if args.save_buffer:
            print("Saving final buffer_state and buffer data (everything needed to recreate replay_buffer)...", flush=True)
            try:
                buffer_path = f"{save_path}/final_buffer.pkl"
                buffer_data = {
                    'buffer_state': buffer_state,
                    'max_replay_size': args.max_replay_size,
                    'batch_size': args.batch_size,
                    'num_envs': args.num_envs,
                    'episode_length': args.episode_length,
                }
                with open(buffer_path, 'wb') as f:
                    pickle.dump(buffer_data, f)
                print(f"Saved replay_buffer to {buffer_path}", flush=True)
            except Exception as e:
                print(f"Error saving final replay buffer: {e}", flush=True)