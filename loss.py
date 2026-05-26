
import jax
import flax.linen as nn
import jax.numpy as jnp


"""
It is necessary to create the loss functions that way, so as to be able to @jax.jit them. 
They need actor, sa_encoder, g_encoder, args and target_entropy upfront as arguments, 
since these are used in the loss functions and are not expected to change during training.
"""
def create_loss_functions(actor, sa_encoder, g_encoder, args, target_entropy):
    
    def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
        obs = transitions.observation           # expected_shape = batch_size, obs_size + goal_size
        state = obs[:, :args.obs_dim]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, args.goal_start_idx : args.goal_end_idx]
        observation = jnp.concatenate([state, goal], axis=1)

        means, log_stds = actor.apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)           # dimension = B

        sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
        sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
        g_repr = g_encoder.apply(g_encoder_params, goal)
        
        sa_repr_norm = sa_repr / (jnp.linalg.norm(sa_repr, axis=-1, keepdims=True) + 1e-6)
        g_repr_norm = g_repr / (jnp.linalg.norm(g_repr, axis=-1, keepdims=True) + 1e-6)

        distances = -jnp.sqrt(jnp.sum((sa_repr_norm - g_repr_norm) ** 2, axis=-1))

        qf_pi = distances / args.loss_temperature

        if args.disable_entropy:
            actor_loss = -jnp.mean(qf_pi)
        else:
            actor_loss = jnp.mean( jnp.exp(log_alpha) * log_prob - (qf_pi) )

        sa_embedding_norm = jnp.sqrt(jnp.sum(sa_repr ** 2, axis=-1)).mean()
        return actor_loss, (log_prob, sa_embedding_norm)


    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))
        return jnp.mean(alpha_loss)


    def critic_loss(critic_params, transitions, key):
        sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
        
        obs = transitions.observation[:, :args.obs_dim]
        action = transitions.action
        
        sa_repr = sa_encoder.apply(sa_encoder_params, obs, action)
        g_repr = g_encoder.apply(g_encoder_params, transitions.observation[:, args.obs_dim:])
            
        sa_repr_norm = sa_repr / (jnp.linalg.norm(sa_repr, axis=-1, keepdims=True) + 1e-6)
        g_repr_norm = g_repr / (jnp.linalg.norm(g_repr, axis=-1, keepdims=True) + 1e-6)

        distances = jnp.sqrt(jnp.sum((sa_repr_norm[:, None, :] - g_repr_norm[None, :, :]) ** 2, axis=-1))

        logits = -distances / args.loss_temperature

        # InfoNCE
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

        # logsumexp regularisation
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

        B = logits.shape[0]
        I = jnp.zeros(1)
        correct = jnp.mean(jnp.argmax(logits, axis=1) == jnp.arange(B))
        logits_pos = jnp.mean(jnp.diag(logits))
        logits_neg = (jnp.sum(logits) - jnp.sum(jnp.diag(logits))) / (B * (B - 1))

        sa_embedding_norm = jnp.sqrt(jnp.sum(sa_repr ** 2, axis=-1)).mean()

        return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg, sa_embedding_norm)


    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        actor_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[:actor_batch_size], 
            transitions
        )
        
        
        (actorloss, (log_prob, actor_sa_embedding_norm)), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(training_state.actor_state.params, training_state.critic_state.params, training_state.alpha_state.params['log_alpha'], transitions, key)
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        actor_grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(lambda s, x: s + jnp.sum(x ** 2), jax.tree_util.tree_leaves(actor_grad), initializer=0.0))

        alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
            "alph_aloss": alphaloss,   
            "log_alpha": training_state.alpha_state.params["log_alpha"],
            "grad_norm_actor": actor_grad_norm,
            "sa_embedding_norm": actor_sa_embedding_norm,
        }

        return training_state, metrics

    @jax.jit
    def update_critic(transitions, training_state, key):
        critic_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[:critic_batch_size], 
            transitions
        )
            
        (loss, (logsumexp, I, correct, logits_pos, logits_neg, critic_sa_embedding_norm)), grad = jax.value_and_grad(critic_loss, has_aux=True)(training_state.critic_state.params, transitions, key)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)

        critic_grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(lambda s, x: s + jnp.sum(x ** 2), jax.tree_util.tree_leaves(grad), initializer=0.0))

        training_state = training_state.replace(critic_state = new_critic_state)

        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
            "grad_norm_critic": critic_grad_norm,
            "sa_embedding_norm": critic_sa_embedding_norm,
        }

        return training_state, metrics
    
    return update_actor_and_alpha, update_critic