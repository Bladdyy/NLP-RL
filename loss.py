
import jax
import flax.linen as nn
import jax.numpy as jnp


def strong_sigreg_loss(z, rng_key, num_slices=1024, num_t_nodes=64, bandwidth=1.0):
    """
    Strong SIGReg loss (LeJEPA). Encourages the distribution of embeddings to
    match a standard normal distribution by minimizing the distance between
    empirical and target characteristic functions.

    Args:
        z: Input embeddings of shape (batch_size, embed_dim).
        rng_key: JAX PRNG key for generating random projection vectors.
        num_slices: Number of random 1D projections (M).
        num_t_nodes: Number of quadrature nodes for ECF integration.
        bandwidth: Bandwidth parameter for the Gaussian weight.

    Returns:
        Scalar loss value.
    """
    batch_size, embed_dim = z.shape

    # Generate M random unit directions on the hypersphere S^(D-1)
    u = jax.random.normal(rng_key, shape=(embed_dim, num_slices))
    u = u / (jnp.linalg.norm(u, axis=0, keepdims=True) + 1e-6)

    # Project high-dimensional embeddings onto the M slices
    h = jnp.dot(z, u)  # (B, M)

    # Define integration nodes t
    t_nodes = jnp.linspace(0.2, 4.0, num_t_nodes)

    # Compute empirical characteristic function per slice at each node
    # t_nodes(T, 1, 1) * h(1, B, M) -> t_h(T, B, M)
    t_h = t_nodes[:, None, None] * h[None, :, :]

    # Average over the batch dimension to get the empirical CF
    # Shape: (T, M)
    phi_emp = jnp.mean(jnp.exp(1j * t_h), axis=1)

    # Compute the target standard-normal CF and Gaussian weighting
    # Shapes: (T, 1) so they broadcast across the M slices
    phi_0 = jnp.exp(-t_nodes**2 / 2.0)[:, None]
    w = jnp.exp(-t_nodes**2 / (2.0 * bandwidth**2))[:, None]

    real_part = jnp.real(phi_emp)
    imag_part = jnp.imag(phi_emp)

    squared_diff = (real_part - phi_0)**2 + imag_part**2
    integrand = squared_diff * w

    # Integrate over the T dimension, then average across all M slices
    slice_losses = jnp.trapz(integrand, x=t_nodes, axis=0)

    return jnp.mean(slice_losses)


def weak_sigreg_loss(z, rng_key, sketch_dim=64):
    """
    Weak SIGReg loss (Covariance Sketching). Encourages the covariance of
    embeddings to be isotropic (close to identity). Uses Johnson-Lindenstrauss
    sketching when embed_dim > sketch_dim for efficiency.

    Args:
        z: Input embeddings of shape (batch_size, embed_dim).
        rng_key: JAX PRNG key for the random sketching matrix.
        sketch_dim: Dimension to sketch down to for covariance estimation.

    Returns:
        Scalar loss value.
    """
    batch_size, embed_dim = z.shape

    # Sketching (dimensionality reduction via Johnson-Lindenstrauss)
    if embed_dim > sketch_dim:
        s = jax.random.normal(rng_key, shape=(sketch_dim, embed_dim)) / jnp.sqrt(embed_dim)
        x_sketched = jnp.dot(z, s.T)
    else:
        sketch_dim = embed_dim
        x_sketched = z

    # Centering and empirical covariance matrix
    x_centered = x_sketched - jnp.mean(x_sketched, axis=0, keepdims=True)
    cov = jnp.dot(x_centered.T, x_centered) / (batch_size - 1.0 + 1e-6)

    # Target is the identity matrix (isotropic)
    target = jnp.eye(sketch_dim)

    # Minimize Frobenius norm distance to identity
    loss = jnp.linalg.norm(cov - target, ord='fro')

    return loss


"""
It is necessary to create the loss functions that way, so as to be able to @jax.jit them. 
They need actor, sa_encoder, g_encoder, args and target_entropy upfront as arguments, 
since these are used in the loss functions and are not expected to change during training.
"""
def create_loss_functions(actor, sa_encoder, g_encoder, args, target_entropy):
    
    def actor_loss(actor_params, critic_params, log_alpha, transitions, key, log_temperature=None):
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

        sa_embedding_norm = jnp.sqrt(jnp.sum(sa_repr ** 2, axis=-1)).mean()
        if args.embed_norm == "l2":
            sa_repr = sa_repr / (jnp.linalg.norm(sa_repr, axis=-1, keepdims=True) + 1e-6)
            g_repr = g_repr / (jnp.linalg.norm(g_repr, axis=-1, keepdims=True) + 1e-6)
            qf_pi = jnp.sum(sa_repr * g_repr, axis=-1)
        else:
            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
        if args.learnable_temperature:
            safe_log_temp = jnp.clip(log_temperature, a_min=-5.0, a_max=5.0)
            qf_pi = qf_pi / jnp.exp(safe_log_temp)
        if args.disable_entropy:
            actor_loss = -jnp.mean(qf_pi)
        else:
            actor_loss = jnp.mean( jnp.exp(log_alpha) * log_prob - (qf_pi) )

        return actor_loss, (log_prob, sa_embedding_norm)


    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))
        return jnp.mean(alpha_loss)


    def critic_loss(critic_params, transitions, key, log_temperature=None):
        sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
        
        obs = transitions.observation[:, :args.obs_dim]
        action = transitions.action
        
        sa_repr = sa_encoder.apply(sa_encoder_params, obs, action)
        g_repr = g_encoder.apply(g_encoder_params, transitions.observation[:, args.obs_dim:])

        sa_embedding_norm = jnp.sqrt(jnp.sum(sa_repr ** 2, axis=-1)).mean()
        if args.embed_norm == "l2":
            sa_repr = sa_repr / (jnp.linalg.norm(sa_repr, axis=-1, keepdims=True) + 1e-6)
            g_repr = g_repr / (jnp.linalg.norm(g_repr, axis=-1, keepdims=True) + 1e-6)
            logits = jnp.sum(sa_repr[:, None, :] * g_repr[None, :, :], axis=-1)
        else:
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
        if args.learnable_temperature:
            safe_log_temp = jnp.clip(log_temperature, a_min=-5.0, a_max=5.0)
            logits = logits / jnp.exp(safe_log_temp)

        # InfoNCE
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

        # logsumexp regularisation
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

        # SIGReg regularization
        sigreg_loss_sa = jnp.zeros(())
        sigreg_loss_g = jnp.zeros(())
        if args.embed_norm == "sigreg":
            sigreg_key_sa, sigreg_key_g = jax.random.split(key)
            sigreg_loss_sa = strong_sigreg_loss(
                sa_repr, sigreg_key_sa,
                num_slices=args.sigreg_num_slices,
                num_t_nodes=args.sigreg_num_t_nodes,
                bandwidth=args.sigreg_bandwidth,
            )
            sigreg_loss_g = strong_sigreg_loss(
                g_repr, sigreg_key_g,
                num_slices=args.sigreg_num_slices,
                num_t_nodes=args.sigreg_num_t_nodes,
                bandwidth=args.sigreg_bandwidth,
            )
        elif args.embed_norm == "weak_sigreg":
            sigreg_key_sa, sigreg_key_g = jax.random.split(key)
            sigreg_loss_sa = weak_sigreg_loss(
                sa_repr, sigreg_key_sa,
                sketch_dim=args.sigreg_sketch_dim,
            )
            sigreg_loss_g = weak_sigreg_loss(
                g_repr, sigreg_key_g,
                sketch_dim=args.sigreg_sketch_dim,
            )
        sigreg_loss_total = sigreg_loss_sa + sigreg_loss_g
        critic_loss += args.sigreg_coeff * sigreg_loss_total

        B = logits.shape[0]
        I = jnp.zeros(1)
        correct = jnp.mean(jnp.argmax(logits, axis=1) == jnp.arange(B))
        logits_pos = jnp.mean(jnp.diag(logits))
        logits_neg = (jnp.sum(logits) - jnp.sum(jnp.diag(logits))) / (B * (B - 1))


        return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg, sa_embedding_norm, sigreg_loss_sa, sigreg_loss_g)


    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        actor_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[:actor_batch_size], 
            transitions
        )
        
        
        log_temperature = training_state.temperature_state.params['log_temperature'] if args.learnable_temperature else None

        (actorloss, (log_prob, actor_sa_embedding_norm)), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(training_state.actor_state.params, training_state.critic_state.params, training_state.alpha_state.params['log_alpha'], transitions, key, log_temperature)
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
        if args.learnable_temperature:
            metrics["log_temperature"] = training_state.temperature_state.params["log_temperature"]

        return training_state, metrics

    @jax.jit
    def update_critic(transitions, training_state, key):
        critic_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[:critic_batch_size], 
            transitions
        )

        if args.learnable_temperature:
            log_temperature = training_state.temperature_state.params['log_temperature']

            def critic_loss_with_temp(critic_params, log_temperature):
                return critic_loss(critic_params, transitions, key, log_temperature)

            (loss, (logsumexp, I, correct, logits_pos, logits_neg, critic_sa_embedding_norm, sigreg_loss_sa, sigreg_loss_g)), (grad, temp_grad) = jax.value_and_grad(critic_loss_with_temp, argnums=(0, 1), has_aux=True)(
                training_state.critic_state.params, log_temperature
            )
            new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
            new_temperature_state = training_state.temperature_state.apply_gradients(grads={'log_temperature': temp_grad})
            training_state = training_state.replace(critic_state=new_critic_state, temperature_state=new_temperature_state)
        else:
            (loss, (logsumexp, I, correct, logits_pos, logits_neg, critic_sa_embedding_norm, sigreg_loss_sa, sigreg_loss_g)), grad = jax.value_and_grad(critic_loss, has_aux=True)(training_state.critic_state.params, transitions, key)
            new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
            training_state = training_state.replace(critic_state=new_critic_state)

        critic_grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(lambda s, x: s + jnp.sum(x ** 2), jax.tree_util.tree_leaves(grad), initializer=0.0))

        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
            "grad_norm_critic": critic_grad_norm,
            "sa_embedding_norm": critic_sa_embedding_norm,
            "sigreg_loss_sa": sigreg_loss_sa,
            "sigreg_loss_g": sigreg_loss_g,
        }
        if args.learnable_temperature:
            metrics["log_temperature"] = training_state.temperature_state.params["log_temperature"]

        return training_state, metrics
    
    return update_actor_and_alpha, update_critic