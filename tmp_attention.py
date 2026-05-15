import jax
import jax.numpy as jnp
from flax import linen as nn

# Based on: 
# https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.ipynb
# https://huggingface.co/blog/rishiraj/attention-in-jax
class Attention(nn.Module):
    d_model: int = 512

    @nn.compact
    def __call__(self, q, k, v, mask=None):
        W_q = nn.Dense(features=self.d_model, use_bias=False)
        W_k = nn.Dense(features=self.d_model, use_bias=False)
        W_v = nn.Dense(features=self.d_model, use_bias=False)
        
        q = W_q(q)
        k = W_k(k)
        v = W_v(v)
        
        keys_transposed = jnp.swapaxes(k, -2, -1)
        logits = jnp.matmul(q, keys_transposed)
        
        scaled_logits = logits /jnp.sqrt(q.shape[-1])
        
        if mask is not None:
            scaled_logits = jnp.where(mask == 0, -1e9, scaled_logits)
        
        attention_distribution = jax.nn.softmax(scaled_logits, axis=-1)
        output = jnp.matmul(attention_distribution, v)
        
        return output
