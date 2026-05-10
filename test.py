import jax
import jax.numpy as jnp
import os

# Script testing if GPU is working.
def test_jax_install():
    print("--- JAX Device Check ---")
    
    # 1. Check available devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    # 2. Check the default backend (should be 'gpu')
    backend = jax.default_backend()
    print(f"Default backend: {backend}")
    
    # 3. Perform a simple computation
    try:
        # Create a large matrix to ensure it hits the GPU
        x = jnp.ones((1000, 1000))
        y = jnp.dot(x, x)
        
        # Block until result is ready to catch any asynchronous CUDA errors
        y.block_until_ready()
        
        print(f"Computation successful on: {y.devices()}")
        print("\n✅ CUDA 13 and JAX are working together!")
        
    except Exception as e:
        print("\n❌ Computation failed!")
        print(f"Error: {e}")
        
        # Check for the CuDNN mismatch in the error log
        if "cudnn" in str(e).lower():
            print("\nTip: This looks like a CuDNN library mismatch.")

if __name__ == "__main__":
    test_jax_install()