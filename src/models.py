import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax
import jax.nn as nn


def init_params(key, k, n):
    """Initialize autoencoder parameters.
    
    Args:
        key: JAX random key
        k: bottleneck dimension (dense dimension)
        n: input dimension (sparse dimension)
        
    Returns:
        Dictionary containing model parameters
    """
    key1, key2 = jax.random.split(key)
    W = jax.random.normal(key1, (k, n)) * 0.01
    b = jnp.zeros((n,))
    return {'W': W, 'b': b}


def forward(params, x):
    """Forward pass through the autoencoder.
    
    Args:
        params: model parameters dictionary
        x: input data
        
    Returns:
        Reconstructed data with ReLU activation
    """
    W, b = params['W'], params['b']
    Wx = W @ x.T  # (k, N)
    x_recon = (W.T @ Wx).T + b  # (N, n)
    x_recon = nn.relu(x_recon)  # Apply ReLU
    return x_recon


def loss_fn(params, X, I):
    """Compute reconstruction loss with feature importance weighting.
    
    Args:
        params: model parameters
        X: input data
        I: feature importance weights
        
    Returns:
        Weighted reconstruction loss
    """
    x_recon = forward(params, X)
    loss = jnp.sum(I * (X - x_recon)**2)
    return loss


@jit
def train_step(params, opt_state, optimizer, X, I):
    """Single training step.
    
    Args:
        params: model parameters
        opt_state: optimizer state
        optimizer: optax optimizer
        X: input data
        I: feature importance weights
        
    Returns:
        Updated parameters, optimizer state, and loss
    """
    loss, grads = value_and_grad(loss_fn)(params, X, I)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train_model(data, I, k, n, num_epochs=10, learning_rate=1e-2, key=None):
    """Train an autoencoder model.
    
    Args:
        data: training data of shape (N, n)
        I: feature importance weights
        k: bottleneck dimension
        n: input dimension
        num_epochs: number of training epochs
        learning_rate: learning rate for Adam optimizer
        key: JAX random key
        
    Returns:
        Trained model parameters
    """
    if key is None:
        key = jax.random.PRNGKey(42)
        
    params = init_params(key, k, n)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    N = data.shape[0]
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(N):
            x_i = data[i, :]
            params, opt_state, loss = train_step(params, opt_state, optimizer, x_i, I)
            epoch_loss += loss
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Avg Loss: {epoch_loss / N:.4f}")
    
    return params