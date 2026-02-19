import numpy as np


def init_params(seed, k, n):
    """Initialize autoencoder parameters.
    
    Args:
        seed: random seed
        k: bottleneck dimension (dense dimension)
        n: input dimension (sparse dimension)
        
    Returns:
        Dictionary containing model parameters
    """
    np.random.seed(seed)
    W = np.random.normal(0, 0.01, size=(k, n))
    b = np.zeros((n,))
    return {'W': W, 'b': b}


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU activation function."""
    return (x > 0).astype(np.float32)


def forward(params, x):
    """Forward pass through the autoencoder.
    
    Args:
        params: model parameters dictionary
        x: input data of shape (batch_size, n) or (n,) for single sample
        
    Returns:
        Reconstructed data with ReLU activation
    """
    W, b = params['W'], params['b']
    
    # Handle single sample vs batch
    if x.ndim == 1:
        x = x.reshape(1, -1)
        single_sample = True
    else:
        single_sample = False
    
    # Forward pass: x -> z -> x_recon
    z = W @ x.T  # (k, batch_size)
    x_linear = (W.T @ z).T + b  # (batch_size, n)
    x_recon = relu(x_linear)
    
    if single_sample:
        return x_recon.flatten()
    return x_recon


def loss_fn(params, X, I):
    """Compute reconstruction loss with feature importance weighting.
    
    Args:
        params: model parameters
        X: input data of shape (batch_size, n) or (n,) for single sample  
        I: feature importance weights of shape (1, n)
        
    Returns:
        Weighted reconstruction loss (scalar)
    """
    x_recon = forward(params, X)
    
    # Handle single sample case
    if X.ndim == 1:
        X = X.reshape(1, -1)
        x_recon = x_recon.reshape(1, -1)
    
    # Compute weighted squared error
    diff = X - x_recon
    loss = np.sum(I * diff**2)
    return loss


def compute_gradients(params, x, I):
    """Compute gradients for a single sample using backpropagation.
    
    Args:
        params: model parameters
        x: single input sample of shape (n,)
        I: feature importance weights of shape (1, n)
        
    Returns:
        Dictionary containing gradients for W and b
    """
    W, b = params['W'], params['b']
    
    # Forward pass (save intermediate values)
    z = W @ x  # (k,)
    x_linear = W.T @ z + b  # (n,)
    x_recon = relu(x_linear)  # (n,)
    
    # Backward pass
    # Loss gradient w.r.t. x_recon: 2 * I * (x_recon - x)
    grad_x_recon = 2 * I.flatten() * (x_recon - x)  # (n,)
    
    # Gradient w.r.t. x_linear (before ReLU)
    grad_x_linear = grad_x_recon * relu_derivative(x_linear)  # (n,)
    
    # Gradient w.r.t. bias b
    grad_b = grad_x_linear  # (n,)
    
    # Gradient w.r.t. z
    grad_z = W @ grad_x_linear  # (k,)
    
    # Gradient w.r.t. W
    # From z = W @ x: dL/dW = outer(grad_z, x) -> shape (k, n)
    # From x_linear = W.T @ z: dL/dW += outer(z, grad_x_linear).T -> shape (k, n) 
    grad_W_from_z = np.outer(grad_z, x)  # (k, n)
    grad_W_from_linear = np.outer(z, grad_x_linear)  # (k, n)
    grad_W = grad_W_from_z + grad_W_from_linear
    
    return {'W': grad_W, 'b': grad_b}


class AdamOptimizer:
    """Adam optimizer implementation."""
    
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # time step
        self.m = {}  # first moment
        self.v = {}  # second moment
        
    def update(self, params, grads):
        """Update parameters using Adam algorithm."""
        self.t += 1
        
        # Initialize moments if first time
        if not self.m:
            self.m = {key: np.zeros_like(param) for key, param in params.items()}
            self.v = {key: np.zeros_like(param) for key, param in params.items()}
        
        updated_params = {}
        
        for key in params:
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second moment estimate  
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params


def train_step(params, optimizer, x, I):
    """Single training step.
    
    Args:
        params: model parameters
        optimizer: AdamOptimizer instance
        x: single input sample
        I: feature importance weights
        
    Returns:
        Updated parameters and loss value
    """
    # Compute gradients
    grads = compute_gradients(params, x, I)
    
    # Compute loss for logging
    loss = loss_fn(params, x, I)
    
    # Update parameters
    params = optimizer.update(params, grads)
    
    return params, loss


def train_model(data, I, k, n, num_epochs=10, learning_rate=1e-2, seed=42):
    """Train an autoencoder model.
    
    Args:
        data: training data of shape (N, n)
        I: feature importance weights
        k: bottleneck dimension
        n: input dimension
        num_epochs: number of training epochs
        learning_rate: learning rate for Adam optimizer
        seed: random seed
        
    Returns:
        Tuple of (trained model parameters, final epoch loss)
    """
    params = init_params(seed, k, n)
    optimizer = AdamOptimizer(learning_rate=learning_rate)
    
    N = data.shape[0]
    final_loss = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(N):
            x_i = data[i, :]
            params, loss = train_step(params, optimizer, x_i, I)
            epoch_loss += loss
        
        final_loss = epoch_loss / N
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Avg Loss: {final_loss:.4f}")
    
    return params, final_loss


# ========== NEW SAE IMPLEMENTATION START ==========
# All SAE-related functions are clearly marked for easy rollback

def get_bottleneck_activations(params, X):
    """Extract bottleneck activations (z) from trained autoencoder.
    
    This function is used to extract the middle layer representations
    that will serve as input to the Sparse Autoencoder.
    
    Args:
        params: trained autoencoder parameters
        X: input data of shape (N, n)
        
    Returns:
        Bottleneck activations of shape (N, k)
    """
    W = params['W']
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Forward pass to bottleneck: x -> z
    z = (W @ X.T).T  # (N, k)
    return z


def init_sae_params(seed, input_dim, hidden_dim):
    """Initialize Sparse Autoencoder parameters.
    
    Args:
        seed: random seed
        input_dim: input dimension (k from original autoencoder)
        hidden_dim: SAE hidden dimension (typically larger than input_dim)
        
    Returns:
        Dictionary containing SAE parameters
    """
    np.random.seed(seed)
    # Encoder: input_dim -> hidden_dim
    W_enc = np.random.normal(0, 0.01, size=(hidden_dim, input_dim))
    b_enc = np.zeros((hidden_dim,))
    
    # Decoder: hidden_dim -> input_dim  
    W_dec = np.random.normal(0, 0.01, size=(input_dim, hidden_dim))
    b_dec = np.zeros((input_dim,))
    
    return {
        'W_enc': W_enc,
        'b_enc': b_enc, 
        'W_dec': W_dec,
        'b_dec': b_dec
    }


def sae_forward(params, x):
    """Forward pass through Sparse Autoencoder.
    
    Args:
        params: SAE parameters dictionary
        x: input data of shape (batch_size, input_dim) or (input_dim,)
        
    Returns:
        Dictionary containing:
        - hidden: hidden layer activations (sparse)
        - recon: reconstructed output
    """
    W_enc, b_enc = params['W_enc'], params['b_enc']
    W_dec, b_dec = params['W_dec'], params['b_dec']
    
    # Handle single sample vs batch
    if x.ndim == 1:
        x = x.reshape(1, -1)
        single_sample = True
    else:
        single_sample = False
    
    # Encoder: x -> hidden (with ReLU activation for sparsity)
    hidden_linear = (W_enc @ x.T).T + b_enc  # (batch_size, hidden_dim)
    hidden = relu(hidden_linear)  # Sparse activations
    
    # Decoder: hidden -> reconstruction
    recon = (W_dec @ hidden.T).T + b_dec  # (batch_size, input_dim)
    
    if single_sample:
        return {
            'hidden': hidden.flatten(),
            'recon': recon.flatten()
        }
    
    return {
        'hidden': hidden,
        'recon': recon
    }


def sae_loss_fn(params, x, lam=0.01):
    """Compute SAE loss with L1 sparsity penalty (Anthropic style).
    
    Args:
        params: SAE parameters
        x: input data
        lam: L1 regularization strength for sparsity
        
    Returns:
        Total loss (reconstruction + L1 penalty)
    """
    forward_result = sae_forward(params, x)
    z = forward_result['hidden']    # hidden code
    x_hat = forward_result['recon'] # reconstruction
    
    # Handle single sample case
    if x.ndim == 1:
        x = x.reshape(1, -1)
        x_hat = x_hat.reshape(1, -1)
        z = z.reshape(1, -1)
    
    # Reconstruction term (MSE)
    recon_loss = np.mean((x - x_hat)**2)
    
    # L1 sparsity penalty on hidden activations
    l1_penalty = np.mean(np.abs(z))
    
    # Total loss
    total_loss = recon_loss + lam * l1_penalty
    return total_loss


def compute_sae_gradients(params, x, lam=0.01):
    """Compute gradients for SAE using backpropagation with L1 sparsity penalty.
    
    Args:
        params: SAE parameters
        x: single input sample
        lam: L1 regularization strength
        
    Returns:
        Dictionary containing gradients for all SAE parameters
    """
    W_enc, b_enc = params['W_enc'], params['b_enc']
    W_dec, b_dec = params['W_dec'], params['b_dec']
    
    # Forward pass (save intermediate values)
    hidden_linear = W_enc @ x + b_enc  # (hidden_dim,)
    z = relu(hidden_linear)            # (hidden_dim,) - hidden code
    x_hat = W_dec @ z + b_dec          # (input_dim,) - reconstruction
    
    # Backward pass
    # Reconstruction loss gradient: 2 * (x_hat - x)
    grad_recon = 2 * (x_hat - x)  # (input_dim,)
    
    # Decoder gradients
    grad_b_dec = grad_recon  # (input_dim,)
    grad_W_dec = np.outer(grad_recon, z)  # (input_dim, hidden_dim)
    
    # Hidden layer gradients from reconstruction
    grad_z_from_recon = W_dec.T @ grad_recon  # (hidden_dim,)
    
    # L1 sparsity penalty gradient: sign(z) for L1 regularization
    grad_z_from_l1 = lam * np.sign(z)  # (hidden_dim,)
    
    # Total gradient w.r.t. hidden activations
    grad_z_total = grad_z_from_recon + grad_z_from_l1
    
    # Apply ReLU derivative
    grad_hidden_linear = grad_z_total * relu_derivative(hidden_linear)
    
    # Encoder gradients
    grad_b_enc = grad_hidden_linear  # (hidden_dim,)
    grad_W_enc = np.outer(grad_hidden_linear, x)  # (hidden_dim, input_dim)
    
    return {
        'W_enc': grad_W_enc,
        'b_enc': grad_b_enc,
        'W_dec': grad_W_dec,
        'b_dec': grad_b_dec
    }


def sae_train_step(params, optimizer, x, lam=0.01):
    """Single SAE training step.
    
    Args:
        params: SAE parameters
        optimizer: AdamOptimizer instance  
        x: single input sample
        lam: L1 regularization strength
        
    Returns:
        Updated parameters and loss value
    """
    # Compute gradients
    grads = compute_sae_gradients(params, x, lam)
    
    # Compute loss for logging
    loss = sae_loss_fn(params, x, lam)
    
    # Update parameters
    params = optimizer.update(params, grads)
    
    return params, loss


def train_sae_batch(activations, input_dim, hidden_dim, num_epochs=50, 
                   learning_rate=1e-3, sparsity_lambda=0.01, sparsity_target=0.05, 
                   batch_size=32, seed=42, verbose=True):
    """Train a Sparse Autoencoder using batch-based KL divergence sparsity.
    
    Args:
        activations: bottleneck activations from original autoencoder (N, k)
        input_dim: input dimension (k from original autoencoder)
        hidden_dim: SAE hidden dimension (typically > input_dim)
        num_epochs: number of training epochs
        learning_rate: learning rate for Adam optimizer
        sparsity_lambda: KL regularization strength for sparsity
        sparsity_target: Target average activation level (rho)
        batch_size: batch size for proper KL divergence computation
        seed: random seed
        verbose: whether to print training progress
        
    Returns:
        Tuple of (trained SAE parameters, final epoch loss)
    """
    params = init_sae_params(seed, input_dim, hidden_dim)
    optimizer = AdamOptimizer(learning_rate=learning_rate)
    
    N = activations.shape[0]
    final_loss = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data each epoch
        indices = np.random.permutation(N)
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_indices = indices[start_idx:end_idx]
            batch = activations[batch_indices]
            
            # Forward pass for entire batch
            batch_outputs = sae_forward(params, batch)
            batch_hidden = batch_outputs['hidden']
            batch_recon = batch_outputs['recon']
            
            # Compute batch loss with proper KL divergence
            recon_loss = np.mean((batch - batch_recon)**2)
            
            # KL divergence with proper batch statistics
            rho_hat = np.mean(batch_hidden, axis=0)  # Average per neuron across batch
            rho = sparsity_target
            epsilon = 1e-8
            
            rho_hat = np.clip(rho_hat, epsilon, 1 - epsilon)
            kl_divergence = rho * np.log(rho / rho_hat) + (1 - rho) * np.log((1 - rho) / (1 - rho_hat))
            sparsity_loss = sparsity_lambda * np.sum(kl_divergence)
            
            batch_loss = recon_loss + sparsity_loss
            
            # Compute batch gradients
            gradients = compute_sae_batch_gradients(params, batch, batch_hidden, batch_recon, 
                                                  sparsity_lambda, sparsity_target, rho_hat)
            
            # Update parameters
            params = optimizer.update(params, gradients)
            
            epoch_loss += batch_loss
            num_batches += 1
        
        final_loss = epoch_loss / num_batches
        if verbose and epoch % 10 == 0:
            print(f"SAE Epoch {epoch}, Avg Loss: {final_loss:.4f}")
    
    return params, final_loss

def compute_sae_batch_gradients(params, batch_x, batch_hidden, batch_recon, 
                               sparsity_lambda, sparsity_target, rho_hat):
    """Compute SAE gradients for a batch with proper KL divergence."""
    W_enc, b_enc = params['W_enc'], params['b_enc']
    W_dec, b_dec = params['W_dec'], params['b_dec']
    
    batch_size = batch_x.shape[0]
    
    # Reconstruction gradients
    grad_recon = 2 * (batch_recon - batch_x) / batch_size  # (batch_size, input_dim)
    
    # Decoder gradients
    grad_b_dec = np.mean(grad_recon, axis=0)  # (input_dim,)
    grad_W_dec = grad_recon.T @ batch_hidden / batch_size  # (input_dim, hidden_dim)
    
    # Hidden layer gradients from reconstruction
    grad_hidden_recon = grad_recon @ W_dec  # (batch_size, hidden_dim)
    
    # KL divergence gradients
    rho = sparsity_target
    epsilon = 1e-8
    rho_hat = np.clip(rho_hat, epsilon, 1 - epsilon)
    
    # Gradient of KL w.r.t. rho_hat: -rho/rho_hat + (1-rho)/(1-rho_hat)
    grad_kl_rho_hat = -rho / rho_hat + (1 - rho) / (1 - rho_hat)
    
    # d_rho_hat / d_hidden = 1/batch_size (since rho_hat = mean(hidden, axis=0))
    grad_hidden_kl = sparsity_lambda * grad_kl_rho_hat / batch_size  # (hidden_dim,)
    
    # Total hidden gradients
    grad_hidden_total = grad_hidden_recon + grad_hidden_kl  # (batch_size, hidden_dim)
    
    # Apply ReLU derivative (need to recompute hidden_linear)
    hidden_linear = (W_enc @ batch_x.T).T + b_enc  # (batch_size, hidden_dim)
    relu_mask = (hidden_linear > 0).astype(float)
    grad_hidden_linear = grad_hidden_total * relu_mask
    
    # Encoder gradients
    grad_b_enc = np.mean(grad_hidden_linear, axis=0)  # (hidden_dim,)
    grad_W_enc = grad_hidden_linear.T @ batch_x / batch_size  # (hidden_dim, input_dim)
    
    return {
        'W_enc': grad_W_enc,
        'b_enc': grad_b_enc,
        'W_dec': grad_W_dec,
        'b_dec': grad_b_dec
    }

def train_sae(activations, input_dim, hidden_dim, num_epochs=50, 
              learning_rate=1e-3, lam=0.01, seed=42, verbose=True):
    """Train a Sparse Autoencoder using L1 sparsity penalty.
    
    Args:
        activations: bottleneck activations from original autoencoder (N, k)
        input_dim: input dimension (k from original autoencoder)
        hidden_dim: SAE hidden dimension (typically > input_dim)
        num_epochs: number of training epochs
        learning_rate: learning rate for Adam optimizer
        lam: L1 regularization strength for sparsity
        seed: random seed
        verbose: whether to print training progress
        
    Returns:
        Tuple of (trained SAE parameters, final epoch loss)
    """
    params = init_sae_params(seed, input_dim, hidden_dim)
    optimizer = AdamOptimizer(learning_rate=learning_rate)
    
    N = activations.shape[0]
    final_loss = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(N):
            x_i = activations[i, :]
            params, loss = sae_train_step(params, optimizer, x_i, lam)
            epoch_loss += loss
        
        final_loss = epoch_loss / N
        if verbose and epoch % 10 == 0:
            print(f"SAE Epoch {epoch}, Avg Loss: {final_loss:.4f}")
    
    return params, final_loss

# ========== NEW SAE IMPLEMENTATION END ==========