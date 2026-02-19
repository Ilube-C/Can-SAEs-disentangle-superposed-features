import numpy as np


def generate_synthetic_data(seed, n, S, N):
    """Generate synthetic sparse data as per the toy models of superposition paper.
    
    Args:
        seed: random seed (int)
        n: sparse dimension 
        S: sparsity parameter (probability of zero)
        N: number of datapoints
        
    Returns:
        Array of shape (N, n) with sparse synthetic data
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create Bernoulli samples (0 or 1) with probability 1-S
    bernoulli_samples = np.random.binomial(1, 1-S, size=(N, n)).astype(np.float32)
    
    # Create uniform random numbers in [0, 1)
    uniform_samples = np.random.uniform(0, 1, size=(N, n))
    
    # Element-wise multiplication
    return uniform_samples * bernoulli_samples


def get_feature_importances(n, decay_factor=0.7):
    """Generate exponentially decaying feature importances.
    
    Args:
        n: number of features
        decay_factor: exponential decay factor
        
    Returns:
        Array of shape (1, n) with feature importances
    """
    feature_importances = decay_factor**np.arange(n)
    return np.reshape(feature_importances, (1, -1))