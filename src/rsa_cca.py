"""Canonical Correlation Analysis (CCA) and Singular Vector CCA (SVCCA) for RSA.

Based on: Raghu et al. (2017), SVCCA: Singular Vector Canonical Correlation Analysis 
for Deep Learning Dynamics and Interpretability (arXiv:1706.05806)
"""

import numpy as np
from typing import Tuple, Optional


def compute_cca(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Compute Canonical Correlation Analysis between two representations.
    
    CCA finds linear projections of X and Y that are maximally correlated.
    Returns the canonical correlations (correlation coefficients for each dimension).
    
    Args:
        X: First representation matrix of shape (n_samples, n_features_x)
        Y: Second representation matrix of shape (n_samples, n_features_y)
        epsilon: Small constant for numerical stability
        
    Returns:
        Canonical correlations, sorted in descending order
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
    n_samples = X.shape[0]
    
    # Center the data
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # Compute covariance matrices
    # Scale by 1/(n-1) for unbiased estimate
    scale = 1.0 / (n_samples - 1)
    
    # Auto-covariance matrices
    C_xx = scale * X.T @ X + epsilon * np.eye(X.shape[1])
    C_yy = scale * Y.T @ Y + epsilon * np.eye(Y.shape[1])
    
    # Cross-covariance matrix
    C_xy = scale * X.T @ Y
    
    # Compute the CCA solution via generalized eigenvalue problem
    # We solve: C_xx^{-1/2} @ C_xy @ C_yy^{-1} @ C_xy.T @ C_xx^{-1/2} @ w = lambda * w
    
    # Compute square root inverses via SVD for numerical stability
    def sqrt_inv(C):
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        s_inv_sqrt = 1.0 / np.sqrt(s + epsilon)
        return U @ np.diag(s_inv_sqrt) @ Vt
    
    C_xx_sqrt_inv = sqrt_inv(C_xx)
    C_yy_sqrt_inv = sqrt_inv(C_yy)
    
    # Form the matrix for eigenvalue decomposition
    T = C_xx_sqrt_inv @ C_xy @ C_yy_sqrt_inv
    
    # Compute SVD of T to get canonical correlations
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    
    # The singular values are the canonical correlations
    # Clip to [0, 1] range to handle numerical errors
    canonical_corrs = np.clip(s, 0, 1)
    
    return canonical_corrs


def compute_svcca(X: np.ndarray, Y: np.ndarray, 
                  threshold: float = 0.99, 
                  epsilon: float = 1e-10) -> Tuple[np.ndarray, dict]:
    """Compute Singular Vector CCA (SVCCA) between two representations.
    
    SVCCA first performs SVD to reduce noise, keeping components that explain
    a threshold amount of variance, then applies CCA to the reduced representations.
    
    Args:
        X: First representation matrix of shape (n_samples, n_features_x)
        Y: Second representation matrix of shape (n_samples, n_features_y)
        threshold: Fraction of variance to retain (default 0.99)
        epsilon: Small constant for numerical stability
        
    Returns:
        Tuple of (mean canonical correlation, info dict with details)
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
    
    # Center the data
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # Perform SVD on each representation
    U_x, s_x, Vt_x = np.linalg.svd(X_centered, full_matrices=False)
    U_y, s_y, Vt_y = np.linalg.svd(Y_centered, full_matrices=False)
    
    # Determine number of components to keep based on variance threshold
    def get_n_components(s, threshold):
        s_squared = s ** 2
        cumsum = np.cumsum(s_squared) / np.sum(s_squared)
        n_components = np.searchsorted(cumsum, threshold) + 1
        return min(n_components, len(s))
    
    n_components_x = get_n_components(s_x, threshold)
    n_components_y = get_n_components(s_y, threshold)
    
    # Project onto principal components
    X_reduced = U_x[:, :n_components_x] * s_x[:n_components_x]
    Y_reduced = U_y[:, :n_components_y] * s_y[:n_components_y]
    
    # Apply CCA to reduced representations
    canonical_corrs = compute_cca(X_reduced, Y_reduced, epsilon)
    
    # Compute mean canonical correlation
    mean_cca = np.mean(canonical_corrs)
    
    info = {
        'canonical_correlations': canonical_corrs,
        'mean_cca': mean_cca,
        'n_components_x': n_components_x,
        'n_components_y': n_components_y,
        'original_dims': (X.shape[1], Y.shape[1]),
        'threshold': threshold
    }
    
    return mean_cca, info


def cca_distance(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute CCA distance between two representations.
    
    Distance is defined as 1 - mean(canonical_correlations).
    
    Args:
        X: First representation matrix of shape (n_samples, n_features_x)
        Y: Second representation matrix of shape (n_samples, n_features_y)
        epsilon: Small constant for numerical stability
        
    Returns:
        CCA distance in [0, 1], where 0 means identical and 1 means orthogonal
    """
    canonical_corrs = compute_cca(X, Y, epsilon)
    return 1.0 - np.mean(canonical_corrs)


def svcca_similarity(X: np.ndarray, Y: np.ndarray, 
                     threshold: float = 0.99,
                     epsilon: float = 1e-10) -> float:
    """Compute SVCCA similarity between two representations.
    
    Returns the mean canonical correlation after SVD denoising.
    
    Args:
        X: First representation matrix of shape (n_samples, n_features_x)
        Y: Second representation matrix of shape (n_samples, n_features_y)
        threshold: Fraction of variance to retain (default 0.99)
        epsilon: Small constant for numerical stability
        
    Returns:
        SVCCA similarity in [0, 1], where 1 means identical and 0 means orthogonal
    """
    mean_cca, _ = compute_svcca(X, Y, threshold, epsilon)
    return mean_cca


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features_x = 20
    n_features_y = 15
    
    # Create correlated representations
    # Shared latent factors
    latent = np.random.randn(n_samples, 5)
    
    # Project to different spaces with some noise
    W_x = np.random.randn(5, n_features_x)
    W_y = np.random.randn(5, n_features_y)
    
    X = latent @ W_x + 0.1 * np.random.randn(n_samples, n_features_x)
    Y = latent @ W_y + 0.1 * np.random.randn(n_samples, n_features_y)
    
    # Test CCA
    print("Testing CCA...")
    canonical_corrs = compute_cca(X, Y)
    print(f"Canonical correlations shape: {canonical_corrs.shape}")
    print(f"Top 5 canonical correlations: {canonical_corrs[:5]}")
    print(f"Mean canonical correlation: {np.mean(canonical_corrs):.4f}")
    print(f"CCA distance: {cca_distance(X, Y):.4f}")
    
    print("\nTesting SVCCA...")
    mean_cca, info = compute_svcca(X, Y)
    print(f"SVCCA similarity: {mean_cca:.4f}")
    print(f"Components kept: X={info['n_components_x']}/{n_features_x}, "
          f"Y={info['n_components_y']}/{n_features_y}")
    
    # Test with orthogonal representations
    print("\nTesting with orthogonal representations...")
    X_orth = np.random.randn(n_samples, n_features_x)
    Y_orth = np.random.randn(n_samples, n_features_y)
    
    print(f"CCA distance (orthogonal): {cca_distance(X_orth, Y_orth):.4f}")
    print(f"SVCCA similarity (orthogonal): {svcca_similarity(X_orth, Y_orth):.4f}")