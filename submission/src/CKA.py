"""Centered Kernel Alignment (CKA) implementation for representation similarity analysis."""

import numpy as np


def linear_cka(X, Y):
    """Compute Linear Centered Kernel Alignment (CKA) between two representations.
    
    CKA measures the similarity between two representations by comparing their
    kernel matrices. Linear CKA uses dot product kernels.
    
    Args:
        X: First representation matrix (N, D1)
        Y: Second representation matrix (N, D2)
        
    Returns:
        CKA similarity score between 0 and 1
    """
    N = X.shape[0]
    
    # Compute Gram matrices (dot product kernels)
    K = X @ X.T  # (N, N)
    L = Y @ Y.T  # (N, N)
    
    # Center the kernel matrices
    H = np.eye(N) - np.ones((N, N)) / N  # Centering matrix
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_XY = np.trace(K_centered @ L_centered) / (N - 1)**2
    hsic_XX = np.trace(K_centered @ K_centered) / (N - 1)**2
    hsic_YY = np.trace(L_centered @ L_centered) / (N - 1)**2
    
    # Compute CKA
    cka = hsic_XY / np.sqrt(hsic_XX * hsic_YY)
    
    return cka


def rbf_cka(X, Y, sigma=None):
    """Compute RBF (Gaussian) Centered Kernel Alignment between two representations.
    
    Args:
        X: First representation matrix (N, D1)
        Y: Second representation matrix (N, D2)
        sigma: RBF kernel bandwidth. If None, uses median heuristic.
        
    Returns:
        CKA similarity score between 0 and 1
    """
    N = X.shape[0]
    
    # Compute RBF kernels
    K = rbf_kernel(X, sigma)
    L = rbf_kernel(Y, sigma)
    
    # Center the kernel matrices
    H = np.eye(N) - np.ones((N, N)) / N
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    # Compute HSIC
    hsic_XY = np.trace(K_centered @ L_centered) / (N - 1)**2
    hsic_XX = np.trace(K_centered @ K_centered) / (N - 1)**2
    hsic_YY = np.trace(L_centered @ L_centered) / (N - 1)**2
    
    # Compute CKA
    cka = hsic_XY / np.sqrt(hsic_XX * hsic_YY)
    
    return cka


def rbf_kernel(X, sigma=None):
    """Compute RBF (Gaussian) kernel matrix.
    
    Args:
        X: Data matrix (N, D)
        sigma: Kernel bandwidth. If None, uses median heuristic.
        
    Returns:
        Kernel matrix (N, N)
    """
    # Compute pairwise squared distances
    XX = np.sum(X**2, axis=1, keepdims=True)
    distances_sq = XX + XX.T - 2 * X @ X.T
    
    # Use median heuristic for sigma if not provided
    if sigma is None:
        sigma = np.sqrt(np.median(distances_sq[distances_sq > 0]))
    
    # Compute RBF kernel
    K = np.exp(-distances_sq / (2 * sigma**2))
    
    return K


def compare_representations(X, Y, method='linear'):
    """Compare two representations using CKA.
    
    Args:
        X: First representation matrix (N, D1)
        Y: Second representation matrix (N, D2)
        method: 'linear' or 'rbf'
        
    Returns:
        Dictionary with CKA score and interpretation
    """
    if method == 'linear':
        cka_score = linear_cka(X, Y)
    elif method == 'rbf':
        cka_score = rbf_cka(X, Y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Interpretation
    if cka_score > 0.9:
        interpretation = "Very high similarity"
    elif cka_score > 0.7:
        interpretation = "High similarity"
    elif cka_score > 0.5:
        interpretation = "Moderate similarity"
    elif cka_score > 0.3:
        interpretation = "Low similarity"
    else:
        interpretation = "Very low similarity"
    
    return {
        'cka': cka_score,
        'interpretation': interpretation,
        'method': method
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample representations
    N, D1, D2 = 100, 50, 30
    X = np.random.randn(N, D1)
    Y = np.random.randn(N, D2)
    
    # Similar representation (linear transformation of X)
    W = np.random.randn(D1, D2)
    Y_similar = X @ W + 0.1 * np.random.randn(N, D2)
    
    # Compare representations
    print("Random vs Random:")
    result = compare_representations(X, Y, 'linear')
    print(f"  Linear CKA: {result['cka']:.4f} - {result['interpretation']}")
    
    print("\nX vs Linear transform of X:")
    result = compare_representations(X, Y_similar, 'linear')
    print(f"  Linear CKA: {result['cka']:.4f} - {result['interpretation']}")
    
    print("\nX vs itself:")
    result = compare_representations(X, X, 'linear')
    print(f"  Linear CKA: {result['cka']:.4f} - {result['interpretation']}")