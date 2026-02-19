"""Procrustes Analysis for measuring alignment between representations.

Based on: Schönemann (1966), A generalized solution of the orthogonal Procrustes problem
and modern applications in neural network analysis.
"""

import numpy as np
from typing import Tuple, Optional, Dict


def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray, 
                          scaling: bool = False,
                          reflection: bool = True) -> Tuple[np.ndarray, float, Dict]:
    """Find the optimal orthogonal transformation to align X to Y.
    
    Solves: min ||Y - XR||_F where R is an orthogonal matrix.
    
    Args:
        X: Source representation matrix of shape (n_samples, n_features)
        Y: Target representation matrix of shape (n_samples, n_features)
        scaling: If True, also find optimal scaling factor
        reflection: If True, allow reflections (det(R) = ±1); 
                   if False, only rotations (det(R) = 1)
        
    Returns:
        Tuple of (R, disparity, info) where:
        - R: Optimal orthogonal transformation matrix
        - disparity: Normalized Procrustes distance after alignment
        - info: Dictionary with additional information
    """
    assert X.shape == Y.shape, "X and Y must have the same shape"
    
    # Center the data
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # Normalize for scale-invariant comparison if not using scaling
    if not scaling:
        X_norm = np.linalg.norm(X_centered, 'fro')
        Y_norm = np.linalg.norm(Y_centered, 'fro')
        
        if X_norm > 0:
            X_centered = X_centered / X_norm
        if Y_norm > 0:
            Y_centered = Y_centered / Y_norm
    
    # Compute the cross-covariance matrix
    H = X_centered.T @ Y_centered
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    
    # Compute optimal rotation
    R = U @ Vt
    
    # If only rotations allowed (no reflections), ensure det(R) = 1
    if not reflection and np.linalg.det(R) < 0:
        # Flip the sign of the smallest singular vector
        U[:, -1] *= -1
        R = U @ Vt
    
    # Apply transformation
    X_aligned = X_centered @ R
    
    # Compute optimal scaling if requested
    scale = 1.0
    if scaling:
        # Optimal scaling: s = tr(X'Y R) / tr(X'X)
        numerator = np.trace(X_centered.T @ Y_centered @ R.T)
        denominator = np.trace(X_centered.T @ X_centered)
        if denominator > 0:
            scale = numerator / denominator
        X_aligned = scale * X_aligned
    
    # Compute disparity (normalized Procrustes distance)
    # d^2 = ||Y - sXR||_F^2 / ||Y||_F^2
    residual = Y_centered - X_aligned
    disparity = np.linalg.norm(residual, 'fro') / np.linalg.norm(Y_centered, 'fro')
    
    info = {
        'rotation_matrix': R,
        'scale': scale,
        'singular_values': S,
        'determinant': np.linalg.det(R),
        'is_rotation': np.linalg.det(R) > 0,
        'frobenius_error': np.linalg.norm(residual, 'fro')
    }
    
    return R, disparity, info


def procrustes_similarity(X: np.ndarray, Y: np.ndarray,
                         scaling: bool = False,
                         reflection: bool = True) -> float:
    """Compute Procrustes similarity between two representations.
    
    Similarity is defined as 1 - normalized_procrustes_distance.
    
    Args:
        X: First representation matrix of shape (n_samples, n_features)
        Y: Second representation matrix of shape (n_samples, n_features)
        scaling: If True, also find optimal scaling factor
        reflection: If True, allow reflections; if False, only rotations
        
    Returns:
        Procrustes similarity in [0, 1], where 1 means perfect alignment
    """
    _, disparity, _ = orthogonal_procrustes(X, Y, scaling, reflection)
    return 1.0 - disparity


def generalized_procrustes(representations: list,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Tuple[list, np.ndarray, float]:
    """Generalized Procrustes Analysis for multiple representations.
    
    Finds optimal alignments for multiple representations to a common space.
    
    Args:
        representations: List of representation matrices, each of shape (n_samples, n_features)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (aligned_representations, mean_shape, final_error)
    """
    assert len(representations) >= 2, "Need at least 2 representations"
    
    # Check all have same shape
    shape = representations[0].shape
    for rep in representations[1:]:
        assert rep.shape == shape, "All representations must have same shape"
    
    n_reps = len(representations)
    
    # Initialize with centered representations
    aligned = []
    for rep in representations:
        centered = rep - rep.mean(axis=0)
        normalized = centered / np.linalg.norm(centered, 'fro')
        aligned.append(normalized.copy())
    
    # Iteratively align to mean
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        # Compute current mean shape
        mean_shape = np.mean(aligned, axis=0)
        mean_shape = mean_shape / np.linalg.norm(mean_shape, 'fro')
        
        # Align each representation to the mean
        new_aligned = []
        total_error = 0
        
        for i, rep in enumerate(representations):
            # Center and normalize
            centered = rep - rep.mean(axis=0)
            normalized = centered / np.linalg.norm(centered, 'fro')
            
            # Align to mean shape
            R, disparity, _ = orthogonal_procrustes(normalized, mean_shape)
            aligned_rep = normalized @ R
            new_aligned.append(aligned_rep)
            total_error += disparity
        
        aligned = new_aligned
        avg_error = total_error / n_reps
        
        # Check convergence
        if abs(prev_error - avg_error) < tolerance:
            break
        prev_error = avg_error
    
    return aligned, mean_shape, avg_error


def angular_procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute angular distance after Procrustes alignment.
    
    This measures the angle between the vectorized representations after
    optimal alignment, which is invariant to orthogonal transformations.
    
    Args:
        X: First representation matrix of shape (n_samples, n_features)
        Y: Second representation matrix of shape (n_samples, n_features)
        
    Returns:
        Angular distance in [0, π/2]
    """
    R, _, _ = orthogonal_procrustes(X, Y, scaling=False)
    
    # Align X to Y
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    X_norm = np.linalg.norm(X_centered, 'fro')
    Y_norm = np.linalg.norm(Y_centered, 'fro')
    
    if X_norm > 0:
        X_centered = X_centered / X_norm
    if Y_norm > 0:
        Y_centered = Y_centered / Y_norm
    
    X_aligned = X_centered @ R
    
    # Compute angle between vectorized representations
    x_vec = X_aligned.flatten()
    y_vec = Y_centered.flatten()
    
    cos_angle = np.clip(np.dot(x_vec, y_vec), -1, 1)
    angle = np.arccos(abs(cos_angle))  # Use abs to ignore sign
    
    return angle


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    # Create a random representation
    X = np.random.randn(n_samples, n_features)
    
    # Create a rotated version with some noise
    theta = np.pi / 4  # 45 degree rotation in 2D subspace
    R_true = np.eye(n_features)
    R_true[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])
    
    Y = X @ R_true + 0.1 * np.random.randn(n_samples, n_features)
    
    # Test orthogonal Procrustes
    print("Testing Orthogonal Procrustes...")
    R, disparity, info = orthogonal_procrustes(X, Y)
    print(f"Procrustes disparity: {disparity:.4f}")
    print(f"Procrustes similarity: {procrustes_similarity(X, Y):.4f}")
    print(f"Is pure rotation: {info['is_rotation']}")
    print(f"Frobenius error: {info['frobenius_error']:.4f}")
    
    # Test with scaling
    print("\nTesting with scaling...")
    Y_scaled = 2.0 * Y  # Scale by factor of 2
    R_scale, disparity_scale, info_scale = orthogonal_procrustes(X, Y_scaled, scaling=True)
    print(f"Recovered scale: {info_scale['scale']:.4f} (true: 2.0)")
    print(f"Disparity with scaling: {disparity_scale:.4f}")
    
    # Test angular distance
    print("\nTesting angular distance...")
    angle = angular_procrustes_distance(X, Y)
    print(f"Angular distance: {angle:.4f} radians ({np.degrees(angle):.2f} degrees)")
    
    # Test with orthogonal representations
    print("\nTesting with orthogonal representations...")
    X_orth = np.random.randn(n_samples, n_features)
    Y_orth = np.random.randn(n_samples, n_features)
    
    similarity_orth = procrustes_similarity(X_orth, Y_orth)
    print(f"Similarity (random): {similarity_orth:.4f}")
    
    # Test generalized Procrustes with multiple representations
    print("\nTesting Generalized Procrustes...")
    representations = [X, Y, X @ R_true.T + 0.05 * np.random.randn(n_samples, n_features)]
    aligned, mean_shape, error = generalized_procrustes(representations)
    print(f"Final alignment error: {error:.4f}")
    print(f"Mean shape: {mean_shape.shape}")