import numpy as np
from scipy.linalg import svd

def subspace_angles(X, Y):
    # Center
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)

    # QR (orthonormal bases)
    Qx, _ = np.linalg.qr(X)
    Qy, _ = np.linalg.qr(Y)

    # Cross-product
    M = Qx.T @ Qy

    # Singular values
    s = svd(M, compute_uv=False)

    # Clamp in case of numerical issues
    s = np.clip(s, -1, 1)

    # Principal angles
    angles = np.arccos(s)
    return angles  # in radians