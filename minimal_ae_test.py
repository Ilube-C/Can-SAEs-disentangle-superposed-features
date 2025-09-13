#!/usr/bin/env python3
"""
Minimal autoencoder test - just the essentials
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src')

from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import train_model, loss_fn

def minimal_test():
    sparsities = [0.85, 0.9, 0.95]
    
    print("Testing 10-5-10 AE at sparsities:", sparsities)
    
    for sparsity in sparsities:
        print(f"\n--- Sparsity {sparsity} ---")
        
        # Generate data
        data = generate_synthetic_data(123, 10, sparsity, 1000)  # Smaller dataset
        I = get_feature_importances(10, 0.7)
        
        print(f"Empirical sparsity: {np.mean(data == 0):.3f}")
        
        # Train AE (fewer epochs)
        params = train_model(data, I, k=5, n=10, num_epochs=2, seed=123)
        
        # Compute loss
        loss = loss_fn(params, data, I) / data.shape[0]
        
        # Get weight matrix
        W = params['W']
        superposition = W.T @ W
        
        print(f"Reconstruction loss: {loss:.4f}")
        print(f"Weight matrix condition: {np.linalg.cond(W):.1f}")
        print(f"Max superposition value: {np.max(np.abs(superposition)):.3f}")
        print(f"Superposition matrix diagonal mean: {np.mean(np.diag(superposition)):.3f}")

if __name__ == "__main__":
    minimal_test()