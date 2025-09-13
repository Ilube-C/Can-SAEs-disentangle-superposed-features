#!/usr/bin/env python3
"""
Diagnostic script to check weight matrices and actual data statistics
"""

import numpy as np
import sys
sys.path.append('src')

from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import train_model

def diagnose_sparsity_effect():
    """Check how often each feature actually appears in data at different sparsities."""
    
    sparsities = [0.3, 0.5, 0.75, 0.85, 0.9, 0.95]
    sparse_dim = 10
    num_samples = 10000  # Increased for more stable feature frequencies
    seed = 123
    
    print("FEATURE ACTIVATION FREQUENCY ANALYSIS")
    print("="*60)
    print(f"Samples: {num_samples}, Features: {sparse_dim}")
    print("="*60)
    
    for sparsity in sparsities:
        print(f"\n--- Sparsity {sparsity} ---")
        
        # Generate data
        data = generate_synthetic_data(seed, sparse_dim, sparsity, num_samples)
        
        # Count non-zero occurrences per feature
        non_zero_counts = np.sum(data > 0, axis=0)
        activation_freq = non_zero_counts / num_samples
        
        # Get importance weights
        I = get_feature_importances(sparse_dim, 0.7)
        
        # Train a small model
        params = train_model(data, I, k=5, n=sparse_dim, num_epochs=3, seed=seed)
        W = params['W']
        
        # Check weight norms per feature (columns of W)
        weight_norms = np.linalg.norm(W, axis=0)
        
        # Print diagnostics
        print(f"Feature | Importance | Actual Freq | Weight Norm")
        print("-" * 50)
        for i in range(min(5, sparse_dim)):  # Show first 5 features
            print(f"   {i:2d}   |   {I[0,i]:.3f}    |    {activation_freq[i]:.3f}    |   {weight_norms[i]:.3f}")
        
        # Special check for feature 0
        if activation_freq[0] < 0.01:
            print(f"\nWARNING: Feature 0 appears in only {activation_freq[0]*100:.1f}% of samples!")
            print(f"   Despite importance={I[0,0]:.3f}, it's effectively dead in the data")
        
        # Check if weight norm correlates with actual frequency more than importance
        actual_freq_correlation = np.corrcoef(activation_freq, weight_norms)[0,1]
        importance_correlation = np.corrcoef(I.flatten(), weight_norms)[0,1]
        
        print(f"\nCorrelations with weight norms:")
        print(f"  Actual frequency: {actual_freq_correlation:.3f}")
        print(f"  Importance weights: {importance_correlation:.3f}")
        
        if actual_freq_correlation > importance_correlation:
            print("  -> Model weights follow actual data frequency more than importance!")

def check_weight_matrices():
    """Quick check of weight matrix structure for 4-3-4 architecture."""
    
    print("\n\n4-3-4 ARCHITECTURE CHECK")
    print("="*60)
    
    # Train 4-3-4 model at high sparsity to match your graphs
    data = generate_synthetic_data(123, 4, 0.95, 10000)
    I = get_feature_importances(4, 0.7)
    
    print("Feature importances:", I.flatten())
    
    params = train_model(data, I, k=3, n=4, num_epochs=5, seed=123)
    W = params['W']
    
    print("\nWeight matrix W (3x4):")
    print(W)
    
    print("\nWeight norms by feature:")
    for i in range(4):
        norm = np.linalg.norm(W[:, i])
        print(f"  Feature {i}: {norm:.3f}")
    
    print("\nSuperposition matrix W.T @ W (4x4):")
    superposition = W.T @ W
    print(superposition)
    
    print("\nDiagonal values (self-reconstruction strength):")
    for i in range(4):
        print(f"  Feature {i}: {superposition[i,i]:.3f}")
        
    # Check feature 0 specifically
    feature_0_activation = np.mean(data[:, 0] > 0)
    print(f"\nFeature 0 activation frequency: {feature_0_activation:.3f}")
    if superposition[0,0] < 0.1:
        print("-> Feature 0 has weak diagonal reconstruction!")

if __name__ == "__main__":
    diagnose_sparsity_effect()
    check_weight_matrices()