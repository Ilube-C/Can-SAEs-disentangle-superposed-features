#!/usr/bin/env python3
"""
Quick test script to verify the NumPy implementation works.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import train_model, forward

def test_data_generation():
    """Test synthetic data generation."""
    print("Testing data generation...")
    
    seed = 42
    n, S, N = 10, 0.5, 100
    
    data = generate_synthetic_data(seed, n, S, N)
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Sparsity (zeros): {np.mean(data == 0):.3f}")
    print(f"Mean non-zero values: {np.mean(data[data > 0]):.3f}")
    
    return data

def test_feature_importances():
    """Test feature importance generation."""
    print("\nTesting feature importances...")
    
    n = 10
    I = get_feature_importances(n)
    print(f"Feature importances shape: {I.shape}")
    print(f"Feature importances: {I.flatten()[:5]}...")
    
    return I

def test_model_training():
    """Test model training with a small example."""
    print("\nTesting model training...")
    
    # Small test case
    n, k, N = 10, 3, 500
    S = 0.3
    seed = 42
    
    # Generate test data
    data = generate_synthetic_data(seed, n, S, N)
    I = get_feature_importances(n)
    
    print(f"Training data shape: {data.shape}")
    print(f"Bottleneck dimension: {k}")
    
    # Train model
    params = train_model(
        data=data,
        I=I, 
        k=k,
        n=n,
        num_epochs=5,
        learning_rate=0.01,
        seed=seed
    )
    
    print(f"Trained W shape: {params['W'].shape}")
    print(f"Trained b shape: {params['b'].shape}")
    
    # Test forward pass
    test_input = data[0]
    reconstruction = forward(params, test_input)
    
    print(f"Test input: {test_input[:5]}...")
    print(f"Reconstruction: {reconstruction[:5]}...")
    print(f"Reconstruction error: {np.mean((test_input - reconstruction)**2):.6f}")
    
    return params

def main():
    print("=== NumPy Implementation Test ===")
    
    try:
        # Test each component
        data = test_data_generation()
        I = test_feature_importances()
        params = test_model_training()
        
        print("\n[SUCCESS] All tests passed! NumPy implementation working.")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())