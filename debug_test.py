#!/usr/bin/env python3
"""
Debug test script for SAE implementation
"""

import os
import sys

# Add local modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))

# Test basic imports
try:
    print("Testing imports...")
    from experiment_config import DEFAULT_CONFIG
    print("+ Config import successful")
    
    from data_generation import generate_synthetic_data, get_feature_importances
    print("+ Data generation import successful")
    
    from models_numpy import train_model, get_bottleneck_activations, train_sae
    print("+ Models import successful")
    
    from analysis import compute_rsa_correlation, analyze_sae_performance
    print("+ Analysis import successful")
    
    print("\nAll imports successful! Running basic functionality test...")
    
    # Create minimal test data
    import numpy as np
    print("Generating test data...")
    data = generate_synthetic_data(42, 10, 0.7, 100)  # Very small test
    I = get_feature_importances(10, 0.7)
    print(f"Data shape: {data.shape}")
    print(f"Feature importances shape: {I.shape}")
    
    # Train minimal autoencoder
    print("Training minimal autoencoder...")
    params = train_model(data, I, k=3, n=10, num_epochs=2, seed=42)
    print("+ Autoencoder training successful")
    
    # Extract activations
    print("Extracting bottleneck activations...")
    activations = get_bottleneck_activations(params, data)
    print(f"Activations shape: {activations.shape}")
    print("+ Activation extraction successful")
    
    # Train minimal SAE
    print("Training minimal SAE...")
    sae_params = train_sae(activations, input_dim=3, hidden_dim=5, 
                          num_epochs=5, verbose=False, seed=42)
    print("+ SAE training successful")
    
    # Test RSA
    print("Testing RSA...")
    rsa_score = compute_rsa_correlation(activations, activations)
    print(f"RSA self-correlation: {rsa_score:.4f}")
    print("+ RSA computation successful")
    
    print("\nSUCCESS: All basic functionality tests passed!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()