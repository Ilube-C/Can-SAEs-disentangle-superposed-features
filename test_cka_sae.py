"""Test CKA implementation with SAE experiment."""

import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')
sys.path.append('config')

from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import train_model, get_bottleneck_activations, train_sae
from analysis import analyze_sae_performance

def quick_cka_sae_test():
    """Quick test comparing RSA and CKA in SAE analysis."""
    
    print("="*60)
    print("Testing CKA vs RSA in SAE Analysis")
    print("="*60)
    
    # Set parameters for quick test
    seed = 42
    sparse_dim = 20
    dense_dim = 5
    sparsity = 0.7
    num_samples = 500  # Small for quick test
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_data(seed, sparse_dim, sparsity, num_samples)
    I = get_feature_importances(sparse_dim, 0.7)
    
    # Train autoencoder
    print("\n2. Training autoencoder...")
    ae_params = train_model(
        data=data,
        I=I,
        k=dense_dim,
        n=sparse_dim,
        num_epochs=10,  # Quick training
        learning_rate=1e-3,
        seed=seed
    )
    
    # Get bottleneck activations
    print("\n3. Extracting bottleneck activations...")
    bottleneck = get_bottleneck_activations(ae_params, data)
    print(f"   Bottleneck shape: {bottleneck.shape}")
    
    # Train SAE
    print("\n4. Training Sparse Autoencoder...")
    sae_params = train_sae(
        activations=bottleneck,
        input_dim=dense_dim,
        hidden_dim=15,
        num_epochs=20,  # Quick training
        learning_rate=1e-3,
        l1_penalty=0.01,
        seed=seed + 1000
    )
    
    # Analyze with CKA (currently set to True in analysis.py)
    print("\n5. Analyzing with CKA...")
    results = analyze_sae_performance(bottleneck, sae_params, verbose=True)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Metric used: {results.get('similarity_metric_name', 'CKA')}")
    print(f"Similarity score: {results['rsa_correlation']:.4f}")
    print(f"Self-check (should be ~1.0): {results['rsa_self_check']:.4f}")
    print(f"Random baseline (should be ~0.0): {results['rsa_random_baseline']:.4f}")
    print(f"Reconstruction MSE: {results['reconstruction_mse']:.6f}")
    print(f"Sparsity improvement: {results['sparsity_improvement']:.2f}x")
    
    # Interpret CKA values
    print("\n" + "="*60)
    print("CKA INTERPRETATION")
    print("="*60)
    cka_value = results['rsa_correlation']
    if cka_value > 0.9:
        interpretation = "Very high similarity - SAE preserves representation structure excellently"
    elif cka_value > 0.7:
        interpretation = "High similarity - SAE preserves representation structure well"
    elif cka_value > 0.5:
        interpretation = "Moderate similarity - SAE somewhat preserves representation structure"
    elif cka_value > 0.3:
        interpretation = "Low similarity - SAE poorly preserves representation structure"
    else:
        interpretation = "Very low similarity - SAE does not preserve representation structure"
    
    print(f"CKA = {cka_value:.4f}: {interpretation}")
    
    return results


if __name__ == "__main__":
    results = quick_cka_sae_test()
    print("\nTest completed successfully!")