#!/usr/bin/env python3
"""
Quick SAE test with minimal parameters
"""

import os
import sys
import numpy as np

# Add local modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import train_model, get_bottleneck_activations, train_sae, sae_forward
from analysis import compute_rsa_correlation, compute_sparsity_metrics

def quick_sae_test():
    """Ultra-fast SAE test with minimal parameters"""
    print("Quick SAE Implementation Test")
    print("=" * 40)
    
    # Ultra-minimal parameters for speed
    sparse_dim = 8
    dense_dim = 3
    num_samples = 2000
    num_epochs = 5
    sae_epochs = 10
    
    print(f"Parameters: {num_samples} samples, {sparse_dim}->{dense_dim} autoencoder, {sae_epochs} SAE epochs")
    
    # Generate data
    print("1. Generating data...")
    data = generate_synthetic_data(42, sparse_dim, 0.7, num_samples)
    I = get_feature_importances(sparse_dim, 0.7)
    
    # Train autoencoder
    print("2. Training autoencoder...")
    autoencoder_params = train_model(data, I, k=dense_dim, n=sparse_dim, 
                                   num_epochs=num_epochs, seed=42)
    
    # Extract activations
    print("3. Extracting activations...")
    activations = get_bottleneck_activations(autoencoder_params, data)
    print(f"   Activations shape: {activations.shape}")
    
    # Train SAE
    print("4. Training SAE...")
    sae_params = train_sae(activations, input_dim=dense_dim, hidden_dim=6,
                          num_epochs=sae_epochs, verbose=False, seed=42)
    
    # Test SAE
    print("5. Testing SAE...")
    sae_result = sae_forward(sae_params, activations)
    sae_hidden = sae_result['hidden']
    sae_recon = sae_result['recon']
    
    # Compute metrics
    print("6. Computing validation metrics...")
    
    # Basic checks
    recon_mse = np.mean((activations - sae_recon)**2)
    print(f"   Reconstruction MSE: {recon_mse:.6f}")
    
    # Sparsity
    orig_sparsity = compute_sparsity_metrics(activations)
    sae_sparsity = compute_sparsity_metrics(sae_hidden)
    sparsity_improvement = orig_sparsity['l0_norm'] / sae_sparsity['l0_norm']
    print(f"   Original L0: {orig_sparsity['l0_norm']:.4f}")
    print(f"   SAE L0: {sae_sparsity['l0_norm']:.4f}")
    print(f"   Sparsity improvement: {sparsity_improvement:.2f}x")
    
    # RSA
    rsa_correlation = compute_rsa_correlation(activations, sae_recon)
    rsa_self = compute_rsa_correlation(activations, activations)
    
    # Random baseline
    np.random.seed(42)
    random_repr = np.random.normal(0, 1, activations.shape)
    rsa_random = compute_rsa_correlation(activations, random_repr)
    
    print(f"   RSA correlation: {rsa_correlation:.4f}")
    print(f"   RSA self-check: {rsa_self:.4f} (should be ~1.0)")
    print(f"   RSA random: {rsa_random:.4f} (should be ~0.0)")
    
    # Validation criteria
    print("\n7. Validation Results:")
    criteria = {
        'RSA reasonable (>0.3)': rsa_correlation > 0.3,
        'RSA self-check (~1.0)': abs(rsa_self - 1.0) < 0.1,
        'RSA random (~0.0)': abs(rsa_random) < 0.3,
        'Sparsity improved (>1.0x)': sparsity_improvement > 1.0,
        'Reconstruction reasonable (<1.0)': recon_mse < 1.0
    }
    
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"   {criterion}: {status}")
    
    overall_success = all(criteria.values())
    print(f"\nOverall Result: {'SUCCESS' if overall_success else 'NEEDS ATTENTION'}")
    
    return overall_success

if __name__ == "__main__":
    success = quick_sae_test()
    if success:
        print("\nThe SAE implementation is working correctly!")
    else:
        print("\nThe SAE implementation needs attention.")