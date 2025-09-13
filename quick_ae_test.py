#!/usr/bin/env python3
"""
Quick test of 10-5-10 autoencoder at sparsities 0.85, 0.9, 0.95
Focus on autoencoder performance and superposition matrices
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Add paths
sys.path.append('src')
sys.path.append('config')

from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import train_model, loss_fn
from analysis import compute_superposition_matrix

def quick_ae_test():
    """Quick test of autoencoder at critical sparsity levels."""
    
    # Configuration
    config = {
        'seed': 777,
        'sparsities': [0.85, 0.9, 0.95],
        'sparse_dim': 10,
        'dense_dim': 5,
        'num_samples': 2000,
        'num_epochs': 3,
        'learning_rate': 0.01,
        'decay_factor': 0.7,
    }
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/quick_ae_test_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nQUICK AUTOENCODER TEST - 10-5-10 ARCHITECTURE")
    print(f"{'='*60}")
    print(f"Testing sparsities: {config['sparsities']}")
    print(f"Architecture: {config['sparse_dim']} -> {config['dense_dim']} -> {config['sparse_dim']}")
    print(f"Results: {results_dir}")
    print(f"{'='*60}")
    
    results = {}
    
    # Train autoencoders
    for sparsity in config['sparsities']:
        print(f"\nTraining autoencoder for sparsity {sparsity}...")
        
        # Generate data
        data = generate_synthetic_data(
            config['seed'], 
            config['sparse_dim'], 
            sparsity, 
            config['num_samples']
        )
        
        # Check empirical sparsity
        empirical_sparsity = float(np.mean(data == 0))
        print(f"  Empirical sparsity: {empirical_sparsity:.3f}")
        
        # Get feature importances
        I = get_feature_importances(config['sparse_dim'], config['decay_factor'])
        
        # Train model
        params = train_model(
            data=data,
            I=I,
            k=config['dense_dim'],
            n=config['sparse_dim'],
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            seed=config['seed']
        )
        
        # Compute metrics
        reconstruction_loss = loss_fn(params, data, I) / data.shape[0]
        superposition_analysis = compute_superposition_matrix(params)
        
        # Compute additional diagnostics
        W = params['W']
        weight_norms = np.linalg.norm(W, axis=0)  # Norm of each column
        condition_number = np.linalg.cond(W)
        
        results[sparsity] = {
            'params': params,
            'data': data,
            'recon_loss': reconstruction_loss,
            'superposition': superposition_analysis,
            'weight_norms': weight_norms,
            'condition_number': condition_number,
            'empirical_sparsity': empirical_sparsity
        }
        
        print(f"  Reconstruction loss: {reconstruction_loss:.4f}")
        print(f"  Weight matrix condition number: {condition_number:.2f}")
        print(f"  Mean weight norm: {np.mean(weight_norms):.4f}")
    
    # Create visualizations
    print(f"\nCREATING VISUALIZATIONS...")
    
    # Plot 1: Superposition matrices comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, sparsity in enumerate(config['sparsities']):
        W = results[sparsity]['params']['W']
        superposition_matrix = W.T @ W
        
        # Normalize for better visualization
        vmax = np.max(np.abs(superposition_matrix))
        
        im = axes[idx].imshow(superposition_matrix, 
                             cmap='coolwarm', 
                             aspect='auto', 
                             vmin=-vmax, 
                             vmax=vmax)
        axes[idx].set_title(f'Sparsity {sparsity}\\nLoss: {results[sparsity]["recon_loss"]:.4f}\\nCond: {results[sparsity]["condition_number"]:.1f}')
        axes[idx].set_xlabel('Feature Index')
        axes[idx].set_ylabel('Feature Index')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # Add grid for better readability
        axes[idx].set_xticks(range(config['sparse_dim']))
        axes[idx].set_yticks(range(config['sparse_dim']))
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Superposition Matrices (W^T @ W) - 10-5-10 Architecture', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/superposition_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Weight norms comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, sparsity in enumerate(config['sparsities']):
        weight_norms = results[sparsity]['weight_norms']
        sorted_norms = np.sort(weight_norms)
        
        bars = axes[idx].bar(range(len(sorted_norms)), sorted_norms, 
                            color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'Sparsity {sparsity}\\nMean norm: {np.mean(weight_norms):.3f}')
        axes[idx].set_xlabel('Feature Index (sorted)')
        axes[idx].set_ylabel('Weight Norm')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Highlight zero/near-zero weights
        for i, (bar, norm) in enumerate(zip(bars, sorted_norms)):
            if norm < 0.01:
                bar.set_color('red')
                bar.set_alpha(0.8)
    
    plt.suptitle('Weight Norms by Feature - 10-5-10 Architecture', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()  
    plt.savefig(f'{results_dir}/weight_norms_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Data distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, sparsity in enumerate(config['sparsities']):
        data = results[sparsity]['data']
        flat_data = data.flatten()
        non_zero_data = flat_data[flat_data > 0]  # Only non-zero values
        
        axes[idx].hist(non_zero_data, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[idx].set_title(f'Sparsity {sparsity}\\nNon-zero values: {len(non_zero_data)}\\nEmpirical: {results[sparsity]["empirical_sparsity"]:.3f}')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Data Distribution (Non-zero values only) - 10-5-10', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/data_distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary table
    print(f"\\n{'='*80}")
    print("AUTOENCODER PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Sparsity':<10} {'AE Loss':<12} {'Condition':<12} {'Mean Norm':<12} {'Empirical':<12}")
    print("-" * 80)
    
    for sparsity in config['sparsities']:
        r = results[sparsity]
        print(f"{sparsity:<10} {r['recon_loss']:<12.4f} {r['condition_number']:<12.1f} {np.mean(r['weight_norms']):<12.4f} {r['empirical_sparsity']:<12.3f}")
    
    print(f"{'='*80}")
    
    # Save summary
    with open(f'{results_dir}/summary.txt', 'w') as f:
        f.write("QUICK AUTOENCODER TEST SUMMARY - 10-5-10 ARCHITECTURE\\n")
        f.write("="*60 + "\\n\\n")
        
        for sparsity in config['sparsities']:
            r = results[sparsity]
            f.write(f"SPARSITY {sparsity}:\\n")
            f.write(f"  Reconstruction Loss: {r['recon_loss']:.4f}\\n")
            f.write(f"  Weight Matrix Condition Number: {r['condition_number']:.2f}\\n")
            f.write(f"  Mean Weight Norm: {np.mean(r['weight_norms']):.4f}\\n")
            f.write(f"  Empirical Sparsity: {r['empirical_sparsity']:.3f}\\n")
            f.write(f"  Near-zero weights: {np.sum(r['weight_norms'] < 0.01)}/{len(r['weight_norms'])}\\n")
            f.write("\\n")
    
    print(f"\\nResults saved to: {results_dir}")
    print("Check the superposition_comparison.png for the key visualization!")
    
    return results

if __name__ == "__main__":
    results = quick_ae_test()