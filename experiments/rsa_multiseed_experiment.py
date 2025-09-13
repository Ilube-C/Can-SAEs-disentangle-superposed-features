"""Multi-seed RSA experiment comparing AE and SAE across different sparsity levels.

This experiment:
1. Runs 5 trials with different random seeds
2. Tests multiple sparsity levels
3. Compares representations using CKA, CCA, and Procrustes metrics
4. Displays results in a 4x5 grid
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models_numpy import LinearAutoencoder
from src.sparse_autoencoder import SparseAutoencoder
from src.data_generation import generate_sparse_data
from src.CKA import compute_linear_cka, compute_rbf_cka
from src.rsa_cca import compute_cca, svcca_similarity
from src.rsa_procrustes import procrustes_similarity

# Configuration
RANDOM_SEEDS = [42, 123, 456, 789, 1011]  # 5 different seeds
SPARSITY_LEVELS = [0.7, 0.8, 0.85, 0.9, 0.95]  # 5 sparsity levels
N_SAMPLES = 5000  # Reduced for faster execution
N_FEATURES = 20
BOTTLENECK_DIM = 5
N_EPOCHS = 8  # Reduced for faster execution
LEARNING_RATE = 0.01
L1_LAMBDA = 0.001
N_TEST_SAMPLES = 500


def train_single_trial(X_train, X_test, feature_importance, seed):
    """Train both models for a single trial."""
    
    # Train AE
    ae = LinearAutoencoder(
        n_features=N_FEATURES,
        n_hidden=BOTTLENECK_DIM,
        learning_rate=LEARNING_RATE,
        random_seed=seed
    )
    
    for epoch in range(N_EPOCHS):
        for i in range(0, len(X_train), 32):
            batch = X_train[i:i+32]
            ae.train_step(batch, feature_importance)
    
    # Train SAE
    sae = SparseAutoencoder(
        input_dim=N_FEATURES,
        hidden_dim=BOTTLENECK_DIM,
        l1_lambda=L1_LAMBDA,
        learning_rate=LEARNING_RATE
    )
    
    # Set random seed for SAE initialization
    np.random.seed(seed)
    sae.W_encoder = np.random.randn(N_FEATURES, BOTTLENECK_DIM) * 0.1
    sae.W_decoder = np.random.randn(BOTTLENECK_DIM, N_FEATURES) * 0.1
    sae.b_encoder = np.zeros(BOTTLENECK_DIM)
    sae.b_decoder = np.zeros(N_FEATURES)
    
    for epoch in range(N_EPOCHS):
        for i in range(0, len(X_train), 32):
            batch = X_train[i:i+32]
            sae.train_step(batch)
    
    return ae, sae


def extract_representations(ae, sae, X):
    """Extract latent representations from both models."""
    z_ae = (ae.W @ X.T).T  # AE latent representation
    z_sae = sae.encode(X)   # SAE latent representation
    return z_ae, z_sae


def compute_metrics(z_ae, z_sae):
    """Compute all RSA metrics between representations."""
    
    # Ensure same dimensionality for Procrustes
    if z_ae.shape[1] != z_sae.shape[1]:
        max_dim = max(z_ae.shape[1], z_sae.shape[1])
        if z_ae.shape[1] < max_dim:
            z_ae_padded = np.pad(z_ae, ((0, 0), (0, max_dim - z_ae.shape[1])), mode='constant')
            z_sae_padded = z_sae
        else:
            z_ae_padded = z_ae
            z_sae_padded = np.pad(z_sae, ((0, 0), (0, max_dim - z_sae.shape[1])), mode='constant')
    else:
        z_ae_padded = z_ae
        z_sae_padded = z_sae
    
    metrics = {
        'cka_linear': compute_linear_cka(z_ae, z_sae),
        'cka_rbf': compute_rbf_cka(z_ae, z_sae),
        'mean_cca': np.mean(compute_cca(z_ae, z_sae)),
        'svcca': svcca_similarity(z_ae, z_sae, threshold=0.99),
        'procrustes': procrustes_similarity(z_ae_padded, z_sae_padded)
    }
    
    return metrics


def run_experiment():
    """Run the complete multi-seed, multi-sparsity experiment."""
    
    print("="*60)
    print("MULTI-SEED RSA EXPERIMENT")
    print(f"Seeds: {RANDOM_SEEDS}")
    print(f"Sparsity levels: {SPARSITY_LEVELS}")
    print("="*60)
    
    # Store results
    results = {
        'cka_linear': np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS))),
        'cka_rbf': np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS))),
        'mean_cca': np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS))),
        'svcca': np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS))),
        'procrustes': np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS)))
    }
    
    # Store reconstruction errors
    ae_recon_errors = np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS)))
    sae_recon_errors = np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS)))
    
    # Run experiments
    for s_idx, sparsity in enumerate(SPARSITY_LEVELS):
        print(f"\nSparsity {sparsity:.2f}:")
        
        for seed_idx, seed in enumerate(RANDOM_SEEDS):
            print(f"  Seed {seed}...", end=" ")
            
            # Generate data with this sparsity and seed
            X_train, feature_importance = generate_sparse_data(
                n_samples=N_SAMPLES,
                n_features=N_FEATURES,
                sparsity=sparsity,
                random_seed=seed
            )
            
            X_test, _ = generate_sparse_data(
                n_samples=N_TEST_SAMPLES,
                n_features=N_FEATURES,
                sparsity=sparsity,
                random_seed=seed + 10000
            )
            
            # Train models
            ae, sae = train_single_trial(X_train, X_test, feature_importance, seed)
            
            # Extract representations
            z_ae, z_sae = extract_representations(ae, sae, X_test)
            
            # Compute metrics
            metrics = compute_metrics(z_ae, z_sae)
            
            # Store results
            for metric_name, value in metrics.items():
                results[metric_name][s_idx, seed_idx] = value
            
            # Compute reconstruction errors
            x_recon_ae = ae.forward(X_test)
            x_recon_sae = sae.forward(X_test)
            ae_recon_errors[s_idx, seed_idx] = np.mean((X_test - x_recon_ae)**2)
            sae_recon_errors[s_idx, seed_idx] = np.mean((X_test - x_recon_sae)**2)
            
            print(f"CKA={metrics['cka_linear']:.3f}")
    
    return results, ae_recon_errors, sae_recon_errors


def plot_results(results, ae_errors, sae_errors):
    """Create a 4x5 grid visualization of results."""
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    # Row 1: Individual metric heatmaps
    metric_names = ['CKA (Linear)', 'CKA (RBF)', 'Mean CCA', 'SVCCA', 'Procrustes']
    metric_keys = ['cka_linear', 'cka_rbf', 'mean_cca', 'svcca', 'procrustes']
    
    for idx, (name, key) in enumerate(zip(metric_names, metric_keys)):
        ax = axes[0, idx]
        data = results[key]
        
        # Create heatmap
        im = ax.imshow(data, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Sparsity' if idx == 0 else '')
        ax.set_yticks(range(len(SPARSITY_LEVELS)))
        ax.set_yticklabels([f'{s:.2f}' for s in SPARSITY_LEVELS] if idx == 0 else [])
        ax.set_xticks(range(len(RANDOM_SEEDS)))
        ax.set_xticklabels([f'S{i+1}' for i in range(len(RANDOM_SEEDS))])
        ax.set_xlabel('Seed')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add text annotations
        for i in range(len(SPARSITY_LEVELS)):
            for j in range(len(RANDOM_SEEDS)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha='center', va='center', color='white' if data[i, j] < 0.5 else 'black',
                             fontsize=8)
    
    # Row 2: Mean and std across seeds
    for idx, (name, key) in enumerate(zip(metric_names, metric_keys)):
        ax = axes[1, idx]
        data = results[key]
        means = np.mean(data, axis=1)
        stds = np.std(data, axis=1)
        
        ax.errorbar(SPARSITY_LEVELS, means, yerr=stds, marker='o', capsize=5, capthick=2)
        ax.set_title(f'{name} (Mean ± Std)', fontsize=10)
        ax.set_xlabel('Sparsity')
        ax.set_ylabel('Similarity' if idx == 0 else '')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add shaded region for std
        ax.fill_between(SPARSITY_LEVELS, means - stds, means + stds, alpha=0.2)
    
    # Row 3: Reconstruction errors and comparisons
    # Plot 1: AE reconstruction errors
    ax = axes[2, 0]
    im = ax.imshow(ae_errors, aspect='auto', cmap='Reds', vmin=0)
    ax.set_title('AE Reconstruction Error', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sparsity')
    ax.set_yticks(range(len(SPARSITY_LEVELS)))
    ax.set_yticklabels([f'{s:.2f}' for s in SPARSITY_LEVELS])
    ax.set_xticks(range(len(RANDOM_SEEDS)))
    ax.set_xticklabels([f'S{i+1}' for i in range(len(RANDOM_SEEDS))])
    ax.set_xlabel('Seed')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot 2: SAE reconstruction errors
    ax = axes[2, 1]
    im = ax.imshow(sae_errors, aspect='auto', cmap='Reds', vmin=0)
    ax.set_title('SAE Reconstruction Error', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(SPARSITY_LEVELS)))
    ax.set_yticklabels([])
    ax.set_xticks(range(len(RANDOM_SEEDS)))
    ax.set_xticklabels([f'S{i+1}' for i in range(len(RANDOM_SEEDS))])
    ax.set_xlabel('Seed')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot 3: Error difference (SAE - AE)
    ax = axes[2, 2]
    error_diff = sae_errors - ae_errors
    im = ax.imshow(error_diff, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.max(np.abs(error_diff)), vmax=np.max(np.abs(error_diff)))
    ax.set_title('Error Difference (SAE - AE)', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(SPARSITY_LEVELS)))
    ax.set_yticklabels([])
    ax.set_xticks(range(len(RANDOM_SEEDS)))
    ax.set_xticklabels([f'S{i+1}' for i in range(len(RANDOM_SEEDS))])
    ax.set_xlabel('Seed')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot 4: Mean reconstruction errors
    ax = axes[2, 3]
    ae_mean = np.mean(ae_errors, axis=1)
    ae_std = np.std(ae_errors, axis=1)
    sae_mean = np.mean(sae_errors, axis=1)
    sae_std = np.std(sae_errors, axis=1)
    
    ax.errorbar(SPARSITY_LEVELS, ae_mean, yerr=ae_std, marker='o', label='AE', capsize=5)
    ax.errorbar(SPARSITY_LEVELS, sae_mean, yerr=sae_std, marker='s', label='SAE', capsize=5)
    ax.set_title('Mean Reconstruction Error', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Correlation matrix between metrics
    ax = axes[2, 4]
    # Flatten all metrics and compute correlation
    metric_data = []
    for key in metric_keys:
        metric_data.append(results[key].flatten())
    metric_data = np.array(metric_data)
    corr_matrix = np.corrcoef(metric_data)
    
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('Metric Correlations', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(metric_keys)))
    ax.set_yticks(range(len(metric_keys)))
    ax.set_xticklabels(['CKA-L', 'CKA-R', 'CCA', 'SVCCA', 'Proc'], rotation=45)
    ax.set_yticklabels(['CKA-L', 'CKA-R', 'CCA', 'SVCCA', 'Proc'])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add correlation values
    for i in range(len(metric_keys)):
        for j in range(len(metric_keys)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                         ha='center', va='center', 
                         color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                         fontsize=8)
    
    # Row 4: Summary statistics and analysis
    # Plot 1: Variance across seeds
    ax = axes[3, 0]
    variances = [np.var(results[key].flatten()) for key in metric_keys]
    bars = ax.bar(range(len(metric_keys)), variances, color='steelblue', alpha=0.7)
    ax.set_title('Metric Variance Across All Trials', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(['CKA-L', 'CKA-R', 'CCA', 'SVCCA', 'Proc'], rotation=45)
    ax.set_ylabel('Variance')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, var in zip(bars, variances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{var:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Sparsity sensitivity (slope of linear fit)
    ax = axes[3, 1]
    sensitivities = []
    for key in metric_keys:
        means = np.mean(results[key], axis=1)
        # Fit linear regression
        coef = np.polyfit(SPARSITY_LEVELS, means, 1)[0]
        sensitivities.append(coef)
    
    bars = ax.bar(range(len(metric_keys)), sensitivities, 
                  color=['green' if s > 0 else 'red' for s in sensitivities], alpha=0.7)
    ax.set_title('Sparsity Sensitivity (Linear Slope)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(['CKA-L', 'CKA-R', 'CCA', 'SVCCA', 'Proc'], rotation=45)
    ax.set_ylabel('Slope')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, sens in zip(bars, sensitivities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{sens:.3f}', ha='center', 
               va='bottom' if sens > 0 else 'top', fontsize=9)
    
    # Plot 3: Box plot of all metrics
    ax = axes[3, 2]
    all_data = [results[key].flatten() for key in metric_keys]
    bp = ax.boxplot(all_data, labels=['CKA-L', 'CKA-R', 'CCA', 'SVCCA', 'Proc'],
                    patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(metric_keys)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Distribution Across All Trials', fontsize=12, fontweight='bold')
    ax.set_ylabel('Similarity Score')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Critical sparsity analysis
    ax = axes[3, 3]
    for key, name in zip(metric_keys, metric_names):
        means = np.mean(results[key], axis=1)
        ax.plot(SPARSITY_LEVELS, means, marker='o', label=name[:5], linewidth=2)
    
    ax.set_title('All Metrics vs Sparsity', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Mean Similarity')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Highlight critical region
    ax.axvspan(0.82, 0.87, alpha=0.2, color='red', label='Critical Region')
    
    # Plot 5: Summary statistics table
    ax = axes[3, 4]
    ax.axis('off')
    
    # Calculate summary statistics
    summary_stats = []
    for key, name in zip(metric_keys, metric_names):
        data = results[key].flatten()
        stats = {
            'Metric': name[:8],
            'Mean': f'{np.mean(data):.3f}',
            'Std': f'{np.std(data):.3f}',
            'Min': f'{np.min(data):.3f}',
            'Max': f'{np.max(data):.3f}'
        }
        summary_stats.append(stats)
    
    # Create table
    cell_text = []
    for stats in summary_stats:
        cell_text.append([stats['Metric'], stats['Mean'], stats['Std'], 
                         stats['Min'], stats['Max']])
    
    table = ax.table(cellText=cell_text,
                    colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_stats) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Main title
    fig.suptitle('Multi-Seed RSA Analysis: AE vs SAE Across Sparsity Levels', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    results_dir = Path("results/rsa_multiseed")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / 'multiseed_analysis_4x5.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved to {results_dir / 'multiseed_analysis_4x5.png'}")


def main():
    """Run the multi-seed experiment and create visualizations."""
    
    # Run experiment
    results, ae_errors, sae_errors = run_experiment()
    
    # Create visualization
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)
    plot_results(results, ae_errors, sae_errors)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for metric in ['cka_linear', 'mean_cca', 'procrustes']:
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Find most stable metric (lowest variance)
    variances = {key: np.var(results[key]) for key in results.keys()}
    most_stable = min(variances, key=variances.get)
    print(f"\nMost stable metric: {most_stable} (variance={variances[most_stable]:.4f})")
    
    # Find metric most sensitive to sparsity
    sensitivities = {}
    for key in results.keys():
        means = np.mean(results[key], axis=1)
        coef = np.polyfit(SPARSITY_LEVELS, means, 1)[0]
        sensitivities[key] = abs(coef)
    
    most_sensitive = max(sensitivities, key=sensitivities.get)
    print(f"Most sparsity-sensitive: {most_sensitive} (|slope|={sensitivities[most_sensitive]:.3f})")
    
    return results, ae_errors, sae_errors


if __name__ == "__main__":
    results, ae_errors, sae_errors = main()