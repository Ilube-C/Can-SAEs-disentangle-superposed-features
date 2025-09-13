"""Fast multi-seed RSA experiment with reduced computational load."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models_numpy import train_model, forward, AdamOptimizer, train_step
from src.models_numpy import train_sae, sae_forward, init_sae_params, sae_train_step
from src.data_generation import generate_synthetic_data, get_feature_importances
from src.CKA import linear_cka
from src.rsa_cca import compute_cca
from src.rsa_procrustes import procrustes_similarity

# Reduced configuration for faster execution
RANDOM_SEEDS = [42, 123, 456, 789, 1011]  # 5 different seeds
SPARSITY_LEVELS = [0.7, 0.8, 0.85, 0.9, 0.95]  # 5 sparsity levels
N_SAMPLES = 1000  # Reduced
N_FEATURES = 20
BOTTLENECK_DIM = 5
N_EPOCHS = 3  # Reduced
LEARNING_RATE = 0.01
L1_LAMBDA = 0.001
N_TEST_SAMPLES = 200  # Reduced
BATCH_SIZE = 50  # Larger batches for faster training


def train_models_fast(X_train, X_test, feature_importance, seed):
    """Fast training of both models."""
    
    # Train standard autoencoder using numpy implementation
    np.random.seed(seed)
    
    # Initialize parameters for standard AE
    W = np.random.randn(BOTTLENECK_DIM, N_FEATURES) * 0.1
    b = np.zeros(N_FEATURES)
    params = {'W': W, 'b': b}
    
    # Create optimizer
    optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)
    
    # Train AE
    for epoch in range(N_EPOCHS):
        for i in range(0, len(X_train), BATCH_SIZE):
            batch = X_train[i:i+BATCH_SIZE]
            for x in batch:
                params, loss = train_step(params, optimizer, x, feature_importance)
    
    # Store AE params
    ae_params = params
    
    # Train SAE with proper initialization
    # Note: SAE typically takes bottleneck activations as input, but here we'll train directly on data
    # Initialize SAE params
    sae_params = init_sae_params(seed, N_FEATURES, BOTTLENECK_DIM)
    sae_optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)
    
    # Train SAE
    for epoch in range(N_EPOCHS):
        for i in range(0, len(X_train), BATCH_SIZE):
            batch = X_train[i:i+BATCH_SIZE]
            for x in batch:
                sae_params, sae_loss = sae_train_step(sae_params, sae_optimizer, x, l1_penalty=L1_LAMBDA)
    
    return ae_params, sae_params


def compute_metrics_fast(ae_params, sae_params, X_test):
    """Compute only essential metrics."""
    # Extract representations
    z_ae = (ae_params['W'] @ X_test.T).T  # AE latent representation
    
    # SAE latent representation (encoder part)
    z_sae = []
    for x in X_test:
        # Forward through SAE encoder to get latent
        W_enc = sae_params['W_enc']  # Shape: (hidden_dim, input_dim)
        b_enc = sae_params['b_enc']  # Shape: (hidden_dim,)
        h = np.maximum(0, W_enc @ x + b_enc)  # ReLU activation (no transpose needed)
        z_sae.append(h)
    z_sae = np.array(z_sae)
    
    # Compute key metrics
    metrics = {
        'cka': linear_cka(z_ae, z_sae),
        'cca': np.mean(compute_cca(z_ae, z_sae)),
        'procrustes': procrustes_similarity(z_ae, z_sae)
    }
    
    # Compute reconstruction errors
    # AE reconstruction
    x_recon_ae = []
    for x in X_test:
        x_recon = forward(ae_params, x)
        x_recon_ae.append(x_recon)
    x_recon_ae = np.array(x_recon_ae)
    
    # SAE reconstruction
    x_recon_sae = []
    for x in X_test:
        result = sae_forward(sae_params, x)
        x_recon = result['recon']  # Extract reconstruction from dictionary
        x_recon_sae.append(x_recon)
    x_recon_sae = np.array(x_recon_sae)
    
    errors = {
        'ae_mse': np.mean((X_test - x_recon_ae)**2),
        'sae_mse': np.mean((X_test - x_recon_sae)**2)
    }
    
    return metrics, errors


def run_fast_experiment():
    """Run the experiment with progress tracking."""
    
    print("="*60)
    print("FAST MULTI-SEED RSA EXPERIMENT")
    print(f"Seeds: {len(RANDOM_SEEDS)}, Sparsities: {len(SPARSITY_LEVELS)}")
    print(f"Total trials: {len(RANDOM_SEEDS) * len(SPARSITY_LEVELS)}")
    print("="*60)
    
    # Initialize result storage
    results = {
        'cka': np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS))),
        'cca': np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS))),
        'procrustes': np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS))),
        'ae_mse': np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS))),
        'sae_mse': np.zeros((len(SPARSITY_LEVELS), len(RANDOM_SEEDS)))
    }
    
    total_trials = len(SPARSITY_LEVELS) * len(RANDOM_SEEDS)
    trial_num = 0
    
    start_time = time.time()
    
    for s_idx, sparsity in enumerate(SPARSITY_LEVELS):
        for seed_idx, seed in enumerate(RANDOM_SEEDS):
            trial_num += 1
            print(f"\rProgress: {trial_num}/{total_trials} "
                  f"(Sparsity={sparsity:.2f}, Seed={seed})...", end="")
            
            # Generate data
            X_train = generate_synthetic_data(seed, N_FEATURES, sparsity, N_SAMPLES)
            feature_importance = get_feature_importances(N_FEATURES, 0.7)
            
            X_test = generate_synthetic_data(seed + 10000, N_FEATURES, sparsity, N_TEST_SAMPLES)
            
            # Train models
            ae_params, sae_params = train_models_fast(X_train, X_test, feature_importance, seed)
            
            # Compute metrics
            metrics, errors = compute_metrics_fast(ae_params, sae_params, X_test)
            
            # Store results
            results['cka'][s_idx, seed_idx] = metrics['cka']
            results['cca'][s_idx, seed_idx] = metrics['cca']
            results['procrustes'][s_idx, seed_idx] = metrics['procrustes']
            results['ae_mse'][s_idx, seed_idx] = errors['ae_mse']
            results['sae_mse'][s_idx, seed_idx] = errors['sae_mse']
    
    elapsed = time.time() - start_time
    print(f"\n\nCompleted in {elapsed:.1f} seconds")
    
    return results


def create_visualization(results):
    """Create the 4x5 grid visualization."""
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    # Configure plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Row 1: CKA heatmap and analysis
    ax = axes[0, 0]
    im = ax.imshow(results['cka'], aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_title('CKA Similarity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sparsity')
    ax.set_yticks(range(len(SPARSITY_LEVELS)))
    ax.set_yticklabels([f'{s:.2f}' for s in SPARSITY_LEVELS])
    ax.set_xticks(range(len(RANDOM_SEEDS)))
    ax.set_xticklabels([f'S{i+1}' for i in range(len(RANDOM_SEEDS))])
    ax.set_xlabel('Seed')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Add text annotations
    for i in range(len(SPARSITY_LEVELS)):
        for j in range(len(RANDOM_SEEDS)):
            text = ax.text(j, i, f'{results["cka"][i, j]:.2f}',
                         ha='center', va='center',
                         color='white' if results["cka"][i, j] < 0.5 else 'black',
                         fontsize=8)
    
    # CCA heatmap
    ax = axes[0, 1]
    im = ax.imshow(results['cca'], aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Mean CCA', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(SPARSITY_LEVELS)))
    ax.set_yticklabels([])
    ax.set_xticks(range(len(RANDOM_SEEDS)))
    ax.set_xticklabels([f'S{i+1}' for i in range(len(RANDOM_SEEDS))])
    ax.set_xlabel('Seed')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    for i in range(len(SPARSITY_LEVELS)):
        for j in range(len(RANDOM_SEEDS)):
            text = ax.text(j, i, f'{results["cca"][i, j]:.2f}',
                         ha='center', va='center',
                         color='white' if results["cca"][i, j] < 0.5 else 'black',
                         fontsize=8)
    
    # Procrustes heatmap
    ax = axes[0, 2]
    im = ax.imshow(results['procrustes'], aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Procrustes Similarity', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(SPARSITY_LEVELS)))
    ax.set_yticklabels([])
    ax.set_xticks(range(len(RANDOM_SEEDS)))
    ax.set_xticklabels([f'S{i+1}' for i in range(len(RANDOM_SEEDS))])
    ax.set_xlabel('Seed')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    for i in range(len(SPARSITY_LEVELS)):
        for j in range(len(RANDOM_SEEDS)):
            text = ax.text(j, i, f'{results["procrustes"][i, j]:.2f}',
                         ha='center', va='center',
                         color='white' if results["procrustes"][i, j] < 0.5 else 'black',
                         fontsize=8)
    
    # AE MSE heatmap
    ax = axes[0, 3]
    im = ax.imshow(results['ae_mse'], aspect='auto', cmap='Reds')
    ax.set_title('AE Reconstruction Error', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(SPARSITY_LEVELS)))
    ax.set_yticklabels([])
    ax.set_xticks(range(len(RANDOM_SEEDS)))
    ax.set_xticklabels([f'S{i+1}' for i in range(len(RANDOM_SEEDS))])
    ax.set_xlabel('Seed')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # SAE MSE heatmap
    ax = axes[0, 4]
    im = ax.imshow(results['sae_mse'], aspect='auto', cmap='Reds')
    ax.set_title('SAE Reconstruction Error', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(SPARSITY_LEVELS)))
    ax.set_yticklabels([])
    ax.set_xticks(range(len(RANDOM_SEEDS)))
    ax.set_xticklabels([f'S{i+1}' for i in range(len(RANDOM_SEEDS))])
    ax.set_xlabel('Seed')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 2: Mean trends across sparsity
    metrics = ['cka', 'cca', 'procrustes', 'ae_mse', 'sae_mse']
    titles = ['CKA Trend', 'CCA Trend', 'Procrustes Trend', 'AE Error Trend', 'SAE Error Trend']
    colors = ['blue', 'green', 'orange', 'red', 'darkred']
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[1, idx]
        data = results[metric]
        means = np.mean(data, axis=1)
        stds = np.std(data, axis=1)
        
        ax.errorbar(SPARSITY_LEVELS, means, yerr=stds, marker='o', 
                   capsize=5, capthick=2, color=color, linewidth=2)
        ax.fill_between(SPARSITY_LEVELS, means - stds, means + stds, alpha=0.2, color=color)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Sparsity')
        ax.set_ylabel('Value' if idx == 0 else '')
        ax.grid(True, alpha=0.3)
        
        if metric in ['cka', 'cca', 'procrustes']:
            ax.set_ylim([0, 1])
    
    # Row 3: Variability analysis
    # Variance across seeds for each metric
    ax = axes[2, 0]
    variances_by_sparsity = {}
    for metric in ['cka', 'cca', 'procrustes']:
        vars = np.var(results[metric], axis=1)
        ax.plot(SPARSITY_LEVELS, vars, marker='o', label=metric.upper(), linewidth=2)
    ax.set_title('Variance Across Seeds', fontsize=11, fontweight='bold')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Coefficient of variation
    ax = axes[2, 1]
    for metric in ['cka', 'cca', 'procrustes']:
        means = np.mean(results[metric], axis=1)
        stds = np.std(results[metric], axis=1)
        cv = stds / (means + 1e-10)
        ax.plot(SPARSITY_LEVELS, cv, marker='s', label=metric.upper(), linewidth=2)
    ax.set_title('Coefficient of Variation', fontsize=11, fontweight='bold')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('CV (std/mean)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Difference plot (SAE - AE errors)
    ax = axes[2, 2]
    error_diff = results['sae_mse'] - results['ae_mse']
    mean_diff = np.mean(error_diff, axis=1)
    std_diff = np.std(error_diff, axis=1)
    ax.errorbar(SPARSITY_LEVELS, mean_diff, yerr=std_diff, 
               marker='D', capsize=5, color='purple', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('MSE Difference (SAE - AE)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('∆MSE')
    ax.grid(True, alpha=0.3)
    
    # All metrics comparison
    ax = axes[2, 3]
    for metric in ['cka', 'cca', 'procrustes']:
        means = np.mean(results[metric], axis=1)
        ax.plot(SPARSITY_LEVELS, means, marker='o', label=metric.upper(), linewidth=2.5)
    ax.set_title('All Similarity Metrics', fontsize=11, fontweight='bold')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Mean Similarity')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Highlight critical region
    ax.axvspan(0.82, 0.87, alpha=0.15, color='red')
    ax.text(0.845, 0.1, 'Critical\nRegion', ha='center', fontsize=9, color='red')
    
    # Box plots
    ax = axes[2, 4]
    data_for_box = [results[m].flatten() for m in ['cka', 'cca', 'procrustes']]
    bp = ax.boxplot(data_for_box, labels=['CKA', 'CCA', 'Procrustes'], patch_artist=True)
    colors_box = ['lightblue', 'lightgreen', 'lightsalmon']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title('Distribution Summary', fontsize=11, fontweight='bold')
    ax.set_ylabel('Similarity Score')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Row 4: Statistical summaries
    # Correlation matrix
    ax = axes[3, 0]
    metric_data = np.array([results[m].flatten() for m in ['cka', 'cca', 'procrustes']])
    corr_matrix = np.corrcoef(metric_data)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('Metric Correlations', fontsize=11, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(['CKA', 'CCA', 'Proc'])
    ax.set_yticklabels(['CKA', 'CCA', 'Proc'])
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                         ha='center', va='center',
                         color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    # Sensitivity to sparsity
    ax = axes[3, 1]
    sensitivities = []
    for metric in ['cka', 'cca', 'procrustes']:
        means = np.mean(results[metric], axis=1)
        coef = np.polyfit(SPARSITY_LEVELS, means, 1)[0]
        sensitivities.append(coef)
    
    bars = ax.bar(['CKA', 'CCA', 'Procrustes'], sensitivities,
                  color=['green' if s > 0 else 'red' for s in sensitivities], alpha=0.7)
    ax.set_title('Sparsity Sensitivity', fontsize=11, fontweight='bold')
    ax.set_ylabel('Linear Slope')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, sens in zip(bars, sensitivities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{sens:.3f}', ha='center',
               va='bottom' if sens > 0 else 'top')
    
    # Mean similarity by sparsity level
    ax = axes[3, 2]
    sparsity_means = []
    for s_idx in range(len(SPARSITY_LEVELS)):
        row_means = []
        for metric in ['cka', 'cca', 'procrustes']:
            row_means.append(np.mean(results[metric][s_idx, :]))
        sparsity_means.append(np.mean(row_means))
    
    bars = ax.bar(range(len(SPARSITY_LEVELS)), sparsity_means, 
                  color=plt.cm.viridis(np.linspace(0.3, 0.9, len(SPARSITY_LEVELS))))
    ax.set_title('Mean Similarity by Sparsity', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(SPARSITY_LEVELS)))
    ax.set_xticklabels([f'{s:.2f}' for s in SPARSITY_LEVELS])
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Mean Similarity')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, sparsity_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom')
    
    # Seed consistency
    ax = axes[3, 3]
    seed_stds = []
    for seed_idx in range(len(RANDOM_SEEDS)):
        col_values = []
        for metric in ['cka', 'cca', 'procrustes']:
            col_values.extend(results[metric][:, seed_idx])
        seed_stds.append(np.std(col_values))
    
    bars = ax.bar([f'S{i+1}' for i in range(len(RANDOM_SEEDS))], seed_stds,
                  color='steelblue', alpha=0.7)
    ax.set_title('Seed Consistency (Std Dev)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Std Dev')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, seed_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom')
    
    # Summary table
    ax = axes[3, 4]
    ax.axis('off')
    
    # Calculate summary statistics
    summary_text = "Summary Statistics\n" + "="*25 + "\n\n"
    
    for metric in ['cka', 'cca', 'procrustes']:
        data = results[metric].flatten()
        summary_text += f"{metric.upper():10s}:\n"
        summary_text += f"  Mean: {np.mean(data):.3f} ± {np.std(data):.3f}\n"
        summary_text += f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]\n\n"
    
    # Add error statistics
    ae_error = results['ae_mse'].flatten()
    sae_error = results['sae_mse'].flatten()
    summary_text += "Reconstruction Errors:\n"
    summary_text += f"  AE:  {np.mean(ae_error):.4f} ± {np.std(ae_error):.4f}\n"
    summary_text += f"  SAE: {np.mean(sae_error):.4f} ± {np.std(sae_error):.4f}\n\n"
    
    # Most stable metric
    variances = {m: np.var(results[m].flatten()) for m in ['cka', 'cca', 'procrustes']}
    most_stable = min(variances, key=variances.get)
    summary_text += f"Most stable: {most_stable.upper()}\n"
    summary_text += f"(var = {variances[most_stable]:.4f})"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Main title
    fig.suptitle('Multi-Seed RSA Analysis: AE vs SAE (5 Seeds × 5 Sparsity Levels)', 
                fontsize=16, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    
    # Save figure
    results_dir = Path("results/rsa_multiseed")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / 'multiseed_4x5_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved to {results_dir / 'multiseed_4x5_grid.png'}")


def main():
    """Run the fast experiment and create visualization."""
    
    # Run experiment
    results = run_fast_experiment()
    
    # Print quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    
    for metric in ['cka', 'cca', 'procrustes']:
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        print(f"{metric.upper():10s}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Create visualization
    print("\n" + "="*60)
    print("GENERATING 4×5 GRID VISUALIZATION")
    print("="*60)
    create_visualization(results)
    
    return results


if __name__ == "__main__":
    results = main()