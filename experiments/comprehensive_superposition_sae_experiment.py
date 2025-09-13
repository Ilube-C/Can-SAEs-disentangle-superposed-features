#!/usr/bin/env python3
"""
Comprehensive Superposition-SAE Experiment
==========================================
Trains autoencoders at critical sparsity levels (0.8, 0.85, 0.9),
then trains SAEs on bottleneck activations and performs RSA/CCA analysis.

Single seed for clean comparison across sparsity levels.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import (
    train_model, get_bottleneck_activations, 
    train_sae, sae_forward, loss_fn
)
from analysis import (
    compute_superposition_matrix, compute_sparsity_metrics,
    analyze_sae_performance
)
from CKA import linear_cka, rbf_cka
from rsa_cca import compute_cca, cca_distance
from rsa_procrustes import procrustes_similarity

def train_autoencoder_single_seed(sparsity, config):
    """Train autoencoder for a single sparsity level."""
    print(f"\n{'='*60}")
    print(f"Training Autoencoder - Sparsity {sparsity}")
    print(f"{'='*60}")
    
    # Generate data
    data = generate_synthetic_data(
        config['seed'], 
        config['sparse_dim'], 
        sparsity, 
        config['num_samples']
    )
    
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
    bottleneck_acts = get_bottleneck_activations(params, data)
    
    # Analyze superposition
    superposition_analysis = compute_superposition_matrix(params)
    
    return {
        'params': params,
        'data': data,
        'bottleneck_activations': bottleneck_acts,
        'reconstruction_loss': reconstruction_loss,
        'superposition_analysis': superposition_analysis,
        'feature_importances': I
    }

def train_sae_on_bottleneck(bottleneck_acts, config):
    """Train SAE on bottleneck activations."""
    print(f"  Training SAE on bottleneck activations...")
    
    sae_params = train_sae(
        activations=bottleneck_acts,
        input_dim=config['sae_input_dim'],  # 5 (bottleneck dim)
        hidden_dim=config['sae_hidden_dim'],  # 20 (expansion)
        num_epochs=config['sae_epochs'],
        learning_rate=config['sae_lr'],
        l1_penalty=config['sae_l1'],
        seed=config['seed'],
        verbose=False
    )
    
    # Get SAE outputs
    sae_result = sae_forward(sae_params, bottleneck_acts)
    sae_hidden = sae_result['hidden']
    sae_recon = sae_result['recon']
    
    # Compute SAE metrics
    sae_mse = np.mean((bottleneck_acts - sae_recon)**2)
    sae_sparsity = compute_sparsity_metrics(sae_hidden)
    
    return {
        'params': sae_params,
        'hidden_activations': sae_hidden,
        'reconstructions': sae_recon,
        'mse': sae_mse,
        'sparsity_metrics': sae_sparsity
    }

def compute_all_similarity_metrics(repr1, repr2, name1="Repr1", name2="Repr2"):
    """Compute all similarity metrics between two representations."""
    print(f"  Computing similarity metrics between {name1} and {name2}...")
    
    metrics = {}
    
    # Linear CKA
    try:
        metrics['linear_cka'] = linear_cka(repr1, repr2)
    except Exception as e:
        print(f"    Warning: Linear CKA failed: {e}")
        metrics['linear_cka'] = np.nan
    
    # RBF CKA
    try:
        metrics['rbf_cka'] = rbf_cka(repr1, repr2)
    except Exception as e:
        print(f"    Warning: RBF CKA failed: {e}")
        metrics['rbf_cka'] = np.nan
    
    # CCA Similarity
    try:
        cca_corrs = compute_cca(repr1, repr2)
        metrics['cca_mean'] = np.mean(cca_corrs)
        metrics['cca_correlations'] = cca_corrs
    except Exception as e:
        print(f"    Warning: CCA failed: {e}")
        metrics['cca_mean'] = np.nan
        metrics['cca_correlations'] = None
    
    # Procrustes Similarity
    try:
        metrics['procrustes'] = procrustes_similarity(repr1, repr2)
    except Exception as e:
        print(f"    Warning: Procrustes failed: {e}")
        metrics['procrustes'] = np.nan
    
    # RSA (from analysis.py)
    try:
        from analysis import compute_rsa_correlation
        metrics['rsa'] = compute_rsa_correlation(repr1, repr2)
    except Exception as e:
        print(f"    Warning: RSA failed: {e}")
        metrics['rsa'] = np.nan
    
    return metrics

def plot_superposition_matrices(results, sparsities, save_dir):
    """Plot superposition matrices for all sparsity levels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (sparsity, ax) in enumerate(zip(sparsities, axes)):
        W = results[sparsity]['autoencoder']['params']['W']
        superposition_matrix = W.T @ W
        
        im = ax.imshow(superposition_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_title(f'Sparsity {sparsity}\nLoss: {results[sparsity]["autoencoder"]["reconstruction_loss"]:.4f}')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Superposition Matrices (W^T @ W)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/superposition_matrices_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_weight_norms_comparison(results, sparsities, save_dir):
    """Plot weight norms for all sparsity levels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (sparsity, ax) in enumerate(zip(sparsities, axes)):
        weight_norms = results[sparsity]['autoencoder']['superposition_analysis']['weight_norms']
        sorted_norms = np.sort(weight_norms)
        
        ax.bar(range(len(sorted_norms)), sorted_norms, color='steelblue', alpha=0.7)
        ax.set_title(f'Sparsity {sparsity}')
        ax.set_xlabel('Feature Index (sorted)')
        ax.set_ylabel('Weight Norm')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Weight Norms Comparison (sorted)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/weight_norms_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_reconstruction_quality(results, sparsities, save_dir):
    """Plot reconstruction quality metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, sparsity in enumerate(sparsities):
        # Autoencoder reconstruction
        ae_data = results[sparsity]['autoencoder']['data']
        ae_params = results[sparsity]['autoencoder']['params']
        I = results[sparsity]['autoencoder']['feature_importances']
        
        # Sample some data points for visualization
        sample_idx = np.random.choice(ae_data.shape[0], 100, replace=False)
        sample_data = ae_data[sample_idx]
        
        # Get reconstructions
        from models_numpy import forward
        recons = forward(ae_params, sample_data)
        
        # Top row: Autoencoder reconstruction scatter
        ax = axes[0, idx]
        ax.scatter(sample_data.flatten(), recons.flatten(), alpha=0.3, s=1)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax.set_xlabel('Original')
        ax.set_ylabel('Reconstructed')
        ax.set_title(f'Sparsity {sparsity}\nAE Loss: {results[sparsity]["autoencoder"]["reconstruction_loss"]:.4f}')
        ax.grid(alpha=0.3)
        
        # Bottom row: SAE reconstruction scatter
        ax = axes[1, idx]
        bottleneck = results[sparsity]['autoencoder']['bottleneck_activations']
        sae_recon = results[sparsity]['sae']['reconstructions']
        
        sample_idx = np.random.choice(bottleneck.shape[0], 100, replace=False)
        ax.scatter(bottleneck[sample_idx].flatten(), sae_recon[sample_idx].flatten(), 
                  alpha=0.3, s=1, color='green')
        ax.plot([bottleneck.min(), bottleneck.max()], 
               [bottleneck.min(), bottleneck.max()], 'r--', alpha=0.5)
        ax.set_xlabel('Original Bottleneck')
        ax.set_ylabel('SAE Reconstructed')
        ax.set_title(f'SAE MSE: {results[sparsity]["sae"]["mse"]:.4f}')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Reconstruction Quality Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/reconstruction_quality.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_metrics_table(results, sparsities, save_dir):
    """Create and display comprehensive metrics table."""
    
    # Prepare data for table
    table_data = []
    
    for sparsity in sparsities:
        row = {'Sparsity': sparsity}
        
        # Autoencoder metrics
        row['AE Recon Loss'] = f"{results[sparsity]['autoencoder']['reconstruction_loss']:.4f}"
        
        # SAE metrics
        row['SAE MSE'] = f"{results[sparsity]['sae']['mse']:.4f}"
        row['SAE L0'] = f"{results[sparsity]['sae']['sparsity_metrics']['l0_norm']:.3f}"
        row['SAE L1'] = f"{results[sparsity]['sae']['sparsity_metrics']['l1_norm']:.3f}"
        
        # Similarity metrics
        metrics = results[sparsity]['similarity_metrics']
        row['Linear CKA'] = f"{metrics['bottleneck_vs_sae_recon']['linear_cka']:.3f}"
        row['RBF CKA'] = f"{metrics['bottleneck_vs_sae_recon']['rbf_cka']:.3f}"
        row['CCA Mean'] = f"{metrics['bottleneck_vs_sae_recon']['cca_mean']:.3f}"
        row['Procrustes'] = f"{metrics['bottleneck_vs_sae_recon']['procrustes']:.3f}"
        row['RSA'] = f"{metrics['bottleneck_vs_sae_recon']['rsa']:.3f}"
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Display table
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    if save_dir:
        df.to_csv(f'{save_dir}/metrics_table.csv', index=False)
        
        # Create formatted HTML table for better visualization
        html_table = df.to_html(index=False, classes='metrics-table')
        with open(f'{save_dir}/metrics_table.html', 'w') as f:
            f.write(f"""
            <html>
            <head>
                <style>
                    .metrics-table {{
                        border-collapse: collapse;
                        width: 100%;
                        font-family: Arial, sans-serif;
                    }}
                    .metrics-table th {{
                        background-color: #4CAF50;
                        color: white;
                        padding: 12px;
                        text-align: left;
                    }}
                    .metrics-table td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                    }}
                    .metrics-table tr:nth-child(even) {{
                        background-color: #f2f2f2;
                    }}
                </style>
            </head>
            <body>
                <h2>Superposition-SAE Experiment Results</h2>
                {html_table}
            </body>
            </html>
            """)
    
    return df

def plot_similarity_metrics_heatmap(results, sparsities, save_dir):
    """Create heatmap of all similarity metrics."""
    
    # Prepare data
    metrics_names = ['Linear CKA', 'RBF CKA', 'CCA Mean', 'Procrustes', 'RSA']
    metrics_keys = ['linear_cka', 'rbf_cka', 'cca_mean', 'procrustes', 'rsa']
    
    data_matrix = []
    for sparsity in sparsities:
        row = []
        for key in metrics_keys:
            value = results[sparsity]['similarity_metrics']['bottleneck_vs_sae_recon'][key]
            row.append(value if not np.isnan(value) else 0)
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix).T  # Transpose for better layout
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data_matrix, 
                xticklabels=[f'{s}' for s in sparsities],
                yticklabels=metrics_names,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                vmin=0, vmax=1,
                cbar_kws={'label': 'Similarity Score'})
    
    plt.title('Similarity Metrics: Bottleneck vs SAE Reconstruction', fontsize=14, fontweight='bold')
    plt.xlabel('Sparsity Level')
    plt.ylabel('Metric')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/similarity_metrics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_comprehensive_experiment():
    """Main experiment runner."""
    
    # Configuration
    config = {
        # General
        'seed': 42,
        'sparsities': [0.8, 0.85, 0.9],
        
        # Autoencoder config
        'sparse_dim': 20,
        'dense_dim': 5,
        'num_samples': 10000,
        'num_epochs': 10,
        'learning_rate': 0.01,
        'decay_factor': 0.7,
        
        # SAE config
        'sae_input_dim': 5,   # bottleneck dimension
        'sae_hidden_dim': 20,  # expansion back to original
        'sae_epochs': 50,
        'sae_lr': 0.001,
        'sae_l1': 0.01,
    }
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/comprehensive_experiment_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE SUPERPOSITION-SAE EXPERIMENT")
    print(f"{'='*80}")
    print(f"Seed: {config['seed']}")
    print(f"Sparsity levels: {config['sparsities']}")
    print(f"Architecture: {config['sparse_dim']} -> {config['dense_dim']} -> {config['sparse_dim']} (Autoencoder)")
    print(f"SAE Architecture: {config['sae_input_dim']} -> {config['sae_hidden_dim']} -> {config['sae_input_dim']}")
    print(f"Results directory: {results_dir}")
    
    # Store all results
    all_results = {}
    
    # Phase 1: Train autoencoders
    print(f"\n{'='*80}")
    print("PHASE 1: TRAINING AUTOENCODERS")
    print(f"{'='*80}")
    
    for sparsity in config['sparsities']:
        all_results[sparsity] = {}
        all_results[sparsity]['autoencoder'] = train_autoencoder_single_seed(sparsity, config)
    
    # Phase 2: Train SAEs on bottleneck activations
    print(f"\n{'='*80}")
    print("PHASE 2: TRAINING SPARSE AUTOENCODERS")
    print(f"{'='*80}")
    
    for sparsity in config['sparsities']:
        print(f"\nTraining SAE for sparsity {sparsity}...")
        bottleneck_acts = all_results[sparsity]['autoencoder']['bottleneck_activations']
        all_results[sparsity]['sae'] = train_sae_on_bottleneck(bottleneck_acts, config)
    
    # Phase 3: Compute similarity metrics
    print(f"\n{'='*80}")
    print("PHASE 3: COMPUTING SIMILARITY METRICS")
    print(f"{'='*80}")
    
    for sparsity in config['sparsities']:
        print(f"\nComputing metrics for sparsity {sparsity}...")
        
        bottleneck = all_results[sparsity]['autoencoder']['bottleneck_activations']
        sae_hidden = all_results[sparsity]['sae']['hidden_activations']
        sae_recon = all_results[sparsity]['sae']['reconstructions']
        
        all_results[sparsity]['similarity_metrics'] = {
            'bottleneck_vs_sae_recon': compute_all_similarity_metrics(
                bottleneck, sae_recon, "Bottleneck", "SAE-Recon"
            ),
            'bottleneck_vs_sae_hidden': compute_all_similarity_metrics(
                bottleneck, sae_hidden, "Bottleneck", "SAE-Hidden"
            ),
        }
    
    # Phase 4: Visualization and reporting
    print(f"\n{'='*80}")
    print("PHASE 4: GENERATING VISUALIZATIONS AND REPORTS")
    print(f"{'='*80}")
    
    # Plot superposition matrices
    plot_superposition_matrices(all_results, config['sparsities'], results_dir)
    
    # Plot weight norms
    plot_weight_norms_comparison(all_results, config['sparsities'], results_dir)
    
    # Plot reconstruction quality
    plot_reconstruction_quality(all_results, config['sparsities'], results_dir)
    
    # Plot similarity metrics heatmap
    plot_similarity_metrics_heatmap(all_results, config['sparsities'], results_dir)
    
    # Create and display metrics table
    metrics_df = create_metrics_table(all_results, config['sparsities'], results_dir)
    
    # Save configuration
    with open(f'{results_dir}/config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f'{key}: {value}\n')
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"All results saved to: {results_dir}")
    
    return all_results, metrics_df

if __name__ == "__main__":
    results, metrics_table = run_comprehensive_experiment()