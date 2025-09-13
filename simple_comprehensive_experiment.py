#!/usr/bin/env python3
"""
Simplified comprehensive experiment - single seed, reduced epochs for testing
Trains autoencoders at sparsity 0.8, 0.85, 0.9, then SAEs, then computes metrics
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Add paths
sys.path.append('src')
sys.path.append('config')

# Core imports
from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import (
    train_model, get_bottleneck_activations, 
    train_sae, sae_forward, loss_fn, forward
)
from analysis import compute_superposition_matrix, compute_sparsity_metrics
from CKA import linear_cka, rbf_cka
from rsa_procrustes import procrustes_similarity

def main(sparse_dim=20, dense_dim=5, sae_hidden_dim=20, seed=123, results_suffix="", run_sae=True, sparsities=None):
    # Configuration - now accepts parameters
    if sparsities is None:
        sparsities = [0.3, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    config = {
        'seed': seed,
        'sparsities': sparsities,
        'sparse_dim': sparse_dim,
        'dense_dim': dense_dim,
        'sae_hidden_dim': sae_hidden_dim,  # New parameter
        'num_samples': 10000,  # More data for stability at high sparsity
        'num_epochs': 10,
        'learning_rate': 0.01,
        'decay_factor': 0.7,
        'sae_epochs': 10,
        'sae_lr': 0.001,
        'sae_l1': 0.01,
        'run_sae': run_sae,  # Flag to control SAE execution
    }
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    arch_name = f"{sparse_dim}-{dense_dim}-{sparse_dim}"
    results_dir = f'results/simple_exp_{arch_name}_{timestamp}{results_suffix}'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nSIMPLE COMPREHENSIVE EXPERIMENT")
    print(f"{'='*60}")
    print(f"Sparsity levels: {config['sparsities']}")
    print(f"Architecture: {sparse_dim} -> {dense_dim} -> {sparse_dim} (AE), {dense_dim} -> {sae_hidden_dim} -> {dense_dim} (SAE)")
    print(f"Seed: {config['seed']}")
    print(f"{'='*60}\n")
    
    # Store results
    results = {}
    
    # Phase 1: Train autoencoders
    print("PHASE 1: TRAINING AUTOENCODERS")
    print("-" * 40)
    
    for sparsity in config['sparsities']:
        print(f"\nTraining autoencoder for sparsity {sparsity}...")
        
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
        
        # Get metrics
        reconstruction_loss = loss_fn(params, data, I) / data.shape[0]
        bottleneck_acts = get_bottleneck_activations(params, data)
        superposition_analysis = compute_superposition_matrix(params)
        
        results[sparsity] = {
            'params': params,
            'data': data,
            'bottleneck': bottleneck_acts,
            'recon_loss': reconstruction_loss,
            'superposition': superposition_analysis,
            'I': I
        }
        
        print(f"  Reconstruction loss: {reconstruction_loss:.4f}")
    
    # Phase 2: Train SAEs (only if flag is set)
    if config['run_sae']:
        print("\n\nPHASE 2: TRAINING SPARSE AUTOENCODERS")
        print("-" * 40)
        
        for sparsity in config['sparsities']:
            print(f"\nTraining SAE for sparsity {sparsity}...")
            
            bottleneck = results[sparsity]['bottleneck']
            
            sae_params = train_sae(
                activations=bottleneck,
                input_dim=config['dense_dim'],
                hidden_dim=config['sae_hidden_dim'],  # Use config parameter
                num_epochs=config['sae_epochs'],
                learning_rate=config['sae_lr'],
                l1_penalty=config['sae_l1'],
                seed=config['seed'],
                verbose=False
            )
            
            # Get SAE outputs
            sae_result = sae_forward(sae_params, bottleneck)
            sae_hidden = sae_result['hidden']
            sae_recon = sae_result['recon']
            
            # Compute SAE metrics
            sae_mse = np.mean((bottleneck - sae_recon)**2)
            sae_sparsity = compute_sparsity_metrics(sae_hidden)
            
            results[sparsity]['sae'] = {
                'params': sae_params,
                'hidden': sae_hidden,
                'recon': sae_recon,
                'mse': sae_mse,
                'sparsity': sae_sparsity
            }
            
            print(f"  SAE MSE: {sae_mse:.4f}")
            print(f"  SAE L0 norm: {sae_sparsity['l0_norm']:.3f}")
    
    # Phase 3: Compute similarity metrics (only if SAE was run)
    if config['run_sae']:
        print("\n\nPHASE 3: COMPUTING SIMILARITY METRICS")
        print("-" * 40)
        
        for sparsity in config['sparsities']:
            print(f"\nComputing metrics for sparsity {sparsity}...")
            
            bottleneck = results[sparsity]['bottleneck']
            sae_recon = results[sparsity]['sae']['recon']
            
            # Compute metrics
            metrics = {}
            
            # Linear CKA
            try:
                metrics['linear_cka'] = linear_cka(bottleneck, sae_recon)
            except:
                metrics['linear_cka'] = np.nan
            
            # RBF CKA  
            try:
                metrics['rbf_cka'] = rbf_cka(bottleneck, sae_recon)
            except:
                metrics['rbf_cka'] = np.nan
            
            # Procrustes
            try:
                metrics['procrustes'] = procrustes_similarity(bottleneck, sae_recon)
            except:
                metrics['procrustes'] = np.nan
            
            # RSA (simple correlation of distance matrices)
            try:
                from scipy.spatial.distance import pdist
                from scipy.stats import pearsonr
                dist1 = pdist(bottleneck, metric='euclidean')
                dist2 = pdist(sae_recon, metric='euclidean')
                metrics['rsa'], _ = pearsonr(dist1, dist2)
            except:
                metrics['rsa'] = np.nan
            
            results[sparsity]['metrics'] = metrics
            
            print(f"  Linear CKA: {metrics['linear_cka']:.3f}")
            print(f"  RSA: {metrics['rsa']:.3f}")
    
    # Phase 4: Create visualizations
    print("\n\nPHASE 4: CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Plot 1: Superposition matrices - always show these
    n_sparsities = len(config['sparsities'])
    fig, axes = plt.subplots(1, n_sparsities, figsize=(5*n_sparsities, 5))
    if n_sparsities == 1:
        axes = [axes]
        
    for idx, sparsity in enumerate(config['sparsities']):
        W = results[sparsity]['params']['W']
        superposition_matrix = W.T @ W
        
        # Normalize color scale
        vmax = np.max(np.abs(superposition_matrix))
        
        im = axes[idx].imshow(superposition_matrix, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)
        
        # Add more info to title
        diag_mean = np.mean(np.diag(superposition_matrix))
        off_diag = superposition_matrix[~np.eye(superposition_matrix.shape[0], dtype=bool)]
        off_diag_mean = np.mean(np.abs(off_diag)) if len(off_diag) > 0 else 0
        
        axes[idx].set_title(f'Sparsity {sparsity}\nLoss: {results[sparsity]["recon_loss"]:.4f}\nDiag: {diag_mean:.3f}, Off: {off_diag_mean:.3f}', 
                           fontsize=9)
        axes[idx].set_xlabel('Feature')
        axes[idx].set_ylabel('Feature')
        
        # Add grid for small architectures
        if config['sparse_dim'] <= 8:
            axes[idx].set_xticks(range(config['sparse_dim']))
            axes[idx].set_yticks(range(config['sparse_dim']))
            axes[idx].grid(True, alpha=0.3, linewidth=0.5)
        
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    arch_name = f"{config['sparse_dim']}-{config['dense_dim']}-{config['sparse_dim']}"
    plt.suptitle(f'Superposition Matrices (W^T @ W) - Architecture {arch_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/superposition_matrices.png', dpi=150)
    plt.close()
    print("  Saved superposition matrices plot")
    
    # Plot 2: Metrics table (only if SAE was run)
    if config['run_sae']:
        table_data = []
        for sparsity in config['sparsities']:
            row = {
                'Sparsity': sparsity,
                'AE Loss': f"{results[sparsity]['recon_loss']:.4f}",
                'SAE MSE': f"{results[sparsity]['sae']['mse']:.4f}",
                'SAE L0': f"{results[sparsity]['sae']['sparsity']['l0_norm']:.3f}",
                'Linear CKA': f"{results[sparsity]['metrics']['linear_cka']:.3f}",
                'RBF CKA': f"{results[sparsity]['metrics']['rbf_cka']:.3f}",
                'Procrustes': f"{results[sparsity]['metrics']['procrustes']:.3f}",
                'RSA': f"{results[sparsity]['metrics']['rsa']:.3f}"
            }
            table_data.append(row)
    else:
        # Simpler table without SAE metrics
        table_data = []
        for sparsity in config['sparsities']:
            W = results[sparsity]['params']['W']
            superposition_matrix = W.T @ W
            diag_mean = np.mean(np.diag(superposition_matrix))
            off_diag = superposition_matrix[~np.eye(superposition_matrix.shape[0], dtype=bool)]
            off_diag_mean = np.mean(np.abs(off_diag)) if len(off_diag) > 0 else 0
            
            row = {
                'Sparsity': sparsity,
                'AE Loss': f"{results[sparsity]['recon_loss']:.4f}",
                'Diag Mean': f"{diag_mean:.3f}",
                'Off-Diag Mean': f"{off_diag_mean:.3f}",
                'Max Value': f"{np.max(np.abs(superposition_matrix)):.3f}"
            }
            table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Display table
    print("\n\nFINAL RESULTS TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # Save table
    df.to_csv(f'{results_dir}/metrics_table.csv', index=False)
    print(f"\n  Results saved to: {results_dir}/")
    
    # Plot 3: Metrics heatmap (only if SAE was run)
    if config['run_sae']:
        metrics_names = ['Linear CKA', 'RBF CKA', 'Procrustes', 'RSA']
        metrics_keys = ['linear_cka', 'rbf_cka', 'procrustes', 'rsa']
        
        data_matrix = []
        for sparsity in config['sparsities']:
            row = []
            for key in metrics_keys:
                value = results[sparsity]['metrics'][key]
                row.append(value if not np.isnan(value) else 0)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix).T
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(data_matrix, 
                    xticklabels=[f'{s}' for s in config['sparsities']],
                    yticklabels=metrics_names,
                    annot=True, 
                    fmt='.3f',
                    cmap='YlOrRd',
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'Similarity Score'})
        
        plt.title('Similarity Metrics: Bottleneck vs SAE Reconstruction', fontsize=12, fontweight='bold')
        plt.xlabel('Sparsity Level')
        plt.ylabel('Metric')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/metrics_heatmap.png', dpi=150)
        plt.close()
        print("  Saved metrics heatmap")
    
    # Plot 4: SAE Activation Analysis (only if SAE was run)
    if config['run_sae']:
        fig, axes = plt.subplots(2, 7, figsize=(35, 10))  # Changed to 7 subplots
        
        for idx, sparsity in enumerate(config['sparsities']):
            sae_hidden = results[sparsity]['sae']['hidden']
            
            # Top row: Number of active nodes per sample
            active_nodes_per_sample = np.sum(sae_hidden > 0, axis=1)
            axes[0, idx].hist(active_nodes_per_sample, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            axes[0, idx].set_title(f'Sparsity {sparsity}\nMean active: {np.mean(active_nodes_per_sample):.1f}')
            axes[0, idx].set_xlabel('# Active Nodes')
            axes[0, idx].set_ylabel('Frequency')
            axes[0, idx].grid(alpha=0.3)
            axes[0, idx].axvline(np.mean(active_nodes_per_sample), color='red', linestyle='--', label=f'Mean: {np.mean(active_nodes_per_sample):.1f}')
            
            # Bottom row: Activation frequency by node
            activation_freq = np.mean(sae_hidden > 0, axis=0)
            sorted_freq = np.sort(activation_freq)[::-1]
            axes[1, idx].bar(range(len(sorted_freq)), sorted_freq, color='coral', alpha=0.7)
            axes[1, idx].set_title(f'Node Activation Frequency')
            axes[1, idx].set_xlabel('Node Index (sorted)')
            axes[1, idx].set_ylabel('Activation Frequency')
            axes[1, idx].grid(alpha=0.3)
            axes[1, idx].axhline(0.1, color='red', linestyle='--', alpha=0.5, label='10% threshold')
            
            # Add statistics to results
            results[sparsity]['sae']['active_nodes_mean'] = np.mean(active_nodes_per_sample)
            results[sparsity]['sae']['active_nodes_std'] = np.std(active_nodes_per_sample)
            results[sparsity]['sae']['dead_neurons'] = np.sum(activation_freq < 0.01)
        
        plt.suptitle('SAE Activation Patterns Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/sae_activation_analysis.png', dpi=150)
        plt.close()
        print("  Saved SAE activation analysis")
        
        # Plot 5: Activation heatmap for a subset of samples
        fig, axes = plt.subplots(1, 7, figsize=(35, 6))  # Changed to 7 subplots
        
        for idx, sparsity in enumerate(config['sparsities']):
            sae_hidden = results[sparsity]['sae']['hidden']
            
            # Take first 100 samples for visualization
            subset = sae_hidden[:100, :]
            
            # Sort nodes by activation frequency for better visualization
            activation_freq = np.mean(sae_hidden > 0, axis=0)
            sorted_indices = np.argsort(activation_freq)[::-1]
            subset_sorted = subset[:, sorted_indices]
            
            im = axes[idx].imshow(subset_sorted.T, aspect='auto', cmap='hot', interpolation='nearest')
            axes[idx].set_title(f'Sparsity {sparsity}\nDead neurons: {results[sparsity]["sae"]["dead_neurons"]}')
            axes[idx].set_xlabel('Sample Index')
            axes[idx].set_ylabel('Node Index (sorted by frequency)')
            plt.colorbar(im, ax=axes[idx], label='Activation')
        
        plt.suptitle('SAE Activation Heatmaps (First 100 Samples)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/sae_activation_heatmap.png', dpi=150)
        plt.close()
        print("  Saved SAE activation heatmap")
    
    # Update the table with activation statistics (only if SAE was run)
    if config['run_sae']:
        print("\n\nSAE ACTIVATION STATISTICS")
        print("=" * 80)
        activation_stats = []
        for sparsity in config['sparsities']:
            stats = {
                'Sparsity': sparsity,
                'Mean Active Nodes': f"{results[sparsity]['sae']['active_nodes_mean']:.1f}",
                'Std Active Nodes': f"{results[sparsity]['sae']['active_nodes_std']:.1f}",
                'Dead Neurons': results[sparsity]['sae']['dead_neurons'],
                'SAE L0': f"{results[sparsity]['sae']['sparsity']['l0_norm']:.3f}"
            }
            activation_stats.append(stats)
        
        stats_df = pd.DataFrame(activation_stats)
        print(stats_df.to_string(index=False))
        print("=" * 80)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    results = main()