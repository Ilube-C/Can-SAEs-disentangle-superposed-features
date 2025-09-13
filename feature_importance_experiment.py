#!/usr/bin/env python3
"""
Feature Importance Experiment
Fixed architecture (20-5-20) but varying feature importance patterns
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

def get_custom_feature_importances(n, pattern_type="exponential", **kwargs):
    """Generate different feature importance patterns."""
    
    if pattern_type == "exponential":
        # Standard exponential decay: 0.7^i
        decay_factor = kwargs.get('decay_factor', 0.7)
        importances = decay_factor ** np.arange(n)
        
    elif pattern_type == "uniform":
        # All features equally important
        importances = np.ones(n)
        
    elif pattern_type == "steep_exponential":
        # Steeper decay: 0.5^i
        decay_factor = kwargs.get('decay_factor', 0.5)
        importances = decay_factor ** np.arange(n)
        
    elif pattern_type == "gentle_exponential":
        # Gentler decay: 0.9^i
        decay_factor = kwargs.get('decay_factor', 0.9)
        importances = decay_factor ** np.arange(n)
        
    elif pattern_type == "step":
        # Step function: first half important, second half not
        split_point = kwargs.get('split_point', n//2)
        high_value = kwargs.get('high_value', 1.0)
        low_value = kwargs.get('low_value', 0.1)
        importances = np.concatenate([
            np.full(split_point, high_value),
            np.full(n - split_point, low_value)
        ])
        
    elif pattern_type == "linear":
        # Linear decay from 1 to 0.1
        importances = np.linspace(1.0, 0.1, n)
        
    elif pattern_type == "u_shaped":
        # U-shaped: high at ends, low in middle
        x = np.linspace(-1, 1, n)
        importances = x**2 * 0.9 + 0.1  # Range from 0.1 to 1.0
        
    elif pattern_type == "random":
        # Random importances (but reproducible with seed)
        seed = kwargs.get('seed', 42)
        np.random.seed(seed)
        importances = np.random.uniform(0.1, 1.0, n)
        np.random.seed()  # Reset seed
        
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")
    
    return importances.reshape(1, -1)

def run_feature_importance_experiment():
    """Run experiment with different feature importance patterns."""
    
    # Configuration
    config = {
        'seed': 123,
        'sparsities': [0.3, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95],
        'sparse_dim': 20,
        'dense_dim': 5,
        'sae_hidden_dim': 20,
        'num_samples': 2000,
        'num_epochs': 5,
        'learning_rate': 0.01,
        'sae_epochs': 20,
        'sae_lr': 0.001,
        'sae_l1': 0.01,
    }
    
    # Feature importance patterns to test
    importance_patterns = [
        {'name': 'Standard_Exp_0.7', 'type': 'exponential', 'decay_factor': 0.7},
        {'name': 'Uniform', 'type': 'uniform'},
        {'name': 'Steep_Exp_0.5', 'type': 'steep_exponential', 'decay_factor': 0.5},
        {'name': 'Gentle_Exp_0.9', 'type': 'gentle_exponential', 'decay_factor': 0.9},
        {'name': 'Step_Function', 'type': 'step', 'split_point': 10, 'high_value': 1.0, 'low_value': 0.1},
        {'name': 'Linear_Decay', 'type': 'linear'},
    ]
    
    # Create master results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_dir = f'results/feature_importance_exp_{timestamp}'
    os.makedirs(master_dir, exist_ok=True)
    
    print(f"\\n{'='*80}")
    print("FEATURE IMPORTANCE PATTERN EXPERIMENT")
    print(f"{'='*80}")
    print(f"Fixed architecture: {config['sparse_dim']}-{config['dense_dim']}-{config['sparse_dim']}")
    print(f"SAE architecture: {config['dense_dim']}-{config['sae_hidden_dim']}-{config['dense_dim']}")
    print(f"Testing patterns: {[p['name'] for p in importance_patterns]}")
    print(f"Sparsity levels: {config['sparsities']}")
    print(f"Results directory: {master_dir}")
    print(f"{'='*80}")
    
    all_results = {}
    combined_data = []
    
    # Run experiment for each feature importance pattern
    for pattern_idx, pattern_config in enumerate(importance_patterns):
        pattern_name = pattern_config['name']
        
        print(f"\\n{'='*60}")
        print(f"RUNNING PATTERN {pattern_idx+1}/{len(importance_patterns)}: {pattern_name}")
        print(f"{'='*60}")
        
        # Generate feature importance weights
        pattern_kwargs = {k: v for k, v in pattern_config.items() if k not in ['name', 'type']}
        I = get_custom_feature_importances(
            config['sparse_dim'], 
            pattern_config['type'], 
            **pattern_kwargs
        )
        
        print(f"Feature importance pattern: {pattern_config['type']}")
        print(f"Importance values: {I.flatten()[:5]}... (showing first 5)")
        print(f"Min: {np.min(I):.3f}, Max: {np.max(I):.3f}, Range: {np.max(I) - np.min(I):.3f}")
        
        pattern_results = {}
        
        # Train models at each sparsity level
        for sparsity in config['sparsities']:
            print(f"\\n  Training at sparsity {sparsity}...")
            
            # Generate data
            data = generate_synthetic_data(
                config['seed'], 
                config['sparse_dim'], 
                sparsity, 
                config['num_samples']
            )
            
            # Train autoencoder
            params = train_model(
                data=data,
                I=I,
                k=config['dense_dim'],
                n=config['sparse_dim'],
                num_epochs=config['num_epochs'],
                learning_rate=config['learning_rate'],
                seed=config['seed']
            )
            
            # Compute AE metrics
            reconstruction_loss = loss_fn(params, data, I) / data.shape[0]
            bottleneck_acts = get_bottleneck_activations(params, data)
            superposition_analysis = compute_superposition_matrix(params)
            
            print(f"    AE Loss: {reconstruction_loss:.4f}")
            
            # Train SAE
            sae_params = train_sae(
                activations=bottleneck_acts,
                input_dim=config['dense_dim'],
                hidden_dim=config['sae_hidden_dim'],
                num_epochs=config['sae_epochs'],
                learning_rate=config['sae_lr'],
                l1_penalty=config['sae_l1'],
                seed=config['seed'],
                verbose=False
            )
            
            # Get SAE outputs and metrics
            sae_result = sae_forward(sae_params, bottleneck_acts)
            sae_hidden = sae_result['hidden']
            sae_recon = sae_result['recon']
            
            sae_mse = np.mean((bottleneck_acts - sae_recon)**2)
            sae_sparsity = compute_sparsity_metrics(sae_hidden)
            
            # SAE activation analysis
            active_nodes_per_sample = np.sum(sae_hidden > 0, axis=1)
            activation_freq = np.mean(sae_hidden > 0, axis=0)
            dead_neurons = np.sum(activation_freq < 0.01)
            
            # Similarity metrics
            try:
                procrustes = procrustes_similarity(bottleneck_acts, sae_recon)
            except:
                procrustes = np.nan
                
            try:
                linear_cka_score = linear_cka(bottleneck_acts, sae_recon)
            except:
                linear_cka_score = np.nan
            
            print(f"    SAE MSE: {sae_mse:.4f}, L0: {sae_sparsity['l0_norm']:.3f}, Dead: {dead_neurons}")
            
            # Store results
            pattern_results[sparsity] = {
                'data': data,
                'feature_importances': I,
                'ae_params': params,
                'ae_loss': reconstruction_loss,
                'bottleneck': bottleneck_acts,
                'superposition': superposition_analysis,
                'sae_params': sae_params,
                'sae_hidden': sae_hidden,
                'sae_recon': sae_recon,
                'sae_mse': sae_mse,
                'sae_sparsity': sae_sparsity,
                'active_nodes_mean': np.mean(active_nodes_per_sample),
                'dead_neurons': dead_neurons,
                'procrustes': procrustes,
                'linear_cka': linear_cka_score
            }
            
            # Add to combined data
            combined_data.append({
                'Pattern': pattern_name,
                'Pattern_Type': pattern_config['type'],
                'Sparsity': sparsity,
                'AE_Loss': reconstruction_loss,
                'SAE_MSE': sae_mse,
                'SAE_L0': sae_sparsity['l0_norm'],
                'Mean_Active_Nodes': np.mean(active_nodes_per_sample),
                'Dead_Neurons': dead_neurons,
                'Procrustes': procrustes,
                'Linear_CKA': linear_cka_score
            })
        
        all_results[pattern_name] = pattern_results
    
    # Create combined analysis
    print(f"\\n{'='*80}")
    print("CREATING COMBINED ANALYSIS")
    print(f"{'='*80}")
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(f'{master_dir}/combined_results.csv', index=False)
    
    # Create summary pivot tables
    print("\\n1. SAE L0 NORM BY PATTERN & SPARSITY")
    print("=" * 60)
    pivot_l0 = combined_df.pivot(index='Sparsity', columns='Pattern', values='SAE_L0')
    print(pivot_l0.round(3))
    
    print("\\n2. DEAD NEURONS BY PATTERN & SPARSITY") 
    print("=" * 60)
    pivot_dead = combined_df.pivot(index='Sparsity', columns='Pattern', values='Dead_Neurons')
    print(pivot_dead)
    
    print("\\n3. PROCRUSTES SIMILARITY BY PATTERN & SPARSITY")
    print("=" * 60)
    pivot_proc = combined_df.pivot(index='Sparsity', columns='Pattern', values='Procrustes')
    print(pivot_proc.round(3))
    
    print("\\n4. AUTOENCODER LOSS BY PATTERN & SPARSITY")
    print("=" * 60)
    pivot_ae = combined_df.pivot(index='Sparsity', columns='Pattern', values='AE_Loss')
    print(pivot_ae.round(4))
    
    # Save summary analysis
    with open(f'{master_dir}/summary_analysis.txt', 'w') as f:
        f.write("FEATURE IMPORTANCE PATTERN EXPERIMENT SUMMARY\\n")
        f.write("="*60 + "\\n\\n")
        
        f.write("1. SAE L0 NORM BY PATTERN & SPARSITY\\n")
        f.write("-"*60 + "\\n")
        f.write(pivot_l0.round(3).to_string())
        f.write("\\n\\n")
        
        f.write("2. DEAD NEURONS BY PATTERN & SPARSITY\\n")
        f.write("-"*60 + "\\n")
        f.write(pivot_dead.to_string())
        f.write("\\n\\n")
        
        f.write("3. PROCRUSTES SIMILARITY BY PATTERN & SPARSITY\\n")
        f.write("-"*60 + "\\n")
        f.write(pivot_proc.round(3).to_string())
        f.write("\\n\\n")
        
        f.write("4. AUTOENCODER LOSS BY PATTERN & SPARSITY\\n")
        f.write("-"*60 + "\\n")
        f.write(pivot_ae.round(4).to_string())
    
    # Create visualizations
    print("\\nCreating visualizations...")
    
    # Plot 1: Feature importance patterns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, pattern_config in enumerate(importance_patterns):
        pattern_kwargs = {k: v for k, v in pattern_config.items() if k not in ['name', 'type']}
        I = get_custom_feature_importances(
            config['sparse_dim'], 
            pattern_config['type'], 
            **pattern_kwargs
        )
        
        axes[idx].bar(range(config['sparse_dim']), I.flatten(), alpha=0.7)
        axes[idx].set_title(f"{pattern_config['name']}")
        axes[idx].set_xlabel('Feature Index')
        axes[idx].set_ylabel('Importance')
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Feature Importance Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{master_dir}/feature_importance_patterns.png', dpi=150)
    plt.close()
    
    # Plot 2: Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # AE Loss by pattern
    for pattern in combined_df['Pattern'].unique():
        subset = combined_df[combined_df['Pattern'] == pattern]
        axes[0, 0].plot(subset['Sparsity'], subset['AE_Loss'], marker='o', label=pattern, linewidth=2)
    axes[0, 0].set_xlabel('Sparsity')
    axes[0, 0].set_ylabel('Autoencoder Loss')
    axes[0, 0].set_title('Autoencoder Loss by Pattern')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Dead neurons by pattern
    for pattern in combined_df['Pattern'].unique():
        subset = combined_df[combined_df['Pattern'] == pattern]
        axes[0, 1].plot(subset['Sparsity'], subset['Dead_Neurons'], marker='s', label=pattern, linewidth=2)
    axes[0, 1].set_xlabel('Sparsity')
    axes[0, 1].set_ylabel('Dead SAE Neurons')
    axes[0, 1].set_title('Dead Neurons by Pattern')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # SAE L0 by pattern
    for pattern in combined_df['Pattern'].unique():
        subset = combined_df[combined_df['Pattern'] == pattern]
        axes[1, 0].plot(subset['Sparsity'], subset['SAE_L0'], marker='^', label=pattern, linewidth=2)
    axes[1, 0].set_xlabel('Sparsity')
    axes[1, 0].set_ylabel('SAE L0 Norm')
    axes[1, 0].set_title('SAE Sparsity by Pattern')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Procrustes similarity by pattern
    for pattern in combined_df['Pattern'].unique():
        subset = combined_df[combined_df['Pattern'] == pattern]
        axes[1, 1].plot(subset['Sparsity'], subset['Procrustes'], marker='d', label=pattern, linewidth=2)
    axes[1, 1].set_xlabel('Sparsity')
    axes[1, 1].set_ylabel('Procrustes Similarity')
    axes[1, 1].set_title('Similarity by Pattern')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Performance Comparison Across Feature Importance Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{master_dir}/performance_comparison.png', dpi=150)
    plt.close()
    
    # Plot 3: Superposition matrices for critical sparsity (0.9) across all patterns
    print("Creating superposition matrix comparison...")
    critical_sparsity = 0.9
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, pattern_config in enumerate(importance_patterns):
        pattern_name = pattern_config['name']
        
        if critical_sparsity in all_results[pattern_name]:
            W = all_results[pattern_name][critical_sparsity]['ae_params']['W']
            superposition_matrix = W.T @ W
            
            # Normalize for consistent color scale
            vmax = np.max(np.abs(superposition_matrix))
            
            im = axes[idx].imshow(superposition_matrix, 
                                 cmap='coolwarm', 
                                 aspect='auto', 
                                 vmin=-vmax, 
                                 vmax=vmax)
            
            ae_loss = all_results[pattern_name][critical_sparsity]['ae_loss']
            dead_neurons = all_results[pattern_name][critical_sparsity]['dead_neurons']
            
            axes[idx].set_title(f'{pattern_name}\\nAE Loss: {ae_loss:.3f}\\nDead SAE: {dead_neurons}', fontsize=10)
            axes[idx].set_xlabel('Feature Index')
            axes[idx].set_ylabel('Feature Index')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        else:
            axes[idx].set_visible(False)
    
    plt.suptitle(f'Superposition Matrices (W^T @ W) at Sparsity {critical_sparsity}\\nAcross Feature Importance Patterns', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{master_dir}/superposition_comparison_sparsity_{critical_sparsity}.png', dpi=150)
    plt.close()
    
    # Plot 4: Superposition matrices for uniform pattern across all sparsities
    print("Creating superposition evolution for uniform pattern...")
    uniform_pattern = 'Uniform'
    
    if uniform_pattern in all_results:
        n_sparsities = len(config['sparsities'])
        fig, axes = plt.subplots(1, n_sparsities, figsize=(5*n_sparsities, 5))
        
        if n_sparsities == 1:
            axes = [axes]
        
        for idx, sparsity in enumerate(config['sparsities']):
            if sparsity in all_results[uniform_pattern]:
                W = all_results[uniform_pattern][sparsity]['ae_params']['W']
                superposition_matrix = W.T @ W
                
                vmax = np.max(np.abs(superposition_matrix))
                
                im = axes[idx].imshow(superposition_matrix, 
                                     cmap='coolwarm', 
                                     aspect='auto', 
                                     vmin=-vmax, 
                                     vmax=vmax)
                
                ae_loss = all_results[uniform_pattern][sparsity]['ae_loss']
                dead_neurons = all_results[uniform_pattern][sparsity]['dead_neurons']
                sae_l0 = all_results[uniform_pattern][sparsity]['sae_sparsity']['l0_norm']
                
                axes[idx].set_title(f'Sparsity {sparsity}\\nAE: {ae_loss:.3f}\\nSAE L0: {sae_l0:.3f}\\nDead: {dead_neurons}', 
                                   fontsize=10)
                axes[idx].set_xlabel('Feature')
                axes[idx].set_ylabel('Feature')
                
                plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Superposition Evolution - {uniform_pattern} Feature Importance', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{master_dir}/superposition_evolution_{uniform_pattern.lower()}.png', dpi=150)
        plt.close()
    
    # Plot 5: Superposition matrices for exponential pattern across all sparsities
    print("Creating superposition evolution for standard exponential pattern...")
    exp_pattern = 'Standard_Exp_0.7'
    
    if exp_pattern in all_results:
        n_sparsities = len(config['sparsities'])
        fig, axes = plt.subplots(1, n_sparsities, figsize=(5*n_sparsities, 5))
        
        if n_sparsities == 1:
            axes = [axes]
        
        for idx, sparsity in enumerate(config['sparsities']):
            if sparsity in all_results[exp_pattern]:
                W = all_results[exp_pattern][sparsity]['ae_params']['W']
                superposition_matrix = W.T @ W
                
                vmax = np.max(np.abs(superposition_matrix))
                
                im = axes[idx].imshow(superposition_matrix, 
                                     cmap='coolwarm', 
                                     aspect='auto', 
                                     vmin=-vmax, 
                                     vmax=vmax)
                
                ae_loss = all_results[exp_pattern][sparsity]['ae_loss']
                dead_neurons = all_results[exp_pattern][sparsity]['dead_neurons']
                sae_l0 = all_results[exp_pattern][sparsity]['sae_sparsity']['l0_norm']
                
                axes[idx].set_title(f'Sparsity {sparsity}\\nAE: {ae_loss:.3f}\\nSAE L0: {sae_l0:.3f}\\nDead: {dead_neurons}', 
                                   fontsize=10)
                axes[idx].set_xlabel('Feature')
                axes[idx].set_ylabel('Feature')
                
                plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Superposition Evolution - {exp_pattern} Feature Importance', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{master_dir}/superposition_evolution_{exp_pattern.lower()}.png', dpi=150)
        plt.close()
    
    print(f"\\n{'='*80}")
    print("FEATURE IMPORTANCE EXPERIMENT COMPLETED!")
    print(f"{'='*80}")
    print(f"Results directory: {master_dir}")
    print("Summary tables in summary_analysis.txt")
    print("Combined data in combined_results.csv")
    print("Visualizations:")
    print("  - feature_importance_patterns.png")
    print("  - performance_comparison.png")
    print("  - superposition_comparison_sparsity_0.9.png")
    print("  - superposition_evolution_uniform.png")
    print("  - superposition_evolution_standard_exp_0.7.png")
    
    return all_results, combined_df

if __name__ == "__main__":
    results, combined_table = run_feature_importance_experiment()