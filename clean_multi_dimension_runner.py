#!/usr/bin/env python3
"""
Clean multi-dimension runner - calls the modified simple experiment 3 times
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from simple_comprehensive_experiment import main

def run_multi_dimension_comparison():
    """Run experiment with 3 different architectures and combine results."""
    
    # Architecture configurations - smaller architectures with different bottlenecks
    architectures = [
        {'name': '10-5-10', 'sparse_dim': 10, 'dense_dim': 5, 'sae_hidden_dim': 10},
        {'name': '20-5-20', 'sparse_dim': 20, 'dense_dim': 5, 'sae_hidden_dim': 20}
    ]
    
    # Create master results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_dir = f'results/multi_dim_clean_{timestamp}'
    os.makedirs(master_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("SMALL ARCHITECTURES COMPARISON - STANDARD EXPONENTIAL DECAY 0.7")
    print(f"{'='*80}")
    print(f"Running architectures: {[arch['name'] for arch in architectures]}")
    print(f"Feature importance: Exponential decay 0.7^i (standard)")
    print(f"Master results: {master_dir}")
    print(f"{'='*80}")
    
    # Define sparsities to test
    sparsities = [0.5, 0.8, 0.85, 0.875, 0.9, 0.925, 0.95]
    
    all_results = {}
    combined_data = []
    
    # Run experiment for each architecture
    for i, arch in enumerate(architectures):
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT {i+1}/4: {arch['name']}")
        print(f"{'='*60}")
        
        # Call the main function with architecture parameters
        results = main(
            sparse_dim=arch['sparse_dim'],
            dense_dim=arch['dense_dim'], 
            sae_hidden_dim=arch['sae_hidden_dim'],
            seed=123,  # Same seed for fair comparison
            results_suffix=f"_{arch['name']}",
            run_sae=True,  # Disable SAE for superposition-only analysis
            sparsities=sparsities  # Pass sparsities explicitly
        )
        
        all_results[arch['name']] = results
        
        # Extract data for combined analysis (no SAE metrics)
        for sparsity in sparsities:
            # Get superposition matrix metrics
            W = results[sparsity]['params']['W']
            superposition_matrix = W.T @ W
            diagonal_mean = np.mean(np.diag(superposition_matrix))
            off_diagonal_mean = np.mean(superposition_matrix[~np.eye(superposition_matrix.shape[0], dtype=bool)])
            
            row = {
                'Architecture': arch['name'],
                'Sparsity': sparsity,
                'AE_Loss': results[sparsity]['recon_loss'],
                'Diagonal_Mean': diagonal_mean,
                'Off_Diagonal_Mean': off_diagonal_mean,
                'Max_Superposition': np.max(np.abs(superposition_matrix))
            }
            
            # Add SAE metrics and Procrustes if available
            if 'sae' in results[sparsity]:
                row['SAE_Loss'] = results[sparsity]['sae'].get('mse', np.nan)
            if 'metrics' in results[sparsity]:
                row['Procrustes'] = results[sparsity]['metrics'].get('procrustes', np.nan)
            
            combined_data.append(row)
    
    # Create combined analysis
    print(f"\n{'='*80}")
    print("CREATING COMBINED ANALYSIS")
    print(f"{'='*80}")
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(f'{master_dir}/combined_results.csv', index=False)
    
    # Create summary pivot tables for requested metrics only
    print("\n1. AUTOENCODER LOSS BY ARCHITECTURE & SPARSITY")
    print("=" * 60)
    pivot_ae_loss = combined_df.pivot(index='Sparsity', columns='Architecture', values='AE_Loss')
    print(pivot_ae_loss.round(4))
    
    if 'SAE_Loss' in combined_df.columns:
        print("\n2. SAE LOSS BY ARCHITECTURE & SPARSITY")
        print("=" * 60)
        pivot_sae_loss = combined_df.pivot(index='Sparsity', columns='Architecture', values='SAE_Loss')
        print(pivot_sae_loss.round(4))
    
    print("\n3. DIAGONAL MEAN (SUPERPOSITION) BY ARCHITECTURE & SPARSITY") 
    print("=" * 60)
    pivot_diag = combined_df.pivot(index='Sparsity', columns='Architecture', values='Diagonal_Mean')
    print(pivot_diag.round(3))
    
    print("\n4. OFF-DIAGONAL MEAN (INTERFERENCE) BY ARCHITECTURE & SPARSITY")
    print("=" * 60)
    pivot_off_diag = combined_df.pivot(index='Sparsity', columns='Architecture', values='Off_Diagonal_Mean')
    print(pivot_off_diag.round(3))
    
    print("\n5. MAX SUPERPOSITION VALUE BY ARCHITECTURE & SPARSITY")
    print("=" * 60)
    pivot_max = combined_df.pivot(index='Sparsity', columns='Architecture', values='Max_Superposition')
    print(pivot_max.round(3))
    
    if 'Procrustes' in combined_df.columns:
        print("\n6. PROCRUSTES SIMILARITY (AE-SAE) BY ARCHITECTURE & SPARSITY")
        print("=" * 60)
        pivot_proc = combined_df.pivot(index='Sparsity', columns='Architecture', values='Procrustes')
        print(pivot_proc.round(3))
    
    # Save pivot tables
    with open(f'{master_dir}/summary_analysis.txt', 'w') as f:
        f.write("SMALL ARCHITECTURES SUPERPOSITION ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. AUTOENCODER LOSS BY ARCHITECTURE & SPARSITY\n")
        f.write("-"*60 + "\n")
        f.write(pivot_ae_loss.round(4).to_string())
        f.write("\n\n")
        
        if 'SAE_Loss' in combined_df.columns:
            f.write("2. SAE LOSS BY ARCHITECTURE & SPARSITY\n")
            f.write("-"*60 + "\n")
            f.write(pivot_sae_loss.round(4).to_string())
            f.write("\n\n")
        
        f.write("3. DIAGONAL MEAN (SUPERPOSITION) BY ARCHITECTURE & SPARSITY\n")  
        f.write("-"*60 + "\n")
        f.write(pivot_diag.round(3).to_string())
        f.write("\n\n")
        
        f.write("4. OFF-DIAGONAL MEAN (INTERFERENCE) BY ARCHITECTURE & SPARSITY\n")
        f.write("-"*60 + "\n")
        f.write(pivot_off_diag.round(3).to_string())
        f.write("\n\n")
        
        f.write("5. MAX SUPERPOSITION VALUE BY ARCHITECTURE & SPARSITY\n")
        f.write("-"*60 + "\n")
        f.write(pivot_max.round(3).to_string())
        
        if 'Procrustes' in combined_df.columns:
            f.write("\n\n")
            f.write("6. PROCRUSTES SIMILARITY (AE-SAE) BY ARCHITECTURE & SPARSITY\n")
            f.write("-"*60 + "\n")
            f.write(pivot_proc.round(3).to_string())
    
    # Create superposition evolution graphs
    print("\n\nCREATING SUPERPOSITION EVOLUTION GRAPHS")
    print("-" * 60)
    
    # Set up style
    sns.set_style("whitegrid")
    
    # Plot 1: AE Loss Evolution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for arch in architectures:
        arch_data = combined_df[combined_df['Architecture'] == arch['name']]
        axes[0].plot(arch_data['Sparsity'], arch_data['AE_Loss'], 
                    marker='o', label=arch['name'], linewidth=2)
    
    axes[0].set_xlabel('Sparsity', fontsize=12)
    axes[0].set_ylabel('AE Reconstruction Loss', fontsize=12)
    axes[0].set_title('Autoencoder Loss vs Sparsity', fontsize=14, fontweight='bold')
    axes[0].legend(title='Architecture', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Diagonal Mean Evolution (Superposition strength)
    for arch in architectures:
        arch_data = combined_df[combined_df['Architecture'] == arch['name']]
        axes[1].plot(arch_data['Sparsity'], arch_data['Diagonal_Mean'], 
                    marker='s', label=arch['name'], linewidth=2)
    
    axes[1].set_xlabel('Sparsity', fontsize=12)
    axes[1].set_ylabel('Diagonal Mean (W^T @ W)', fontsize=12)
    axes[1].set_title('Superposition Strength vs Sparsity', fontsize=14, fontweight='bold')
    axes[1].legend(title='Architecture', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Recovery')
    
    plt.suptitle('Superposition Evolution Across Architectures', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{master_dir}/superposition_evolution.png', dpi=150)
    plt.close()
    print("  Saved superposition_evolution.png")
    
    # Plot 2: Procrustes Evolution (if SAE was run)
    if 'Procrustes' in combined_df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        for arch in architectures:
            arch_data = combined_df[combined_df['Architecture'] == arch['name']]
            ax.plot(arch_data['Sparsity'], arch_data['Procrustes'], 
                   marker='D', label=arch['name'], linewidth=2)
        
        ax.set_xlabel('Sparsity', fontsize=12)
        ax.set_ylabel('Procrustes Similarity', fontsize=12)
        ax.set_title('AE-SAE Representation Similarity vs Sparsity', fontsize=14, fontweight='bold')
        ax.legend(title='Architecture', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect Alignment')
        ax.axhline(y=0.0, color='r', linestyle='--', alpha=0.5, label='No Correlation')
        ax.set_ylim([-0.5, 1.1])
        
        plt.tight_layout()
        plt.savefig(f'{master_dir}/procrustes_evolution.png', dpi=150)
        plt.close()
        print("  Saved procrustes_evolution.png")
    
    # Plot 3: Superposition matrices comparison
    n_archs = len(architectures)
    fig, axes = plt.subplots(n_archs, len(sparsities), 
                             figsize=(3*len(sparsities), 3*n_archs))
    
    for i, arch in enumerate(architectures):
        for j, sparsity in enumerate(sparsities):
            W = all_results[arch['name']][sparsity]['params']['W']
            superposition_matrix = W.T @ W
            
            # Normalize color scale
            vmax = np.max(np.abs(superposition_matrix))
            
            ax = axes[i, j] if n_archs > 1 else axes[j]
            im = ax.imshow(superposition_matrix, cmap='coolwarm', 
                          aspect='auto', vmin=-vmax, vmax=vmax)
            
            if i == 0:
                ax.set_title(f'S={sparsity}', fontsize=10)
            if j == 0:
                ax.set_ylabel(arch['name'], fontsize=10, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Superposition Matrices (W^T @ W) Across Architectures and Sparsities', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{master_dir}/superposition_matrices_grid.png', dpi=150)
    plt.close()
    print("  Saved superposition_matrices_grid.png")
    
    print(f"\n{'='*80}")
    print("MULTI-DIMENSION EXPERIMENT COMPLETED!")
    print(f"{'='*80}")
    print(f"Master results: {master_dir}")
    print("Individual experiments in separate directories")
    print("Combined analysis in combined_results.csv")
    print("Summary tables in summary_analysis.txt")
    print("Evolution graphs in superposition_evolution.png and procrustes_evolution.png")
    print("Matrix grid in superposition_matrices_grid.png")
    
    return all_results, combined_df

if __name__ == "__main__":
    results, combined_table = run_multi_dimension_comparison()