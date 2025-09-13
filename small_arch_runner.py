#!/usr/bin/env python3
"""
Small architectures runner - focused on superposition analysis
Tests 3-2-3, 5-2-5, 4-3-4, 8-3-8 with exponential decay 0.7
"""

import pandas as pd
from datetime import datetime
import os
from simple_comprehensive_experiment import main

def run_small_architectures():
    """Run experiment with small architectures, superposition only."""
    
    # Small architecture configurations  
    architectures = [
        {'name': '3-2-3', 'sparse_dim': 3, 'dense_dim': 2},
        {'name': '5-2-5', 'sparse_dim': 5, 'dense_dim': 2},
        {'name': '4-3-4', 'sparse_dim': 4, 'dense_dim': 3},
        {'name': '8-3-8', 'sparse_dim': 8, 'dense_dim': 3}
    ]
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_dir = f'results/small_arch_{timestamp}'
    os.makedirs(master_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("SMALL ARCHITECTURES SUPERPOSITION ANALYSIS")
    print(f"{'='*80}")
    print(f"Architectures: {[arch['name'] for arch in architectures]}")
    print(f"Feature importance: Exponential decay 0.7^i")
    print(f"Results: {master_dir}")
    print(f"{'='*80}")
    
    all_results = {}
    combined_data = []
    
    # Run each architecture
    for i, arch in enumerate(architectures):
        print(f"\n{'='*60}")
        print(f"ARCHITECTURE {i+1}/4: {arch['name']}")
        print(f"{'='*60}")
        
        # Run experiment without SAE
        results = main(
            sparse_dim=arch['sparse_dim'],
            dense_dim=arch['dense_dim'],
            sae_hidden_dim=arch['sparse_dim'],  # Would be used if SAE enabled
            seed=123,
            results_suffix=f"_{arch['name']}",
            run_sae=False  # Focus on superposition only
        )
        
        all_results[arch['name']] = results
        
        # Extract superposition metrics
        for sparsity in [0.3, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]:
            W = results[sparsity]['params']['W']
            superposition_matrix = W.T @ W
            
            # Compute metrics
            diag_mean = np.mean(np.diag(superposition_matrix))
            off_diag = superposition_matrix[~np.eye(superposition_matrix.shape[0], dtype=bool)]
            off_diag_mean = np.mean(np.abs(off_diag)) if len(off_diag) > 0 else 0
            max_val = np.max(np.abs(superposition_matrix))
            
            combined_data.append({
                'Architecture': arch['name'],
                'Sparse_Dim': arch['sparse_dim'],
                'Dense_Dim': arch['dense_dim'],
                'Compression': f"{arch['sparse_dim']}/{arch['dense_dim']} = {arch['sparse_dim']/arch['dense_dim']:.1f}",
                'Sparsity': sparsity,
                'AE_Loss': results[sparsity]['recon_loss'],
                'Diag_Mean': diag_mean,
                'Off_Diag_Mean': off_diag_mean,
                'Max_Value': max_val,
                'Diag_Dominance': diag_mean / (off_diag_mean + 1e-8)  # Ratio showing orthogonality
            })
    
    # Create analysis
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(f'{master_dir}/combined_results.csv', index=False)
    
    print(f"\n{'='*80}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*80}")
    
    # Key comparisons
    print("\n1. RECONSTRUCTION LOSS BY ARCHITECTURE")
    print("-" * 60)
    pivot_loss = combined_df.pivot(index='Sparsity', columns='Architecture', values='AE_Loss')
    print(pivot_loss.round(4))
    
    print("\n2. DIAGONAL DOMINANCE (Higher = More Orthogonal)")  
    print("-" * 60)
    pivot_dom = combined_df.pivot(index='Sparsity', columns='Architecture', values='Diag_Dominance')
    print(pivot_dom.round(2))
    
    print("\n3. OFF-DIAGONAL INTERFERENCE")
    print("-" * 60)
    pivot_off = combined_df.pivot(index='Sparsity', columns='Architecture', values='Off_Diag_Mean')
    print(pivot_off.round(3))
    
    # Save summary
    with open(f'{master_dir}/summary.txt', 'w') as f:
        f.write("SMALL ARCHITECTURES SUPERPOSITION ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write("Architectures tested:\n")
        for arch in architectures:
            f.write(f"  {arch['name']}: {arch['sparse_dim']} features compressed to {arch['dense_dim']} dimensions\n")
        f.write(f"\nFeature importance: Exponential decay 0.7^i\n")
        f.write(f"Random seed: 123\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-"*60 + "\n")
        
        # Find architecture with highest loss at 0.9 sparsity
        loss_at_09 = combined_df[combined_df['Sparsity'] == 0.9][['Architecture', 'AE_Loss']]
        worst_arch = loss_at_09.loc[loss_at_09['AE_Loss'].idxmax(), 'Architecture']
        worst_loss = loss_at_09['AE_Loss'].max()
        f.write(f"Worst performance at 0.9 sparsity: {worst_arch} (loss={worst_loss:.4f})\n")
        
        # Find most orthogonal architecture
        mean_dominance = combined_df.groupby('Architecture')['Diag_Dominance'].mean()
        most_orthogonal = mean_dominance.idxmax()
        f.write(f"Most orthogonal representations: {most_orthogonal} (avg dominance={mean_dominance[most_orthogonal]:.2f})\n\n")
        
        f.write("DETAILED TABLES:\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. RECONSTRUCTION LOSS\n")
        f.write("-"*60 + "\n")
        f.write(pivot_loss.round(4).to_string())
        f.write("\n\n2. DIAGONAL DOMINANCE\n")
        f.write("-"*60 + "\n")
        f.write(pivot_dom.round(2).to_string())
        f.write("\n\n3. OFF-DIAGONAL INTERFERENCE\n")
        f.write("-"*60 + "\n")
        f.write(pivot_off.round(3).to_string())
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved to: {master_dir}")
    print("Check superposition_matrices.png in each architecture folder!")
    
    return all_results, combined_df

if __name__ == "__main__":
    import numpy as np
    results, df = run_small_architectures()