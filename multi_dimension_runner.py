#!/usr/bin/env python3
"""
Multi-dimension runner that calls simple_comprehensive_experiment.py 
with different architecture dimensions: 10-5-10, 20-5-20, 30-5-30
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add paths
sys.path.append('src')
sys.path.append('config')

# Import the main function from our existing experiment
from simple_comprehensive_experiment import main

def run_multi_dimension_experiment():
    """Run the experiment with different dimensions and combine results."""
    
    # Architecture configurations
    architectures = [
        {'name': '10-5-10', 'sparse_dim': 10, 'dense_dim': 5, 'sae_hidden_dim': 10},
        {'name': '20-5-20', 'sparse_dim': 20, 'dense_dim': 5, 'sae_hidden_dim': 20},
        {'name': '30-5-30', 'sparse_dim': 30, 'dense_dim': 5, 'sae_hidden_dim': 30}
    ]
    
    # Create master results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_results_dir = f'results/multi_dim_comparison_{timestamp}'
    os.makedirs(master_results_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("MULTI-DIMENSION ARCHITECTURE COMPARISON")
    print(f"{'='*80}")
    print(f"Running architectures: {[arch['name'] for arch in architectures]}")
    print(f"Master results directory: {master_results_dir}")
    print(f"{'='*80}")
    
    all_results = {}
    combined_data = []
    
    # Run experiment for each architecture
    for i, arch in enumerate(architectures):
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT {i+1}/3: {arch['name']}")
        print(f"{'='*60}")
        
        # Temporarily modify the main function's config
        import simple_comprehensive_experiment as exp_module
        
        # Store original main function
        original_main = exp_module.main
        
        def modified_main():
            # Configuration with current architecture
            config = {
                'seed': 123,
                'sparsities': [0.3, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95],
                'sparse_dim': arch['sparse_dim'],
                'dense_dim': arch['dense_dim'],
                'num_samples': 2000,
                'num_epochs': 5,
                'learning_rate': 0.01,
                'decay_factor': 0.7,
                'sae_epochs': 20,
                'sae_lr': 0.001,
                'sae_l1': 0.01,
            }
            
            # Update SAE hidden dimension to match sparse dimension
            # This requires modifying the train_sae calls in the original code
            # We'll need to inject this into the experiment
            
            # Import everything we need
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            from datetime import datetime
            
            from data_generation import generate_synthetic_data, get_feature_importances
            from models_numpy import (
                train_model, get_bottleneck_activations, 
                train_sae, sae_forward, loss_fn, forward
            )
            from analysis import compute_superposition_matrix, compute_sparsity_metrics
            from CKA import linear_cka, rbf_cka
            from rsa_procrustes import procrustes_similarity
            
            # Create results directory for this architecture
            arch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = f'{master_results_dir}/{arch["name"].replace("-", "_")}_{arch_timestamp}'
            os.makedirs(results_dir, exist_ok=True)
            
            print(f"\nARCHITECTURE: {arch['name']}")
            print(f"AE: {config['sparse_dim']} -> {config['dense_dim']} -> {config['sparse_dim']}")
            print(f"SAE: {config['dense_dim']} -> {arch['sae_hidden_dim']} -> {config['dense_dim']}")
            print(f"Results: {results_dir}")
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
            
            # Phase 2: Train SAEs with architecture-specific hidden dimension
            print("\n\nPHASE 2: TRAINING SPARSE AUTOENCODERS")
            print("-" * 40)
            
            for sparsity in config['sparsities']:
                print(f"\nTraining SAE for sparsity {sparsity}...")
                
                bottleneck = results[sparsity]['bottleneck']
                
                sae_params = train_sae(
                    activations=bottleneck,
                    input_dim=config['dense_dim'],
                    hidden_dim=arch['sae_hidden_dim'],  # Use architecture-specific dimension
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
            
            # Phase 3: Compute similarity metrics
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
                
                # RSA
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
            
            # Save individual results and create basic plots
            # (Abbreviated version - just key plots)
            
            # Add activation statistics
            for sparsity in config['sparsities']:
                sae_hidden = results[sparsity]['sae']['hidden']
                active_nodes_per_sample = np.sum(sae_hidden > 0, axis=1)
                activation_freq = np.mean(sae_hidden > 0, axis=0)
                
                results[sparsity]['sae']['active_nodes_mean'] = np.mean(active_nodes_per_sample)
                results[sparsity]['sae']['active_nodes_std'] = np.std(active_nodes_per_sample)
                results[sparsity]['sae']['dead_neurons'] = np.sum(activation_freq < 0.01)
            
            # Create results table
            table_data = []
            for sparsity in config['sparsities']:
                row = {
                    'Architecture': arch['name'],
                    'Sparsity': sparsity,
                    'AE Loss': f"{results[sparsity]['recon_loss']:.4f}",
                    'SAE MSE': f"{results[sparsity]['sae']['mse']:.4f}",
                    'SAE L0': f"{results[sparsity]['sae']['sparsity']['l0_norm']:.3f}",
                    'Mean Active': f"{results[sparsity]['sae']['active_nodes_mean']:.1f}",
                    'Dead Neurons': results[sparsity]['sae']['dead_neurons'],
                    'Linear CKA': f"{results[sparsity]['metrics']['linear_cka']:.3f}",
                    'Procrustes': f"{results[sparsity]['metrics']['procrustes']:.3f}",
                    'RSA': f"{results[sparsity]['metrics']['rsa']:.3f}"
                }
                table_data.append(row)
            
            df = pd.DataFrame(table_data)
            df.to_csv(f'{results_dir}/results_table.csv', index=False)
            
            print(f"\nRESULTS TABLE - {arch['name']}")
            print("=" * 80)
            print(df.to_string(index=False))
            print("=" * 80)
            
            return results
        
        # Run modified experiment
        exp_module.main = modified_main
        arch_results = exp_module.main()
        
        # Restore original main function
        exp_module.main = original_main
        
        # Store results
        all_results[arch['name']] = arch_results
        
        # Add to combined data
        for sparsity in [0.3, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]:
            row = {
                'Architecture': arch['name'],
                'Sparse_Dim': arch['sparse_dim'],
                'SAE_Hidden_Dim': arch['sae_hidden_dim'],
                'Sparsity': sparsity,
                'AE_Loss': arch_results[sparsity]['recon_loss'],
                'SAE_MSE': arch_results[sparsity]['sae']['mse'],
                'SAE_L0': arch_results[sparsity]['sae']['sparsity']['l0_norm'],
                'Mean_Active_Nodes': arch_results[sparsity]['sae']['active_nodes_mean'],
                'Dead_Neurons': arch_results[sparsity]['sae']['dead_neurons'],
                'Procrustes': arch_results[sparsity]['metrics']['procrustes']
            }
            combined_data.append(row)
    
    # Create combined analysis
    print(f"\n{'='*80}")
    print("CREATING COMBINED ANALYSIS")
    print(f"{'='*80}")
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(f'{master_results_dir}/combined_all_architectures.csv', index=False)
    
    # Summary comparison
    print("\nCOMBINED RESULTS SUMMARY")
    print("=" * 100)
    summary = combined_df.groupby(['Architecture', 'Sparsity']).agg({
        'Mean_Active_Nodes': 'mean',
        'Dead_Neurons': 'mean', 
        'Procrustes': 'mean'
    }).round(3)
    print(summary)
    print("=" * 100)
    
    print(f"\n{'='*80}")
    print("MULTI-DIMENSION EXPERIMENT COMPLETED!")
    print(f"{'='*80}")
    print(f"Master results directory: {master_results_dir}")
    print("Individual architecture results in subdirectories")
    print("Combined analysis in combined_all_architectures.csv")
    
    return all_results, combined_df

if __name__ == "__main__":
    results, combined_table = run_multi_dimension_experiment()