#!/usr/bin/env python3
"""
Quick test version of comprehensive experiment with reduced epochs
"""

import os
import sys

# Add paths
sys.path.append('src')
sys.path.append('config')
sys.path.append('experiments')

from comprehensive_superposition_sae_experiment import run_comprehensive_experiment

# Monkey patch the config to use fewer epochs for faster testing
import experiments.comprehensive_superposition_sae_experiment as exp_module

def run_quick_test():
    """Run with reduced epochs for quick testing."""
    
    # Override config in the function
    original_func = exp_module.run_comprehensive_experiment
    
    def quick_experiment():
        # Configuration with reduced epochs
        config = {
            # General
            'seed': 42,
            'sparsities': [0.8, 0.85, 0.9],
            
            # Autoencoder config
            'sparse_dim': 20,
            'dense_dim': 5,
            'num_samples': 1000,  # Reduced from 10000
            'num_epochs': 3,       # Reduced from 10
            'learning_rate': 0.01,
            'decay_factor': 0.7,
            
            # SAE config
            'sae_input_dim': 5,   
            'sae_hidden_dim': 20,  
            'sae_epochs': 10,      # Reduced from 50
            'sae_lr': 0.001,
            'sae_l1': 0.01,
        }
        
        # Temporarily store the config in module
        exp_module.config = config
        return original_func()
    
    # Patch the config inside the function
    import experiments.comprehensive_superposition_sae_experiment as mod
    old_run = mod.run_comprehensive_experiment
    
    def patched_run():
        import os
        import sys
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        # Configuration with reduced epochs
        config = {
            'seed': 42,
            'sparsities': [0.8, 0.85, 0.9],
            'sparse_dim': 20,
            'dense_dim': 5,
            'num_samples': 1000,  # Reduced
            'num_epochs': 3,      # Reduced
            'learning_rate': 0.01,
            'decay_factor': 0.7,
            'sae_input_dim': 5,   
            'sae_hidden_dim': 20,  
            'sae_epochs': 10,     # Reduced
            'sae_lr': 0.001,
            'sae_l1': 0.01,
        }
        
        # Create results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f'results/quick_test_{timestamp}'
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"QUICK TEST: COMPREHENSIVE SUPERPOSITION-SAE EXPERIMENT")
        print(f"{'='*80}")
        print(f"Using reduced samples/epochs for faster testing")
        print(f"Samples: {config['num_samples']}, AE Epochs: {config['num_epochs']}, SAE Epochs: {config['sae_epochs']}")
        
        # Import the functions we need
        from experiments.comprehensive_superposition_sae_experiment import (
            train_autoencoder_single_seed,
            train_sae_on_bottleneck,
            compute_all_similarity_metrics,
            plot_superposition_matrices,
            plot_weight_norms_comparison,
            plot_reconstruction_quality,
            plot_similarity_metrics_heatmap,
            create_metrics_table
        )
        
        # Store all results
        all_results = {}
        
        # Phase 1: Train autoencoders
        print(f"\n{'='*80}")
        print("PHASE 1: TRAINING AUTOENCODERS")
        print(f"{'='*80}")
        
        for sparsity in config['sparsities']:
            all_results[sparsity] = {}
            all_results[sparsity]['autoencoder'] = train_autoencoder_single_seed(sparsity, config)
        
        # Phase 2: Train SAEs
        print(f"\n{'='*80}")
        print("PHASE 2: TRAINING SPARSE AUTOENCODERS")
        print(f"{'='*80}")
        
        for sparsity in config['sparsities']:
            print(f"\nTraining SAE for sparsity {sparsity}...")
            bottleneck_acts = all_results[sparsity]['autoencoder']['bottleneck_activations']
            all_results[sparsity]['sae'] = train_sae_on_bottleneck(bottleneck_acts, config)
        
        # Phase 3: Compute metrics
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
        
        # Phase 4: Visualizations
        print(f"\n{'='*80}")
        print("PHASE 4: GENERATING VISUALIZATIONS")
        print(f"{'='*80}")
        
        plot_superposition_matrices(all_results, config['sparsities'], results_dir)
        plot_weight_norms_comparison(all_results, config['sparsities'], results_dir)
        plot_reconstruction_quality(all_results, config['sparsities'], results_dir)
        plot_similarity_metrics_heatmap(all_results, config['sparsities'], results_dir)
        metrics_df = create_metrics_table(all_results, config['sparsities'], results_dir)
        
        # Save config
        with open(f'{results_dir}/config.txt', 'w') as f:
            for key, value in config.items():
                f.write(f'{key}: {value}\n')
        
        print(f"\n{'='*80}")
        print("QUICK TEST COMPLETED!")
        print(f"{'='*80}")
        print(f"Results saved to: {results_dir}")
        
        return all_results, metrics_df
    
    mod.run_comprehensive_experiment = patched_run
    return mod.run_comprehensive_experiment()

if __name__ == "__main__":
    print("Running quick test with reduced epochs...")
    results, metrics = run_quick_test()
    print("\nQuick test completed!")