#!/usr/bin/env python3
"""
Main entry point for superposition research experiments.
"""

import os
import sys
import argparse

# Add local modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))

from experiment_config import ExperimentConfig, DEFAULT_CONFIG
from superposition_experiment import run_superposition_experiment
# ========== NEW SAE IMPORT START ==========
from sae_experiment import run_sae_experiment, run_multi_sparsity_sae_experiment, test_sae_implementation
# ========== NEW SAE IMPORT END ==========


def create_custom_config(args):
    """Create experiment config from command line arguments."""
    config = ExperimentConfig()
    
    if args.sparsity:
        config.sparsity_levels = args.sparsity
    if args.sparse_dim:
        config.sparse_dim = args.sparse_dim
    if args.dense_dim:
        config.dense_dim = args.dense_dim
    if args.num_samples:
        config.num_samples = args.num_samples
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.random_seed:
        config.random_seed = args.random_seed
    if args.results_dir:
        config.results_dir = args.results_dir
    
    config.save_plots = not args.no_plots
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Run superposition research experiments')
    
    # ========== NEW SAE ARGUMENTS START ==========
    # Experiment type selection
    parser.add_argument('--experiment-type', type=str, default='superposition',
                       choices=['superposition', 'sae', 'sae-multi', 'sae-test'],
                       help='Type of experiment to run (default: superposition)')
    # ========== NEW SAE ARGUMENTS END ==========
    
    # Experiment parameters
    parser.add_argument('--sparsity', type=float, nargs='+', 
                       help='Sparsity levels to test (default: [0.1, 0.3, 0.7])')
    parser.add_argument('--sparse-dim', type=int, default=20,
                       help='Sparse dimension (default: 20)')
    parser.add_argument('--dense-dim', type=int, default=5,
                       help='Dense/bottleneck dimension (default: 5)')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of training samples (default: 10000)')
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=1e-2,
                       help='Learning rate (default: 0.01)')
    
    # ========== NEW SAE PARAMETERS START ==========
    # SAE-specific parameters
    parser.add_argument('--sae-hidden-dim', type=int, default=20,
                       help='SAE hidden dimension (default: 20)')
    parser.add_argument('--sae-epochs', type=int, default=50,
                       help='SAE training epochs (default: 50)')
    parser.add_argument('--sae-lr', type=float, default=1e-3,
                       help='SAE learning rate (default: 0.001)')
    parser.add_argument('--sae-l1', type=float, default=0.01,
                       help='SAE L1 penalty (default: 0.01)')
    parser.add_argument('--sae-ablation', action='store_true',
                       help='Run SAE ablation study')
    # ========== NEW SAE PARAMETERS END ==========
    
    # Other parameters
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory (default: results)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable saving plots')
    
    args = parser.parse_args()
    
    # Create configuration
    if any([args.sparsity, args.sparse_dim != 20, args.dense_dim != 5, 
            args.num_samples != 10000, args.num_epochs != 10, 
            args.learning_rate != 1e-2, args.random_seed != 42,
            args.results_dir != 'results', args.no_plots]):
        config = create_custom_config(args)
        print("Using custom configuration")
    else:
        config = DEFAULT_CONFIG
        print("Using default configuration")
    
    # ========== NEW SAE EXPERIMENT LOGIC START ==========
    # Run experiment based on type
    if args.experiment_type == 'superposition':
        # Original superposition experiment
        results = run_superposition_experiment(config)
        print(f"\nResults saved to: {config.results_dir}")
        print("Superposition experiment completed successfully!")
        
    elif args.experiment_type == 'sae-test':
        # Quick SAE implementation test
        print("Running SAE implementation test...")
        success = test_sae_implementation(verbose=True)
        if success:
            print("\n✓ SAE implementation is working correctly!")
        else:
            print("\n✗ SAE implementation needs attention. Check the output above.")
        return
        
    elif args.experiment_type == 'sae':
        # Single SAE experiment
        sae_config = {
            'hidden_dim': args.sae_hidden_dim,
            'num_epochs': args.sae_epochs,
            'learning_rate': args.sae_lr,
            'l1_penalty': args.sae_l1,
            'run_ablation': args.sae_ablation,
            'ablation_hidden_dims': [10, 15, 20, 25, 30],
            'ablation_l1_penalties': [0.001, 0.01, 0.1]
        }
        
        results = run_sae_experiment(config, sae_config)
        
        if results['overall_success']:
            print(f"\n✓ SAE experiment completed successfully!")
        else:
            print(f"\n⚠ SAE experiment completed with some validation failures.")
        
        print(f"Results saved to: {config.results_dir}/sae_results")
        
    elif args.experiment_type == 'sae-multi':
        # Multi-sparsity SAE experiment
        sae_config = {
            'hidden_dim': args.sae_hidden_dim,
            'num_epochs': args.sae_epochs,
            'learning_rate': args.sae_lr,
            'l1_penalty': args.sae_l1
        }
        
        results = run_multi_sparsity_sae_experiment(config, sae_config)
        
        # Count successful sparsity levels
        successes = sum(1 for r in results.values() if r['overall_success'])
        total = len(results)
        
        print(f"\n✓ Multi-sparsity SAE experiment completed!")
        print(f"Success rate: {successes}/{total} sparsity levels passed validation")
        print(f"Results saved to: {config.results_dir}/sae_results")
    
    # ========== NEW SAE EXPERIMENT LOGIC END ==========


if __name__ == "__main__":
    main()