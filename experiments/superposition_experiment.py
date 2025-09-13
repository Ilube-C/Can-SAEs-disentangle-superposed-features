import os
import sys
from datetime import datetime
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import train_model
from analysis import analyze_sparsity_effects, plot_data_histograms
from experiment_config import DEFAULT_CONFIG


def run_superposition_experiment(config=None):
    """Run the main superposition experiment.
    
    Args:
        config: ExperimentConfig object, uses DEFAULT_CONFIG if None
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    print("Starting Superposition Experiment")
    print("="*50)
    print(f"Sparsity levels: {config.sparsity_levels}")
    print(f"Sparse dimension: {config.sparse_dim}")
    print(f"Dense dimension: {config.dense_dim}")
    print(f"Number of samples: {config.num_samples}")
    print(f"Training epochs: {config.num_epochs}")
    print("="*50)
    
    # Set up random seed
    if hasattr(config, 'random_seed'):
        seed = config.random_seed
    else:
        # Use timestamp if no seed specified
        now = datetime.now()
        seed = int(now.timestamp())
    
    # Generate synthetic data for each sparsity level
    print("Generating synthetic data...")
    synth_data = []
    for s in config.sparsity_levels:
        data = generate_synthetic_data(seed, config.sparse_dim, s, config.num_samples)
        synth_data.append(data)
        
        # Check empirical sparsity
        empirical_sparsity = float(np.mean(data == 0))
        print(f"Sparsity = {s:.1f}, Empirical sparsity = {empirical_sparsity:.4f}")
    
    # Generate feature importances
    I = get_feature_importances(config.sparse_dim, config.decay_factor)
    print(f"Feature importances shape: {I.shape}")
    
    # Create results directory if it doesn't exist and we're saving plots
    results_dir = None
    if config.save_plots:
        results_dir = config.results_dir
        os.makedirs(results_dir, exist_ok=True)
        print(f"Saving results to: {results_dir}")
    
    # Plot data histograms
    print("\nPlotting data distributions...")
    plot_data_histograms(synth_data, config.sparsity_levels, results_dir)
    
    # Train models for each sparsity level
    print("\nTraining models...")
    models = []
    for i, (data, sparsity) in enumerate(zip(synth_data, config.sparsity_levels)):
        print(f"\nTraining model for sparsity = {sparsity}")
        
        # Use different seed for each model
        model_seed = seed + i
        
        params = train_model(
            data=data,
            I=I,
            k=config.dense_dim,
            n=config.sparse_dim,
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            seed=model_seed
        )
        models.append(params)
    
    # Analyze superposition effects
    print("\nAnalyzing superposition effects...")
    analysis_results = analyze_sparsity_effects(models, config.sparsity_levels, results_dir)
    
    print("\nExperiment completed successfully!")
    return {
        'models': models,
        'synth_data': synth_data,
        'feature_importances': I,
        'analysis_results': analysis_results,
        'config': config
    }


if __name__ == "__main__":
    # Run with default configuration
    results = run_superposition_experiment()