import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

# ========== NEW SAE EXPERIMENT START ==========
# This entire file is new and can be easily removed for rollback

from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import train_model, get_bottleneck_activations, train_sae
from analysis import (analyze_sae_performance, plot_activation_comparison, 
                     plot_sae_feature_correlation, run_sae_ablation_study)
from experiment_config import DEFAULT_CONFIG


def run_sae_experiment(config=None, sae_config=None):
    """Run SAE experiment on trained autoencoder representations.
    
    This experiment:
    1. Trains a standard autoencoder on sparse data
    2. Extracts bottleneck activations 
    3. Trains a Sparse Autoencoder on these activations
    4. Analyzes the SAE using RSA and other metrics
    
    Args:
        config: ExperimentConfig for base autoencoder (uses DEFAULT_CONFIG if None)
        sae_config: Dictionary with SAE-specific configuration
        
    Returns:
        Dictionary containing all experimental results
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Default SAE configuration
    if sae_config is None:
        sae_config = {
            'hidden_dim': 20,  # SAE hidden dimension (typically > bottleneck_dim)
            'num_epochs': 50,  # SAE training epochs
            'learning_rate': 1e-3,  # SAE learning rate
            'l1_penalty': 0.01,  # L1 sparsity penalty
            'run_ablation': False,  # Whether to run ablation study
            'ablation_hidden_dims': [10, 15, 20, 25, 30],
            'ablation_l1_penalties': [0.001, 0.01, 0.1]
        }
    
    print("Starting SAE Experiment")
    print("="*50)
    print(f"Base autoencoder config:")
    print(f"  Sparsity levels: {config.sparsity_levels}")
    print(f"  Sparse dimension: {config.sparse_dim}")
    print(f"  Dense dimension: {config.dense_dim}")
    print(f"SAE config:")
    print(f"  Hidden dimension: {sae_config['hidden_dim']}")
    print(f"  L1 penalty: {sae_config['l1_penalty']}")
    print(f"  Training epochs: {sae_config['num_epochs']}")
    print("="*50)
    
    # Set up random seed
    if hasattr(config, 'random_seed'):
        seed = config.random_seed
    else:
        now = datetime.now()
        seed = int(now.timestamp())
    
    # Create results directory
    results_dir = None
    if config.save_plots:
        results_dir = config.results_dir
        os.makedirs(results_dir, exist_ok=True)
        # Create SAE-specific subdirectory
        sae_results_dir = os.path.join(results_dir, 'sae_results')
        os.makedirs(sae_results_dir, exist_ok=True)
        print(f"SAE results will be saved to: {sae_results_dir}")
    else:
        sae_results_dir = None
    
    # Step 1: Train base autoencoder (use first sparsity level for simplicity)
    print("\n=== Step 1: Training Base Autoencoder ===")
    sparsity = config.sparsity_levels[0]
    print(f"Using sparsity level: {sparsity}")
    
    # Generate synthetic data
    data = generate_synthetic_data(seed, config.sparse_dim, sparsity, config.num_samples)
    empirical_sparsity = float(np.mean(data == 0))
    print(f"Empirical sparsity: {empirical_sparsity:.4f}")
    
    # Generate feature importances
    I = get_feature_importances(config.sparse_dim, config.decay_factor)
    
    # Train autoencoder
    print("Training autoencoder...")
    autoencoder_params = train_model(
        data=data,
        I=I,
        k=config.dense_dim,
        n=config.sparse_dim,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        seed=seed
    )
    
    # Step 2: Extract bottleneck activations
    print("\n=== Step 2: Extracting Bottleneck Activations ===")
    bottleneck_activations = get_bottleneck_activations(autoencoder_params, data)
    print(f"Bottleneck activations shape: {bottleneck_activations.shape}")
    print(f"Bottleneck statistics:")
    print(f"  Mean: {np.mean(bottleneck_activations):.4f}")
    print(f"  Std: {np.std(bottleneck_activations):.4f}")
    print(f"  Min: {np.min(bottleneck_activations):.4f}")
    print(f"  Max: {np.max(bottleneck_activations):.4f}")
    
    # Step 3: Train Sparse Autoencoder
    print("\n=== Step 3: Training Sparse Autoencoder ===")
    sae_params = train_sae(
        activations=bottleneck_activations,
        input_dim=config.dense_dim,
        hidden_dim=sae_config['hidden_dim'],
        num_epochs=sae_config['num_epochs'],
        learning_rate=sae_config['learning_rate'],
        l1_penalty=sae_config['l1_penalty'],
        seed=seed + 1000  # Different seed for SAE
    )
    
    # Step 4: Analyze SAE Performance
    print("\n=== Step 4: SAE Performance Analysis ===")
    sae_analysis = analyze_sae_performance(bottleneck_activations, sae_params, verbose=True)
    
    # Step 4b: Compute Comprehensive RSA Metrics
    print("\n=== Step 4b: Computing All RSA Metrics ===")
    
    # Get SAE reconstructions for metrics
    from models_numpy import sae_forward
    sae_result = sae_forward(sae_params, bottleneck_activations)
    sae_reconstructions = sae_result['recon']
    sae_hidden = sae_result['hidden']
    
    # Import additional RSA metrics if available
    rsa_metrics = {}
    
    # Basic RSA (already computed in analyze_sae_performance)
    rsa_metrics['rsa_correlation'] = sae_analysis['rsa_correlation']
    print(f"1. RSA Correlation: {rsa_metrics['rsa_correlation']:.4f}")
    
    # Try to import and compute CKA metrics
    try:
        from src.CKA import compute_linear_cka, compute_rbf_cka
        linear_cka = compute_linear_cka(bottleneck_activations, sae_reconstructions)
        rbf_cka = compute_rbf_cka(bottleneck_activations, sae_reconstructions)
        rsa_metrics['cka_linear'] = linear_cka
        rsa_metrics['cka_rbf'] = rbf_cka
        print(f"2. Linear CKA: {linear_cka:.4f}")
        print(f"3. RBF CKA: {rbf_cka:.4f}")
    except ImportError:
        print("2-3. CKA metrics: Not available (module not found)")
    
    # Try to import and compute CCA/SVCCA metrics
    try:
        from src.rsa_cca import compute_cca, cca_distance, svcca_similarity
        canonical_corrs = compute_cca(bottleneck_activations, sae_reconstructions)
        mean_cca = np.mean(canonical_corrs)
        cca_dist = cca_distance(bottleneck_activations, sae_reconstructions)
        svcca = svcca_similarity(bottleneck_activations, sae_reconstructions, threshold=0.99)
        rsa_metrics['mean_cca'] = mean_cca
        rsa_metrics['cca_distance'] = cca_dist
        rsa_metrics['svcca_similarity'] = svcca
        print(f"4. Mean CCA: {mean_cca:.4f}")
        print(f"5. CCA Distance: {cca_dist:.4f}")
        print(f"6. SVCCA Similarity: {svcca:.4f}")
        print(f"   Top 5 canonical correlations: {canonical_corrs[:5]}")
    except ImportError:
        print("4-6. CCA/SVCCA metrics: Not available (module not found)")
    
    # Try to import and compute Procrustes metrics
    try:
        from src.rsa_procrustes import orthogonal_procrustes, angular_procrustes_distance
        R, proc_dist, info = orthogonal_procrustes(bottleneck_activations, sae_reconstructions)
        angular_dist = angular_procrustes_distance(bottleneck_activations, sae_reconstructions)
        rsa_metrics['procrustes_distance'] = proc_dist
        rsa_metrics['angular_procrustes'] = angular_dist
        print(f"7. Procrustes Distance: {proc_dist:.4f}")
        print(f"8. Angular Procrustes: {angular_dist:.4f}")
    except ImportError:
        print("7-8. Procrustes metrics: Not available (module not found)")
    
    # Add metrics to sae_analysis
    sae_analysis['all_rsa_metrics'] = rsa_metrics
    
    print("\n=== RSA Metrics Summary ===")
    print(f"Metrics computed: {len(rsa_metrics)}")
    for metric, value in rsa_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    # Step 5: Generate Diagnostic Plots
    print("\n=== Step 5: Generating Diagnostic Plots ===")
    
    # Activation comparison plot
    save_path = f"{sae_results_dir}/sae_activation_comparison.png" if sae_results_dir else None
    plot_activation_comparison(bottleneck_activations, sae_params, save_path)
    
    # Feature correlation plot
    save_path = f"{sae_results_dir}/sae_feature_correlation.png" if sae_results_dir else None
    correlation_matrix = plot_sae_feature_correlation(bottleneck_activations, sae_params, save_path)
    
    # Step 6: Validation Checks
    print("\n=== Step 6: Validation Checks ===")
    
    # Check if implementation worked successfully
    success_criteria = {
        'rsa_correlation_reasonable': sae_analysis['rsa_correlation'] > 0.3,
        'rsa_self_check_passed': abs(sae_analysis['rsa_self_check'] - 1.0) < 0.1,
        'rsa_random_check_passed': abs(sae_analysis['rsa_random_baseline']) < 0.3,
        'sparsity_improved': sae_analysis['sparsity_improvement'] > 1.0,
        'reconstruction_reasonable': sae_analysis['reconstruction_mse'] < 1.0
    }
    
    print("Validation Results:")
    for criterion, passed in success_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {criterion}: {status}")
    
    overall_success = all(success_criteria.values())
    print(f"\nOverall SAE Implementation: {'✓ SUCCESS' if overall_success else '✗ NEEDS ATTENTION'}")
    
    # Step 7: Optional Ablation Study
    ablation_results = None
    if sae_config.get('run_ablation', False):
        print("\n=== Step 7: Running Ablation Study ===")
        ablation_results = run_sae_ablation_study(
            original_activations=bottleneck_activations,
            hidden_dims=sae_config['ablation_hidden_dims'],
            l1_penalties=sae_config['ablation_l1_penalties'],
            input_dim=config.dense_dim,
            num_epochs=20,  # Shorter for ablation
            save_dir=sae_results_dir
        )
        print("Ablation study completed!")
    
    # Compile final results
    results = {
        'autoencoder_params': autoencoder_params,
        'sae_params': sae_params,
        'original_data': data,
        'bottleneck_activations': bottleneck_activations,
        'sae_analysis': sae_analysis,
        'correlation_matrix': correlation_matrix,
        'success_criteria': success_criteria,
        'overall_success': overall_success,
        'ablation_results': ablation_results,
        'config': config,
        'sae_config': sae_config,
        'feature_importances': I
    }
    
    print(f"\nSAE Experiment completed!")
    if sae_results_dir:
        print(f"Results saved to: {sae_results_dir}")
    
    return results


def format_metrics_table(all_results, config):
    """Format RSA metrics as a pretty table."""
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False
    
    # Check which metrics are available
    first_result = all_results[config.sparsity_levels[0]]
    has_metrics = 'all_rsa_metrics' in first_result['sae_analysis']
    
    if not has_metrics:
        return None
    
    # Prepare data for table
    headers = ['Sparsity', 'Recon MSE', 'Sparse↑']
    available_metrics = list(first_result['sae_analysis']['all_rsa_metrics'].keys())
    
    # Add metric headers (shortened names)
    metric_names = {
        'rsa_correlation': 'RSA',
        'cka_linear': 'Lin CKA',
        'cka_rbf': 'RBF CKA',
        'mean_cca': 'Mean CCA',
        'cca_distance': 'CCA Dist↓',
        'svcca_similarity': 'SVCCA',
        'procrustes_distance': 'Procrust↓',
        'angular_procrustes': 'Angular↓'
    }
    
    for metric in available_metrics:
        headers.append(metric_names.get(metric, metric[:8]))
    
    # Collect data rows
    rows = []
    for sparsity in config.sparsity_levels:
        analysis = all_results[sparsity]['sae_analysis']
        row = [
            f"{sparsity:.2f}",
            f"{analysis['reconstruction_mse']:.4f}",
            f"{analysis['sparsity_improvement']:.2f}x"
        ]
        
        for metric in available_metrics:
            value = analysis['all_rsa_metrics'].get(metric, 0)
            row.append(f"{value:.4f}")
        
        rows.append(row)
    
    # Add summary row with best/worst indicators
    summary_row = ['Best@', '-', '-']
    for i, metric in enumerate(available_metrics):
        values = [all_results[s]['sae_analysis']['all_rsa_metrics'].get(metric, 0) 
                 for s in config.sparsity_levels]
        
        # Determine if higher or lower is better
        lower_better = 'distance' in metric or 'angular' in metric
        best_idx = values.index(min(values)) if lower_better else values.index(max(values))
        summary_row.append(f"s={config.sparsity_levels[best_idx]:.2f}")
    
    rows.append(['---'] * len(headers))  # Separator
    rows.append(summary_row)
    
    if use_tabulate:
        return tabulate(rows, headers=headers, tablefmt='grid', floatfmt='.4f')
    else:
        # Fallback to simple formatting
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 
                     for i in range(len(headers))]
        
        lines = []
        # Header
        header_line = ''.join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append('-' * sum(col_widths))
        
        # Data rows
        for row in rows:
            line = ''.join(f"{str(v):<{w}}" for v, w in zip(row, col_widths))
            lines.append(line)
        
        return '\n'.join(lines)


def plot_metrics_heatmap(all_results, config, save_path=None):
    """Create a heatmap showing how RSA metrics vary across sparsity levels."""
    
    # Check if metrics are available
    first_result = all_results[config.sparsity_levels[0]]
    if 'all_rsa_metrics' not in first_result['sae_analysis']:
        return None
    
    # Collect data for heatmap
    available_metrics = list(first_result['sae_analysis']['all_rsa_metrics'].keys())
    sparsities = config.sparsity_levels
    
    # Create matrix: rows = metrics, cols = sparsity levels
    data_matrix = []
    metric_names = []
    
    for metric in available_metrics:
        values = [all_results[s]['sae_analysis']['all_rsa_metrics'].get(metric, 0) 
                 for s in sparsities]
        data_matrix.append(values)
        
        # Use short names for display
        short_names = {
            'rsa_correlation': 'RSA Correlation',
            'cka_linear': 'Linear CKA',
            'cka_rbf': 'RBF CKA', 
            'mean_cca': 'Mean CCA',
            'cca_distance': 'CCA Distance',
            'svcca_similarity': 'SVCCA',
            'procrustes_distance': 'Procrustes Dist',
            'angular_procrustes': 'Angular Proc'
        }
        metric_names.append(short_names.get(metric, metric))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize each metric to 0-1 for better color comparison
    normalized_data = []
    for row in data_matrix:
        row_array = np.array(row)
        if np.max(row_array) - np.min(row_array) > 0:
            normalized = (row_array - np.min(row_array)) / (np.max(row_array) - np.min(row_array))
        else:
            normalized = np.ones_like(row_array) * 0.5
        normalized_data.append(normalized)
    
    # Create heatmap
    im = ax.imshow(normalized_data, cmap='RdYlGn', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(sparsities)))
    ax.set_xticklabels([f'{s:.2f}' for s in sparsities])
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names)
    
    # Add text annotations with actual values
    for i in range(len(metric_names)):
        for j in range(len(sparsities)):
            text = f'{data_matrix[i][j]:.3f}'
            ax.text(j, i, text, ha="center", va="center", 
                   color="black" if normalized_data[i][j] > 0.5 else "white",
                   fontsize=8)
    
    # Labels and title
    ax.set_xlabel('Sparsity Level')
    ax.set_ylabel('RSA Metrics')
    ax.set_title('RSA Metrics Across Sparsity Levels\n(Normalized by metric, actual values shown)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Normalized Value (0=min, 1=max for each metric)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics heatmap saved to: {save_path}")
    
    plt.show()
    
    return fig


def run_multi_sparsity_sae_experiment(config=None, sae_config=None):
    """Run SAE experiment across multiple sparsity levels.
    
    This extends the basic SAE experiment to analyze how SAE performance
    varies across different sparsity regimes, particularly focusing on
    the critical sparsity transition region (0.82-0.87).
    
    Args:
        config: ExperimentConfig for base autoencoder 
        sae_config: Dictionary with SAE-specific configuration
        
    Returns:
        Dictionary containing results for all sparsity levels
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if sae_config is None:
        sae_config = {
            'hidden_dim': 20,
            'num_epochs': 30,  # Shorter for multi-sparsity study
            'learning_rate': 1e-3,
            'l1_penalty': 0.01
        }
    
    print("Starting Multi-Sparsity SAE Experiment")
    print("="*60)
    print(f"Testing sparsity levels: {config.sparsity_levels}")
    print("="*60)
    
    all_results = {}
    
    for sparsity in config.sparsity_levels:
        print(f"\n{'='*20} SPARSITY {sparsity} {'='*20}")
        
        # Create temporary config for this sparsity level
        temp_config = type(config)()
        for attr in dir(config):
            if not attr.startswith('_'):
                setattr(temp_config, attr, getattr(config, attr))
        temp_config.sparsity_levels = [sparsity]
        
        # Run SAE experiment for this sparsity level
        results = run_sae_experiment(temp_config, sae_config)
        all_results[sparsity] = results
        
        # Print summary for this sparsity level
        analysis = results['sae_analysis']
        print(f"\nSparsity {sparsity} Summary:")
        print(f"  RSA Correlation: {analysis['rsa_correlation']:.4f}")
        print(f"  Reconstruction MSE: {analysis['reconstruction_mse']:.6f}")
        print(f"  Sparsity Improvement: {analysis['sparsity_improvement']:.2f}x")
        print(f"  Overall Success: {'✓' if results['overall_success'] else '✗'}")
    
    # Generate formatted table
    print(f"\n{'='*20} METRICS TABLE {'='*20}")
    
    table = format_metrics_table(all_results, config)
    if table:
        print("\n" + table + "\n")
        print("Note: ↑ = higher is better, ↓ = lower is better")
    
    # Generate summary comparison
    print(f"\n{'='*20} SUMMARY COMPARISON {'='*20}")
    
    # Check which metrics are available
    first_result = all_results[config.sparsity_levels[0]]
    has_extra_metrics = 'all_rsa_metrics' in first_result['sae_analysis']
    
    if has_extra_metrics:
        # Extended summary with all RSA metrics
        available_metrics = list(first_result['sae_analysis']['all_rsa_metrics'].keys())
        
        # Print header
        print(f"{'Sparsity':<10} {'Recon MSE':<12} {'Sparse Imp':<12}", end="")
        for metric in available_metrics[:4]:  # Show first 4 metrics to fit
            short_name = metric.replace('_', ' ').replace('correlation', 'corr')[:8]
            print(f" {short_name:<8}", end="")
        print()
        print("-" * 80)
        
        # Print values for each sparsity
        for sparsity in config.sparsity_levels:
            analysis = all_results[sparsity]['sae_analysis']
            print(f"{sparsity:<10.2f} {analysis['reconstruction_mse']:<12.6f} "
                  f"{analysis['sparsity_improvement']:<12.2f}", end="")
            
            if 'all_rsa_metrics' in analysis:
                for metric in available_metrics[:4]:
                    value = analysis['all_rsa_metrics'].get(metric, 0)
                    print(f" {value:<8.4f}", end="")
            print()
        
        # Identify trends
        print(f"\n{'='*20} METRIC TRENDS ACROSS SPARSITY {'='*20}")
        for metric in available_metrics:
            values = [all_results[s]['sae_analysis']['all_rsa_metrics'].get(metric, 0) 
                     for s in config.sparsity_levels]
            min_val = min(values)
            max_val = max(values)
            min_idx = values.index(min_val)
            max_idx = values.index(max_val)
            print(f"{metric:25s}: Min={min_val:.4f} @ s={config.sparsity_levels[min_idx]:.2f}, "
                  f"Max={max_val:.4f} @ s={config.sparsity_levels[max_idx]:.2f}")
        
        # Generate heatmap visualization
        if config.save_plots:
            print(f"\n{'='*20} GENERATING METRICS HEATMAP {'='*20}")
            sae_results_dir = os.path.join(config.results_dir, 'sae_results')
            heatmap_path = os.path.join(sae_results_dir, 'rsa_metrics_heatmap.png')
            plot_metrics_heatmap(all_results, config, heatmap_path)
    else:
        # Original summary (fallback)
        print(f"{'Sparsity':<10} {'RSA':<8} {'Recon MSE':<12} {'Sparse Imp':<12} {'Success':<8}")
        print("-" * 55)
        
        for sparsity in config.sparsity_levels:
            analysis = all_results[sparsity]['sae_analysis']
            success = "✓" if all_results[sparsity]['overall_success'] else "✗"
            print(f"{sparsity:<10.2f} {analysis['rsa_correlation']:<8.4f} "
                  f"{analysis['reconstruction_mse']:<12.6f} "
                  f"{analysis['sparsity_improvement']:<12.2f} {success:<8}")
    
    return all_results


# Simple function to test SAE implementation quickly
def test_sae_implementation(verbose=True):
    """Quick test to verify SAE implementation works correctly.
    
    Args:
        verbose: whether to print detailed output
        
    Returns:
        Boolean indicating if test passed
    """
    print("Running SAE Implementation Test...")
    
    # Create minimal test configuration
    test_config = DEFAULT_CONFIG
    test_config.sparsity_levels = [0.7]  # Single sparsity level
    test_config.num_samples = 1000       # Small dataset
    test_config.num_epochs = 5           # Quick training
    test_config.save_plots = False       # No plots for test
    
    sae_config = {
        'hidden_dim': 10,
        'num_epochs': 20,
        'learning_rate': 1e-3,
        'l1_penalty': 0.01,
        'run_ablation': False
    }
    
    try:
        results = run_sae_experiment(test_config, sae_config)
        success = results['overall_success']
        
        if verbose:
            if success:
                print("✓ SAE implementation test PASSED!")
            else:
                print("✗ SAE implementation test FAILED!")
                print("Check the validation criteria in the detailed output above.")
        
        return success
        
    except Exception as e:
        if verbose:
            print(f"✗ SAE implementation test FAILED with error: {e}")
        return False


if __name__ == "__main__":
    # Run a quick test by default
    test_sae_implementation()

# ========== NEW SAE EXPERIMENT END ==========