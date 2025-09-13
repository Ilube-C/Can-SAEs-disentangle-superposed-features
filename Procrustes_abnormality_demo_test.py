"""
Procrustes Abnormality Demo with Linear CKA Analysis and Superposition Matrix Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from src.models_numpy import init_params, train_model, init_sae_params, train_sae, get_bottleneck_activations, sae_forward
from src.data_generation import generate_synthetic_data, get_feature_importances
from src.analysis import compute_superposition_matrix
from CKA import linear_cka, rbf_cka
from src.rsa_procrustes import procrustes_similarity


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
        
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")
    
    return importances.reshape(1, -1)



def train_autoencoder(sparsity, seed, n=20, k=5, num_samples=10000, num_epochs=10, test_ratio=0.2, importance_pattern="exponential", **importance_kwargs):
    """Train an autoencoder with train/test split for proper evaluation."""
    # Generate total data (train + test)
    total_samples = int(num_samples / (1 - test_ratio))
    X_total = generate_synthetic_data(seed, n, sparsity, total_samples)
    I = get_custom_feature_importances(n, importance_pattern, **importance_kwargs)
    
    # Split into train/test
    n_test = int(total_samples * test_ratio)
    n_train = total_samples - n_test
    
    # Use numpy random state for reproducible splits
    np.random.seed(seed + 1000)
    indices = np.random.permutation(total_samples)
    
    train_indices = indices[:n_train] 
    test_indices = indices[n_train:]
    
    X_train = X_total[train_indices]
    X_test = X_total[test_indices]
    
    print(f"  AE data split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train model on training data only
    params, train_loss = train_model(
        X_train, I, k, n,
        num_epochs=num_epochs,
        learning_rate=0.01,
        seed=seed
    )
    
    # Evaluate on test data
    from src.models_numpy import forward
    test_reconstructions = forward(params, X_test)
    test_loss = np.mean(I * (X_test - test_reconstructions) ** 2)
    
    print(f"  AE Test loss: {test_loss:.6f}")
    
    return params, test_loss, X_train, X_test, I


# Use the imported get_bottleneck_activations from models_numpy

def train_sae_on_activations(activations, hidden_dim=20, lam=0.1, seed=42, num_epochs=10, test_ratio=0.2):
    """Train a Sparse Autoencoder on bottleneck activations with train/test split."""
    input_dim = activations.shape[1]
    total_samples = activations.shape[0]
    
    # Split activations into train/test
    n_test = int(total_samples * test_ratio)
    n_train = total_samples - n_test
    
    # Use numpy random state for reproducible splits
    np.random.seed(seed + 2000)
    indices = np.random.permutation(total_samples)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    activations_train = activations[train_indices]
    activations_test = activations[test_indices]
    
    print(f"  SAE data split: {len(activations_train)} train, {len(activations_test)} test activations")
    
    # Train SAE on training activations only
    sae_params, train_loss = train_sae(
        activations_train,
        input_dim,
        hidden_dim,
        num_epochs=num_epochs,
        learning_rate=0.001,
        lam=lam,
        seed=seed,
        verbose=False
    )
    
    # Evaluate on test activations
    test_outputs = sae_forward(sae_params, activations_test)
    test_reconstructions = test_outputs['recon']
    test_recon_error = np.mean((activations_test - test_reconstructions) ** 2)
    
    print(f"  SAE Test reconstruction error: {test_recon_error:.6f}")
    
    return sae_params, test_recon_error, activations_train, activations_test


def plot_superposition_matrices_and_angles(results, sparsities, architectures, results_dir):
    """Plot superposition matrices with bias vectors."""
    for arch in architectures:
        n_sparsities = len(sparsities)
        # Create single row layout for superposition matrices only
        fig, axes = plt.subplots(1, n_sparsities, figsize=(6*n_sparsities, 6))
        
        # Handle single sparsity case
        if n_sparsities == 1:
            axes = [axes]
        
        for idx, sparsity in enumerate(sparsities):
            key = f"{arch['name']}_s{sparsity}"
            params = results[key]['ae_params']
            W = params['W']  # Shape: (hidden_dim, sparse_dim)
            b = params['b']  # Get bias vector
            
            # === SUPERPOSITION MATRICES ===
            superposition_matrix = W.T @ W
            
            # Create combined matrix: bias vector as leftmost column + superposition matrix
            bias_column = b.reshape(-1, 1)
            combined_matrix = np.hstack([bias_column, superposition_matrix])
            
            # Plot superposition matrix
            im1 = axes[idx].imshow(combined_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            # Add info to superposition matrix title
            diag_mean = np.mean(np.diag(superposition_matrix))
            off_diag = superposition_matrix[~np.eye(superposition_matrix.shape[0], dtype=bool)]
            off_diag_mean = np.mean(np.abs(off_diag)) if len(off_diag) > 0 else 0
            bias_stats = f"b: {np.mean(b):.2f}±{np.std(b):.2f}"
            
            axes[idx].set_title(f'Superposition Matrix\nSparsity {sparsity} | Loss: {results[key]["ae_test_loss"]:.4f}\nDiag: {diag_mean:.3f}, Off: {off_diag_mean:.3f} | {bias_stats}', 
                                  fontsize=9)
            
            # Set up x-axis labels for superposition matrix
            if arch['sparse_dim'] <= 30:
                x_labels = ['b'] + [str(i) for i in range(arch['sparse_dim'])]
                axes[idx].set_xticks(range(len(x_labels)))
                axes[idx].set_xticklabels(x_labels, rotation=0 if arch['sparse_dim'] <= 10 else 45)
                axes[idx].set_yticks(range(arch['sparse_dim']))
                axes[idx].set_yticklabels(range(arch['sparse_dim']))
                axes[idx].axvline(x=0.5, color='black', linewidth=2, alpha=0.7)
                axes[idx].grid(True, alpha=0.3, linewidth=0.5)
            
            axes[idx].set_xlabel('Feature (b | W^T@W columns)')
            axes[idx].set_ylabel('Feature')
            plt.colorbar(im1, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # Add overall title
        fig.suptitle(f'Superposition Analysis - Architecture {arch["name"]}\nSuperposition Matrices (W^T@W with bias)', 
                     fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Make room for suptitle
        
        # Save combined plot
        arch_results_dir = os.path.join(results_dir, arch['name'])
        os.makedirs(arch_results_dir, exist_ok=True)
        plt.savefig(f'{arch_results_dir}/superposition_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved superposition matrices plot for {arch['name']}")


def plot_metrics_heatmap(results, sparsities, architectures, results_dir):
    """Plot metrics heatmap for all architectures with mean ± std annotations."""
    for arch in architectures:
        # Test-based metrics only
        metrics_names = ['Linear CKA', 'RBF CKA', 'Procrustes', 'RSA', 
                        'AE Test Loss', 'SAE Test Recon', 'Diag Strength', 'Off-Diag Inter', 'SAE L0']
        metrics_keys = ['linear_cka', 'rbf_cka', 'procrustes', 'rsa',
                       'ae_test_loss', 'sae_test_recon_error', 'diag_strength', 'off_diag_interference', 'sae_l0']
        
        data_matrix = []
        annotation_matrix = []
        
        for sparsity in sparsities:
            key = f"{arch['name']}_s{sparsity}"
            row = []
            annot_row = []
            
            for metric_key in metrics_keys:
                # Get mean and std values
                if metric_key in ['diag_strength', 'off_diag_interference', 'sae_l0']:
                    # These are stored in analysis_metrics
                    mean_val = results[key].get('analysis_metrics', {}).get(metric_key, np.nan)
                    std_val = results[key].get('analysis_metrics_std', {}).get(metric_key, np.nan)
                elif metric_key in ['ae_test_loss', 'sae_test_recon_error']:
                    # These are stored directly
                    mean_val = results[key].get(metric_key, np.nan)
                    std_val = results[key].get(f"{metric_key}_std", np.nan)
                    # Invert loss values for heatmap (lower loss = higher score)
                    if not np.isnan(mean_val) and mean_val > 0:
                        mean_val = 1.0 / (1.0 + mean_val)  # Transform to [0,1] range where higher = better
                        # Note: std transformation is approximate for visualization
                        if not np.isnan(std_val):
                            std_val = std_val / (1.0 + results[key].get(metric_key, 1.0))**2
                else:
                    # Similarity metrics
                    mean_val = results[key]['metrics'].get(metric_key, np.nan)
                    std_val = results[key]['metrics_std'].get(metric_key, np.nan)
                
                # Create annotation string with mean ± std
                if np.isnan(mean_val):
                    annot_str = "NaN"
                    display_val = 0
                elif np.isnan(std_val):
                    annot_str = f"{mean_val:.3f}"
                    display_val = mean_val
                else:
                    annot_str = f"{mean_val:.3f}\n±{std_val:.3f}"
                    display_val = mean_val
                    
                row.append(display_val if not np.isnan(display_val) else 0)
                annot_row.append(annot_str)
                
            data_matrix.append(row)
            annotation_matrix.append(annot_row)
        
        data_matrix = np.array(data_matrix).T
        annotation_matrix = np.array(annotation_matrix).T
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(data_matrix, 
                    xticklabels=[f'{s}' for s in sparsities],
                    yticklabels=metrics_names,
                    annot=annotation_matrix, 
                    fmt='',
                    cmap='YlOrRd',
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'Normalized Score (Mean)'},
                    annot_kws={'fontsize': 8})
        
        plt.title(f'Extended Similarity & Analysis Metrics (Mean ± Std)\nArchitecture: {arch["name"]} | Seeds: {len(seeds)}', 
                  fontsize=12, fontweight='bold')
        plt.xlabel('Sparsity Level')
        plt.ylabel('Metric')
        plt.tight_layout()
        
        # Save plot
        arch_results_dir = os.path.join(results_dir, arch['name'])
        os.makedirs(arch_results_dir, exist_ok=True)
        plt.savefig(f'{arch_results_dir}/metrics_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved metrics heatmap for {arch['name']}")


def permutation_test_significance(ae_representations, sae_representations, n_permutations=500):
    """
    Perform permutation test to assess statistical significance of similarity metrics.
    
    Shuffles SAE representations n_permutations times to build null distribution,
    then computes p-values and z-scores for all similarity metrics.
    
    Args:
        ae_representations: AE test bottleneck activations (n_samples, n_features)
        sae_representations: SAE reconstructions (n_samples, n_features)
        n_permutations: Number of permutations for null distribution
    
    Returns:
        dict: Contains observed scores, null distributions, p-values, and z-scores
    """
    print(f"Running permutation test with {n_permutations} permutations...")
    
    # Compute observed scores
    observed_scores = {}
    
    # Linear CKA
    try:
        observed_scores['linear_cka'] = linear_cka(ae_representations, sae_representations)
    except:
        observed_scores['linear_cka'] = np.nan
        
    # RBF CKA
    try:
        observed_scores['rbf_cka'] = rbf_cka(ae_representations, sae_representations)
    except:
        observed_scores['rbf_cka'] = np.nan
        
    # Procrustes
    try:
        observed_scores['procrustes'] = procrustes_similarity(ae_representations, sae_representations)
    except:
        observed_scores['procrustes'] = np.nan
        
    # RSA
    try:
        dist1 = pdist(ae_representations, metric='euclidean')
        dist2 = pdist(sae_representations, metric='euclidean')
        observed_scores['rsa'], _ = pearsonr(dist1, dist2)
    except:
        observed_scores['rsa'] = np.nan
    
    # Initialize null distributions
    null_distributions = {metric: [] for metric in observed_scores.keys()}
    
    # Run permutations
    np.random.seed(42)  # For reproducibility
    for i in range(n_permutations):
        if i % 100 == 0:
            print(f"  Permutation {i}/{n_permutations}")
        
        # Shuffle SAE representations (permute rows)
        perm_indices = np.random.permutation(len(sae_representations))
        sae_permuted = sae_representations[perm_indices]
        
        # Compute metrics for permuted data
        # Linear CKA
        try:
            null_cka_linear = linear_cka(ae_representations, sae_permuted)
            null_distributions['linear_cka'].append(null_cka_linear)
        except:
            pass
            
        # RBF CKA
        try:
            null_cka_rbf = rbf_cka(ae_representations, sae_permuted)
            null_distributions['rbf_cka'].append(null_cka_rbf)
        except:
            pass
            
        # Procrustes
        try:
            null_procrustes = procrustes_similarity(ae_representations, sae_permuted)
            null_distributions['procrustes'].append(null_procrustes)
        except:
            pass
            
        # RSA
        try:
            dist1 = pdist(ae_representations, metric='euclidean')
            dist2_perm = pdist(sae_permuted, metric='euclidean')
            null_rsa, _ = pearsonr(dist1, dist2_perm)
            null_distributions['rsa'].append(null_rsa)
        except:
            pass
    
    # Compute statistics
    results = {
        'observed_scores': observed_scores,
        'null_distributions': null_distributions,
        'p_values': {},
        'z_scores': {},
        'n_permutations': n_permutations
    }
    
    for metric in observed_scores.keys():
        if len(null_distributions[metric]) > 0:
            null_array = np.array(null_distributions[metric])
            observed = observed_scores[metric]
            
            if not np.isnan(observed):
                # P-value: fraction of null scores >= observed score
                p_value = np.mean(null_array >= observed)
                results['p_values'][metric] = p_value
                
                # Z-score: (observed - null_mean) / null_std
                null_mean = np.mean(null_array)
                null_std = np.std(null_array)
                if null_std > 0:
                    z_score = (observed - null_mean) / null_std
                    results['z_scores'][metric] = z_score
                else:
                    results['z_scores'][metric] = np.nan
            else:
                results['p_values'][metric] = np.nan
                results['z_scores'][metric] = np.nan
        else:
            results['p_values'][metric] = np.nan
            results['z_scores'][metric] = np.nan
    
    return results


def lambda_to_l0_sweep(sparsity=0.85, seed=123, n=20, k=5, num_samples=2000, ae_epochs=5, sae_epochs=10, target_l0s=[1, 2, 3, 4]):
    """
    Sweep lambda values to achieve target L0 sparsity levels for SAE.
    
    Uses architecture 20-5-20 near critical sparsity (0.85) to test whether we can
    control SAE sparsity via the lambda penalty parameter and observe expected behavior.
    
    Args:
        sparsity: Data sparsity level (default near critical point)
        seed: Random seed for reproducibility
        n: Sparse dimension (input/output size)
        k: Dense dimension (bottleneck size) 
        num_samples: Number of training samples
        ae_epochs: AE training epochs
        sae_epochs: SAE training epochs
        target_l0s: Target mean L0 values to achieve
    
    Returns:
        dict: Results for each lambda value tried
    """
    print(f"Running Lambda -> L0 sweep for sparsity {sparsity}, target L0s: {target_l0s}")
    
    # Train autoencoder first
    print("Training autoencoder...")
    params, ae_test_loss, X_train, X_test, I = train_autoencoder(
        sparsity=sparsity,
        seed=seed,
        n=n,
        k=k,
        num_samples=num_samples,
        num_epochs=ae_epochs
    )
    
    # Get bottleneck activations
    bottleneck_activations_test = get_bottleneck_activations(params, X_test)
    print(f"Bottleneck activations shape: {bottleneck_activations_test.shape}")
    
    # Split for SAE training 
    n_sae_test = len(bottleneck_activations_test) // 5
    n_sae_train = len(bottleneck_activations_test) - n_sae_test
    
    np.random.seed(seed + 2000)
    sae_indices = np.random.permutation(len(bottleneck_activations_test))
    sae_train_indices = sae_indices[:n_sae_train]
    sae_test_indices = sae_indices[n_sae_train:]
    
    activations_train = bottleneck_activations_test[sae_train_indices]
    activations_test = bottleneck_activations_test[sae_test_indices]
    
    # Define lambda values to try - start with broader range then narrow down
    lambda_candidates = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    results = {}
    achieved_l0s = {}
    
    print(f"Testing lambda values: {lambda_candidates}")
    
    # Test each lambda value
    for lam in lambda_candidates:
        print(f"\nTesting lambda = {lam}")
        
        # Train SAE with this lambda
        sae_params, sae_train_loss = train_sae(
            activations_train,
            activations_train.shape[1], 
            n,  # sae_hidden_dim = sparse_dim
            num_epochs=sae_epochs,
            learning_rate=0.001,
            lam=lam,
            seed=seed,
            verbose=False
        )
        
        # Evaluate on test set
        sae_test_outputs = sae_forward(sae_params, activations_test)
        sae_reconstructions = sae_test_outputs['recon']
        sae_hidden = sae_test_outputs['hidden']
        
        # Compute L0 norm (mean number of active neurons per sample)
        threshold = 1e-6
        active_per_sample = np.sum(sae_hidden > threshold, axis=1)
        mean_l0 = np.mean(active_per_sample)
        
        # Compute similarity metrics
        try:
            linear_cka_score = linear_cka(activations_test, sae_reconstructions)
        except:
            linear_cka_score = np.nan
            
        try:
            rbf_cka_score = rbf_cka(activations_test, sae_reconstructions)
        except:
            rbf_cka_score = np.nan
            
        try:
            procrustes_score = procrustes_similarity(activations_test, sae_reconstructions)
        except:
            procrustes_score = np.nan
            
        try:
            dist1 = pdist(activations_test, metric='euclidean')
            dist2 = pdist(sae_reconstructions, metric='euclidean')
            rsa_score, _ = pearsonr(dist1, dist2)
        except:
            rsa_score = np.nan
        
        # Reconstruction error
        recon_error = np.mean((activations_test - sae_reconstructions) ** 2)
        
        print(f"  lambda={lam}: Mean L0={mean_l0:.2f}, Recon Error={recon_error:.6f}")
        print(f"    Linear CKA={linear_cka_score:.3f}, Procrustes={procrustes_score:.3f}")
        
        results[lam] = {
            'lambda': lam,
            'mean_l0': mean_l0,
            'recon_error': recon_error,
            'linear_cka': linear_cka_score,
            'rbf_cka': rbf_cka_score,
            'procrustes': procrustes_score,
            'rsa': rsa_score,
            'sae_params': sae_params
        }
        
        achieved_l0s[lam] = mean_l0
    
    # Find best lambda for each target L0
    print(f"\nFinding best lambda values for target L0s: {target_l0s}")
    best_matches = {}
    
    for target_l0 in target_l0s:
        # Find lambda that gives closest L0 to target
        best_lambda = None
        best_diff = float('inf')
        
        for lam, achieved_l0 in achieved_l0s.items():
            diff = abs(achieved_l0 - target_l0)
            if diff < best_diff:
                best_diff = diff
                best_lambda = lam
        
        if best_lambda is not None:
            best_matches[target_l0] = {
                'lambda': best_lambda,
                'achieved_l0': achieved_l0s[best_lambda],
                'difference': best_diff,
                'results': results[best_lambda]
            }
            print(f"  Target L0={target_l0}: Best lambda={best_lambda} (achieved L0={achieved_l0s[best_lambda]:.2f}, diff={best_diff:.2f})")
    
    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    lambdas = sorted(results.keys())
    l0_values = [results[lam]['mean_l0'] for lam in lambdas]
    recon_errors = [results[lam]['recon_error'] for lam in lambdas]
    linear_cka_values = [results[lam]['linear_cka'] for lam in lambdas]
    procrustes_values = [results[lam]['procrustes'] for lam in lambdas]
    
    # L0 vs lambda
    ax1.semilogx(lambdas, l0_values, 'bo-')
    ax1.set_xlabel('Lambda (log scale)')
    ax1.set_ylabel('Mean L0 (Active Neurons)')
    ax1.set_title('SAE Sparsity vs Lambda Parameter')
    ax1.grid(True, alpha=0.3)
    
    # Add target L0 lines
    for target_l0 in target_l0s:
        ax1.axhline(y=target_l0, color='red', linestyle='--', alpha=0.7, label=f'Target L0={target_l0}')
    ax1.legend()
    
    # Reconstruction error vs lambda  
    ax2.semilogx(lambdas, recon_errors, 'ro-')
    ax2.set_xlabel('Lambda (log scale)')
    ax2.set_ylabel('Reconstruction Error')
    ax2.set_title('SAE Reconstruction Error vs Lambda')
    ax2.grid(True, alpha=0.3)
    
    # Linear CKA vs L0
    valid_cka = [(l0, cka) for l0, cka in zip(l0_values, linear_cka_values) if not np.isnan(cka)]
    if valid_cka:
        l0_vals, cka_vals = zip(*valid_cka)
        ax3.plot(l0_vals, cka_vals, 'go-')
    ax3.set_xlabel('Mean L0')
    ax3.set_ylabel('Linear CKA')
    ax3.set_title('Linear CKA vs SAE Sparsity')
    ax3.grid(True, alpha=0.3)
    
    # Procrustes vs L0
    valid_proc = [(l0, proc) for l0, proc in zip(l0_values, procrustes_values) if not np.isnan(proc)]
    if valid_proc:
        l0_vals, proc_vals = zip(*valid_proc)
        ax4.plot(l0_vals, proc_vals, 'mo-')
    ax4.set_xlabel('Mean L0')
    ax4.set_ylabel('Procrustes Similarity')
    ax4.set_title('Procrustes vs SAE Sparsity')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return {
        'lambda_results': results,
        'best_matches': best_matches,
        'target_l0s': target_l0s,
        'figure': fig
    }


def rotation_baseline_test(sparsity, seed, n=20, k=5, num_samples=2000, ae_epochs=10, sae_epochs=20):
    """
    Rotation baseline control experiment.
    
    Applies random orthogonal rotation to AE representations, then trains SAE
    to test against alignment by chance.
    
    Args:
        sparsity: Data sparsity level
        seed: Random seed
        n, k: Architecture dimensions
        num_samples, ae_epochs, sae_epochs: Training parameters
    
    Returns:
        dict: Comparison of true SAE vs rotation baseline metrics
    """
    print(f"Running rotation baseline test at sparsity {sparsity}")
    
    # Train autoencoder first
    print("Training autoencoder...")
    params, ae_test_loss, X_train, X_test, I = train_autoencoder(
        sparsity=sparsity,
        seed=seed,
        n=n,
        k=k,
        num_samples=num_samples,
        num_epochs=ae_epochs,
        test_ratio=0.2
    )
    
    # Get bottleneck activations
    bottleneck_activations_test = get_bottleneck_activations(params, X_test)
    print(f"Bottleneck activations shape: {bottleneck_activations_test.shape}")
    
    # Split for SAE training 
    n_sae_test = len(bottleneck_activations_test) // 5
    n_sae_train = len(bottleneck_activations_test) - n_sae_test
    
    np.random.seed(seed + 2000)
    sae_indices = np.random.permutation(len(bottleneck_activations_test))
    sae_train_indices = sae_indices[:n_sae_train]
    sae_test_indices = sae_indices[n_sae_train:]
    
    activations_train = bottleneck_activations_test[sae_train_indices]
    activations_test = bottleneck_activations_test[sae_test_indices]
    
    # Train normal SAE
    print("Training normal SAE...")
    sae_params_normal, _ = train_sae(
        activations_train,
        activations_train.shape[1],
        n,  # sae_hidden_dim = sparse_dim
        num_epochs=sae_epochs,
        learning_rate=0.001,
        lam=0.03,
        seed=seed,
        verbose=False
    )
    
    # Generate random orthogonal rotation
    print("Training SAE with rotated inputs...")
    np.random.seed(seed + 3000)
    Q, _ = np.linalg.qr(np.random.randn(activations_train.shape[1], activations_train.shape[1]))
    
    # Apply rotation to training and test data
    activations_train_rotated = activations_train @ Q
    activations_test_rotated = activations_test @ Q
    
    # Train SAE on rotated data
    sae_params_rotated, _ = train_sae(
        activations_train_rotated,
        activations_train_rotated.shape[1],
        n,  # sae_hidden_dim = sparse_dim
        num_epochs=sae_epochs,
        learning_rate=0.001,
        lam=0.03,
        seed=seed,
        verbose=False
    )
    
    # Evaluate both SAEs on original (unrotated) test data
    sae_outputs_normal = sae_forward(sae_params_normal, activations_test)
    sae_recon_normal = sae_outputs_normal['recon']
    
    # For rotated SAE, we need to be careful about evaluation
    # We trained on rotated data, so we evaluate vs rotated targets
    sae_outputs_rotated = sae_forward(sae_params_rotated, activations_test_rotated)
    sae_recon_rotated = sae_outputs_rotated['recon']
    
    # Compute metrics for normal SAE (original vs original)
    def compute_metrics(ae_activations, sae_reconstructions, label):
        metrics = {'label': label}
        
        # Reconstruction error
        metrics['recon_error'] = np.mean((ae_activations - sae_reconstructions) ** 2)
        
        # Linear CKA
        try:
            metrics['linear_cka'] = linear_cka(ae_activations, sae_reconstructions)
        except:
            metrics['linear_cka'] = np.nan
            
        # RBF CKA
        try:
            metrics['rbf_cka'] = rbf_cka(ae_activations, sae_reconstructions)
        except:
            metrics['rbf_cka'] = np.nan
            
        # Procrustes
        try:
            metrics['procrustes'] = procrustes_similarity(ae_activations, sae_reconstructions)
        except:
            metrics['procrustes'] = np.nan
            
        # RSA
        try:
            dist1 = pdist(ae_activations, metric='euclidean')
            dist2 = pdist(sae_reconstructions, metric='euclidean')
            metrics['rsa'], _ = pearsonr(dist1, dist2)
        except:
            metrics['rsa'] = np.nan
            
        return metrics
    
    # Compare normal SAE vs rotated baseline
    normal_metrics = compute_metrics(activations_test, sae_recon_normal, "Normal SAE")
    rotated_metrics = compute_metrics(activations_test_rotated, sae_recon_rotated, "Rotated Baseline")
    
    print(f"Normal SAE - Linear CKA: {normal_metrics['linear_cka']:.3f}, Procrustes: {normal_metrics['procrustes']:.3f}")
    print(f"Rotated Baseline - Linear CKA: {rotated_metrics['linear_cka']:.3f}, Procrustes: {rotated_metrics['procrustes']:.3f}")
    
    return {
        'sparsity': sparsity,
        'normal_sae': normal_metrics,
        'rotated_baseline': rotated_metrics,
        'architecture': f"{n}-{k}-{n}"
    }


if __name__ == "__main__":
    # Create timestamped results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/procrustes_demo_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Experiment setup - Test run with multiple importance patterns
    sparsities = [0.84, 0.85, 0.86]  # Multiple sparsities across phase transition
    architectures = [
        {'name': '20-5-20', 'sparse_dim': 20, 'dense_dim': 5, 'sae_hidden_dim': 20}]
    
    # Define importance patterns to test
    importance_patterns = [
        {"name": "exponential", "type": "exponential", "kwargs": {}},
        {"name": "uniform", "type": "uniform", "kwargs": {}},
        {"name": "steep_exp", "type": "steep_exponential", "kwargs": {}},
        {"name": "gentle_exp", "type": "gentle_exponential", "kwargs": {}},
        {"name": "step", "type": "step", "kwargs": {}},
        {"name": "linear", "type": "linear", "kwargs": {}},
        {"name": "u_shaped", "type": "u_shaped", "kwargs": {}},
    ]
    
    # Single seed for testing (no random seeds)
    seeds = [1]  # Single deterministic seed
    print(f"Running TEST version with {len(seeds)} seed: {seeds}")
    
    # Store results for all seeds
    all_seed_results = {}  # Format: {seed: {arch_sparsity: {...}}}
    
    # Final averaged results
    results = {}
    
    # Run experiments for each seed
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed_idx+1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")
        
        seed_results = {}
        
        for importance_pattern in importance_patterns:
            pattern_name = importance_pattern["name"]
            pattern_type = importance_pattern["type"]
            pattern_kwargs = importance_pattern["kwargs"]
            
            print(f"\n{'='*60}")
            print(f"IMPORTANCE PATTERN: {pattern_name}")
            print(f"{'='*60}")
            
            for arch in architectures:
                print(f"\n{'='*50}")
                print(f"Architecture: {arch['name']} (Seed: {seed}, Pattern: {pattern_name})")
                print(f"{'='*50}")
                
                for sparsity in sparsities:
                    print(f"\n--- Sparsity: {sparsity} (Seed: {seed}, Pattern: {pattern_name}) ---")
                    
                    # Train autoencoder with train/test split (2500 total -> 2000 train + 500 test)
                    params, ae_test_loss, X_train, X_test, I = train_autoencoder(
                        sparsity=sparsity, 
                        seed=seed,
                        n=arch['sparse_dim'],
                        k=arch['dense_dim'],
                        num_samples=2000,  # Train samples (will get 2500 total with test_ratio=0.2)
                        num_epochs=10,     # Full production run
                        test_ratio=0.2,    # 500 test samples from 2500 total
                        importance_pattern=pattern_type,
                        **pattern_kwargs
                    )
                
                # Get bottleneck activations from test data (for unbiased analysis)
                bottleneck_activations_test = get_bottleneck_activations(params, X_test)
                print(f"Test bottleneck activations shape: {bottleneck_activations_test.shape}")
            
                # Train SAE using all test bottleneck activations (no further split)
                # We already have test data from AE, so we can use it directly for SAE training
                sae_hidden_dim = arch['sparse_dim']  # Inverted: bottleneck -> original sparse dim
                
                # For SAE, we'll use a simple 80/20 split of the test activations
                n_sae_test = len(bottleneck_activations_test) // 5  # 20% for SAE test
                n_sae_train = len(bottleneck_activations_test) - n_sae_test
                
                # Split the test activations for SAE training
                np.random.seed(seed + 2000)
                sae_indices = np.random.permutation(len(bottleneck_activations_test))
                sae_train_indices = sae_indices[:n_sae_train]
                sae_test_indices = sae_indices[n_sae_train:]
                
                activations_train = bottleneck_activations_test[sae_train_indices]
                activations_test = bottleneck_activations_test[sae_test_indices]
                
                print(f"  SAE data split: {len(activations_train)} train, {len(activations_test)} test activations")
                
                # Train SAE on training subset of test activations
                sae_params, sae_train_loss = train_sae(
                    activations_train,
                    activations_train.shape[1],
                    sae_hidden_dim,
                    num_epochs=20,  # Full production run
                    learning_rate=0.001,
                    lam=0.03,
                    seed=seed,
                    verbose=False
                )
                
                # Evaluate SAE on test subset
                sae_test_outputs = sae_forward(sae_params, activations_test)
                sae_test_reconstructions = sae_test_outputs['recon']
                sae_test_recon_error = np.mean((activations_test - sae_test_reconstructions) ** 2)
                
                print(f"  SAE Test reconstruction error: {sae_test_recon_error:.6f}")
                print(f"SAE architecture: {arch['dense_dim']}-{sae_hidden_dim}-{arch['dense_dim']}")
                
                # For consistency in comparisons, use the SAE test subset indices to get matching AE data
                X_test_sae_subset = X_test[sae_test_indices]
                from src.models_numpy import forward
                ae_reconstructions = forward(params, X_test_sae_subset)
                
                # Get SAE representations of the test activations subset
                sae_outputs = sae_forward(sae_params, activations_test)
                sae_reconstructions = sae_outputs['recon']
                sae_hidden = sae_outputs['hidden']  # This is the SAE bottleneck (hidden layer)
                
                # Compute similarity metrics between test bottleneck and SAE reconstruction
                metrics = {}
                
                # Linear CKA
                try:
                    metrics['linear_cka'] = linear_cka(activations_test, sae_reconstructions)
                except:
                    metrics['linear_cka'] = np.nan
                    
                # RBF CKA  
                try:
                    metrics['rbf_cka'] = rbf_cka(activations_test, sae_reconstructions)
                except:
                    metrics['rbf_cka'] = np.nan
                    
                # Procrustes
                try:
                    metrics['procrustes'] = procrustes_similarity(activations_test, sae_reconstructions)
                except:
                    metrics['procrustes'] = np.nan
                    
                # RSA (simple correlation of distance matrices)
                try:
                    dist1 = pdist(activations_test, metric='euclidean')
                    dist2 = pdist(sae_reconstructions, metric='euclidean')
                    metrics['rsa'], _ = pearsonr(dist1, dist2)
                except:
                    metrics['rsa'] = np.nan
                
                # CKA between AE reconstructions and SAE hidden (both on test data)
                cka_ae_sae = linear_cka(ae_reconstructions, sae_hidden)
                print(f"Linear CKA (test bottleneck vs SAE recon): {metrics['linear_cka']:.4f}")
                print(f"RBF CKA (test bottleneck vs SAE recon): {metrics['rbf_cka']:.4f}")
                print(f"Procrustes (test bottleneck vs SAE recon): {metrics['procrustes']:.4f}")
                print(f"RSA (test bottleneck vs SAE recon): {metrics['rsa']:.4f}")
                print(f"CKA (AE test recon vs SAE hidden): {cka_ae_sae:.4f}")
                
                # Compute advanced analysis metrics
                analysis_metrics = {}
                
                # 1. Diagonal strength vs RSA correlation (normalized by dimension)
                W = params['W']
                superposition_matrix = W.T @ W
                diag_strength = np.sum(np.diag(superposition_matrix)) / arch['sparse_dim']  # Normalize by dimension
                analysis_metrics['diag_strength'] = diag_strength
                
                # 2. Off-diagonal interference
                off_diag = superposition_matrix[~np.eye(superposition_matrix.shape[0], dtype=bool)]
                off_diag_interference = np.mean(np.abs(off_diag)) if len(off_diag) > 0 else 0
                analysis_metrics['off_diag_interference'] = off_diag_interference
                
                # 3. SAE L0 norm (empirical sparsity)
                sae_l0 = np.mean(np.sum(sae_hidden > 1e-6, axis=1)) / sae_hidden.shape[1]  # Normalized by dimension
                analysis_metrics['sae_l0'] = sae_l0
                
                # 4. Store test reconstruction errors
                analysis_metrics['ae_test_loss'] = ae_test_loss
                analysis_metrics['sae_test_recon_error'] = sae_test_recon_error
                
                # Compute mean number of active neurons (neurons with activations > threshold)
                threshold = 1e-6
                ae_active = np.mean(np.sum(ae_reconstructions > threshold, axis=1))
                sae_active = np.mean(np.sum(sae_hidden > threshold, axis=1))
                
                # Print test metrics only
                print(f"Mean active neurons - AE: {ae_active:.1f}, SAE: {sae_active:.1f}")
                print(f"Diagonal strength: {diag_strength:.4f}")
                print(f"Off-diagonal interference: {off_diag_interference:.4f}")
                print(f"SAE L0 norm: {sae_l0:.4f}")
                
                # Store results with test metrics for this seed
                key = f"{arch['name']}_s{sparsity}_{pattern_name}"
                seed_results[key] = {
                    'ae_params': params,
                    'ae_test_loss': ae_test_loss,
                    'sae_params': sae_params,
                    'sae_test_recon_error': sae_test_recon_error,
                    'activations': activations_test,  # Store test activations
                    'cka_ae_sae': cka_ae_sae,
                    'ae_active': ae_active,
                    'sae_active': sae_active,
                    'metrics': metrics,  # Similarity metrics on test data
                    'analysis_metrics': analysis_metrics  # Analysis metrics including test losses
                }
                
                # Add separator line after each sparsity
                print()
            
            # Add separator line after each architecture
            print()
        
        # Store this seed's results
        all_seed_results[seed] = seed_results
        print(f"\nSeed {seed} completed!")
    
    # Print simple summary for test version
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE - TEST VERSION")
    print(f"{'='*60}")
    
    print("Results stored in all_seed_results with keys including pattern names.")
    print("Generating basic comparison plots...")
    
    # Create a simple comparison plot of key metrics across importance patterns
    import matplotlib.pyplot as plt
    
    # Generate comparison plots for each sparsity level
    seed = 1  # We only have one seed
    arch_name = '20-5-20'
    
    for sparsity in sparsities:
        print(f"Creating plots for sparsity {sparsity}...")
        
        pattern_names = []
        ae_losses = []
        sae_losses = []
        linear_cka_scores = []
        procrustes_scores = []
        
        for pattern in importance_patterns:
            pattern_name = pattern["name"]
            key = f"{arch_name}_s{sparsity}_{pattern_name}"
            
            if key in all_seed_results[seed]:
                result = all_seed_results[seed][key]
                pattern_names.append(pattern_name)
                ae_losses.append(result['ae_test_loss'])
                sae_losses.append(result['analysis_metrics']['sae_test_recon_error'])
                linear_cka_scores.append(result['metrics']['linear_cka'])
                procrustes_scores.append(result['metrics']['procrustes'])
        
        # Create 2x2 subplot for this sparsity
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: AE Test Loss
        ax1.bar(range(len(pattern_names)), ae_losses, color='skyblue')
        ax1.set_title(f'Autoencoder Test Loss (Sparsity {sparsity})')
        ax1.set_ylabel('Test Loss')
        ax1.set_xticks(range(len(pattern_names)))
        ax1.set_xticklabels(pattern_names, rotation=45)
        
        # Plot 2: SAE Reconstruction Error
        ax2.bar(range(len(pattern_names)), sae_losses, color='lightcoral')
        ax2.set_title(f'SAE Reconstruction Error (Sparsity {sparsity})')
        ax2.set_ylabel('Reconstruction Error')
        ax2.set_xticks(range(len(pattern_names)))
        ax2.set_xticklabels(pattern_names, rotation=45)
        
        # Plot 3: Linear CKA Scores
        ax3.bar(range(len(pattern_names)), linear_cka_scores, color='lightgreen')
        ax3.set_title(f'Linear CKA (Sparsity {sparsity})')
        ax3.set_ylabel('Linear CKA Score')
        ax3.set_xticks(range(len(pattern_names)))
        ax3.set_xticklabels(pattern_names, rotation=45)
        
        # Plot 4: Procrustes Scores
        ax4.bar(range(len(pattern_names)), procrustes_scores, color='gold')
        ax4.set_title(f'Procrustes Similarity (Sparsity {sparsity})')
        ax4.set_ylabel('Procrustes Score')
        ax4.set_xticks(range(len(pattern_names)))
        ax4.set_xticklabels(pattern_names, rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/patterns_comparison_s{sparsity}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot for sparsity {sparsity}")
    
    print(f"Saved all sparsity-specific comparison plots")
    
    # Generate metric heatmaps for each sparsity level
    print("Generating metric heatmaps for each sparsity level...")
    
    for sparsity in sparsities:
        print(f"Creating heatmaps for sparsity {sparsity}...")
        
        # Collect all metrics for heatmap
        heatmap_data = []
        heatmap_labels = []
        
        for pattern in importance_patterns:
            pattern_name = pattern["name"]
            key = f"{arch_name}_s{sparsity}_{pattern_name}"
            
            if key in all_seed_results[seed]:
                result = all_seed_results[seed][key]
                
                # Collect normalized metrics for heatmap
                metrics_row = [
                    result['ae_test_loss'],  # Lower is better
                    result['analysis_metrics']['sae_test_recon_error'],  # Lower is better  
                    result['metrics']['linear_cka'],  # Higher is better
                    result['metrics']['rbf_cka'],  # Higher is better
                    result['metrics']['procrustes'],  # Higher is better
                    result['metrics']['rsa'],  # Higher is better
                    result['analysis_metrics']['diag_strength'],  # Context dependent
                    result['analysis_metrics']['off_diag_interference'],  # Lower is better
                    result['analysis_metrics']['sae_l0'],  # Context dependent
                    result['ae_active'],  # Context dependent
                    result['sae_active']  # Context dependent
                ]
                
                heatmap_data.append(metrics_row)
                heatmap_labels.append(pattern_name)
        
        # Convert to numpy array for easier manipulation
        import numpy as np
        
        if len(heatmap_data) == 0:
            print(f"No data found for sparsity {sparsity}, skipping heatmaps...")
            continue
            
        heatmap_data = np.array(heatmap_data)
        print(f"Heatmap data shape for sparsity {sparsity}: {heatmap_data.shape}")
        
        # Normalize each column to [0,1] for better visualization
        normalized_data = np.zeros_like(heatmap_data)
        for col in range(heatmap_data.shape[1]):
            col_data = heatmap_data[:, col]
            col_min, col_max = np.min(col_data), np.max(col_data)
            if col_max > col_min:  # Avoid division by zero
                normalized_data[:, col] = (col_data - col_min) / (col_max - col_min)
            else:
                normalized_data[:, col] = 0.5  # If all values are the same
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        
        metric_names = [
            'AE Test Loss', 'SAE Recon Error', 'Linear CKA', 'RBF CKA', 
            'Procrustes', 'RSA', 'Diag Strength', 'Off-Diag Inter', 
            'SAE L0', 'AE Active', 'SAE Active'
        ]
        
        import seaborn as sns
        
        # Create the heatmap
        ax = sns.heatmap(normalized_data, 
                         xticklabels=metric_names,
                         yticklabels=heatmap_labels,
                         annot=True,
                         fmt='.3f',
                         cmap='RdYlBu_r',
                         cbar_kws={'label': 'Normalized Score (0=min, 1=max)'})
        
        plt.title(f'Normalized Metrics Heatmap Across Feature Importance Patterns\\n{arch_name} Architecture (Sparsity {sparsity})', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Feature Importance Patterns', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/metrics_heatmap_s{sparsity}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved metrics heatmap for sparsity {sparsity}")
        
        # Generate individual pattern comparison heatmap with raw values
        plt.figure(figsize=(14, 8))
        
        # Use raw values for annotation but normalized for color
        ax = sns.heatmap(normalized_data, 
                         xticklabels=metric_names,
                         yticklabels=heatmap_labels,
                         annot=heatmap_data,  # Show raw values
                         fmt='.4f',
                         cmap='viridis',
                         cbar_kws={'label': 'Normalized Color Scale'})
        
        plt.title(f'Raw Metrics Values with Normalized Color Scale\\n{arch_name} Architecture (Sparsity {sparsity})', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Feature Importance Patterns', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/raw_metrics_heatmap_s{sparsity}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved raw metrics heatmap for sparsity {sparsity}")
    
    print("All heatmaps generated!")
    
    # Create a summary table for all sparsity levels
    print("\\n" + "="*100)
    print("COMPREHENSIVE METRICS SUMMARY TABLE BY IMPORTANCE PATTERN AND SPARSITY")
    print("="*100)
    print(f"{'Pattern':<12} {'Sparsity':<8} {'AE Loss':<10} {'SAE Error':<10} {'Linear CKA':<11} {'Procrustes':<10} {'Diag Str':<9} {'Off-Diag':<9}")
    print("-" * 100)
    
    for sparsity in sparsities:
        for pattern in importance_patterns:
            pattern_name = pattern["name"]
            key = f"{arch_name}_s{sparsity}_{pattern_name}"
            
            if key in all_seed_results[seed]:
                result = all_seed_results[seed][key]
                ae_loss = result['ae_test_loss']
                sae_error = result['analysis_metrics']['sae_test_recon_error']
                linear_cka = result['metrics']['linear_cka']
                procrustes = result['metrics']['procrustes']
                diag_str = result['analysis_metrics']['diag_strength']
                off_diag = result['analysis_metrics']['off_diag_interference']
                
                print(f"{pattern_name:<12} {sparsity:<8.2f} {ae_loss:<10.6f} {sae_error:<10.6f} {linear_cka:<11.4f} {procrustes:<10.4f} {diag_str:<9.4f} {off_diag:<9.4f}")
        
        # Add separator line between sparsity levels
        if sparsity != sparsities[-1]:
            print("-" * 100)
    
    print("\\nTest version complete with comprehensive multi-sparsity visualizations!")
