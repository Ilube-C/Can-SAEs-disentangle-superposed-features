import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def compute_superposition_matrix(params):
    """Compute superposition matrix W.T @ W and related metrics.
    
    Args:
        params: model parameters dictionary
        
    Returns:
        Dictionary containing superposition analysis results
    """
    W = params['W']
    superposition_matrix = W.T @ W
    
    # Compute weight norms
    weight_norms = np.array([np.linalg.norm(col, ord=2) for col in superposition_matrix])
    
    # Compute pairwise dot products
    n = superposition_matrix.shape[0]
    dot_products = {}
    for i in range(n):
        dot_products[i] = []
        for j in range(n):
            score = np.dot(superposition_matrix[i], superposition_matrix[j])
            dot_products[i].append(float(score))
    
    return {
        'superposition_matrix': superposition_matrix,
        'weight_norms': weight_norms,
        'dot_products': dot_products,
        'W': W
    }


def plot_weight_norms(analysis_results, sparsity, save_path=None):
    """Plot weight norms in sorted order.
    
    Args:
        analysis_results: results from compute_superposition_matrix
        sparsity: sparsity level for the title
        save_path: optional path to save the plot
    """
    weight_norms = analysis_results['weight_norms']
    W = analysis_results['W']
    
    # Sort weight norms
    reorder = np.argsort(weight_norms)
    weight_norms_sorted = weight_norms[reorder]
    
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(W.shape[1]), weight_norms_sorted)
    plt.title(f"Weight Norms (sorted), Sparsity = {sparsity}")
    plt.xlabel("Feature Index (sorted by norm)")
    plt.ylabel("L2 Norm")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_superposition_heatmap(analysis_results, sparsity, save_path=None):
    """Plot superposition matrix as heatmap.
    
    Args:
        analysis_results: results from compute_superposition_matrix
        sparsity: sparsity level for the title
        save_path: optional path to save the plot
    """
    dot_products = analysis_results['dot_products']
    
    df = pd.DataFrame(dot_products)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, cmap="coolwarm", center=0)
    plt.title(f"Superposition Matrix (W.T @ W), Sparsity = {sparsity}")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def analyze_sparsity_effects(models, sparsities, save_dir=None):
    """Analyze superposition effects across different sparsity levels.
    
    Args:
        models: list of trained model parameters
        sparsities: list of sparsity levels
        save_dir: optional directory to save plots
    """
    results = []
    
    for i, (params, sparsity) in enumerate(zip(models, sparsities)):
        print(f"\nAnalyzing model with sparsity = {sparsity}")
        
        analysis = compute_superposition_matrix(params)
        results.append(analysis)
        
        # Plot weight norms
        save_path = f"{save_dir}/weight_norms_s{sparsity}.png" if save_dir else None
        plot_weight_norms(analysis, sparsity, save_path)
        
        # Plot superposition heatmap  
        save_path = f"{save_dir}/superposition_s{sparsity}.png" if save_dir else None
        plot_superposition_heatmap(analysis, sparsity, save_path)
    
    return results


def plot_data_histograms(synth_data, sparsities, save_dir=None):
    """Plot histograms of synthetic data for different sparsity levels.
    
    Args:
        synth_data: list of synthetic datasets
        sparsities: list of sparsity levels
        save_dir: optional directory to save plots
    """
    for i, (data, sparsity) in enumerate(zip(synth_data, sparsities)):
        flat_data = data.flatten()
        
        plt.figure(figsize=(8, 6))
        plt.hist(flat_data, bins=50, alpha=0.7)
        plt.title(f"Data Distribution, Sparsity = {sparsity}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        
        if save_dir:
            plt.savefig(f"{save_dir}/data_hist_s{sparsity}.png")
        plt.show()


# ========== NEW SAE ANALYSIS IMPLEMENTATION START ==========
# All SAE analysis functions are clearly marked for easy rollback

def compute_rsa_correlation(repr1, repr2):
    """Compute Representational Similarity Analysis (RSA) correlation.
    
    RSA measures how similar two representation spaces are by computing
    the correlation between their representational distance matrices.
    
    Args:
        repr1: first representation matrix (N, D1)
        repr2: second representation matrix (N, D2)
        
    Returns:
        RSA correlation coefficient (scalar)
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import pearsonr
    
    # Compute pairwise distances for each representation
    dist1 = pdist(repr1, metric='euclidean')
    dist2 = pdist(repr2, metric='euclidean')
    
    # Compute correlation between distance vectors
    correlation, p_value = pearsonr(dist1, dist2)
    
    return correlation


def compute_sparsity_metrics(activations):
    """Compute sparsity metrics for neural activations.
    
    Args:
        activations: activation matrix (N, D)
        
    Returns:
        Dictionary containing sparsity metrics
    """
    # L0 norm (fraction of non-zero elements)
    l0_norm = np.mean(activations != 0)
    
    # L1 norm (average absolute activation)
    l1_norm = np.mean(np.abs(activations))
    
    # Gini coefficient (measure of inequality)
    flat_acts = np.abs(activations.flatten())
    sorted_acts = np.sort(flat_acts)
    n = len(sorted_acts)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_acts)) / (n * np.sum(sorted_acts)) - (n + 1) / n
    
    # Maximum activation value
    max_activation = np.max(np.abs(activations))
    
    return {
        'l0_norm': l0_norm,
        'l1_norm': l1_norm, 
        'gini_coefficient': gini,
        'max_activation': max_activation,
        'mean_activation': np.mean(activations),
        'std_activation': np.std(activations)
    }


def analyze_sae_performance(original_activations, sae_params, verbose=True):
    """Comprehensive analysis of SAE performance.
    
    Args:
        original_activations: original autoencoder bottleneck activations (N, k)
        sae_params: trained SAE parameters
        verbose: whether to print detailed results
        
    Returns:
        Dictionary containing performance metrics
    """
    # Import SAE functions
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from models_numpy import sae_forward
    
    # Get SAE reconstructions and hidden activations
    sae_result = sae_forward(sae_params, original_activations)
    sae_hidden = sae_result['hidden']
    sae_recon = sae_result['recon']
    
    # 1. Basic functionality checks
    recon_mse = np.mean((original_activations - sae_recon)**2)
    
    # 2. Sparsity analysis
    original_sparsity = compute_sparsity_metrics(original_activations)
    sae_sparsity = compute_sparsity_metrics(sae_hidden)
    
    # 3. Similarity metrics
    # Toggle between RSA and CKA
    use_cka = True  # Set to False to use RSA instead
    
    if use_cka:
        # Use CKA for similarity measurement
        from CKA import linear_cka
        similarity_metric = linear_cka(original_activations, sae_recon)
        self_check = linear_cka(original_activations, original_activations)
        
        # Generate random baseline for comparison
        np.random.seed(42)
        random_repr = np.random.normal(0, 1, original_activations.shape)
        random_baseline = linear_cka(original_activations, random_repr)
        
        metric_name = "CKA"
    else:
        # Use RSA for similarity measurement
        similarity_metric = compute_rsa_correlation(original_activations, sae_recon)
        self_check = compute_rsa_correlation(original_activations, original_activations)
        
        # Generate random baseline for comparison
        np.random.seed(42)
        random_repr = np.random.normal(0, 1, original_activations.shape)
        random_baseline = compute_rsa_correlation(original_activations, random_repr)
        
        metric_name = "RSA"
    
    results = {
        'reconstruction_mse': recon_mse,
        'similarity_metric': similarity_metric,
        'similarity_metric_name': metric_name,
        'self_check': self_check,
        'random_baseline': random_baseline,
        # Keep old field names for backwards compatibility
        'rsa_correlation': similarity_metric,
        'rsa_self_check': self_check,
        'rsa_random_baseline': random_baseline,
        'original_sparsity': original_sparsity,
        'sae_sparsity': sae_sparsity,
        'sparsity_improvement': original_sparsity['l0_norm'] / sae_sparsity['l0_norm']
    }
    
    if verbose:
        print("=== SAE Performance Analysis ===")
        print(f"Using similarity metric: {metric_name}")
        print(f"Reconstruction MSE: {recon_mse:.6f}")
        print(f"{metric_name} Similarity: {similarity_metric:.4f}")
        print(f"{metric_name} Self-check: {self_check:.4f} (should be ~1.0)")
        print(f"{metric_name} Random baseline: {random_baseline:.4f} (should be ~0.0)")
        print(f"\nSparsity Analysis:")
        print(f"Original L0: {original_sparsity['l0_norm']:.4f}")
        print(f"SAE L0: {sae_sparsity['l0_norm']:.4f}")
        print(f"Sparsity improvement: {results['sparsity_improvement']:.2f}x")
        print(f"Original L1: {original_sparsity['l1_norm']:.4f}")
        print(f"SAE L1: {sae_sparsity['l1_norm']:.4f}")
    
    return results


def plot_activation_comparison(original_activations, sae_params, save_path=None):
    """Plot comparison of original vs SAE activations.
    
    Args:
        original_activations: original autoencoder activations (N, k)
        sae_params: trained SAE parameters
        save_path: optional path to save the plot
    """
    # Import SAE functions
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from models_numpy import sae_forward
    
    sae_result = sae_forward(sae_params, original_activations)
    sae_hidden = sae_result['hidden']
    sae_recon = sae_result['recon']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original activations histogram
    axes[0, 0].hist(original_activations.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Original Activations Distribution')
    axes[0, 0].set_xlabel('Activation Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # SAE hidden activations histogram
    axes[0, 1].hist(sae_hidden.flatten(), bins=50, alpha=0.7, color='red')
    axes[0, 1].set_title('SAE Hidden Activations Distribution')
    axes[0, 1].set_xlabel('Activation Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Reconstruction comparison scatter
    axes[1, 0].scatter(original_activations.flatten(), sae_recon.flatten(), 
                      alpha=0.5, s=1)
    axes[1, 0].plot([original_activations.min(), original_activations.max()],
                   [original_activations.min(), original_activations.max()], 
                   'r--', label='Perfect reconstruction')
    axes[1, 0].set_xlabel('Original Activations')
    axes[1, 0].set_ylabel('SAE Reconstructions')
    axes[1, 0].set_title('Reconstruction Quality')
    axes[1, 0].legend()
    
    # Sparsity comparison
    orig_l0 = np.mean(original_activations != 0, axis=0)
    sae_l0 = np.mean(sae_hidden != 0, axis=0)
    
    x_pos = np.arange(len(orig_l0))
    axes[1, 1].bar(x_pos - 0.2, orig_l0, 0.4, label='Original', alpha=0.7)
    # Note: SAE has different dimensions, so we show mean sparsity
    sae_mean_l0 = np.mean(sae_l0)
    axes[1, 1].axhline(y=sae_mean_l0, color='red', linestyle='--', 
                      label=f'SAE Mean L0: {sae_mean_l0:.3f}')
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Activation Frequency')
    axes[1, 1].set_title('Sparsity Comparison')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_sae_feature_correlation(original_activations, sae_params, save_path=None):
    """Plot correlation between original features and SAE features.
    
    Args:
        original_activations: original autoencoder activations (N, k)
        sae_params: trained SAE parameters
        save_path: optional path to save the plot
    """
    # Import SAE functions
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from models_numpy import sae_forward
    
    sae_result = sae_forward(sae_params, original_activations)
    sae_hidden = sae_result['hidden']
    
    print("=== SAE DIMENSION DEBUG ===")
    print(f"Original activations shape: {original_activations.shape}")
    print(f"SAE hidden activations shape: {sae_hidden.shape}")
    print(f"SAE W_enc shape: {sae_params['W_enc'].shape}")
    print(f"SAE W_dec shape: {sae_params['W_dec'].shape}")
    
    # Compute correlation matrix between original and SAE features
    correlation_matrix = np.corrcoef(original_activations.T, sae_hidden.T)
    
    print("correlation_matrix =========================")
    print(f"Full correlation matrix shape: {correlation_matrix.shape}")
    print(f"Expected: ({original_activations.shape[1]} + {sae_hidden.shape[1]}) x ({original_activations.shape[1]} + {sae_hidden.shape[1]})")


    # Extract the cross-correlation part (original vs SAE)
    k = original_activations.shape[1]

    print("K =========================")
    print(k)

    cross_corr = correlation_matrix[:k, k:]

    print("cross corr =========================")
    print(f"Cross-correlation shape: {cross_corr.shape}")
    print(f"This is: {k} original features x {cross_corr.shape[1]} SAE features")
    print("=============================")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cross_corr, cmap='coolwarm', center=0, 
                xticklabels=[f'SAE_{i}' for i in range(cross_corr.shape[1])],
                yticklabels=[f'Orig_{i}' for i in range(cross_corr.shape[0])])
    plt.title('Feature Correlation: Original vs SAE')
    plt.xlabel('SAE Features')
    plt.ylabel('Original Features')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return cross_corr


def run_sae_ablation_study(original_activations, hidden_dims, l1_penalties, 
                          input_dim, num_epochs=30, save_dir=None):
    """Run ablation study over SAE hyperparameters.
    
    Args:
        original_activations: original autoencoder activations (N, k)
        hidden_dims: list of hidden dimensions to test
        l1_penalties: list of L1 penalties to test
        input_dim: input dimension (k from original autoencoder)
        num_epochs: number of training epochs per configuration
        save_dir: optional directory to save results
        
    Returns:
        DataFrame containing ablation results
    """
    # Import SAE functions
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from models_numpy import train_sae
    
    results = []
    
    for hidden_dim in hidden_dims:
        for l1_penalty in l1_penalties:
            print(f"Testing hidden_dim={hidden_dim}, l1_penalty={l1_penalty}")
            
            # Train SAE with current hyperparameters
            sae_params = train_sae(
                original_activations, 
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_epochs=num_epochs,
                l1_penalty=l1_penalty,
                verbose=False
            )
            
            # Analyze performance
            analysis = analyze_sae_performance(
                original_activations, 
                sae_params, 
                verbose=False
            )
            
            # Store results
            result = {
                'hidden_dim': hidden_dim,
                'l1_penalty': l1_penalty,
                'reconstruction_mse': analysis['reconstruction_mse'],
                'rsa_correlation': analysis['rsa_correlation'],
                'sae_l0_norm': analysis['sae_sparsity']['l0_norm'],
                'sparsity_improvement': analysis['sparsity_improvement']
            }
            results.append(result)
    
    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    
    if save_dir:
        results_df.to_csv(f"{save_dir}/sae_ablation_results.csv", index=False)
        
        # Plot ablation results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RSA vs hidden dimension
        for l1 in l1_penalties:
            subset = results_df[results_df['l1_penalty'] == l1]
            axes[0, 0].plot(subset['hidden_dim'], subset['rsa_correlation'], 
                           marker='o', label=f'L1={l1}')
        axes[0, 0].set_xlabel('Hidden Dimension')
        axes[0, 0].set_ylabel('RSA Correlation')
        axes[0, 0].set_title('RSA vs Hidden Dimension')
        axes[0, 0].legend()
        
        # Sparsity vs L1 penalty
        for hd in hidden_dims:
            subset = results_df[results_df['hidden_dim'] == hd]
            axes[0, 1].plot(subset['l1_penalty'], subset['sae_l0_norm'], 
                           marker='o', label=f'HD={hd}')
        axes[0, 1].set_xlabel('L1 Penalty')
        axes[0, 1].set_ylabel('SAE L0 Norm')
        axes[0, 1].set_title('Sparsity vs L1 Penalty')
        axes[0, 1].legend()
        
        # Reconstruction MSE heatmap
        pivot_mse = results_df.pivot('hidden_dim', 'l1_penalty', 'reconstruction_mse')
        sns.heatmap(pivot_mse, annot=True, fmt='.4f', ax=axes[1, 0])
        axes[1, 0].set_title('Reconstruction MSE')
        
        # RSA correlation heatmap  
        pivot_rsa = results_df.pivot('hidden_dim', 'l1_penalty', 'rsa_correlation')
        sns.heatmap(pivot_rsa, annot=True, fmt='.3f', ax=axes[1, 1])
        axes[1, 1].set_title('RSA Correlation')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sae_ablation_plots.png")
        plt.show()
    
    return results_df

# ========== NEW SAE ANALYSIS IMPLEMENTATION END ==========