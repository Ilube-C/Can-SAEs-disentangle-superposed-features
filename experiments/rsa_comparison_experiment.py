"""Comprehensive RSA experiment comparing Autoencoder and Sparse Autoencoder representations.

This experiment trains both AE and SAE models and compares their learned representations
using multiple RSA metrics:
1. Centered Kernel Alignment (CKA)
2. Canonical Correlation Analysis (CCA/SVCCA)
3. Procrustes Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models_numpy import LinearAutoencoder
from src.sparse_autoencoder import SparseAutoencoder
from src.data_generation import generate_sparse_data
from src.CKA import compute_cka, compute_linear_cka, compute_rbf_cka
from src.rsa_cca import compute_cca, compute_svcca, cca_distance, svcca_similarity
from src.rsa_procrustes import orthogonal_procrustes, procrustes_similarity, angular_procrustes_distance

# Configuration
RANDOM_SEED = 42
N_SAMPLES = 10000
N_FEATURES = 20
BOTTLENECK_DIM = 5
SPARSITY = 0.95
N_EPOCHS = 10
LEARNING_RATE = 0.01
L1_LAMBDA = 0.001  # For SAE

# For RSA analysis
N_TEST_SAMPLES = 1000


def train_models(X_train, feature_importance):
    """Train both AE and SAE models on the same data."""
    
    print("Training Standard Autoencoder...")
    ae = LinearAutoencoder(
        n_features=N_FEATURES,
        n_hidden=BOTTLENECK_DIM,
        learning_rate=LEARNING_RATE,
        random_seed=RANDOM_SEED
    )
    
    ae_losses = []
    for epoch in range(N_EPOCHS):
        epoch_losses = []
        for i in range(0, len(X_train), 32):  # Batch size 32
            batch = X_train[i:i+32]
            loss = ae.train_step(batch, feature_importance)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        ae_losses.append(avg_loss)
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    print("\nTraining Sparse Autoencoder...")
    sae = SparseAutoencoder(
        input_dim=N_FEATURES,
        hidden_dim=BOTTLENECK_DIM,
        l1_lambda=L1_LAMBDA,
        learning_rate=LEARNING_RATE
    )
    
    sae_losses = []
    for epoch in range(N_EPOCHS):
        epoch_losses = []
        for i in range(0, len(X_train), 32):  # Batch size 32
            batch = X_train[i:i+32]
            loss, _ = sae.train_step(batch)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        sae_losses.append(avg_loss)
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    return ae, sae, ae_losses, sae_losses


def extract_representations(model, X, model_type='ae'):
    """Extract latent representations from a model."""
    if model_type == 'ae':
        # For standard autoencoder
        z = model.W @ X.T  # Shape: (bottleneck_dim, n_samples)
        return z.T  # Return (n_samples, bottleneck_dim)
    elif model_type == 'sae':
        # For sparse autoencoder
        z = model.encode(X)  # Already returns (n_samples, hidden_dim)
        return z
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_all_rsa_metrics(rep1, rep2, labels=['Model 1', 'Model 2']):
    """Compute all RSA metrics between two representations."""
    
    results = {}
    
    print(f"\nComparing {labels[0]} vs {labels[1]}:")
    print(f"  Representation shapes: {rep1.shape} vs {rep2.shape}")
    
    # 1. CKA metrics
    print("\n  CKA Metrics:")
    cka_linear = compute_linear_cka(rep1, rep2)
    cka_rbf = compute_rbf_cka(rep1, rep2)
    print(f"    Linear CKA: {cka_linear:.4f}")
    print(f"    RBF CKA: {cka_rbf:.4f}")
    
    results['cka_linear'] = cka_linear
    results['cka_rbf'] = cka_rbf
    
    # 2. CCA/SVCCA metrics
    print("\n  CCA/SVCCA Metrics:")
    canonical_corrs = compute_cca(rep1, rep2)
    mean_cca = np.mean(canonical_corrs)
    cca_dist = cca_distance(rep1, rep2)
    svcca_sim = svcca_similarity(rep1, rep2, threshold=0.99)
    
    print(f"    Mean CCA: {mean_cca:.4f}")
    print(f"    CCA Distance: {cca_dist:.4f}")
    print(f"    SVCCA Similarity: {svcca_sim:.4f}")
    print(f"    Top 5 canonical correlations: {canonical_corrs[:5]}")
    
    results['mean_cca'] = mean_cca
    results['cca_distance'] = cca_dist
    results['svcca_similarity'] = svcca_sim
    results['canonical_correlations'] = canonical_corrs
    
    # 3. Procrustes Analysis
    print("\n  Procrustes Analysis:")
    
    # For Procrustes, we need same dimensionality
    # If different, pad the smaller one with zeros
    if rep1.shape[1] != rep2.shape[1]:
        max_dim = max(rep1.shape[1], rep2.shape[1])
        if rep1.shape[1] < max_dim:
            rep1_padded = np.pad(rep1, ((0, 0), (0, max_dim - rep1.shape[1])), mode='constant')
            rep2_padded = rep2
        else:
            rep1_padded = rep1
            rep2_padded = np.pad(rep2, ((0, 0), (0, max_dim - rep2.shape[1])), mode='constant')
    else:
        rep1_padded = rep1
        rep2_padded = rep2
    
    proc_sim = procrustes_similarity(rep1_padded, rep2_padded)
    proc_sim_scaled = procrustes_similarity(rep1_padded, rep2_padded, scaling=True)
    angular_dist = angular_procrustes_distance(rep1_padded, rep2_padded)
    
    print(f"    Procrustes Similarity: {proc_sim:.4f}")
    print(f"    Procrustes Similarity (with scaling): {proc_sim_scaled:.4f}")
    print(f"    Angular Distance: {angular_dist:.4f} rad ({np.degrees(angular_dist):.2f}°)")
    
    results['procrustes_similarity'] = proc_sim
    results['procrustes_similarity_scaled'] = proc_sim_scaled
    results['angular_distance'] = angular_dist
    
    return results


def analyze_weight_matrices(ae, sae):
    """Analyze and compare the weight matrices directly."""
    
    print("\n" + "="*60)
    print("WEIGHT MATRIX ANALYSIS")
    print("="*60)
    
    # Extract encoder and decoder matrices
    W_ae = ae.W  # Shape: (bottleneck_dim, n_features)
    W_ae_decoder = ae.W.T  # Shape: (n_features, bottleneck_dim)
    
    W_sae = sae.W_encoder.T  # Shape: (hidden_dim, input_dim) - transpose to match AE format
    W_sae_decoder = sae.W_decoder  # Shape: (input_dim, hidden_dim)
    
    print(f"\nWeight matrix shapes:")
    print(f"  AE encoder: {W_ae.shape}")
    print(f"  AE decoder: {W_ae_decoder.shape}")
    print(f"  SAE encoder: {W_sae.shape}")
    print(f"  SAE decoder: {W_sae_decoder.shape}")
    
    # Compare encoder weights as representations
    print("\nEncoder Weight Comparison:")
    encoder_results = compute_all_rsa_metrics(
        W_ae.T,  # Transpose to (n_features, bottleneck_dim) for RSA
        W_sae.T,  # Transpose to (n_features, hidden_dim) for RSA
        labels=['AE Encoder', 'SAE Encoder']
    )
    
    # Compare decoder weights as representations
    print("\nDecoder Weight Comparison:")
    decoder_results = compute_all_rsa_metrics(
        W_ae_decoder,
        W_sae_decoder,
        labels=['AE Decoder', 'SAE Decoder']
    )
    
    return encoder_results, decoder_results


def visualize_results(ae, sae, X_test, all_results):
    """Create comprehensive visualizations of RSA results."""
    
    results_dir = Path("results/rsa_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract test representations
    z_ae = extract_representations(ae, X_test, model_type='ae')
    z_sae = extract_representations(sae, X_test, model_type='sae')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Weight matrices heatmaps
    ax1 = plt.subplot(3, 4, 1)
    sns.heatmap(ae.W, cmap='coolwarm', center=0, cbar=True, ax=ax1)
    ax1.set_title('AE Encoder Weights')
    ax1.set_xlabel('Input Features')
    ax1.set_ylabel('Hidden Units')
    
    ax2 = plt.subplot(3, 4, 2)
    sns.heatmap(sae.W_encoder.T, cmap='coolwarm', center=0, cbar=True, ax=ax2)
    ax2.set_title('SAE Encoder Weights')
    ax2.set_xlabel('Input Features')
    ax2.set_ylabel('Hidden Units')
    
    # 2. Latent activations distribution
    ax3 = plt.subplot(3, 4, 3)
    ax3.hist(z_ae.flatten(), bins=50, alpha=0.7, label='AE', density=True)
    ax3.hist(z_sae.flatten(), bins=50, alpha=0.7, label='SAE', density=True)
    ax3.set_xlabel('Activation Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Latent Activation Distributions')
    ax3.legend()
    
    # 3. Sparsity of latent representations
    ax4 = plt.subplot(3, 4, 4)
    ae_sparsity = np.mean(np.abs(z_ae) < 0.01, axis=0)  # Fraction near zero per dimension
    sae_sparsity = np.mean(np.abs(z_sae) < 0.01, axis=0)
    x_pos = np.arange(len(ae_sparsity))
    width = 0.35
    ax4.bar(x_pos - width/2, ae_sparsity, width, label='AE', alpha=0.7)
    ax4.bar(x_pos + width/2, sae_sparsity, width, label='SAE', alpha=0.7)
    ax4.set_xlabel('Hidden Dimension')
    ax4.set_ylabel('Fraction Near Zero')
    ax4.set_title('Sparsity per Hidden Dimension')
    ax4.legend()
    
    # 4. RSA metrics summary
    ax5 = plt.subplot(3, 4, 5)
    metrics_names = ['CKA\n(Linear)', 'CKA\n(RBF)', 'Mean\nCCA', 'SVCCA', 'Procrustes']
    latent_scores = [
        all_results['latent']['cka_linear'],
        all_results['latent']['cka_rbf'],
        all_results['latent']['mean_cca'],
        all_results['latent']['svcca_similarity'],
        all_results['latent']['procrustes_similarity']
    ]
    bars = ax5.bar(metrics_names, latent_scores, alpha=0.7, color='steelblue')
    ax5.set_ylabel('Similarity Score')
    ax5.set_title('RSA Metrics: Latent Representations')
    ax5.set_ylim([0, 1])
    ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, latent_scores):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Canonical correlations
    ax6 = plt.subplot(3, 4, 6)
    canonical_corrs = all_results['latent']['canonical_correlations']
    ax6.plot(canonical_corrs, 'o-', alpha=0.7, linewidth=2, markersize=6)
    ax6.set_xlabel('Canonical Dimension')
    ax6.set_ylabel('Correlation')
    ax6.set_title('Canonical Correlations (CCA)')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    # 6. Weight norms comparison
    ax7 = plt.subplot(3, 4, 7)
    ae_encoder_norms = np.linalg.norm(ae.W, axis=1)
    sae_encoder_norms = np.linalg.norm(sae.W_encoder.T, axis=1)
    x_pos = np.arange(BOTTLENECK_DIM)
    width = 0.35
    ax7.bar(x_pos - width/2, sorted(ae_encoder_norms, reverse=True), width, label='AE', alpha=0.7)
    ax7.bar(x_pos + width/2, sorted(sae_encoder_norms, reverse=True), width, label='SAE', alpha=0.7)
    ax7.set_xlabel('Hidden Unit (sorted)')
    ax7.set_ylabel('Weight Norm')
    ax7.set_title('Encoder Weight Norms')
    ax7.legend()
    
    # 7. Reconstruction quality comparison
    ax8 = plt.subplot(3, 4, 8)
    # Reconstruct test data
    x_recon_ae = ae.forward(X_test)
    x_recon_sae = sae.forward(X_test)
    
    # Calculate per-feature reconstruction error
    ae_errors = np.mean((X_test - x_recon_ae)**2, axis=0)
    sae_errors = np.mean((X_test - x_recon_sae)**2, axis=0)
    
    feature_idx = np.arange(N_FEATURES)
    ax8.plot(feature_idx, ae_errors, 'o-', label='AE', alpha=0.7)
    ax8.plot(feature_idx, sae_errors, 's-', label='SAE', alpha=0.7)
    ax8.set_xlabel('Feature Index')
    ax8.set_ylabel('Mean Squared Error')
    ax8.set_title('Per-Feature Reconstruction Error')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 8. Encoder weights RSA metrics
    ax9 = plt.subplot(3, 4, 9)
    encoder_scores = [
        all_results['encoder']['cka_linear'],
        all_results['encoder']['cka_rbf'],
        all_results['encoder']['mean_cca'],
        all_results['encoder']['svcca_similarity'],
        all_results['encoder']['procrustes_similarity']
    ]
    bars = ax9.bar(metrics_names, encoder_scores, alpha=0.7, color='coral')
    ax9.set_ylabel('Similarity Score')
    ax9.set_title('RSA Metrics: Encoder Weights')
    ax9.set_ylim([0, 1])
    ax9.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    for bar, score in zip(bars, encoder_scores):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 9. Decoder weights RSA metrics
    ax10 = plt.subplot(3, 4, 10)
    decoder_scores = [
        all_results['decoder']['cka_linear'],
        all_results['decoder']['cka_rbf'],
        all_results['decoder']['mean_cca'],
        all_results['decoder']['svcca_similarity'],
        all_results['decoder']['procrustes_similarity']
    ]
    bars = ax10.bar(metrics_names, decoder_scores, alpha=0.7, color='seagreen')
    ax10.set_ylabel('Similarity Score')
    ax10.set_title('RSA Metrics: Decoder Weights')
    ax10.set_ylim([0, 1])
    ax10.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    for bar, score in zip(bars, decoder_scores):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 10. L1 penalty effect (SAE specific)
    ax11 = plt.subplot(3, 4, 11)
    l1_norms_per_sample = np.mean(np.abs(z_sae), axis=1)
    ax11.hist(l1_norms_per_sample, bins=50, alpha=0.7, color='purple')
    ax11.set_xlabel('Mean L1 Norm per Sample')
    ax11.set_ylabel('Count')
    ax11.set_title(f'SAE L1 Regularization Effect (λ={L1_LAMBDA})')
    ax11.axvline(x=np.mean(l1_norms_per_sample), color='red', linestyle='--', 
                label=f'Mean: {np.mean(l1_norms_per_sample):.3f}')
    ax11.legend()
    
    # 11. Summary text
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary_text = f"""RSA Comparison Summary
    
Training Configuration:
• Samples: {N_SAMPLES}
• Features: {N_FEATURES}
• Bottleneck: {BOTTLENECK_DIM}
• Sparsity: {SPARSITY}
• Epochs: {N_EPOCHS}
• L1 Lambda (SAE): {L1_LAMBDA}

Key Findings:
• Highest similarity: {max(latent_scores):.3f}
  ({metrics_names[np.argmax(latent_scores)]})
• Mean CCA: {all_results['latent']['mean_cca']:.3f}
• Angular distance: {np.degrees(all_results['latent']['angular_distance']):.1f}°

Reconstruction:
• AE MSE: {np.mean(ae_errors):.4f}
• SAE MSE: {np.mean(sae_errors):.4f}
• SAE sparsity: {np.mean(np.abs(z_sae) < 0.01):.3f}
"""
    
    ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Comprehensive RSA Analysis: Autoencoder vs Sparse Autoencoder', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(results_dir / 'rsa_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved to {results_dir / 'rsa_comprehensive_analysis.png'}")


def main():
    """Run the comprehensive RSA comparison experiment."""
    
    print("="*60)
    print("RSA COMPARISON EXPERIMENT: AE vs SAE")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Generate training data
    print(f"\nGenerating sparse data with sparsity={SPARSITY}...")
    X_train, feature_importance = generate_sparse_data(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        sparsity=SPARSITY,
        random_seed=RANDOM_SEED
    )
    
    # Generate test data for RSA analysis
    X_test, _ = generate_sparse_data(
        n_samples=N_TEST_SAMPLES,
        n_features=N_FEATURES,
        sparsity=SPARSITY,
        random_seed=RANDOM_SEED + 1
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Feature importance decay: 0.7^i")
    
    # Train both models
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    ae, sae, ae_losses, sae_losses = train_models(X_train, feature_importance)
    
    print(f"\nFinal losses:")
    print(f"  AE: {ae_losses[-1]:.6f}")
    print(f"  SAE: {sae_losses[-1]:.6f}")
    
    # Extract representations for test data
    print("\n" + "="*60)
    print("EXTRACTING REPRESENTATIONS")
    print("="*60)
    
    z_ae_test = extract_representations(ae, X_test, model_type='ae')
    z_sae_test = extract_representations(sae, X_test, model_type='sae')
    
    print(f"AE latent shape: {z_ae_test.shape}")
    print(f"SAE latent shape: {z_sae_test.shape}")
    
    # Compute RSA metrics on latent representations
    print("\n" + "="*60)
    print("RSA ANALYSIS: LATENT REPRESENTATIONS")
    print("="*60)
    
    latent_results = compute_all_rsa_metrics(
        z_ae_test, z_sae_test,
        labels=['AE Latent', 'SAE Latent']
    )
    
    # Analyze weight matrices
    encoder_results, decoder_results = analyze_weight_matrices(ae, sae)
    
    # Compile all results
    all_results = {
        'latent': latent_results,
        'encoder': encoder_results,
        'decoder': decoder_results
    }
    
    # Create visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    visualize_results(ae, sae, X_test, all_results)
    
    # Print final summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print(f"1. Latent Representation Similarity (Linear CKA): {latent_results['cka_linear']:.4f}")
    print(f"2. Weight Space Alignment (Procrustes): {latent_results['procrustes_similarity']:.4f}")
    print(f"3. Linear Relationship (Mean CCA): {latent_results['mean_cca']:.4f}")
    print(f"4. Angular Distance: {np.degrees(latent_results['angular_distance']):.2f}°")
    
    if latent_results['cka_linear'] > 0.7:
        print("\n→ High similarity: Models learn similar representations despite different objectives")
    elif latent_results['cka_linear'] > 0.4:
        print("\n→ Moderate similarity: Some shared structure but notable differences")
    else:
        print("\n→ Low similarity: Models learn substantially different representations")
    
    return ae, sae, all_results


if __name__ == "__main__":
    ae, sae, results = main()