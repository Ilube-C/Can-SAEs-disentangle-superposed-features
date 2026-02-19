"""
Geometric Visualizations for Autoencoder Superposition Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from src.models_numpy import init_params, train_model, get_bottleneck_activations, forward
from src.data_generation import generate_synthetic_data, get_feature_importances
from src.analysis import compute_superposition_matrix
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D, art3d


def plot_2d_geometry(W, architecture_name, sparsity, save_dir, params=None):
    """
    Plot both superposition matrix and 2D feature geometry side by side
    W: weight matrix (hidden_dim x feature_dim)
    params: dict containing model parameters (optional, for bias vector)
    """
    hidden_dim, feature_dim = W.shape
    
    if hidden_dim != 2:
        print(f"Skipping {architecture_name} - not 2D hidden layer (dim={hidden_dim})")
        return
    
    # Compute superposition matrix
    superposition_matrix = W.T @ W
    
    # Include bias vector if provided
    if params and 'b' in params:
        b = params['b']  # Get bias vector
        bias_column = b.reshape(-1, 1)
        display_matrix = np.hstack([bias_column, superposition_matrix])
        feature_labels = ['b'] + [f'F{i}' for i in range(feature_dim)]
        matrix_dim = feature_dim + 1
        
        # Bias statistics for title
        bias_norm = np.linalg.norm(b)
        bias_max = np.max(np.abs(b))
        bias_info = f" | Bias norm: {bias_norm:.3f}, max: {bias_max:.3f}"
    else:
        display_matrix = superposition_matrix
        feature_labels = [f'F{i}' for i in range(feature_dim)]
        matrix_dim = feature_dim
        bias_info = ""
    
    # Keep original feature vectors (columns of W)
    feature_vectors = []
    feature_norms = []
    for i in range(feature_dim):
        vec = W[:, i]  # Column i is feature i's representation
        norm = np.linalg.norm(vec)
        feature_vectors.append(vec)
        feature_norms.append(norm)
    
    feature_vectors = np.array(feature_vectors)  # shape: (feature_dim, 2)
    
    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === LEFT PLOT: Superposition Matrix ===
    im = ax1.imshow(display_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('W^T @ W (with bias)' if params and 'b' in params else 'W^T @ W', fontsize=12)
    
    # Add text annotations
    for i in range(feature_dim):
        for j in range(matrix_dim):
            text = ax1.text(j, i, f'{display_matrix[i, j]:.2f}', 
                           ha='center', va='center', fontweight='bold',
                           color='white' if abs(display_matrix[i, j]) > 0.5 else 'black')
    
    # Add vertical separator line after bias column if present
    if params and 'b' in params:
        ax1.axvline(x=0.5, color='red', linewidth=2, alpha=0.7)
    
    ax1.set_title(f'Superposition Matrix W^T @ W\n{architecture_name} (Sparsity {sparsity}){bias_info}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature j', fontsize=12)
    ax1.set_ylabel('Feature i', fontsize=12)
    ax1.set_xticks(range(matrix_dim))
    ax1.set_yticks(range(feature_dim))
    ax1.set_xticklabels(feature_labels)
    ax1.set_yticklabels([f'F{i}' for i in range(feature_dim)])
    
    # === RIGHT PLOT: Feature Geometry ===
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='lightgray', linestyle='--', alpha=0.7)
    ax2.add_patch(circle)
    
    # Plot feature vectors as arrows
    colors = plt.cm.tab10(np.linspace(0, 1, feature_dim))
    
    for i, (vec, norm, color) in enumerate(zip(feature_vectors, feature_norms, colors)):
        if norm > 1e-8:  # Only plot non-zero vectors
            ax2.arrow(0, 0, vec[0], vec[1], head_width=0.05, head_length=0.05, 
                     fc=color, ec=color, linewidth=3, alpha=0.8, 
                     label=f'Feature {i}')
            # Add feature label at tip with norm info
            label_pos = vec * 1.15 if norm > 0.1 else vec * 3
            ax2.text(label_pos[0], label_pos[1], f'F{i}\n({norm:.2f})', fontsize=10, 
                    ha='center', va='center', color=color, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Compute and plot convex hull if we have enough non-zero points
    non_zero_points = feature_vectors[np.linalg.norm(feature_vectors, axis=1) > 1e-8]
    
    if len(non_zero_points) >= 3:
        try:
            hull = ConvexHull(non_zero_points)
            hull_points = non_zero_points[hull.vertices]
            polygon = Polygon(hull_points, fill=False, edgecolor='red', 
                            linewidth=2, alpha=0.7, linestyle='-')
            ax2.add_patch(polygon)
            ax2.text(0, -1.3, f'Convex Hull: {len(hull_points)}-sided polygon', 
                    ha='center', color='red', fontweight='bold')
        except:
            print(f"Could not compute convex hull for {architecture_name}")
    
    # Formatting for geometry plot
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    ax2.set_title(f'Feature Geometry in 2D Hidden Space\n{architecture_name} (Sparsity {sparsity})', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hidden Dimension 1', fontsize=12)
    ax2.set_ylabel('Hidden Dimension 2', fontsize=12)
    
    # Add interpretation text
    interpretation_text = (
        "Vector length = Feature strength\n"
        "Axis-aligned = No superposition\n"  
        "Regular polygon = Superposition"
    )
    ax2.text(-1.4, 1.4, interpretation_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # Save plot
    plt.tight_layout()
    # Use architecture name from the parameter, replacing any problematic characters for filenames
    safe_arch_name = architecture_name.replace('-', '_')
    plt.savefig(f'{save_dir}/2d_geometry_{safe_arch_name}_s{sparsity}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_geometry(W, architecture_name, sparsity, save_dir, params=None):
    """
    Plot both superposition matrix and 3D feature geometry side by side
    W: weight matrix (hidden_dim x feature_dim)
    params: dict containing model parameters (optional, for bias vector)
    """
    hidden_dim, feature_dim = W.shape
    
    if hidden_dim != 3:
        print(f"Skipping {architecture_name} - not 3D hidden layer (dim={hidden_dim})")
        return
    
    # Extract feature vectors and normalize
    V_raw = W.T  # Shape: (n_features, 3)
    strengths = np.linalg.norm(V_raw, axis=1)
    U = np.zeros_like(V_raw)
    for i in range(feature_dim):
        if strengths[i] > 1e-8:
            U[i] = V_raw[i] / strengths[i]
    
    # Compute superposition matrix
    S = V_raw @ V_raw.T
    
    # Include bias vector if provided
    if params and 'b' in params:
        b = params['b']  # Get bias vector
        bias_column = b.reshape(-1, 1)
        display_matrix = np.hstack([bias_column, S])
        feature_labels = ['b'] + [f'F{i}' for i in range(feature_dim)]
        matrix_dim = feature_dim + 1
        
        # Bias statistics for title
        bias_norm = np.linalg.norm(b)
        bias_max = np.max(np.abs(b))
        bias_info = f" | Bias norm: {bias_norm:.3f}, max: {bias_max:.3f}"
    else:
        display_matrix = S
        feature_labels = [f'F{i}' for i in range(feature_dim)]
        matrix_dim = feature_dim
        bias_info = ""
    
    # Create figure
    fig = plt.figure(figsize=(18, 8))
    
    # === LEFT PANEL: Superposition Matrix ===
    ax1 = plt.subplot(1, 2, 1)
    im = ax1.imshow(display_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Dot Product (with bias)' if params and 'b' in params else 'Dot Product', fontsize=12)
    
    # Add value annotations
    for i in range(feature_dim):
        for j in range(matrix_dim):
            color = 'white' if abs(display_matrix[i, j]) > 0.5 else 'black'
            ax1.text(j, i, f'{display_matrix[i, j]:.2f}', ha='center', va='center', 
                    fontweight='bold', color=color, fontsize=9)
    
    # Add vertical separator line after bias column if present
    if params and 'b' in params:
        ax1.axvline(x=0.5, color='red', linewidth=2, alpha=0.7)
    
    ax1.set_title(f'Superposition Matrix\n{architecture_name} (Sparsity {sparsity}){bias_info}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature j', fontsize=12)
    ax1.set_ylabel('Feature i', fontsize=12)
    ax1.set_xticks(range(matrix_dim))
    ax1.set_yticks(range(feature_dim))
    ax1.set_xticklabels(feature_labels)
    ax1.set_yticklabels([f'F{i}' for i in range(feature_dim)])
    
    # === RIGHT PANEL: 3D Geometry ===
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Draw unit sphere wireframe
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v) 
    zs = np.cos(v)
    ax2.plot_wireframe(xs, ys, zs, linewidth=0.5, alpha=0.2, color='lightsteelblue')
    
    # Plot feature vectors
    colors = plt.cm.tab10(np.linspace(0, 1, feature_dim))
    
    for i in range(feature_dim):
        if strengths[i] > 1e-8:
            endpoint = U[i]
            color = colors[i]
            
            # Draw arrow from origin
            ax2.plot([0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]], 
                    linewidth=3, color=color, alpha=0.8)
            
            # Add endpoint marker
            ax2.scatter(endpoint[0], endpoint[1], endpoint[2], 
                       s=60, c=[color], alpha=0.9)
            
            # Add label
            ax2.text(endpoint[0]*1.1, endpoint[1]*1.1, endpoint[2]*1.1,
                    f'F{i}\n({strengths[i]:.2f})', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Try to compute convex hull
    non_zero_points = U[strengths > 1e-8]
    if len(non_zero_points) >= 4:
        try:
            hull = ConvexHull(non_zero_points)
            for simplex in hull.simplices:
                face = non_zero_points[simplex]
                poly = art3d.Poly3DCollection([face], alpha=0.2, facecolor='lightblue', 
                                            edgecolor='darkred', linewidth=0.5)
                ax2.add_collection3d(poly)
        except Exception as e:
            print(f"Could not compute 3D convex hull: {e}")
    
    # Formatting
    ax2.set_xlim([-1.2, 1.2])
    ax2.set_ylim([-1.2, 1.2]) 
    ax2.set_zlim([-1.2, 1.2])
    ax2.set_box_aspect((1,1,1))
    ax2.set_xlabel('Hidden Dim 1', fontsize=12)
    ax2.set_ylabel('Hidden Dim 2', fontsize=12)
    ax2.set_zlabel('Hidden Dim 3', fontsize=12)
    ax2.grid(True, alpha=0.25, color='lightsteelblue')
    
    ax2.set_title(f'3D Feature Geometry\n{architecture_name} (Sparsity {sparsity})', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    # Use architecture name from the parameter, replacing any problematic characters for filenames
    safe_arch_name = architecture_name.replace('-', '_')
    plt.savefig(f'{save_dir}/3d_geometry_{safe_arch_name}_s{sparsity}.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_autoencoder(sparsity, seed, n=20, k=5, num_samples=10000, num_epochs=10, test_ratio=0.2):
    """Train an autoencoder with train/test split for proper evaluation."""
    # Generate total data (train + test)
    total_samples = int(num_samples / (1 - test_ratio))  # Ensure we get the requested training samples
    X_total = generate_synthetic_data(seed, n, sparsity, total_samples)
    I = get_feature_importances(n)
    
    # Split into train/test
    n_test = int(total_samples * test_ratio)
    n_train = total_samples - n_test
    
    # Use numpy random state for reproducible splits
    np.random.seed(seed + 1000)  # Different seed for split to avoid correlation
    indices = np.random.permutation(total_samples)
    
    train_indices = indices[:n_train] 
    test_indices = indices[n_train:]
    
    X_train = X_total[train_indices]
    X_test = X_total[test_indices]
    
    print(f"  Data split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train model on training data only
    params, train_loss = train_model(
        X_train, I, k, n,
        num_epochs=num_epochs,
        learning_rate=0.01,
        seed=seed
    )
    
    # Evaluate on test data
    test_reconstructions = forward(params, X_test)
    test_loss = np.mean(I * (X_test - test_reconstructions) ** 2)
    
    print(f"  Test loss: {test_loss:.6f}")
    
    return params, train_loss, test_loss, X_train, X_test, I




def plot_superposition_matrices(results, sparsities, architectures, results_dir):
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
    """Plot metrics heatmap for autoencoder analysis."""
    for arch in architectures:
        # Autoencoder metrics focused on test performance
        metrics_names = ['AE Test Loss', 'Diag Strength', 'Off-Diag Inter', 'Active Features']
        metrics_keys = ['ae_test_loss', 'diag_strength', 'off_diag_interference', 'ae_active']
        
        data_matrix = []
        for sparsity in sparsities:
            key = f"{arch['name']}_s{sparsity}"
            row = []
            for metric_key in metrics_keys:
                if metric_key == 'ae_test_loss':
                    # Get test loss from main results dict
                    value = results[key].get(metric_key, np.nan)
                else:
                    # Get other metrics from analysis_metrics
                    value = results[key].get('analysis_metrics', {}).get(metric_key, np.nan)
                
                # Invert loss values for heatmap (lower loss = higher score)
                if metric_key == 'ae_test_loss' and not np.isnan(value) and value > 0:
                    value = 1.0 / (1.0 + value)  # Transform to [0,1] range where higher = better
                elif metric_key == 'ae_active':
                    # Normalize active features by dimension
                    value = value / arch['sparse_dim'] if not np.isnan(value) else 0
                    
                row.append(value if not np.isnan(value) else 0)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix).T
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(data_matrix, 
                    xticklabels=[f'{s}' for s in sparsities],
                    yticklabels=metrics_names,
                    annot=True, 
                    fmt='.3f',
                    cmap='YlOrRd',
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'Normalized Score'})
        
        plt.title(f'Autoencoder Analysis Metrics\nArchitecture: {arch["name"]}', 
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


if __name__ == "__main__":
    # Create timestamped results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/geometric_viz_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    #experiment setup
    #sparsities = [0.5, 0.6, 0.7, 0.8, 0.9]
    sparsities = [0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89]
    architectures = [
        # 2D hidden layer architectures
        {'name': '3-2-3', 'sparse_dim': 3, 'dense_dim': 2},
        {'name': '5-2-5', 'sparse_dim': 5, 'dense_dim': 2},
        {'name': '7-2-7', 'sparse_dim': 7, 'dense_dim': 2},
        # 3D hidden layer architectures
        {'name': '4-3-4', 'sparse_dim': 4, 'dense_dim': 3},
        {'name': '7-3-7', 'sparse_dim': 7, 'dense_dim': 3},
        {'name': '8-3-8', 'sparse_dim': 8, 'dense_dim': 3},
    ]
    seeds = [345, 629, 801]

    results = {}
    
    for seed in seeds:
        # Create seed-specific results directory
        seed_results_dir = os.path.join(results_dir, f'seed_{seed}')
        os.makedirs(seed_results_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"SEED: {seed}")
        print(f"{'='*60}")
        
        seed_results = {}  # Store results for this seed
        
        for arch in architectures:
            print(f"\n{'='*50}")
            print(f"Architecture: {arch['name']} (Seed: {seed})")
            print(f"{'='*50}")
            
            for sparsity in sparsities:
                print(f"\n--- Sparsity: {sparsity} (Seed: {seed}) ---")
                
                # Train autoencoder with train/test split
                params, train_loss, test_loss, X_train, X_test, I = train_autoencoder(
                    sparsity=sparsity, 
                    seed=seed,
                    n=arch['sparse_dim'],
                    k=arch['dense_dim'],
                    num_samples=4000,
                    num_epochs=10
                )
                
                # Use test data for analysis and visualization (unbiased)
                # Get bottleneck activations from test data
                bottleneck_activations = get_bottleneck_activations(params, X_test)
                print(f"Test bottleneck activations shape: {bottleneck_activations.shape}")
                
                # Get autoencoder reconstructions from test data
                ae_reconstructions = forward(params, X_test)
                
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
                
                # 3. AE losses (train and test)
                analysis_metrics['ae_train_loss'] = train_loss
                analysis_metrics['ae_test_loss'] = test_loss
                
                # Compute mean number of active features (features with activations > threshold)
                threshold = 1e-6
                ae_active = np.mean(np.sum(ae_reconstructions > threshold, axis=1))
                analysis_metrics['ae_active'] = ae_active
                
                # Print metrics
                print(f"Mean active features: {ae_active:.1f}")
                print(f"Diagonal strength: {diag_strength:.4f}")
                print(f"Off-diagonal interference: {off_diag_interference:.4f}")
                
                # Store results with seed information
                key = f"{arch['name']}_s{sparsity}"
                seed_results[key] = {
                    'ae_params': params,
                    'ae_train_loss': train_loss,
                    'ae_test_loss': test_loss,
                    'activations': bottleneck_activations,
                    'analysis_metrics': analysis_metrics
                }
                
                # Add geometric visualization based on dense dimension
                W = params['W']  # Weight matrix (dense_dim x sparse_dim)
                if arch['dense_dim'] == 2:
                    print(f"Creating 2D geometric visualization (Seed: {seed})...")
                    plot_2d_geometry(W, arch['name'], sparsity, seed_results_dir, params)
                elif arch['dense_dim'] == 3:
                    print(f"Creating 3D geometric visualization (Seed: {seed})...")
                    plot_3d_geometry(W, arch['name'], sparsity, seed_results_dir, params)
                else:
                    print(f"Skipping geometric visualization (dense_dim={arch['dense_dim']}, only 2D and 3D supported)")
            
            # Add separator line after each architecture
            print()
        
        print(f"\n{'='*50}")
        print(f"Experiment Complete for Seed {seed}!")
        print(f"{'='*50}")
        
        # Store this seed's results in main results dict
        results[f'seed_{seed}'] = seed_results
        
        # Generate superposition matrix plots
        print(f"\n{'='*50}")
        print(f"Generating Superposition Matrix Plots (Seed: {seed})...")
        print(f"{'='*50}")
        plot_superposition_matrices(seed_results, sparsities, architectures, seed_results_dir)
        
        # Generate metrics heatmap plots
        print(f"\n{'='*50}")
        print(f"Generating Metrics Heatmap Plots (Seed: {seed})...")
        print(f"{'='*50}")
        plot_metrics_heatmap(seed_results, sparsities, architectures, seed_results_dir)
        
        # Print results table
        print(f"\n{'='*80}")
        print(f"RESULTS TABLE (Seed: {seed})")
        print(f"{'='*80}")
        
        # Print header
        print(f"{'Architecture':<12} {'Sparsity':<8} {'Test Loss':<10} {'Active Feat':<11} {'Diag Str':<9} {'Off-Diag':<9}")
        print(f"{'-'*12} {'-'*8} {'-'*10} {'-'*11} {'-'*9} {'-'*9}")
        
        # Print results for each architecture and sparsity
        for i, arch in enumerate(architectures):
            for sparsity in sparsities:
                key = f"{arch['name']}_s{sparsity}"
                test_loss = seed_results[key]['ae_test_loss']
                ae_active = seed_results[key]['analysis_metrics']['ae_active']
                diag_strength = seed_results[key]['analysis_metrics']['diag_strength']
                off_diag = seed_results[key]['analysis_metrics']['off_diag_interference']
                print(f"{arch['name']:<12} {sparsity:<8.2f} {test_loss:<10.6f} {ae_active:<11.1f} {diag_strength:<9.4f} {off_diag:<9.4f}")
            
            # Add separator line after each architecture (except the last one)
            if i < len(architectures) - 1:
                print(f"{'-'*12} {'-'*8} {'-'*10} {'-'*11} {'-'*9} {'-'*9}")
        
        
        # Save metrics table to file
        print(f"\n{'='*50}")
        print(f"Saving Metrics Table (Seed: {seed})...")
        print(f"{'='*50}")
        
        table_filename = os.path.join(seed_results_dir, f'autoencoder_metrics_table_seed_{seed}.txt')
        with open(table_filename, 'w') as f:
            # Write header
            f.write("="*80 + "\n")
            f.write("AUTOENCODER METRICS TABLE\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Experiment: Geometric Visualizations\n")
            f.write(f"Architectures: {[arch['name'] for arch in architectures]}\n")
            f.write(f"Sparsities: {sparsities}\n")
            f.write("="*80 + "\n\n")
            
            # Write table header
            f.write(f"{'Architecture':<12} {'Sparsity':<8} {'Test Loss':<10} {'Active Feat':<11} {'Diag Str':<9} {'Off-Diag':<9}\n")
            f.write(f"{'-'*12} {'-'*8} {'-'*10} {'-'*11} {'-'*9} {'-'*9}\n")
            
            # Write data rows
            for i, arch in enumerate(architectures):
                for sparsity in sparsities:
                    key = f"{arch['name']}_s{sparsity}"
                    test_loss = seed_results[key]['ae_test_loss']
                    ae_active = seed_results[key]['analysis_metrics']['ae_active']
                    diag_strength = seed_results[key]['analysis_metrics']['diag_strength']
                    off_diag = seed_results[key]['analysis_metrics']['off_diag_interference']
                    
                    f.write(f"{arch['name']:<12} {sparsity:<8.2f} {test_loss:<10.6f} {ae_active:<11.1f} {diag_strength:<9.4f} {off_diag:<9.4f}\n")
                
                # Add separator line after each architecture (except the last one)
                if i < len(architectures) - 1:
                    f.write(f"{'-'*12} {'-'*8} {'-'*10} {'-'*11} {'-'*9} {'-'*9}\n")
            
            # Add metric explanations
            f.write(f"\n\n{'='*80}\n")
            f.write("METRIC EXPLANATIONS\n")
            f.write("="*80 + "\n")
            f.write("Test Loss:       Autoencoder reconstruction loss on held-out test data\n")
            f.write("Active Feat:     Mean number of active features per sample\n")
            f.write("Diag Str:        Diagonal strength of W^T @ W (normalized by dimension)\n")
            f.write("Off-Diag:        Off-diagonal interference in W^T @ W (mean absolute)\n")
        
        print(f"  Saved metrics table to: {table_filename}")
        
        print(f"\n{'='*50}")
        print(f"Seed {seed} results saved to: {seed_results_dir}")
        print(f"{'='*50}")
    
    # After all seeds are complete, print summary
    print(f"\n{'='*60}")
    print("ALL SEEDS COMPLETE - SUMMARY")
    print(f"{'='*60}")
    print(f"Total seeds run: {len(seeds)}")
    print(f"Seeds: {seeds}")
    print(f"Main results directory: {results_dir}")
    for seed in seeds:
        print(f"  - Seed {seed}: {os.path.join(results_dir, f'seed_{seed}')}")
    print(f"{'='*60}")