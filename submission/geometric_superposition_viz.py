#!/usr/bin/env python3
"""
Geometric superposition visualization - inspired by Anthropic's toy model paper
Shows how features arrange in 2D hidden space as polygons when in superposition
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import sys
sys.path.append('src')

from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import train_model

def plot_combined_visualization(W, architecture_name, sparsity, save_dir, params=None):
    """
    Plot both superposition matrix and feature geometry side by side
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
        combined_matrix = np.hstack([bias_column, superposition_matrix])
        feature_labels = ['b'] + [f'F{i}' for i in range(feature_dim)]
        display_matrix = combined_matrix
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
    
    # Keep original feature vectors (columns of W) - don't normalize to unit length
    feature_vectors = []
    feature_norms = []
    for i in range(feature_dim):
        vec = W[:, i]  # Column i is feature i's representation
        norm = np.linalg.norm(vec)
        feature_vectors.append(vec)  # Keep original magnitude
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
    
    max_norm = max(feature_norms) if feature_norms else 1
    
    for i, (vec, norm, color) in enumerate(zip(feature_vectors, feature_norms, colors)):
        if norm > 1e-8:  # Only plot non-zero vectors
            ax2.arrow(0, 0, vec[0], vec[1], head_width=0.05, head_length=0.05, 
                     fc=color, ec=color, linewidth=3, alpha=0.8, 
                     label=f'Feature {i}')
            # Add feature label at tip with norm info
            label_pos = vec * 1.15 if norm > 0.1 else vec * 3  # Push small vectors further out
            ax2.text(label_pos[0], label_pos[1], f'F{i}\n({norm:.2f})', fontsize=10, 
                    ha='center', va='center', color=color, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Compute and plot convex hull if we have enough non-zero points
    non_zero_points = feature_vectors[np.linalg.norm(feature_vectors, axis=1) > 1e-8]
    
    if len(non_zero_points) >= 3:
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(non_zero_points)
            # Plot convex hull
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
    
    # Title and labels for geometry plot
    ax2.set_title(f'Feature Geometry in 2D Hidden Space\n{architecture_name} (Sparsity {sparsity})', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hidden Dimension 1', fontsize=12)
    ax2.set_ylabel('Hidden Dimension 2', fontsize=12)
    
    # Add interpretation text with norm info
    interpretation_text = (
        "Vector length = Feature strength\n"
        "Axis-aligned = No superposition\n"  
        "Regular polygon = Superposition"
    )
    ax2.text(-1.4, 1.4, interpretation_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # Add connection explanation
    fig.suptitle(f'Superposition Analysis: Matrix ↔ Geometry Connection', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Save plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'{save_dir}/combined_s{sparsity}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_geometric_superposition():
    """Run geometric analysis on 3-2-3 and 5-2-5 architectures"""
    
    # Architectures with 2D hidden layer
    architectures = [
        {'name': '3-2-3', 'sparse_dim': 3, 'dense_dim': 2},
        {'name': '5-2-5', 'sparse_dim': 5, 'dense_dim': 2}
    ]
    
    sparsities = [0.3, 0.5, 0.75, 0.85, 0.9, 0.95]
    seed = 123
    
    print("GEOMETRIC SUPERPOSITION ANALYSIS")
    print("="*60)
    print("Following Anthropic's toy model paper approach")
    print("Looking for polygon structures in 2D hidden space")
    print("="*60)
    
    for arch in architectures:
        print(f"\nAnalyzing {arch['name']} architecture...")
        
        # Create directory for this architecture
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f"results/geometric_{arch['name']}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        for sparsity in sparsities:
            print(f"  Training at sparsity {sparsity}...")
            
            # Generate data and train (reduced for speed)
            data = generate_synthetic_data(seed, arch['sparse_dim'], sparsity, 2000)
            I = get_feature_importances(arch['sparse_dim'], 0.7)
            
            # Train autoencoder (fewer epochs for speed)
            params = train_model(data, I, k=arch['dense_dim'], n=arch['sparse_dim'], 
                               num_epochs=3, seed=seed)
            W = params['W']  # Shape: (hidden_dim, feature_dim)
            
            # Create combined visualization
            plot_combined_visualization(W, arch['name'], sparsity, save_dir, params)
            
        print(f"  Saved geometric plots to: {save_dir}")
    
    print(f"\n{'='*60}")
    print("GEOMETRIC ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print("Look for these patterns:")
    print("• Low sparsity: Features near coordinate axes (no superposition)")
    print("• High sparsity: Features arranged in regular polygons (superposition)")
    print("• Triangle for 3 features, Pentagon for 5 features")

if __name__ == "__main__":
    analyze_geometric_superposition()