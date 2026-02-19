#!/usr/bin/env python3
"""
3D Geometric superposition visualization for 3D hidden layers
Following the detailed instructions for 4-3-4 and 8-3-8 architectures
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Polygon
# PCA not needed for 3D hidden layers
from scipy.spatial import ConvexHull
import sys
sys.path.append('src')

from data_generation import generate_synthetic_data, get_feature_importances
from models_numpy import train_model

def order_ring(points):
    """Order points by angle around z-axis to form a proper ring."""
    angles = np.arctan2(points[:,1], points[:,0])
    return points[np.argsort(angles)]

def plot_antiprism_enhanced(U, ax, architecture_name, sparsity):
    """
    Enhanced antiprism visualization for 8-feature geometries with GPT-5 suggestions.
    Separates top/bottom squares and shows stagger connections clearly.
    """
    if len(U) != 8:
        return False  # Only works for 8 features
    
    # Normalize to unit sphere
    U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-8)
    
    # Split by z-coordinate into top and bottom
    top_idx = np.where(U[:,2] >= 0)[0]
    bottom_idx = np.where(U[:,2] < 0)[0]
    
    if len(top_idx) != 4 or len(bottom_idx) != 4:
        # If not naturally split 4+4, use a different approach
        # Sort by z and take top/bottom 4
        z_sorted_idx = np.argsort(U[:,2])
        bottom_idx = z_sorted_idx[:4]
        top_idx = z_sorted_idx[4:]
    
    top = order_ring(U[top_idx])
    bottom = order_ring(U[bottom_idx])
    
    # Draw enhanced unit sphere wireframe
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_wireframe(xs, ys, zs, linewidth=0.4, alpha=0.12, color='lightsteelblue')
    
    # Plot points with different colors for top/bottom
    ax.scatter(top[:,0], top[:,1], top[:,2], s=60, color="#1f77b4", 
              depthshade=False, alpha=0.9, edgecolor='white', linewidth=1)
    ax.scatter(bottom[:,0], bottom[:,1], bottom[:,2], s=60, color="#ff7f0e", 
              depthshade=False, alpha=0.9, edgecolor='white', linewidth=1)
    
    def edges_ring(P, color, label=None):
        """Draw closed polygon edges with bold lines."""
        k = len(P)
        for i in range(k):
            a, b = P[i], P[(i+1) % k]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 
                   color=color, linewidth=2.5, alpha=0.9, 
                   label=label if i == 0 else None)
    
    # Draw square edges for top and bottom
    edges_ring(top, "#1f77b4", "Top Square")
    edges_ring(bottom, "#ff7f0e", "Bottom Square")
    
    # Draw dashed stagger edges (antiprism connections)
    for i in range(4):
        a = top[i]
        b = bottom[(i+1) % 4]  # Staggered connection
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 
               color="crimson", linewidth=2, linestyle="--", alpha=0.8,
               label="Stagger Edges" if i == 0 else None)
    
    # Optional: very light triangular faces
    hull = ConvexHull(U)
    faces = [U[s] for s in hull.simplices]
    poly = art3d.Poly3DCollection(faces, alpha=0.1, edgecolor="none", 
                                 facecolor="lightgray")
    ax.add_collection3d(poly)
    
    # Minimal feature labels with simplified indexing
    for i, p in enumerate(U):
        ax.text(p[0]*1.15, p[1]*1.15, p[2]*1.15, f'F{i}', 
               fontsize=10, ha="center", va="center", fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Add antiprism info
    ax.text2D(0.02, 0.02, f'Square Antiprism: 8V, {len(hull.simplices)}F',
             transform=ax.transAxes, fontsize=10, color='crimson', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    # Add legend
    ax.legend(loc="upper left", fontsize=9)
    
    return True

def plot_3d_superposition(W_enc, architecture_name, sparsity, save_dir, params=None):
    """
    Create 2-panel figure: (A) superposition matrix W^T @ W, (B) 3D feature geometry on unit sphere
    
    Args:
        W_enc: Encoder weight matrix (m_hidden, n_features)
        params: dict containing model parameters (optional, for bias vector)
    """
    m_hidden, n_features = W_enc.shape
    
    print(f"Processing {architecture_name}: {m_hidden}D hidden, {n_features} features")
    
    # Step 1: Extract feature direction vectors
    V_raw = W_enc.T  # Shape: (n_features, m_hidden), rows are feature vectors
    
    # Step 2: Skip whitening for simplicity (can add later if needed)
    
    # Step 3: Reduce to 3D if needed
    if m_hidden == 3:
        V_3d = V_raw  # Perfect! Already 3D
    elif m_hidden > 3:
        print(f"  Error: {m_hidden}D > 3D not supported (would need PCA)")
        return
    else:
        print(f"  Error: {m_hidden}D < 3D not supported (need 3D for this viz)")
        return
    
    # Step 4: Normalize and record strengths
    strengths = np.linalg.norm(V_3d, axis=1)  # Feature strengths s_i
    U = np.zeros_like(V_3d)  # Unit directions
    for i in range(n_features):
        if strengths[i] > 1e-8:
            U[i] = V_3d[i] / strengths[i]
        else:
            U[i] = V_3d[i]  # Keep zero vectors as zero
    
    # Step 5: Compute superposition matrix in the space we're plotting
    if m_hidden > 3:
        # Use PCA-projected vectors for consistency with geometry plot
        Wf = V_3d  # Use the 3D projected vectors
    else:
        Wf = V_raw  # Use original 3D vectors
    
    S = Wf @ Wf.T  # Superposition matrix: (n_features, n_features)
    
    # Include bias vector if provided
    if params and 'b' in params:
        b = params['b']  # Get bias vector
        bias_column = b.reshape(-1, 1)
        display_matrix = np.hstack([bias_column, S])
        feature_labels = ['b'] + [f'F{i}' for i in range(n_features)]
        matrix_dim = n_features + 1
        
        # Bias statistics for title
        bias_norm = np.linalg.norm(b)
        bias_max = np.max(np.abs(b))
        bias_info = f" | Bias norm: {bias_norm:.3f}, max: {bias_max:.3f}"
    else:
        display_matrix = S
        feature_labels = [f'F{i}' for i in range(n_features)]
        matrix_dim = n_features
        bias_info = ""
    
    # Create the figure
    fig = plt.figure(figsize=(18, 8))
    
    # === LEFT PANEL: Superposition Matrix ===
    ax1 = plt.subplot(1, 2, 1)
    im = ax1.imshow(display_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Dot Product (with bias)' if params and 'b' in params else 'Dot Product', fontsize=12)
    
    # Add value annotations
    for i in range(n_features):
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
    ax1.set_yticks(range(n_features))
    ax1.set_xticklabels(feature_labels)
    ax1.set_yticklabels([f'F{i}' for i in range(n_features)])
    
    # === RIGHT PANEL: 3D Geometry ===
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Try enhanced antiprism visualization for 8-feature architectures
    if n_features == 8 and plot_antiprism_enhanced(U, ax2, architecture_name, sparsity):
        # Successfully used antiprism visualization
        pass
    else:
        # Fallback to original visualization
        # Step 6: Draw unit sphere wireframe for reference
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v) 
        zs = np.cos(v)
        ax2.plot_wireframe(xs, ys, zs, linewidth=0.5, alpha=0.2, color='lightsteelblue')
        
        # Plot feature vectors as arrows
        colors = plt.cm.tab10(np.linspace(0, 1, n_features))
        
        for i in range(n_features):
            if strengths[i] > 1e-8:  # Only plot non-zero vectors
                endpoint = U[i]
                color = colors[i]
                
                # Draw arrow from origin to endpoint
                ax2.plot([0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]], 
                        linewidth=3, color=color, alpha=0.8)
                
                # Add endpoint marker
                ax2.scatter(endpoint[0], endpoint[1], endpoint[2], 
                           s=60, c=[color], alpha=0.9)
                
                # Add label with strength
                ax2.text(endpoint[0]*1.1, endpoint[1]*1.1, endpoint[2]*1.1,
                        f'F{i}\n({strengths[i]:.2f})', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color=color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Step 7: Enhanced convex hull with highlighted edges
    non_zero_points = U[strengths > 1e-8]
    if len(non_zero_points) >= 4:  # Need at least 4 points for 3D hull
        try:
            hull = ConvexHull(non_zero_points)
            
            # Color faces systematically for better visibility
            face_colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 
                          'lightpink', 'lightgray', 'wheat', 'lavender']
            n_faces = len(hull.simplices)
            
            # For tetrahedron (4 faces) or other simple polyhedra, use distinct colors
            if n_faces <= 4:
                colors_to_use = face_colors[:n_faces]
            else:
                # For complex polyhedra, cycle through colors
                colors_to_use = [face_colors[i % len(face_colors)] for i in range(n_faces)]
            
            # Draw faces with distinct colors
            for i, simplex in enumerate(hull.simplices):
                face = non_zero_points[simplex]
                color = colors_to_use[i]
                poly = art3d.Poly3DCollection([face], alpha=0.3, facecolor=color, 
                                            edgecolor='darkred', linewidth=0.5)
                ax2.add_collection3d(poly)
            
            # Highlight edges with depth-based coloring (blue=background, red=foreground)
            # Get current view direction (camera position) - use simple heuristic
            all_edges = []
            for simplex in hull.simplices:
                # Draw edges of each triangular face
                for i in range(len(simplex)):
                    start_point = non_zero_points[simplex[i]]
                    end_point = non_zero_points[simplex[(i+1) % len(simplex)]]
                    
                    # Calculate edge midpoint and distance from viewer
                    midpoint = (start_point + end_point) / 2
                    # Simple depth heuristic: distance from origin projected along view direction
                    # Assume viewer is looking from positive x,y,z direction
                    depth = midpoint[0] + midpoint[1] + midpoint[2]  # Simple depth proxy
                    
                    all_edges.append((start_point, end_point, depth))
            
            # Sort edges by depth and determine foreground/background
            all_edges.sort(key=lambda x: x[2])  # Sort by depth
            n_edges = len(all_edges)
            
            for i, (start_point, end_point, depth) in enumerate(all_edges):
                # Background edges (further) in blue, foreground edges (closer) in red
                if i < n_edges // 2:  # Background half
                    color = 'steelblue'
                    alpha = 0.6
                else:  # Foreground half
                    color = 'red' 
                    alpha = 0.9
                
                ax2.plot([start_point[0], end_point[0]], 
                        [start_point[1], end_point[1]], 
                        [start_point[2], end_point[2]], 
                        color=color, linestyle='--', linewidth=2, alpha=alpha)
            
            # Count vertices and faces for title
            n_vertices = len(non_zero_points)
            n_faces = len(hull.simplices)
            
            # Determine polyhedron type
            if n_vertices == 4:
                shape_name = "Tetrahedron"
            elif n_vertices == 6:
                shape_name = "Octahedron"
            elif n_vertices == 8:
                shape_name = "Complex Polyhedron"
            else:
                shape_name = f"{n_vertices}-vertex Polyhedron"
            
            ax2.text2D(0.02, 0.02, f'{shape_name}: {n_vertices}V, {n_faces}F',
                      transform=ax2.transAxes, fontsize=10, color='red', fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        except Exception as e:
            print(f"Could not compute 3D convex hull: {e}")
    
    # Formatting with enhanced grid
    ax2.set_xlim([-1.2, 1.2])
    ax2.set_ylim([-1.2, 1.2]) 
    ax2.set_zlim([-1.2, 1.2])
    ax2.set_box_aspect((1,1,1))
    
    # Add coordinate axes for reference in blue
    ax2.plot([0, 1], [0, 0], [0, 0], 'steelblue', linestyle=':', alpha=0.4, linewidth=1.5)  # X axis
    ax2.plot([0, 0], [0, 1], [0, 0], 'steelblue', linestyle=':', alpha=0.4, linewidth=1.5)  # Y axis  
    ax2.plot([0, 0], [0, 0], [0, 1], 'steelblue', linestyle=':', alpha=0.4, linewidth=1.5)  # Z axis
    
    ax2.set_xlabel('Hidden Dim 1', fontsize=12)
    ax2.set_ylabel('Hidden Dim 2', fontsize=12)
    ax2.set_zlabel('Hidden Dim 3', fontsize=12)
    
    # Improve grid visibility with blue color
    ax2.grid(True, alpha=0.25, color='lightsteelblue')
    
    title_suffix = f" (PCA from {m_hidden}D)" if m_hidden > 3 else ""
    ax2.set_title(f'3D Feature Geometry{title_suffix}\n{architecture_name} (Sparsity {sparsity})', 
                  fontsize=14, fontweight='bold')
    
    # Step 8: Overall layout
    fig.suptitle(f'3D Superposition Analysis: {architecture_name} at Sparsity {sparsity}',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'{save_dir}/3d_combined_s{sparsity}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return S, U, strengths

def analyze_3d_superposition():
    """Run 3D geometric analysis on 4-3-4 and 8-3-8 architectures"""
    
    # Architectures with 3D hidden layer (only 3D architectures for this script)
    architectures = [
        {'name': '4-3-4', 'sparse_dim': 4, 'dense_dim': 3},
        {'name': '8-3-8', 'sparse_dim': 8, 'dense_dim': 3}
    ]
    
    sparsities = sorted([0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89])  # Ensure ascending order
    seeds = [123, 459, 724]
    
    print("3D GEOMETRIC SUPERPOSITION ANALYSIS")
    print("="*60)
    print("Creating superposition matrix + 3D geometry visualizations")
    print("Following matplotlib 3D gallery instructions")
    print("="*60)
    
    results = {}
    
    for seed in seeds:
        print(f"SEED: {seed}")
        for arch in architectures:
            print(f"\nAnalyzing {arch['name']} architecture...")
            
            # Create directory for this architecture
            import os
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = f"results/3d_geometric_{arch['name']}_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            
            arch_results = {}
            
            for sparsity in sparsities:
                print(f"  Training at sparsity {sparsity}...")
                
                # Generate data and train (reduced samples for speed)
                data = generate_synthetic_data(seed, arch['sparse_dim'], sparsity, 2000)
                I = get_feature_importances(arch['sparse_dim'], 0.7)
                
                # Train autoencoder
                params, final_loss = train_model(data, I, k=arch['dense_dim'], n=arch['sparse_dim'], 
                                num_epochs=3, seed=seed)
                W_enc = params['W']  # Shape: (hidden_dim, feature_dim)
                
                # Create 3D visualization
                S, U, strengths = plot_3d_superposition(W_enc, arch['name'], sparsity, save_dir, params)
                
                # Store results
                arch_results[sparsity] = {
                    'superposition_matrix': S,
                    'unit_directions': U,
                    'feature_strengths': strengths,
                    'weight_matrix': W_enc
                }
                
            results[arch['name']] = arch_results
            print(f"  Saved 3D plots to: {save_dir}")
        
        print(f"\n{'='*60}")
        print("3D GEOMETRIC ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print("Expectations:")
        print("• 4-3-4: Features may be orthogonal or near-orthogonal (tetrahedral)")
        print("• 8-3-8: Features will form complex 3D arrangements (8 points on sphere)")
        print("• High sparsity: Regular polyhedra or uniform sphere packings")
        print("• Convex hulls reveal geometric structure of superposition")
        
        return results

if __name__ == "__main__":
    results = analyze_3d_superposition()