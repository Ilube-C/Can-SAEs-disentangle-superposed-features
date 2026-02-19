# Superposition Research Project

## Overview
Implementation of toy models studying superposition in neural networks, based on Anthropic's superposition research. The project explores how neural networks represent more features than they have dimensions through geometric analysis and visualization.

## Key Components

### Core Demonstrations
- **Procrustes_abnormality_demo.py**: Demonstrates phase transitions in superposition using Procrustes analysis to measure geometric distortions
- **geometric_visualisations.py**: 2D/3D visualizations of feature representations in bottleneck layers
- **geometric_3d_viz.py**: 3D geometric analysis of superposition structures
- **geometric_superposition_viz.py**: Visualizes superposition patterns and weight matrices

### Modules (src/)
- **models_numpy.py**: Pure NumPy autoencoder implementation with manual gradients
- **data_generation.py**: Synthetic sparse data generation
- **analysis.py**: Superposition analysis and metrics
- **CKA.py**: Centered Kernel Alignment for representation similarity
- **rsa_procrustes.py**: Procrustes analysis for geometric alignment

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run any of the demo scripts directly:
```bash
python Procrustes_abnormality_demo.py
python geometric_visualisations.py
```

## Results
The `procrustes_demo_*` and `geometric_viz_*` folders contain experimental results showing phase transitions in superposition as sparsity varies.