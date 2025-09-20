Can-SAEs-disentangle-superposed-features - Dissertation Project

  Repository Structure

  This repository contains research code for a dissertation studying superposition in neural networks,
   based on Anthropic's toy models of superposition.

  Note: This repository has two branches:
  - main - Contains the dissertation document and writing
  - master - Contains the research code implementation (current branch)

  Project Overview

  Superposition is a behaviour exhibited by neural networks, where one dimension is uesed to encode multiple features when data is projected to a lower dimensionality. In [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html), Anthropic use a Sparse Autoencoder (SAE) to separate the superposed features of a language model into individual interpretable features. This project implements the toy models experiment from Anthropic's [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) and extends it by using SAEs of inverted dimensionality to try to recover the original features from the input data, in order to test how effectively SAEs are able to disentangle superposed features. The codebase was converted from Jupyter notebooks to a structured Python project with pure NumPy implementations for complete mathematical transparency.

  Key Research Findings

  Superposition Phase Transitions

  The experiments revealed critical phase transitions in how autoencoders handle feature
  superposition:

  - Sparsity 0.1-0.7: Normal reconstruction with moderate superposition
  - Sparsity 0.82-0.87: Critical transition region - reconstruction loss peaks as model struggles with    
   superposition/orthogonality tradeoff
  - Sparsity 0.9+: Efficient sparse regime - model successfully adapts to represent sparse features       

  This phase transition around sparsity 0.85 represents a fundamental shift in how the model balances     
  feature interference versus efficient representation.

  

  Implementation Details

  Pure NumPy Architecture

  - Linear autoencoder with bottleneck: x → W @ x → ReLU(W.T @ z + b) → x_recon
  - Manual gradient computation using chain rule
  - Custom Adam optimizer implementation
  - No external autodiff dependencies

  Key Components

  - Data Generation: Synthetic sparse data with Bernoulli × Uniform distribution
  - Feature Importance: Exponential decay weighting (0.7^i)
  - Loss Function: Importance-weighted MSE
  - Visualization: Comprehensive analysis of weight matrices and superposition structure

  Installation & Usage

  Setup

  python -m venv venv
  source venv/Scripts/activate  # Windows
  # source venv/bin/activate    # Linux/Mac
  pip install -r requirements.txt

  Running Experiments

  # Basic experiment
  python main.py

  # Test critical sparsity region
  python main.py --sparsity 0.82 0.85 0.87 --num-epochs 5

  # Custom architecture
  python main.py --sparse-dim 30 --dense-dim 8

  Generated Outputs

  Results are saved to results/ directory:
  - Data distribution histograms
  - Weight norm analysis
  - Superposition matrix heatmaps (W.T @ W)
  - Loss curves and metrics

  Project Structure

  superposition_research/
  ├── src/                    # Core implementation
  │   ├── data_generation.py  # Sparse data synthesis
  │   ├── models_numpy.py     # Pure NumPy autoencoder
  │   └── analysis.py         # Visualization and metrics
  ├── experiments/            # Experiment runners
  ├── config/                 # Experiment configuration
  ├── results/                # Generated plots and analysis
  └── main.py                # CLI interface

  Mathematical Framework

  The autoencoder learns to compress n-dimensional sparse features through a k-dimensional bottleneck     
  (k < n), resulting in superposition - features sharing directions in the latent space. The degree of    
   superposition depends critically on data sparsity and feature importance distribution.

  Contributing

  This is active dissertation research. For questions or collaboration, please open an issue.

  Citation

  This work builds upon:
  - Anthropic's "Toy Models of Superposition" (2022)
  - Related mechanistic interpretability research
