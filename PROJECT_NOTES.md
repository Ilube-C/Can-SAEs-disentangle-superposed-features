# Superposition Research Project - Key Findings and Notes

## Project Overview
This dissertation project studies superposition in neural networks, implementing toy models based on Anthropic's superposition research. Originally converted from Jupyter notebook to full Python project with pure NumPy implementation for transparency.

---

## Critical Discovery: SAE Failure at Specific Architecture-Sparsity Combinations

### Finding Summary
**Date**: August 28, 2025  
**Discovery**: Complete Sparse Autoencoder (SAE) failure at sparsity 0.9 specifically on 10-5-10 architecture  
**Investigation**: `quick_ae_test.py` in `results/0.9_sparsity_10-5-10_abnormality/`

### The Abnormality

#### Architecture-Specific SAE Failure Pattern
Testing 10-5-10 autoencoder architecture across sparsity levels 0.85, 0.9, 0.95 revealed:

**Sparsity 0.85 (Normal)**:
- Mean Active SAE Nodes: ~6
- Dead SAE Neurons: 2-4/10
- Procrustes Similarity: ~0.95+
- SAE L0 Norm: ~0.4-0.6

**Sparsity 0.9 (COMPLETE FAILURE)**:
- Mean Active SAE Nodes: 0.0
- Dead SAE Neurons: 10/10 (all dead)
- Procrustes Similarity: negative or NaN
- SAE L0 Norm: 0.000

**Sparsity 0.95 (Normal)**:
- Mean Active SAE Nodes: ~3-4
- Dead SAE Neurons: 3-5/10
- Procrustes Similarity: ~0.92+
- SAE L0 Norm: ~0.2-0.3

### Key Insights

#### 1. Autoencoder Performance is NOT the Issue
Contrary to initial hypothesis, the underlying autoencoder performs **better** at 0.9 sparsity:
- **0.85**: AE Loss 0.0708, Condition 1.8
- **0.9**: AE Loss 0.0280 (**lowest loss**), Condition 1.6  
- **0.95**: AE Loss 0.0371, Condition 63.6

#### 2. Superposition Structure Drives SAE Success
The failure correlates with specific learned representation structure:
- **0.9**: Diagonal dominance 0.734 (too orthogonal for SAE)
- **0.85**: Diagonal dominance 0.543 (balanced superposition)
- **0.95**: Diagonal dominance 0.375 (weak diagonal, high condition number)

#### 3. Non-Monotonic Difficulty Curve
Sparsity effects are not monotonic - moderate sparsity (0.9) can be harder than extreme sparsity (0.95) due to:
- **Phase transition effects**: 0.9 hits critical superposition transition region
- **Representation mismatch**: SAE expects correlated features, gets orthogonal ones
- **Optimization landscape**: L1 penalty interacts poorly with specific activation patterns

### Scientific Significance

#### For Superposition Research
1. **Architecture Sensitivity**: Small architectural changes (10 vs 20 dimensions) can dramatically affect interpretability
2. **Critical Sparsity Regions**: Specific sparsity levels may be particularly challenging for interpretability tools
3. **Representation Quality ≠ Interpretability**: Good reconstruction doesn't guarantee interpretable representations

#### For Sparse Autoencoder Development  
1. **Adaptive Regularization**: L1 penalty may need adjustment based on input sparsity and architecture size
2. **Initialization Strategies**: Different initialization may be needed for different sparsity regimes
3. **Architecture Matching**: SAE capacity must be carefully matched to input representation structure

#### For Neural Network Interpretability
1. **Context Dependency**: Interpretability tools may fail unpredictably at specific operating points
2. **Systematic Testing**: Need to test interpretability across full parameter space, not just extremes
3. **Failure Mode Analysis**: Understanding when/why tools fail is as important as when they succeed

### Broader Implications

This finding suggests that the relationship between **neural network representations** and **interpretability tools** is more complex than previously understood. The failure occurs despite:
- ✅ Good autoencoder performance
- ✅ Reasonable data distribution
- ✅ Standard hyperparameters
- ✅ Successful operation at nearby sparsity levels

This indicates fundamental challenges in developing robust interpretability tools that work across all operating regimes.

---

## Next Research Directions

1. **Systematic Architecture Study**: Test failure pattern across more architecture sizes (5-3-5, 15-7-15, etc.)
2. **Adaptive SAE Design**: Develop SAE variants that can handle orthogonal representations
3. **Phase Transition Mapping**: Identify all critical sparsity regions for different architectures
4. **Alternative Similarity Metrics**: Find metrics that remain stable during SAE failure modes

---

## Experimental Evidence

### File Locations
- **Main Investigation**: `quick_ae_test.py`
- **Results**: `results/0.9_sparsity_10-5-10_abnormality/`
- **Multi-Architecture Comparison**: `results/multi_dim_clean_*/`
- **Original Finding**: `results/simple_exp_10-5-10_*/metrics_table.csv`

### Reproducibility
- **Seed**: 123 (consistent across experiments)
- **Architecture**: 10-5-10 autoencoder, 5-10-5 SAE
- **Hyperparameters**: L1=0.01, SAE LR=0.001, SAE epochs=20
- **Critical Sparsity**: 0.9 (empirical sparsity ~0.905)

This discovery represents a significant contribution to understanding the limitations and failure modes of current interpretability approaches.