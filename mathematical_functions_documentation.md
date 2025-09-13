# Mathematical Functions Documentation
## Superposition Research Project

This document contains all mathematical functions implemented in the codebase, with both the code implementation and LaTeX mathematical representation.

---

## 1. Activation Functions

### ReLU (Rectified Linear Unit)
**File:** `src/models_numpy.py:21-23`

```python
def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)
```

**LaTeX:**
$$\text{ReLU}(x) = \max(0, x)$$

### ReLU Derivative
**File:** `src/models_numpy.py:26-28`

```python
def relu_derivative(x):
    """Derivative of ReLU activation function."""
    return (x > 0).astype(np.float32)
```

**LaTeX:**
$$\frac{\partial \text{ReLU}}{\partial x} = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

---

## 2. Autoencoder Architecture

### Weight Initialization
**File:** `src/models_numpy.py:15-17`

```python
W = np.random.normal(0, 0.01, size=(k, n))
b = np.zeros((n,))
```

**LaTeX:**
$$W \sim \mathcal{N}(0, 0.01^2), \quad W \in \mathbb{R}^{k \times n}$$
$$b = \vec{0}, \quad b \in \mathbb{R}^n$$

### Forward Pass
**File:** `src/models_numpy.py:50-53`

```python
# Forward pass: x -> z -> x_recon
z = W @ x.T  # (k, batch_size)
x_linear = (W.T @ z).T + b  # (batch_size, n)
x_recon = relu(x_linear)
```

**LaTeX:**
$$z = Wx$$
$$\hat{x}_{\text{linear}} = W^T z + b$$
$$\hat{x} = \text{ReLU}(\hat{x}_{\text{linear}})$$

### Bottleneck Extraction
**File:** `src/models_numpy.py:245`

```python
z = (W @ X.T).T  # (N, k)
```

**LaTeX:**
$$Z = XW^T$$
where $X \in \mathbb{R}^{N \times n}$ and $Z \in \mathbb{R}^{N \times k}$

---

## 3. Loss Functions

### Weighted Reconstruction Loss
**File:** `src/models_numpy.py:79-80`

```python
diff = X - x_recon
loss = np.sum(I * diff**2)
```

**LaTeX:**
$$\mathcal{L} = \sum_{i=1}^{n} I_i \cdot (x_i - \hat{x}_i)^2$$

where $I_i$ are feature importance weights.

### Feature Importance Weights
**File:** `src/data_generation.py:39`

```python
feature_importances = decay_factor**np.arange(n)
```

**LaTeX:**
$$I_i = \alpha^{i-1}, \quad \text{where } \alpha = 0.7$$

---

## 4. Gradient Computation (Backpropagation)

### Loss Gradient w.r.t. Reconstruction
**File:** `src/models_numpy.py:104`

```python
grad_x_recon = 2 * I.flatten() * (x_recon - x)  # (n,)
```

**LaTeX:**
$$\frac{\partial \mathcal{L}}{\partial \hat{x}} = 2I \odot (\hat{x} - x)$$

### Gradient Through ReLU
**File:** `src/models_numpy.py:107`

```python
grad_x_linear = grad_x_recon * relu_derivative(x_linear)  # (n,)
```

**LaTeX:**
$$\frac{\partial \mathcal{L}}{\partial \hat{x}_{\text{linear}}} = \frac{\partial \mathcal{L}}{\partial \hat{x}} \odot \mathbb{1}[\hat{x}_{\text{linear}} > 0]$$

### Gradient w.r.t. Bias
**File:** `src/models_numpy.py:110`

```python
grad_b = grad_x_linear  # (n,)
```

**LaTeX:**
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial \hat{x}_{\text{linear}}}$$

### Gradient w.r.t. Bottleneck
**File:** `src/models_numpy.py:113`

```python
grad_z = W @ grad_x_linear  # (k,)
```

**LaTeX:**
$$\frac{\partial \mathcal{L}}{\partial z} = W \cdot \frac{\partial \mathcal{L}}{\partial \hat{x}_{\text{linear}}}$$

### Gradient w.r.t. Weight Matrix
**File:** `src/models_numpy.py:118-120`

```python
grad_W_from_z = np.outer(grad_z, x)  # (k, n)
grad_W_from_linear = np.outer(z, grad_x_linear)  # (k, n)
grad_W = grad_W_from_z + grad_W_from_linear
```

**LaTeX:**
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial z} \otimes x + z \otimes \frac{\partial \mathcal{L}}{\partial \hat{x}_{\text{linear}}}$$

---

## 5. Adam Optimizer

### Moment Updates
**File:** `src/models_numpy.py:150-153`

```python
self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
```

**LaTeX:**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

### Bias Correction
**File:** `src/models_numpy.py:156-159`

```python
m_hat = self.m[key] / (1 - self.beta1 ** self.t)
v_hat = self.v[key] / (1 - self.beta2 ** self.t)
```

**LaTeX:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### Parameter Update
**File:** `src/models_numpy.py:162`

```python
updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

**LaTeX:**
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---

## 6. Sparse Autoencoder (SAE)

### SAE Forward Pass (Encoder)
**File:** `src/models_numpy.py:300-301`

```python
hidden_linear = (W_enc @ x.T).T + b_enc  # (batch_size, hidden_dim)
hidden = relu(hidden_linear)  # Sparse activations
```

**LaTeX:**
$$h_{\text{linear}} = W_{\text{enc}} x + b_{\text{enc}}$$
$$h = \text{ReLU}(h_{\text{linear}})$$

### SAE Forward Pass (Decoder)
**File:** `src/models_numpy.py:304`

```python
recon = (W_dec @ hidden.T).T + b_dec  # (batch_size, input_dim)
```

**LaTeX:**
$$\hat{z} = W_{\text{dec}} h + b_{\text{dec}}$$

### SAE Loss Function
**File:** `src/models_numpy.py:340-345`

```python
# Reconstruction loss (MSE)
recon_loss = np.mean((x - recon)**2)
# L1 sparsity penalty
sparsity_loss = l1_penalty * np.mean(np.abs(hidden))
total_loss = recon_loss + sparsity_loss
```

**LaTeX:**
$$\mathcal{L}_{\text{SAE}} = \frac{1}{N}\sum_{i=1}^{N} \|z_i - \hat{z}_i\|^2 + \lambda \frac{1}{N}\sum_{i=1}^{N} \|h_i\|_1$$

where $\lambda$ is the L1 penalty coefficient.

### SAE Gradient Computation
**File:** `src/models_numpy.py:371-381`

```python
# Reconstruction loss gradient
grad_recon = 2 * (recon - x)
# L1 sparsity penalty gradient
grad_hidden_from_l1 = l1_penalty * np.sign(hidden)
```

**LaTeX:**
$$\frac{\partial \mathcal{L}_{\text{recon}}}{\partial \hat{z}} = 2(\hat{z} - z)$$
$$\frac{\partial \mathcal{L}_{\text{sparse}}}{\partial h} = \lambda \cdot \text{sign}(h)$$

---

## 7. Data Generation

### Synthetic Sparse Data
**File:** `src/data_generation.py:20-26`

```python
# Create Bernoulli samples (0 or 1) with probability 1-S
bernoulli_samples = np.random.binomial(1, 1-S, size=(N, n))
# Create uniform random numbers in [0, 1)
uniform_samples = np.random.uniform(0, 1, size=(N, n))
# Element-wise multiplication
return uniform_samples * bernoulli_samples
```

**LaTeX:**
$$x_{ij} = B_{ij} \cdot U_{ij}$$

where:
- $B_{ij} \sim \text{Bernoulli}(1-S)$ (sparsity mask)
- $U_{ij} \sim \text{Uniform}(0, 1)$ (values)
- $S$ is the sparsity parameter

---

## 8. Analysis Functions

### Superposition Matrix
**File:** `src/analysis.py:17`

```python
superposition_matrix = W.T @ W
```

**LaTeX:**
$$M = W^T W$$

where $M \in \mathbb{R}^{n \times n}$ measures feature interference.

### Weight Norms (L2)
**File:** `src/analysis.py:20`

```python
weight_norms = np.array([np.linalg.norm(col, ord=2) for col in superposition_matrix])
```

**LaTeX:**
$$\|w_i\| = \sqrt{\sum_{j=1}^{k} (W^T W)_{ij}^2}$$

### Pairwise Dot Products
**File:** `src/analysis.py:28`

```python
score = np.dot(superposition_matrix[i], superposition_matrix[j])
```

**LaTeX:**
$$s_{ij} = \langle (W^T W)_i, (W^T W)_j \rangle$$

### RSA (Representational Similarity Analysis) Correlation
**File:** `src/analysis.py:157-161`

```python
# Compute pairwise distances for each representation
dist1 = pdist(repr1, metric='euclidean')
dist2 = pdist(repr2, metric='euclidean')
# Compute correlation between distance vectors
correlation, p_value = pearsonr(dist1, dist2)
```

**LaTeX:**
$$d_1^{(ij)} = \|r_1^{(i)} - r_1^{(j)}\|_2$$
$$d_2^{(ij)} = \|r_2^{(i)} - r_2^{(j)}\|_2$$
$$\text{RSA} = \text{Corr}(d_1, d_2)$$

### CKA (Centered Kernel Alignment)
**File:** `src/CKA.py:6-37`

```python
def linear_cka(X, Y):
    # Compute Gram matrices (dot product kernels)
    K = X @ X.T  # (N, N)
    L = Y @ Y.T  # (N, N)
    
    # Center the kernel matrices
    H = np.eye(N) - np.ones((N, N)) / N  # Centering matrix
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_XY = np.trace(K_centered @ L_centered) / (N - 1)**2
    hsic_XX = np.trace(K_centered @ K_centered) / (N - 1)**2
    hsic_YY = np.trace(L_centered @ L_centered) / (N - 1)**2
    
    # Compute CKA
    cka = hsic_XY / np.sqrt(hsic_XX * hsic_YY)
```

**LaTeX:**

Given two representation matrices $X \in \mathbb{R}^{N \times D_1}$ and $Y \in \mathbb{R}^{N \times D_2}$:

**Gram Matrices:**
$$K = XX^T, \quad L = YY^T$$

**Centering Matrix:**
$$H = I_N - \frac{1}{N}\mathbf{1}_N\mathbf{1}_N^T$$

**Centered Kernels:**
$$\tilde{K} = HKH, \quad \tilde{L} = HLH$$

**HSIC (Hilbert-Schmidt Independence Criterion):**
$$\text{HSIC}(X,Y) = \frac{1}{(N-1)^2} \text{tr}(\tilde{K}\tilde{L})$$
$$\text{HSIC}(X,X) = \frac{1}{(N-1)^2} \text{tr}(\tilde{K}^2)$$
$$\text{HSIC}(Y,Y) = \frac{1}{(N-1)^2} \text{tr}(\tilde{L}^2)$$

**Linear CKA:**
$$\text{CKA}(X,Y) = \frac{\text{HSIC}(X,Y)}{\sqrt{\text{HSIC}(X,X) \cdot \text{HSIC}(Y,Y)}}$$

**Properties:**
- $\text{CKA}(X,X) = 1$ (self-similarity)
- $\text{CKA}(X,Y) = \text{CKA}(Y,X)$ (symmetry)
- $0 \leq \text{CKA}(X,Y) \leq 1$ (bounded)
- Invariant to orthogonal transformations and isotropic scaling

### Sparsity Metrics - L0 Norm
**File:** `src/analysis.py:176`

```python
l0_norm = np.mean(activations != 0)
```

**LaTeX:**
$$\|A\|_0 = \frac{1}{N \cdot D} \sum_{i,j} \mathbb{1}[A_{ij} \neq 0]$$

### Sparsity Metrics - L1 Norm
**File:** `src/analysis.py:179`

```python
l1_norm = np.mean(np.abs(activations))
```

**LaTeX:**
$$\|A\|_1 = \frac{1}{N \cdot D} \sum_{i,j} |A_{ij}|$$

### Gini Coefficient
**File:** `src/analysis.py:182-186`

```python
flat_acts = np.abs(activations.flatten())
sorted_acts = np.sort(flat_acts)
n = len(sorted_acts)
index = np.arange(1, n + 1)
gini = (2 * np.sum(index * sorted_acts)) / (n * np.sum(sorted_acts)) - (n + 1) / n
```

**LaTeX:**
$$G = \frac{2\sum_{i=1}^{n} i \cdot a_i}{n \sum_{i=1}^{n} a_i} - \frac{n+1}{n}$$

where $a_i$ are the sorted absolute activation values.

### Reconstruction MSE
**File:** `src/analysis.py:224`

```python
recon_mse = np.mean((original_activations - sae_recon)**2)
```

**LaTeX:**
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \|z_i - \hat{z}_i\|^2$$

---

## Summary of Key Mathematical Concepts

### 1. **Autoencoder Architecture**
- Linear encoder: $z = Wx$
- Linear decoder with ReLU: $\hat{x} = \text{ReLU}(W^T z + b)$
- Bottleneck dimension $k < n$ enforces compression

### 2. **Superposition Phenomenon**
- When $k < n$, features must share capacity
- Superposition matrix $W^T W$ reveals feature interference
- Sparsity allows more features than dimensions

### 3. **Training Objective**
- Weighted MSE loss with exponential feature importance
- Encourages accurate reconstruction of important features
- Adam optimizer with manual gradient computation

### 4. **Sparse Autoencoder Extension**
- Adds hidden layer with L1 penalty for sparsity
- Disentangles representations in bottleneck space
- Trade-off between reconstruction and sparsity

### 5. **Phase Transitions**
- Critical sparsity region (0.82-0.87) shows dramatic loss changes
- Model transitions between different representation regimes
- High sparsity (>0.9) enables efficient sparse coding

This mathematical framework implements the toy models from Anthropic's superposition research, providing full transparency through manual gradient computation and pure NumPy implementation.