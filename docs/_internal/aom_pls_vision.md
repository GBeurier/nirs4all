# AOM-PLS: Current State, Limitations, and the Next Generation

## High-Level Description of AOM-PLS
Near-Infrared Spectroscopy (NIRS) data is plagued by physical artifacts: light scattering, baseline shifts, and overlapping chemical peaks. Traditionally, finding the optimal preprocessing pipeline requires an exhaustive grid search over millions of combinations (e.g., 10 scatter corrections × 50 Savitzky-Golay filter variations × 30 PLS components = 15,000 model fits). This is computationally prohibitive and prone to overfitting.

AOM-PLS (Adaptive Operator-Mixture Partial Least Squares) fundamentally changes this paradigm. Instead of applying a static preprocessing pipeline to the data *before* training, AOM-PLS embeds the preprocessing selection directly into the PLS training loop. By treating preprocessings as a bank of linear operators, it dynamically evaluates and selects the best "lens" through which to view the data during the extraction of *each* PLS latent variable. This aims to deliver an optimized, component-specific PLS model in seconds rather than hours, bridging the gap between exhaustive search and instant, optimal fitting.

## Detailed Description of Key Points of AOM-PLS
1. **Linear Operator Bank**: The core of AOM-PLS is a predefined bank of linear transformations (e.g., Savitzky-Golay filters, Detrending, Finite Differences, Wavelet Projections). Because these are strictly linear, they can be represented mathematically as $p \times p$ matrices $A_b$. For example, a detrending operator is a projection matrix $A = I - Q Q^T$.
2. **The Adjoint Trick**: Explicitly transforming the entire dataset $X$ ($n \times p$) for every operator in the bank would cost $O(n \cdot p^2)$, which is too slow. AOM-PLS uses the adjoint (transpose) of the operators. In PLS, we need the covariance vector $c = X^T y$. Instead of computing $(X A_b)^T y$, we compute $A_b^T (X^T y) = A_b^T c$. This reduces the complexity to $O(p)$ per sample, making the evaluation of hundreds of operators nearly instantaneous.
3. **NIPALS Integration**: AOM-PLS modifies the standard NIPALS (Non-linear Iterative Partial Least Squares) algorithm. At each deflation step ($k=1 \dots K$), the algorithm evaluates all operators on the current residual $X_{res}$. The data is then deflated ($X_{res} \leftarrow X_{res} - t p^T$) taking into account the specific operator chosen for that component.
4. **Gating Mechanisms**: 
   - **Hard Gating**: Evaluates all operators and strictly selects the one that yields the best validation RMSE or highest first-component $R^2$.
   - **Sparsemax Gating**: A soft-selection approach that computes a sparse weight distribution ($\gamma$) over the operator bank. It creates a custom, mixed preprocessing operator on the fly: $W_{mixed} = \sum \gamma_b A_b W_b$.
5. **Component-Specific Preprocessing**: Because selection happens inside the NIPALS loop, component 1 might use a 1st derivative to capture a specific chemical peak, while component 2 might use a baseline correction to model physical scatter. This is a massive advantage over traditional PLS, which forces a single preprocessing pipeline for all components.

## Limitations: What Can Be Done Better
While AOM-PLS works well in ~80% of cases, it falls short of the "5-second Graal" due to several structural and mathematical limitations:
1. **Strict Linearity Constraint**: The adjoint trick strictly requires linear operators. Crucial NIRS preprocessings like SNV (Standard Normal Variate) and MSC (Multiplicative Scatter Correction) are non-linear (they depend on sample-wise statistics) and cannot be natively integrated into the fast adjoint-based selection.
2. **Greedy Extraction**: NIPALS extracts components greedily. The operator that maximizes covariance for Component 1 might actually sub-optimize the overall predictive power of a $K$-component model. It lacks a global view of the optimization landscape.
3. **Still Reliant on Fine-Tuning**: As seen in current benchmarks (`run_reg_aom.py`), AOM-PLS still relies on external hyperparameter tuners (like TPE/Optuna) to find the optimal `n_components`, `n_orth`, and sometimes forces the `operator_index`. This defeats the purpose of a zero-shot, instant fit.
4. **Bank Size Bottleneck**: The computational cost scales linearly with the number of operators in the bank. Expanding the bank to cover all edge cases (e.g., thousands of SG window/poly combinations) slows down the algorithm significantly.

## 8 Suggestions for Improvement (The "Graal" of PLS)

### Category A: Single-Dataset / Fit-Time Enhancements (No External Data Required)
These solutions focus on improving the algorithm for a single dataset at `fit()` time, without requiring a pre-trained external model.

#### 1. Pseudo-Linearization of Non-Linear Operators (SNV/MSC)
- **Concept**: SNV and MSC are non-linear because they divide by a sample-dependent standard deviation. However, for a *fixed* dataset $X$, this division is just a diagonal scaling matrix $D$. 
- **The Graal**: We can pre-compute the sample-wise means and standard deviations once. During the NIPALS loop, we treat SNV/MSC as dynamic linear operators $A(X) = D(X) \cdot (I - J)$. This allows us to use the adjoint trick on SNV and MSC, bringing the most powerful NIRS preprocessings into the AOM-PLS bank without losing the $O(p)$ speed.

#### 2. Joint Optimization via Block-Coordinate Descent (BCD)
- **Concept**: Abandon the greedy, component-by-component NIPALS extraction. Instead, formulate the entire PLS objective $||Y - X W(\theta) Q||^2$ as a single global loss function, where $\theta$ represents continuous preprocessing parameters (e.g., fractional derivatives, continuous smoothing windows).
- **The Graal**: Use Block-Coordinate Descent on the single dataset: alternate between solving for the PLS weights $W, Q$ (which has a fast closed-form solution) and taking a gradient step to update the preprocessing parameters $\theta$. This guarantees a globally optimal pipeline for the specific dataset in seconds, avoiding greedy sub-optimization.

#### 3. Multi-Armed Bandit Operator Selection
- **Concept**: Instead of evaluating all 120+ operators for every PLS component, treat operator selection as a Multi-Armed Bandit problem (e.g., using UCB - Upper Confidence Bound).
- **The Graal**: As NIPALS progresses, the algorithm quickly learns which *families* of operators (e.g., SG derivatives vs. Detrending) are yielding the best $R^2$ drops. It dynamically prunes the search space, sampling only the most promising operators. This allows the bank size to grow to $10^4$ operators while keeping the fit time under 1 second.

### Category B: Deep Learning & Meta-Learning (Requires Synthetic/External Data)
These solutions leverage your synthetic data generator to train models that generalize across datasets.

#### 4. Differentiable PLS with Implicit Layers (End-to-End Non-Linearity)
- **Concept**: Implement the entire preprocessing pipeline and PLS solver as a differentiable computational graph in PyTorch/JAX. The PLS algorithm itself is treated as an implicit layer.
- **The Graal**: We can use gradient descent to jointly optimize the continuous parameters of the preprocessing and the PLS weights simultaneously across a batch of synthetic datasets, learning a highly robust, generalized preprocessing strategy.

#### 5. Zero-Shot Preprocessing Router (Meta-Learning)
- **Concept**: Train a lightweight neural network (e.g., a 1D-CNN or Transformer) on thousands of synthetic NIRS datasets. It takes the raw $X$ matrix as input and instantly outputs the optimal preprocessing pipeline configuration and `n_components`.
- **The Graal**: When a user calls `fit()`, the Router looks at the data and predicts the best pipeline in 10 milliseconds. The PLS is then fitted exactly once. (Inspired by TabPFN).

#### 6. Mixture of Preprocessing Experts (MoE)
- **Concept**: Pass the raw spectrum $X$ through multiple parallel branches (Experts), where each branch is a different non-linear preprocessing. A lightweight Gating Network looks at each individual sample and assigns weights to the experts.
- **The Graal**: Allows *sample-wise* preprocessing. A sample with high baseline shift might be routed heavily through the MSC expert, while a clean sample bypasses it. The outputs are linearly combined and fed into a standard fast PLS.

#### 7. Latent Space PLS (Deep Representation Learning)
- **Concept**: Train an Autoencoder or a 1D-ResNet where the encoder acts as a universal, non-linear "preprocessor". The latent space $Z$ is optimized such that a standard, un-preprocessed PLS applied to $Z$ yields perfect predictions.
- **The Graal**: By using Contrastive Learning with physical NIRS distortions (scatter, baseline shifts, noise) as augmentations, the encoder learns a representation inherently invariant to these artifacts.

#### 8. Continuous Relaxation of the Operator Bank (DARTS for NIRS)
- **Concept**: Borrowing from Differentiable Architecture Search (DARTS), relax the discrete choice of operators into a continuous optimization problem using a softmax-weighted sum of all possible operators.
- **The Graal**: During a rapid warm-up phase, use gradient descent to optimize the softmax weights $\alpha$. The operators with the highest weights are "snapped" into a final discrete pipeline.

## Implementation Roadmap

To bring these solutions to life in `nirs4all`, here is how we will implement them:

### Phase 1: Fit-Time Enhancements (Immediate Value, No Training)
*   **Implement Pseudo-Linear SNV/MSC (Suggestion 1)**: 
    *   Create `PseudoLinearSNVOperator` and `PseudoLinearMSCOperator` inheriting from `LinearOperator`.
    *   Override the `initialize(X)` method to pre-compute the sample-wise standard deviations and means for the *training* set.
    *   Implement `apply_adjoint` using these cached vectors.
*   **Implement Bandit-based NIPALS (Suggestion 3)**:
    *   Modify `_aompls_fit_numpy` in `aom_pls.py`.
    *   Introduce a `ThompsonSampler` or `UCB` tracker that updates the probability of selecting an operator based on the residual variance drop in previous components.
*   **Implement BCD Joint Optimization (Suggestion 2)**:
    *   Create a new PyTorch-based estimator `ContinuousAOMPLS`.
    *   Define preprocessings as `nn.Module` with learnable parameters (e.g., `self.window_size = nn.Parameter(...)`).
    *   Write a custom training loop that alternates between `torch.linalg.lstsq` (for PLS weights) and `loss.backward()` (for preprocessing parameters).

### Phase 2: Deep Learning & Meta-Learning (Using Synthetic Generator)
*   **Implement Differentiable PLS & DARTS (Suggestions 4 & 8)**:
    *   Build a `DifferentiablePLS` layer using PyTorch implicit gradients (e.g., using `torch.autograd.Function` for the SVD/Eigenvalue decomposition).
    *   Implement the DARTS super-network where `forward(X) = sum(softmax(alpha) * op(X))`.
*   **Implement Zero-Shot Router & MoE (Suggestions 5 & 6)**:
    *   Generate a massive dataset using `nirs4all.generate`.
    *   Train a `PreprocessingRouter` (a simple MLP or 1D-CNN) that predicts the optimal `operator_index` and `n_components` from the raw spectra.
    *   Integrate this into a new `ZeroShotPLSRegressor` that calls the router before fitting.
*   **Implement Latent Space PLS (Suggestion 7)**:
    *   Train a `ContrastiveSpectraEncoder` using SimCLR-style augmentations (adding synthetic scatter, noise, baseline shifts).
    *   Wrap this in a `LatentPLSRegressor` that transforms $X \rightarrow Z$ before applying standard `sklearn.cross_decomposition.PLSRegression`.
