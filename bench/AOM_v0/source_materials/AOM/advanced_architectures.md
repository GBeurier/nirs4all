# Advanced AOM-PLS Architectures: Analysis and Future Directions

This document provides a detailed analysis of five advanced prototype architectures developed to improve the performance and flexibility of the AOM-PLS (Adaptive Operator Matching PLS) framework. For each prototype, we describe its current implementation, its underlying philosophy, and concrete steps for future improvements to achieve state-of-the-art results.

---

## 1. Enhanced AOM-PLS (`enhanced_aom.py`)

### What it does
The Enhanced AOM-PLS is a direct extension of the baseline AOM-PLS model. The baseline AOM-PLS uses a lean operator bank (primarily Savitzky-Golay filters and Detrending) to avoid diluting the selection signal during the NIPALS extraction loop. The Enhanced version injects additional, highly effective operators into this default bank—most notably, the `PseudoLinearSNVOperator`. By doing so, it allows the AOM-PLS algorithm to dynamically select Standard Normal Variate (SNV) scatter correction on a per-component basis, directly within the native C/NumPy backend.

### How to improve it for better results
*   **Operator Synergy Analysis**: Currently, operators are added naively. We should analyze which combinations of operators (e.g., SNV followed by SG derivative) provide the most orthogonal information and only include those synergistic pairs in the bank.
*   **Dynamic Bank Sizing**: Instead of a fixed enhanced bank, the model could perform a rapid pre-screening of operators on the first latent variable and permanently prune those that offer zero correlation with the target, speeding up subsequent extractions.
*   **Orthogonal Signal Correction (OSC) Integration**: Previous attempts to include OSC caused instability. A mathematically rigorous integration of OSC that respects the NIPALS deflation constraints could drastically improve the model's ability to ignore structured noise.

---

## 2. Pseudo-Linear SNV AOM (`pseudo_linear_aom.py`)

### What it does
Standard Normal Variate (SNV) is a powerful scatter correction technique, but it is inherently non-linear because it divides each spectrum by its own standard deviation. The AOM-PLS framework requires linear operators to compute adjoints efficiently during the NIPALS loop. The `PseudoLinearSNVOperator` solves this by implementing the *exact analytical adjoint* of the SNV transform. It treats the per-sample mean and standard deviation as fixed constants during the backward pass, allowing SNV to act as a "pseudo-linear" operator. This enables the AOM-PLS model to evaluate the utility of SNV at every step of the latent variable extraction.

### How to improve it for better results
*   **Full Jacobian Integration**: While the current adjoint is exact for a fixed standard deviation, a true differentiable approach would propagate gradients through the standard deviation calculation itself. This would require transitioning the core NIPALS loop to a fully differentiable backend (like PyTorch or JAX).
*   **Localized SNV**: Instead of applying SNV across the entire spectrum, we could implement a "Windowed Pseudo-Linear SNV" that corrects scatter locally. This is particularly useful for spectra with localized scattering artifacts.
*   **Multiplicative Scatter Correction (MSC) Adjoint**: The success of the SNV adjoint suggests we should derive and implement the exact analytical adjoint for MSC, allowing both premier scatter correction techniques to compete natively in the AOM-PLS bank.

---

## 3. Zero-Shot Preprocessing Router (`zero_shot_router.py`)

### What it does
The Zero-Shot Router acts as a hard-gating mechanism that selects a single, optimal preprocessing pipeline for an entire dataset *before* any PLS training occurs. It is "zero-shot" because it does not require a training phase to learn how to route. Instead, it relies on fast, statistical heuristics computed directly from the raw spectra—such as the ratio of raw variance to derivative variance (indicating baseline drift vs. high-frequency noise) and overall skewness. Based on these heuristics, it routes the data through a specific pipeline (e.g., SNV + SG vs. Detrend + SNV) and then fits a standard AOM-PLS model.

### How to improve it for better results
*   **Data-Driven Thresholds**: The current heuristic thresholds (e.g., `ratio > 2000`) are hardcoded based on domain intuition. These should be optimized using a meta-learning approach across a large corpus of synthetic and real datasets to find the true optimal decision boundaries.
*   **Soft Routing**: Instead of hard-routing to a single pipeline, the router could output a probability distribution over pipelines, allowing the model to blend the outputs of the top two or three pipelines.
*   **Feature-Level Routing**: Rather than routing the entire dataset, the router could segment the spectrum into distinct wavelength regions and apply different preprocessing heuristics to each region before concatenating them for the PLS model.

---

## 4. Mixture of Preprocessing Experts (MoE PLS) (`moe_pls.py`)

### What it does
The MoE PLS model is an ensemble approach inspired by Mixture of Experts architectures. It defines a set of distinct preprocessing "experts" (e.g., one expert uses SNV, another uses SG derivatives, another uses MSC). During training, it fits a separate AOM-PLS model for each expert pipeline. It then uses a cross-validation strategy to generate out-of-fold predictions for each expert. Finally, a meta-model (a low-rank PLS regression) is trained to combine these out-of-fold predictions into a single, robust final prediction.

### How to improve it for better results
*   **Learnable Gating Network**: Currently, the meta-model simply performs a linear combination of the experts' predictions. A true MoE architecture would use a Gating Network (e.g., a small MLP) that looks at the *raw input spectrum* and dynamically outputs weights for each expert on a *per-sample* basis.
*   **End-to-End Training**: The current implementation trains the experts and the meta-model in disjoint steps. By moving to a differentiable backend, the experts and the gating network could be trained jointly, allowing the experts to specialize in specific types of spectral variance.
*   **Expert Pruning**: To reduce inference time, the meta-model could utilize L1 regularization (Lasso) to prune experts that do not contribute significantly to the final prediction, resulting in a leaner, faster ensemble.

---

## 5. DARTS PLS (Differentiable Architecture Search) (`darts_pls.py`)

### What it does
DARTS PLS applies the concept of Differentiable Architecture Search to the problem of preprocessing selection. It defines a discrete bank of preprocessing operators and applies all of them to the input data. It then maintains a set of continuous, learnable weights ($\alpha$) over these operators. Using a custom, differentiable implementation of the PLS1 algorithm written in PyTorch (`torch_pls1`), it optimizes these weights via gradient descent to minimize the final prediction error. Once the optimal weights are learned, they are used to create a static, mixed preprocessing pipeline, which is then fed into a standard, fast AOM-PLS model for the final fit.

### How to improve it for better results
*   **Memory Efficiency**: Currently, DARTS applies all operators and stores the results in memory (`X_ops`), which scales poorly with large datasets and large operator banks. This needs to be optimized, potentially by computing the operator applications on-the-fly during the forward pass.
*   **True Differentiable NIPALS**: The `torch_pls1` implementation is a simplified proxy. To achieve maximum performance, we need a highly optimized, numerically stable, and fully differentiable implementation of the exact NIPALS algorithm, potentially utilizing custom CUDA kernels or implicit differentiation techniques to avoid unrolling the entire extraction loop.
*   **Temperature Annealing**: The softmax weights over the operators often remain too "soft" (e.g., blending 5 operators equally). Implementing Gumbel-Softmax with temperature annealing would force the model to make "harder" choices as training progresses, resulting in a more interpretable and physically meaningful final pipeline.
