# Operator-Adaptive Partial Least Squares for Near-Infrared Spectroscopy

## Abstract

Near-infrared spectroscopy relies heavily on preprocessing choices before
Partial Least Squares regression. These choices are usually selected through
external grid searches, making calibration expensive, unstable on small
datasets, and difficult to interpret. We introduce Operator-Adaptive Partial
Least Squares, a unified framework in which spectral preprocessing operators
are selected inside the PLS extraction process. The framework covers standard
PLS, global Adaptive Operator-Mixture PLS, per-component POP-PLS, soft operator
mixtures, and superblock multi-view PLS. We provide both NIPALS and SIMPLS
formulations, including a covariance-space SIMPLS engine based on the identity
`(X A^T)^T Y = A X^T Y`. We extend the framework to PLS-DA with class-balanced
latent-space coding and leakage-safe probability calibration. On the
TabPFN/NIRS benchmark protocol, we compare all AOM/POP variants against
standard PLS and the published TabPFN reference results. The study reports
predictive performance, selected operators, component counts, computational
cost, and probability calibration. The implementation is provided in NumPy and
PyTorch with equivalence tests against materialized references.

## 1. Introduction

NIRS calibration is rarely a pure model-fitting problem. The predictive quality
of PLS depends on preprocessing decisions such as scatter correction, smoothing,
baseline removal, and derivatives. These operations are usually treated as
external hyperparameters. As the number of candidate pipelines grows, the
selection problem becomes combinatorial and can dominate the scientific
workflow.

This paper asks whether preprocessing can be moved from an external search
space into the PLS extraction itself. The answer is a family of
Operator-Adaptive PLS algorithms. Instead of fitting a separate PLS model for
each complete preprocessing pipeline, we define a bank of spectral operators
and select operators globally or per latent component.

The contributions are:

1. A unified mathematical formulation of standard PLS, AOM-PLS, POP-PLS, soft
   operator mixtures, and superblock PLS.
2. NIPALS and SIMPLS implementations, including covariance-space SIMPLS for
   linear operators.
3. A PLS-DA extension with class-balanced coding and probability calibration.
4. A NumPy and PyTorch implementation with materialized-reference equivalence
   tests.
5. A benchmark against the existing TabPFN NIRS regression results and a
   classification cohort.

## 2. Related Work

Discuss PLS and chemometrics, preprocessing in NIRS, SIMPLS, PLS-DA, TabPFN for
tabular foundation models, and learnable spectral kernels such as FCK-PLS.

## 3. Operator-Adaptive PLS

Let `X in R^{n x p}` and `Y in R^{n x q}`. A strict linear spectral operator
`A_b in R^{p x p}` transforms samples by `X_b = X A_b^T`. The central covariance
identity is:

```text
X_b^T Y = A_b X^T Y
```

Operator-Adaptive PLS selects a sequence of operators:

```text
A^(a) in {A_1, ..., A_B}
```

for components `a = 1, ..., K`. Standard PLS is recovered when all operators are
identity. AOM global selection uses a single operator for all components. POP
selection chooses a potentially different operator per component. Soft mixture
uses convex combinations of operators. Superblock PLS concatenates all
operator-transformed views.

## 4. Algorithms

### 4.1 NIPALS Materialized and Adjoint Forms

The materialized reference builds `X A_b^T` and fits standard NIPALS. The fast
adjoint form avoids materializing transformed matrices by applying operators to
cross-covariance vectors and mapping transformed directions to original-space
effective weights.

### 4.2 SIMPLS Materialized and Covariance Forms

SIMPLS can evaluate candidate operators directly in covariance space because
`S_b = A_b S`, where `S = X^T Y`. This produces a fast operator selection engine
for both global AOM and per-component POP.

### 4.3 Orthogonalization

Two orthogonalization modes are required. Transformed-space orthogonalization is
the materialized-reference mode for fixed operators. Original-space
orthogonalization maps all component directions back to effective weights in
the original spectral space and is the default when operators vary by component.

## 5. Classification

For PLS-DA, labels are encoded as a class-balanced one-hot matrix and the
operator-adaptive PLS2 engine extracts discriminant latent variables. A
multinomial logistic model is fitted on latent scores to provide probabilities.
All calibration steps are performed inside the training fold during benchmark
selection.

## 6. Experiments

Regression uses the `master_results.csv` protocol from the TabPFN NIRS
benchmark. AOM/POP rows use the same schema and are joined against PLS,
TabPFN-Raw, and TabPFN-opt. Classification uses the local
`bench/tabpfn_paper/data/classification` cohort with balanced accuracy as the
primary metric.

## 7. Results

Populate from generated tables:

- `publication/tables/table_regression_main.tex`
- `publication/tables/table_classification_main.tex`
- `publication/tables/table_ablation.tex`

Discuss both absolute RMSEP/accuracy and deltas against PLS and TabPFN.

## 8. Discussion

Interpret when operator adaptivity helps, when identity wins, and whether POP
selects meaningful multi-stage operator sequences. Discuss the role of
covariance-space SIMPLS as both a speed optimization and a cleaner conceptual
formulation.

## 9. Limitations

SNV and MSC are not strict fixed linear operators. Soft mixtures may collapse to
hard selection under covariance objectives. Classification conclusions are
limited by the size of the available classification cohort. No generalization
dominance is claimed beyond measured benchmark results.

## 10. Reproducibility

Include exact commands, commit hash, package versions, random seeds, benchmark
cohorts, and result CSV paths.

## 11. Conclusion

Operator-Adaptive PLS turns preprocessing selection into a traceable component
of the PLS model. The resulting framework unifies AOM, POP, NIPALS, SIMPLS, and
PLS-DA while preserving materialized-reference tests and producing benchmark
outputs compatible with the existing TabPFN study.
