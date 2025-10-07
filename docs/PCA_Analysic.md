# PCA Analysis for preprocessing evaluation

## Purpose

Evaluate whether each preprocessing preserves or enhances the intrinsic structure of NIRS data. Do this **without labels**. Use raw PCA as reference. Compare it to PCA after preprocessing.

## Inputs

* `raw_data`: per-dataset raw matrices `(n_samples, n_features)`.
* `pp_data`: for each preprocessing, the corresponding per-dataset matrices. Sample order must match the raw dataset.

## Method

1. For each dataset, run PCA on raw data. This gives:

   * Scores (low-dim embeddings) capturing dominant variance.
   * Loadings defining the reference subspace.
   * Cumulative explained variance (EVR).
2. For each preprocessing and dataset, run PCA again.
3. Compare the **reference** (raw) and **candidate** (preprocessed) representations at two levels:

   * **Subspace level**: compare PCA loading spaces.
   * **Embedding level**: compare PCA scores and neighborhoods.

## Metrics

* **Explained variance ratio (EVR)**: how much variance top PCs capture after preprocessing. Higher suggests better compression of structure.
* **Grassmann distance** (principal angles between subspaces): similarity of raw vs preprocessed PCA subspaces. Lower means subspaces align. Skipped if feature counts differ.
* **Procrustes disparity** (on scores): rigid alignment error between raw and preprocessed embeddings. Lower is better.
* **Linear CKA** and **RV-coefficient**: correlation of representational geometry. Higher means global structure is preserved.
* **Trustworthiness** (k-NN): stability of local neighborhoods from raw to preprocessed embeddings. Higher means local geometry is preserved.

## What is analyzed

* **Global geometry**: are main variance directions similar (EVR, CKA, RV, Procrustes)?
* **Local geometry**: do nearest neighbors remain neighbors (Trustworthiness)?
* **Model space shift**: do preprocessing steps rotate or distort the information-bearing subspace (Grassmann)?

## Results

* A per-dataset × preprocessing table of metrics.
* Cached 2D PCA projections for visual checks.
* Optional plots:

  * Summary bars across preprocessings.
  * Overlaid PCA scatters (raw vs preprocessed, Procrustes-aligned).
  * Similarity network between preprocessings.

## How to read it

* Prefer preprocessings with **high** CKA, RV, Trustworthiness, EVR and **low** Procrustes and Grassmann.
* Consistent gains across datasets are stronger evidence.
* Use as an **unsupervised filter** before supervised evaluation on `y`.

## Caveats

* Unsupervised agreement with raw does not guarantee better prediction. Always validate downstream.
* Grassmann needs equal feature spaces; it is set to NaN otherwise.
* PCA focuses on variance, which may include noise; metrics should be read jointly, not in isolation.
