# NIRS Universal Spectral Embedding — Onboarding & Implementation Spec (for Claude Code / Opus 4.5)

## 1) Objective (detailed)

### 1.1 Goal

Build a **universal spectral encoder** that produces a **fixed-size embedding** (z \in \mathbb{R}^d) from **1D NIR spectra** with:

* **Variable length**: ~250 to ~2100 wavelength points
* **Variable wavelength grids**: different ranges and step sizes across instruments (irregular or regular)
* **Multiple representations**: absorbance, reflectance, transmittance, etc.
* **Heterogeneous instruments**: portable / lab / industrial, different manufacturers, acquisition setups

The embedding must be:

* **Generic** (transferable across datasets/instruments)
* **Robust** to baseline/scatter and common instrument artifacts
* Usable for **downstream regression/classification** with lightweight heads (ridge/PLS/MLP) and also **TabPFN/TabICL** after embedding extraction.

### 1.2 Datasets

We have ~**45 real datasets**, each with:

* (L) (wavelength points): **250–2100**
* (N) samples: **60–15,000**
* Instrument-specific wavelength grids (step/range differ)
* Targets may exist (regression/classification) but **pretraining must not require labels**

We also have a **synthetic generator** that can produce **credible** NIR-like spectra (not necessarily perfect chemistry, but structurally plausible). The generator should be used to:

* Increase diversity
* Enforce meaningful priors (baseline/scatter/noise/peak interactions)
* Support multi-instrument simulation (grid/range/step variation)

### 1.3 High-level approach

Use a **ViT-style encoder** adapted to 1D signals and variable grids:

* Treat spectra as sequences of points ((\lambda_i, x_i))
* Encode wavelength using **positional encoding on real (\lambda)** (often better with `log(λ)`)
* Use a **latent bottleneck** architecture (Perceiver-style) to handle variable length efficiently:

  * Cross-attention from fixed latents to variable input tokens
  * Self-attention among latents
* Output embedding via pooling over latents (mean or CLS-latent)

Pretraining uses **Masked Spectral Modeling (MAE-like)** + **learned invariances** via augmentations and an auxiliary **AsLS baseline-consistency loss**.

---

## 2) Technical Specifications (models, data providers, training logic)

### 2.1 Repository layout (recommended)

```
spectral_encoder/
  data/
    real/<dataset_id>/
      spectra.npy
      wavelengths.npy
      targets.npy           # optional
      metadata.json
      baseline_asls.npy     # precomputed
    splits/
      intra_dataset/
      leave_one_dataset_out/
    synthetic/
      generator.py
      configs/
  augmentations/
    baseline.py
    scatter.py
    noise.py
    masking.py
    compose.py
  preprocessing/
    normalization.py
    asls.py
    meta_encoding.py
  models/
    encoder/
      spectral_perceiver.py
      positional_encoding.py
      metadata_embedding.py
    decoder/
      mae_decoder.py
    heads/
      regression.py
      classification.py
  losses/
    mae.py
    baseline_consistency.py
    contrastive.py          # optional
    adversarial_dataset.py  # optional
  training/
    pretrain.py
    finetune.py
    utils.py
    configs/
  evaluation/
    downstream.py
    ablations.py
    lodo.py
```

### 2.2 Data format & contracts

#### 2.2.1 Sample contract

Each sample must be representable as:

* `x`: Tensor[Li] — spectral intensities
* `lambda`: Tensor[Li] — wavelengths (float, real units, not indices)
* `baseline`: Tensor[Li] — **precomputed AsLS baseline** (same grid)
* `meta`: dict — dataset-level metadata (may be missing)

**Important**: `Li` varies across samples/datasets.

#### 2.2.2 Metadata contract (sparse allowed)

`metadata.json` (dataset-level):

```json
{
  "dataset_id": "dataset_12",
  "instrument": "FOSS",
  "manufacturer": "FOSS",
  "instrument_type": "industrial",
  "mode": "absorbance",
  "spectral_range": [850, 2500],
  "spectral_step": 2.0
}
```

Rules:

* Missing categorical values → `"UNKNOWN"`
* Missing numeric values → omit field or set null; encode with `(value, mask)` downstream.

#### 2.2.3 Precomputed AsLS baseline cache

We want **fast training**, so baseline is computed offline.

* Provide a CLI script `preprocessing/precompute_asls.py` to compute and store `baseline_asls.npy`.
* If `baseline_asls.npy` is missing, training should still run but baseline-consistency loss is disabled (warn).

### 2.3 Minimal preprocessing (do not overdo)

Before augmentation/model:

* **Per-spectrum normalization only** for numerical stability:

  * `(x - mean)/std` or robust `(x - median)/MAD`
* Do **not** apply deterministic baseline removal (AsLS), SNV, MSC, derivatives, detrend, etc. as fixed preprocessing.

  * Those should become **augmentations** and/or **auxiliary invariance losses**.

### 2.4 Augmentations (classic preprocessing → learned invariances)

#### 2.4.1 Mapping: preprocessing → augmentation

* AsLS baseline removal → **random smooth additive baseline drift**
* SNV / MSC → **multiplicative scatter + additive offset**
* Detrend → **random linear tilt**
* Derivatives → do not impose; emulate by adding/removing slow trends + mild smoothing noise

#### 2.4.2 Required augmentations

Implement these as composable transforms:

1. **Baseline drift** (smooth additive)

* Generate smooth function on wavelength grid (poly/spline)
* Add with small amplitude

2. **Scatter** (multiplicative + offset)

* `x <- a*x + c`, with `a ~ N(1,σ_a)`, `c ~ N(0,σ_c)`

3. **Noise**

* Add Gaussian; optionally wavelength-dependent

4. **Band dropout** (optional)

* Mask small spectral windows (not the MAE mask)

### 2.5 Masked Spectral Modeling (MAE-like)

#### 2.5.1 Masking strategy

* Mask **60–80%** of points by **contiguous blocks** in wavelength space.
* Avoid IID masking only; block masking is more realistic.

Outputs:

* `mask`: Tensor[Li] with 1=visible, 0=masked
* `x_masked`: replace masked points by 0 or learnable mask token value (implementation choice)

#### 2.5.2 MAE loss

Reconstruct **only masked points**:

* `L_mae = MSE(x_hat[mask==0], x_true[mask==0])`
  Optionally weighted by per-wavelength variance.

### 2.6 Encoder model: Spectral Perceiver (ViT-style)

#### 2.6.1 Tokenization (point-based, variable length)

For each point (i):

* input features: `[x_i, PE(λ_i), meta_embed]`
* Project to `D` via MLP/Linear

**Positional encoding**:

* Use real wavelengths: `PE(λ)` or `PE(log(λ))`
* Provide implementation in `positional_encoding.py`

#### 2.6.2 Metadata embedding (robust to missing)

* Categorical: learned embeddings + `"UNKNOWN"`
* Numeric: encode as `[value, mask]` and project to `D_meta`
* Inject metadata:

  * Preferred: inject into **latents** (additive bias or concat+MLP)
  * Acceptable: concat to each input token

#### 2.6.3 Latent bottleneck

* Learnable latents: `M=64` (configurable)
* Latent dimension: `D=128` or `256`

Layers:

1. **Cross-attention**: `latents <- Attn(Q=latents, K=tokens, V=tokens)`
2. **Latent self-attention blocks**: 2–4 blocks
3. Output:

   * `z_latents`: Tensor[B, M, D]
   * `embedding z`: mean over latents or CLS-latent

Complexity: ~O(L*M) per forward, good for variable L.

### 2.7 MAE Decoder

* Input: `z_latents`, `lambda`, `mask`
* Output: reconstructed `x_hat` aligned with original wavelength grid
  Implementation options:
* Simple per-point MLP conditioned on `PE(λ)` and a pooled latent summary
* Or lightweight cross-attention from masked points to latents (heavier)

Keep decoder small; encoder is the main asset.

### 2.8 Auxiliary losses

#### 2.8.1 Baseline-consistency loss (AsLS as auxiliary)

Goal: make embeddings invariant to AsLS baseline component without applying AsLS at inference.

Offline: `b* = AsLS(x)`
Then:

* `z = encoder(x)`
* `z' = encoder(x - b*)`
  Loss:
* `L_base = MSE(z, z')` (use pooled embedding) OR MSE of latents mean

Weight: `λ_baseline ~ 0.05–0.2` (tune)

#### 2.8.2 Optional: contrastive invariance

Two augmented views `x1, x2` of same spectrum:

* `L_contrast = InfoNCE(emb(x1), emb(x2))`
  This stabilizes representation but increases training cost.

#### 2.8.3 Optional: dataset-adversarial invariance

Classifier predicts `dataset_id` from embedding; use gradient reversal to encourage dataset-agnostic embeddings.

* Useful if LODO generalization is weak.

### 2.9 Training logic

#### 2.9.1 Multi-dataset sampling

Do not sample proportional to dataset size (would bias toward large datasets).
Use **uniform-by-dataset** sampling:

* pick dataset uniformly
* pick sample uniformly within dataset
  Optionally mix synthetic with probability `p_synth` (e.g. 0.3).

#### 2.9.2 Mixed precision & efficiency

* Use AMP (fp16/bf16)
* Precompute AsLS baseline
* Consider gradient checkpointing on attention blocks if memory tight

#### 2.9.3 Training loop (requirements)

For each batch:

1. normalize per spectrum
2. augment
3. MAE mask + forward encoder/decoder
4. compute `L_mae`
5. compute `L_base` using raw and baseline-corrected inputs (double encoder pass)
6. optional losses
7. backward + optimizer step

Total loss:
`L = L_mae + λb*L_base + λc*L_contrast + λa*L_adv`

### 2.10 DataLoader specification

#### 2.10.1 Unit datasets

* `SpectralDataset(root)` loads real dataset files
* `SyntheticSpectralDataset(generator, size)` generates samples on the fly

#### 2.10.2 Multi-dataset wrapper

* `MultiSpectralDataset(real_datasets, synth_dataset, p_synth)`
* implements uniform-by-dataset sampling

#### 2.10.3 Collate function (no padding)

Return lists:

* `x: List[Tensor[Li]]`, `lambda: List[Tensor[Li]]`, etc.

Encoder must accept lists and internally stack/loop (or implement a pack strategy). Keep it simple first.

### 2.11 Recommended default hyperparameters (starting point)

* Latents `M=64`
* Dim `D=128`
* Cross-attn layers: 1–2
* Latent self-attn blocks: 3
* Heads: 4–8
* MAE mask ratio: 0.7
* Optim: AdamW, lr=1e-4, wd=0.05
* Scheduler: warmup + cosine
* Batch size:

  * 3090: 64
  * 4090: 96–128
  * A100: 128–256

### 2.12 Training duration (expected)

For MAE + baseline loss:

* **~150–250 epochs** recommended
* Ballpark wall time:

  * RTX 4090: ~5–8 hours for 200 epochs (depending on total samples/steps)
  * RTX 3090: ~8–10 hours for 200 epochs
  * A100: about half

Stop criteria:

* embedding stability metrics plateau
* quick downstream proxy (ridge on 2–3 datasets) no longer improves

---

## 3) Experimental Planning (paper-ready)

### 3.1 Phase 0 — Reference baselines (fast)

On 5–10 representative datasets:

* `AsLS + SNV + PLS`
* `AsLS + SNV + ridge`
* `PCA + ridge`
* `PCA + TabPFN`

Metrics:

* RMSE/R² (intra-dataset)
* sensitivity to artificial baseline/scatter injected at test time

Deliverables:

* baseline table
* robustness plot

### 3.2 Phase 1 — Minimal encoder feasibility

Train:

* `Encoder + MAE only` (no aug, no meta, real-only)
  Evaluate:
* embedding + ridge vs PCA + ridge

Goal: show representation is usable.

### 3.3 Phase 2 — Core ablation: baseline vs learned

Compare:
A) classical preprocessing pipeline
B) encoder MAE (no aug)
C) encoder MAE + augmentations
D) encoder MAE + aug + **AsLS baseline-consistency loss**

Robustness tests:

* inject baseline drift at test
* inject scatter at test

Deliverables:

* ablation table
* robustness curves

### 3.4 Phase 3 — Generalization: Leave-One-Dataset-Out (LODO)

For each dataset i:

* train downstream on all other datasets’ embeddings
* test on dataset i

Compare:

* classical baseline + PLS
* PCA + ridge
* encoder embedding + ridge
* encoder embedding + TabPFN

Deliverables:

* LODO boxplots
* train→test heatmap

### 3.5 Phase 4 — Synthetic data impact

Pretrain variants:

* real-only
* real + synth (low)
* real + synth (high)

Evaluate LODO and robustness.
Goal: show synth improves generalization, not just in-dataset scores.

### 3.6 Phase 5 — Metadata impact (and missingness)

Variants:

* no meta
* meta on
* meta on + 50% meta dropout

Goal: meta helps but does not become required.

### 3.7 Phase 6 — Downstream final

Show encoder simplifies downstream:

* encoder + ridge/PLS/TabPFN competitive with heavy pipelines
* engineering complexity reduced

---

## Implementation priorities (for Claude Code)

1. Data contracts + loaders + collation (real + synthetic)
2. Precompute AsLS baseline caching script
3. Augmentation library
4. Spectral Perceiver encoder + positional encoding + meta embedding
5. MAE decoder + MAE loss
6. Baseline-consistency loss
7. Pretraining script with config system
8. Evaluation scripts (downstream ridge/PLS, LODO, ablations)
9. Optional: contrastive, dataset-adversarial

## Non-negotiable constraints

* Must support variable lengths and wavelength grids
* Must not require metadata at inference
* Must keep preprocessing minimal; invariances learned via training (aug + auxiliary losses)
* Must implement uniform-by-dataset sampling to avoid dominance of large datasets
* Provide deterministic runs (seed, config logging)

## Deliverables to generate automatically

* Saved encoder checkpoint
* Embedding dumps per dataset (numpy)
* Metrics JSON per experiment
* Plots: robustness curves, LODO boxplots, heatmaps

---
