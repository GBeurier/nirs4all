# nicon / decon — Weakness Analysis

This document enumerates the failure modes of the existing `nicon` and `decon` 1-D CNNs in `nirs4all.operators.models.tensorflow.nicon` and `.pytorch.nicon`. Each weakness is paired with literature evidence (see `source_materials/literature_review/LITERATURE_REVIEW.md`) and a remediation lead that flows into `docs/HYPOTHESES.md`.

The analysis was produced after reading the actual source files (TF, PyTorch, JAX) and confronting them with the gap analysis G1-G4 of the literature review.

## Severity scale

* **C — Critical.** Causes wrong-output mode or loss of orders of magnitude of performance on at least one dataset.
* **H — High.** Likely worth ≥ 5 % RMSEP across the cohort.
* **M — Medium.** Likely worth 1-5 %; or only matters on subsets.
* **L — Low / hygiene.** No measurable performance impact but obstructs cleanliness or reproducibility.

---

## W1 — Sigmoid output head saturates the regression target [C]

`nicon` ends in `Dense(1, activation="sigmoid")` (TF) / `nn.Sigmoid()` (PyTorch). For regression this assumes the target is already scaled to (0, 1) and even then the gradient through `σ` is **≤ 0.25** for all inputs, with most of the mass clamped to 0 or 1 for any prediction outside ±2.2.

Concrete consequence: on Beer original-extract (target range ~ 4-13), if `y` is min-max-scaled to [0, 1], any sample at the extremes (e.g. y=1.0) only receives a learning signal through σ′ ≤ 0.25 once the network's pre-σ activation crosses 2.2; below that the gradient explodes. This is one of the primary reasons NICON collapses on small-n datasets where outliers drive RMSE.

**Evidence.** Cui & Fearn (2018), DeepSpectra (Zhang 2019) and BEST-1DConvNet (Wang 2024) all use **linear** outputs for regression. Sigmoid is reserved for classification logits.

**Remediation lead → H1.** Replace sigmoid with identity for regression heads; expose a `head_activation` parameter; keep sigmoid only for binary classification.

## W2 — Mixed activations break SELU's self-normalisation invariant [H]

The `_build_nicon` body sets:

```
Conv1D(8 …) → SELU                       # SELU + AlphaDropout (correct)
Dropout 0.2 (AlphaDropout in TF) →
Conv1D(64 …) → ReLU + BatchNorm
Conv1D(32 …) → ELU + BatchNorm
Dense(16, sigmoid) → Dense(1, sigmoid)
```

SELU + AlphaDropout requires the *entire* network to maintain the LeCun-normal init + SELU + AlphaDropout invariant. Mixing in ReLU + BatchNorm + sigmoid + ELU breaks the invariant, so the SELU layer's only contribution becomes the activation curve itself, while AlphaDropout (which is calibrated for SELU's mean/variance) silently mis-regularises subsequent layers.

**Evidence.** Klambauer et al. (2017) on SELU; Helin (2022), Cui & Fearn (2018), Mishra/Passos (2021-2023) all use one activation throughout (ReLU / GELU / ELU + BatchNorm or LayerNorm), no SELU.

**Remediation lead → H2.** Commit to a single activation discipline. Recommend ELU or GELU + LayerNorm (small-batch friendly) + standard dropout. Drop AlphaDropout.

## W3 — Stride-only downsampling with huge kernels truncates short spectra [H]

`_build_nicon` strides are 5, 3, 3 with kernels 15, 21, 5. For an input of length `L`, the output length is

```
L_out = floor( ( floor( ( floor( (L - 15)/5 + 1 ) - 21)/3 + 1 ) - 5)/3 + 1 )
```

Concrete examples:
* `L = 700` (Tablet) → 397 → 125 → 40
* `L = 576` (Beer)  → 113 → 31 → 9
* `L = 401` (DIESEL) → 78 → 20 → 6

After 3 strided convs Beer has only 9 effective time steps with 32 channels = 288 features fed into Dense(16). DIESEL has 6 → 192. The receptive field has eaten almost the entire spectrum, leaving little room for the dense head to discriminate.

**Evidence.** Cui & Fearn (2018) advocate **smaller kernels (3-7)** with stride 1 + max-pool 2; Mishra & Passos (2022) report concat-derivatives + GAP wins consistently; DeepSpectra uses an Inception block to keep multiple receptive fields in parallel.

**Remediation lead → H3.** Replace 3 large strided convs with 3-5 small-kernel (3-7) blocks + max-pool 2 + Global Average Pooling head (Cui-Fearn / DeepSpectra style). Kernel sizes scale to spectrum length.

## W4 — Capacity / head parameterisation blow-up [H]

Two related sub-failures:

**W4.a — `Flatten` + double-sigmoid in NICON.** After `Flatten()` the dense head is `Dense(16, sigmoid) → Dense(1, sigmoid)`. Two sigmoids in series is unusual and creates a vanishing-gradient bottleneck (chain rule multiplies `σ′·σ′ ≤ 1/16`). On a typical short spectrum the post-conv flat dim is small enough that the bottleneck dominates the learnable expressivity.

**W4.b — Channel-multiplier blow-up in DECON.** DECON multiplies channels ×2 each conv (1 → 2 → 4 → 8 → 16 → 32 → 64). With kernel 7-9 each conv has roughly `2 · kernel · in` depthwise params; the depthwise stack accumulates ~ `2 × (7 · 2 + 7 · 4 + 5 · 8 + 5 · 16 + 9 · 32 + 9 · 64) ≈ 1880` depthwise params, plus ~4096 final separable, plus 128·32 + 32·1 ≈ 4128 dense. On Beer (n_train = 40) with 576 features the model has more parameters than training samples, so even with regularisation the variance of fitted predictions is huge.

**Evidence.** Cui & Fearn (2018) Section 3.2: GAP head + single linear projection consistently outperforms `Flatten + Dense + Dense` on three industrial NIR datasets. Walsh et al. (2023) tabulation confirms. Padarian et al. 2019 ten-thousand-sample heuristic flags small-n overparameterisation as the dominant CNN failure mode.

**Remediation lead → H3 / H4.** Use Global Average Pooling → linear projection (regression) or softmax (classification). Cap channel multiplier; introduce a parameter budget heuristic (`params ≤ 5 · n_train · n_features` *or* `≤ 1e6`, whichever larger).

## W5 — No concatenated derivatives at the input [H]

`nicon` and `decon` both ingest the spectrum as a single 1-D channel. Mishra & Passos (2022, *TRAC*) explicitly recommend the concatenated-derivatives input: `[raw, SG-1st-derivative, SG-2nd-derivative]` as 3 parallel channels, citing this as the single most consistent input scheme for 1-D CNN regression on NIR.

**Evidence.** Mishra/Passos (2022) review; Mishra et al. (2023) augmentation paper; Walsh et al. (2023) tabulation.

**Remediation lead → H5.** Add a deterministic differentiable preprocessing front-end that concatenates raw + 1st-SG + 2nd-SG (and optionally SNV / MSC) as input channels. Optionally make the SG window learnable.

## W6 — No principled augmentation in the model itself [H]

The current `nicon` / `decon` models contain no built-in augmentation. The pipeline operators offer Gaussian noise / wavelength shift / mixup, but these are applied *outside* the model, so a researcher running the model standalone (or in finetuning) gets no augmentation.

**Evidence.** Bjerrum, Glahder & Skov (2017, arXiv 1710.01927) — EMSC-parameter augmentation (offset / slope / multiplicative) applied per epoch is the strongest single trick for small-n NIRS deep models. Mishra et al. (2023) confirms. Yao et al. (2022) C-Mixup is unused in NIRS literature.

**Remediation lead → H6 (Bjerrum-style augmentation) and H7 (C-Mixup).**

## W7 — No skip connections / multi-scale features [M]

Both architectures are pure feed-forward stacks. For NIR data with overlapping absorption bands (combination + overtones spanning octaves) it is well known that multi-scale receptive fields help.

**Evidence.** DeepSpectra (Zhang 2019) Inception block; 1D-Inception-ResNet (Tian 2023); Transformer-CNN (Wang 2024) all show multi-scale gains on n > 500 NIR data.

**Remediation lead → H8.** Add an Inception or ResNet-1D block in the trunk.

## W8 — Hard-coded BatchNorm with potentially small batches [M]

`_build_nicon` uses BatchNormalization in conv blocks 2 and 3. With training batches of 8-16 (typical for NIRS where n=40-300) BatchNorm statistics are noisy.

**Evidence.** Wu & He (2018) GroupNorm; Ba et al. (2016) LayerNorm; Helin (2022) NIRS practice.

**Remediation lead → H4.** Use LayerNorm (per-channel) or GroupNorm; configurable.

## W9 — No uncertainty quantification [M]

Both models output a point estimate. Padarian et al. (2022, *Geoderma* 425) on LUCAS soil shows MC-dropout PIs cover ~74 % vs nominal 90 %, which is mediocre but better than nothing.

**Evidence.** Padarian (2022); Mishra (2025) UQ comparison; Liland (2025) conformal NIR.

**Remediation lead → H9.** Add MC-dropout inference, deep ensembles (5 networks), and conformal calibration on a holdout split.

## W10 — No learnable preprocessing [M]

ACT (AAAI 2024) and Helin (2022) show that bolted-in *learnable* preprocessing (EMSC as a layer; learnable SG window) can replace classical preprocessing entirely. NICON has none.

**Remediation lead → H10.** Add an optional `LearnableEMSC` layer (Helin 2022) and `LearnableSG` window (small differentiable convolution with a smoothness penalty).

## W11 — Output activation sigmoid hard-coded for classification [L]

The classification variant uses `Dense(num_classes, softmax)` for `num_classes>=3` but `Dense(1, sigmoid)` for binary — fine but inconsistent with the regression head's misuse of sigmoid.

**Remediation lead.** Unify by always emitting logits, applying `sigmoid_with_logits` / `softmax_cross_entropy` in the loss.

## W12 — No principled handling of repeated measurements [M]

The dataset contract (`SpectroDataset.set_repetition('Sample_ID')`) is honoured only when the pipeline configuration explicitly invokes it. The CNN has no awareness of repetition, so cross-validated R² on datasets with multiple spectra per physical sample is inflated.

**Evidence.** Walsh et al. (2023) explicitly flags this as the chronic mistake of the deep-NIR literature.

**Remediation lead.** Always cross-validate at the physical-sample level; aggregate predictions per sample at evaluation time.

## W13 — Hyperparameter ranges in tuning template are too wide [L]

The `nicon_sample_finetune` dictionary spans `filters ∈ {4, 8, …, 256}`, `kernel ∈ {3, …, 15}`, `strides ∈ {1, …, 5}` with all combinations Cartesian. For a 5-conv-layer search this is > 10^7 configurations; defaults to random search; rarely converges to a good model under any realistic budget.

**Remediation lead.** Provide focused search spaces driven by literature priors (kernel ∈ {3, 5, 7}; filters in fixed pyramid 16/32/64/128; etc.).

## W14 — Length robustness: model crashes / collapses on short spectra [H]

Beyond the receptive-field issue (W3), there is a hard-correctness invariant: every variant must produce non-empty output for the minimum cohort `p`. Concretely, with NICON's `(stride=5, kernel=15) → (stride=3, kernel=21) → (stride=3, kernel=5)`:

* `p = 401` (DIESEL): post-stage outputs `78 → 20 → 6` — works, but the final 6 timesteps × 32 channels = 192 features are then `Flatten()`-fed to `Dense(16, sigmoid)`, giving a 192→16 projection trained on at most 113 samples (DIESEL_b-a).
* `p = 250` (some ARABIDOPSIS variants): post-stage outputs `47 → 9 → 1` — the final layer would have a single timestep; `Flatten` fed to Dense becomes a 32-feature linear head with negligible capacity.
* `p = 200`: `37 → 5 → ⌊(5−5)/3+1⌋ = 1` — borderline; any further reduction crashes.

The benchmark protocol (BENCHMARK_PROTOCOL §7) requires every model variant to pass forward+backward sanity tests on the spectrum-length set `{401, 576, 700, 1154, 2151}` and report the effective receptive field. This is enforced by `tests/test_length_robustness.py`.

**Evidence.** Cui & Fearn (2018) and Mishra/Passos (2022) advocate small-kernel + max-pool with same-padding to keep variable-length spectra well-conditioned.

**Remediation lead → H3.** Switch to small kernels with max-pool 2 + GAP; the receptive field is then a function of depth (deterministic), and length collapse cannot occur because GAP averages over whatever timesteps remain.

## W15 — No ensemble support; no calibrated transfer baseline [L]

Both models train a single network. Lakshminarayanan (2017) deep ensembles + Huang (2017) snapshot ensembles are well-established.

**Remediation lead → H9.**

---

## Cross-reference table

| ID | Severity | Affects | Hypothesis |
|----|----------|---------|------------|
| W1 | C | Regression head | H1 — linear output |
| W2 | H | All | H2 — single activation discipline |
| W3 | H | Short spectra | H3 — small-kernel + GAP backbone |
| W4 | H | Small-n cohorts | H3 / H4 — GAP head |
| W5 | H | All | H5 — concat-derivatives |
| W6 | H | All | H6 / H7 — Bjerrum + C-Mixup |
| W7 | M | Medium-n cohorts | H8 — Inception / ResNet-1D |
| W8 | M | Small batch | H4 — LayerNorm/GroupNorm |
| W9 | M | All | H9 — UQ + ensembles |
| W10 | M | All | H10 — learnable EMSC / SG |
| W11 | L | Classification | hygiene |
| W12 | M | Repetition cohorts | physical-sample CV |
| W13 | L | Tuning | focused search space |
| W14 | H | Short spectra | length-robustness gate |
| W15 | L | All | H9 — ensembles |

The first iteration (Phase 1) addresses **W1, W3, W4, W5** (C + H severity, low risk) and produces a baseline that is already expected to outperform plain `nicon` on the entire cohort. Later iterations add augmentation (H6/H7), multi-scale (H8), and uncertainty (H9).
