# Round 12 — beyond CNN: Transformer / Mixer / U-Net / Distillation

User direction (2026-05-01): "lâche-toi au niveau archi, sors des CNN si
nécessaire". V2L's architecture-only ceiling is ≈+18 % vs Ridge / +35 % vs
AOM-PLS-best — Codex round 9 said this is the limit for pure CNN. Round 12
explores **non-CNN** architectures and **training-time** improvements that
keep a single-model predict path.

## Constraints (still in force)

* **No stacking at predict time.** Single forward pass per sample.
  AOM-Ridge / PLS may appear as **training-time teachers** (knowledge
  distillation, regularisation), but their predictions are not summed at
  inference.
* All variants benchmarked on the user's 10-dataset representative cohort.
* Codex review every round.

## Variant proposals

### V3 — AOM-Transformer (CNN front + Transformer trunk)

```
Input (N, 1, L)
  → 11 strict-linear AOM branches (= V2L front-end)
  → per-branch RMS norm + branch SE
  → 1 ResConvBlock(11→64, kernel=7) + MaxPool(2)        (one CNN block to reduce L)
  → Transformer encoder (2 layers, d_model=64, heads=4, FF=128)
       — input: (N, L', 64); self-attention over wavelengths
  → mean pool over L' → (N, 64)
  → Dropout(0.3) → Linear(64, 1)
```

Different from the failed V2P-attnHead: the transformer is in the **trunk**
(after one conv block), not at the head. Attention learns long-range
wavelength-band dependencies that the local convs cannot.

### V4 — AOM-Mixer (MLP-Mixer 1-D)

```
Input (N, 1, L) → AOM branches → (N, 11, L)
  → patch the L axis: split into K=16 patches of size L/K, each gets an MLP
    embedding → (N, 11, K, d_patch)
  → 2 Mixer blocks: alternating
       (a) token-MLP: mix across K patches per (channel, dim)
       (b) channel-MLP: mix across 11+d_patch per (patch, position)
  → mean pool over patches → (N, d_patch)
  → Linear(d_patch, 1)
```

Lightweight, no attention, tractable param budget.

### V5 — U-Net 1-D (encoder-decoder with skip connections)

```
Input (N, 1, L) → AOM branches → (N, 11, L)
  Encoder (3 levels): conv → pool ×2; channel doubling
  Bottleneck: 1 ResConvBlock
  Decoder (3 levels): upsample + conv + skip-add from encoder
  GAP → Linear
```

The encoder-decoder gives strong gradient flow and integrates information
across multiple resolutions. Provenance from segmentation.

### V6 — Distillation from AOM-PLS (training-only teacher)

```
At train time, for each fold:
  1. fit AOM-PLS-best (compact bank, simpls covariance) on the train fold
  2. compute teacher predictions z_teacher on train+val
  3. CNN training loss = MSE(y_true, y_pred) + λ · MSE(z_teacher, y_pred)
At predict time: only the CNN forward pass — single model.
```

Single CNN forward pass at inference. Adds minor wall-clock at train
time (the AOM-PLS teacher is fast, fold-local). λ tunable per-dataset
or fixed at 0.3.

### V7 — Test-Time Augmentation (TTA)

```
At predict time:
  1. Generate K=5 Bjerrum-augmented copies of X_test
  2. Run CNN forward on each
  3. Mean over K → final prediction
```

Single model, no extra training. Moderate compute at predict time.

## Training-time improvements (orthogonal to architecture)

* Cosine annealing with restarts (currently OneCycleLR).
* Stochastic Weight Averaging (SWA) over the last N epochs.
* Higher epoch budget (currently 200) with longer patience.
* Mixup / Manifold mixup at the post-AOM-front level (concat + interpolate).
* Mean teacher (EMA of CNN weights).

## Implementation order (highest EV first)

1. **V6 — Distillation from AOM-PLS** (highest EV, lowest risk; training-only change).
2. **V3 — AOM-Transformer trunk** (clean architecture extension, novel).
3. **V7 — TTA** (predict-time only; cheap).
4. **V5 — U-Net 1-D** (proven for spectra in segmentation).
5. **V4 — AOM-Mixer** (lightweight, may not beat V2L).

V6 + V3 + V7 in round 12; V4 / V5 in round 13 if needed.

## Stop criterion

Same as round 11: smoke gate is ≥ 10 % wins or median ratio ≤ 1.05 vs
AOM-PLS-best. Round 11 V2M-deeper achieved 1/10 wins; we aim to push to
3-5/10 in round 12 with V6 distillation alone.
