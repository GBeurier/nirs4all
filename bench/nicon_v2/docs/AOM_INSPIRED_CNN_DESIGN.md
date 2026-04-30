# AOM-inspired CNN architectures — design proposal (round-robin Codex review before coding)

## Motivation

The full 38-dataset comparison shows our `Stack-Ridge-PLS-V1c` is **at the
mid-tier of NIR baselines** but loses to AOM-PLS-best (+15 %) and
AOM-Ridge-best (+11 %) on every dataset. The pure CNN (`NiconV1c-concat-bjerrum`)
is far worse — we beat it by 34 % via stacking.

The user is convinced a stronger pure CNN is reachable. The AOM line of work
is the obvious source of inspiration: AOM's success comes from running PLS /
Ridge through a **bank of linear spectral operators** (SG / SNV / MSC / EMSC /
OSC / Detrend / Whittaker / Gaussian) and combining them via superblock
kernels, MKL weights, or per-component selection. The CNN analogue is a
**multi-branch front-end** where each branch applies a different (fixed or
learnable) preprocessing, then a shared trunk learns from the stacked
features.

This document proposes three concrete designs (V2A, V2B, V2C) and asks Codex
for review **before** implementation. We will iterate based on the review and
benchmark on the **full 61-dataset cohort**.

## Inspiration map: AOM-PLS / AOM-Ridge → CNN

| AOM concept | CNN translation |
|-------------|------------------|
| Operator bank (Identity, SG-1, SG-2, FD, NW, Detrend, Whittaker, Gaussian-deriv) | Multi-branch input where each branch is a fixed Conv1D filter initialised from the corresponding SG / Gaussian / detrend kernel |
| Superblock (K = sum_b s_b² K_b) | Branch concatenation on the channel dim with learnable per-branch scaling |
| MKL block weights (KTA-supervised) | Soft per-branch gating learnt by a tiny attention head reading post-trunk features |
| Active superblock (top-K KTA) | Stochastic-depth-style branch dropout that prunes low-KTA branches at training time |
| Per-component selection (POP-PLS) | Per-block selection: each conv block reads from a different branch (multi-head conv) |
| Branch-global (SNV / MSC outer wrapper) | The SNV / MSC operator is a pre-conv layer; we let it be a separate branch |

## V2A — Multi-Branch Frozen Operators + Shared Trunk (baseline AOM-CNN)

```
Input (N, 1, L)
├─ Branch 1 [Identity]                               → (N, 1, L)
├─ Branch 2 [Conv1D init SG(w=11, p=2, d=1)]         → (N, 1, L)
├─ Branch 3 [Conv1D init SG(w=11, p=2, d=2)]         → (N, 1, L)
├─ Branch 4 [Conv1D init Gaussian(σ=1.5)]            → (N, 1, L)
├─ Branch 5 [SNV(in-place)]                          → (N, 1, L)
├─ Branch 6 [MSC(learnable ref)]                     → (N, 1, L)
├─ Branch 7 [Detrend(poly=1)]                        → (N, 1, L)
├─ Branch 8 [Conv1D init Whittaker(λ=10)]            → (N, 1, L)
↓
Concatenate along channel dim                         → (N, 8, L)
↓
[Shared trunk]
  Conv1D(8 → 32, kernel=7, padding=3) → LayerNorm → GELU → Dropout1d(0.1) → MaxPool(2)
  Conv1D(32 → 64, kernel=5, padding=2) → LayerNorm → GELU → Dropout1d(0.1) → MaxPool(2)
  Conv1D(64 → 128, kernel=3, padding=1) → LayerNorm → GELU → MaxPool(2)
↓
GAP1D → Flatten → Dropout(0.2) → Linear(128 → 1)
```

* Operators are **frozen** (Conv1D weights = SG/Gaussian/Whittaker kernels, `requires_grad=False`).
* SNV, MSC are stateful PyTorch modules with `fit_state` (MSC reference
  computed on `A` only).
* `~50 K parameters` (small, suitable for `n_train ≥ 40`).
* Naturally length-invariant (GAP head).

## V2B — Same as V2A but operators are *learnable* + L2-from-init regulariser

```
Same architecture, but:
  - Each operator branch: Conv1D weight is initialised from the SG/Gaussian/Whittaker kernel
    BUT requires_grad = True.
  - Add a regulariser  λ_op · Σ_b ‖W_b − W_b^init‖² to the loss
    (penalises drift from the chemometric-meaningful initialisation).
  - λ_op tuned on smoke; default 1e-3.
```

This is V2A but with the Helin 2022 (learnable EMSC) trick generalised to all
operators. Falls back to V2A if `λ_op = ∞`.

## V2C — Branch gating (MKL-CNN)

```
[multi-branch frozen ops, same as V2A]
↓
[Per-branch shallow conv stem: 1 → 4 channels each]
↓
[Branch gate]
  GAP per branch → 4 features per branch → 32 features total
  → Linear(32 → 8) → softmax → 8 weights w_b ∈ Δ^8
  → broadcast to branch features  (multiply each branch's 4 channels by its weight)
↓
[Concatenation: (N, 32, L)]
↓
[Shared trunk: Conv1D(32 → 64, k=5) → LayerNorm → GELU → ... → GAP → Linear]
```

The branch gate is a **soft KTA**: it learns which operators matter for the
current dataset. For interpretability we record `w_b` per dataset.

## Open design questions for Codex

1. **Operator bank choice.** The 8-operator set {Identity, SG-1, SG-2,
   Gaussian, SNV, MSC, Detrend, Whittaker} is taken from AOM-PLS `compact`.
   Are there operators that are **chemometrically valuable but
   non-trivial in PyTorch** (OSC, EMSC degree=2, ASLSBaseline)? OSC requires
   a fitted reference response; EMSC is a closed-form least-squares fit.
   Which should be included as a frozen-first-then-learnable layer?

2. **Branch-fusion mechanism.** Three options:
   - (a) plain concat (V2A);
   - (b) gated weighted sum after a shared 1-channel projection (MKL-CNN, V2C);
   - (c) Inception-style multi-scale parallel convs **on top of** the
     concatenated branches (DeepSpectra 2019).

   Which has highest EV given 40-2151 features and 40-2000 train samples?

3. **Augmentation in the multi-branch model.** Bjerrum 2017 augmentation
   alters the raw spectrum (offset / slope / multiplicative). Should the
   augmentation happen
   - (i) **before** the branches (so all branches see the augmented spectrum), or
   - (ii) **after** the branches (so each branch sees its original
     preprocessed view, augmented in branch-space)?

   We default to (i); is there a chemometric reason to prefer (ii)?

4. **Length robustness on short spectra.** Datasets like DIESEL (`p=401`)
   and Beer (`p=576`) are smaller than ECOSIS (`p=2151`). The 4-block
   max-pool stack reduces length by 16×, leaving `L/16 ∈ [25, 134]` for GAP.
   Acceptable? Or should we cap at 3 blocks?

5. **Loss function and regularisation.** Cross-entropy/MSE only? Or add an
   auxiliary loss matching the post-branch features to a frozen PLS
   prediction (knowledge distillation from AOM-PLS-best as a teacher)?
   The latter is risky but could lift small-n performance.

6. **Training schedule.** Same OneCycleLR + AdamW + early stopping (200 epochs
   / patience 20)? Or use a longer warmup since the multi-branch front has
   more to learn (or to *not* drift, in V2B)?

7. **Stop criterion.** A new pure-CNN model is worth pursuing only if it can
   beat AOM-PLS-best on at least 30 % of the cohort while still passing the
   length-robustness invariants and not blowing up on `n_train ≤ 50` (the
   `Biscuit_Sucrose` failure mode that the stack also hits).

## Plan

1. Submit this proposal to Codex.
2. Apply Codex feedback.
3. Implement the chosen variant (V2A first, then V2B if V2A is competitive).
4. Run on the full 61-dataset cohort, 1 seed.
5. Codex review of code + first results.
6. Iterate (V2C MKL-CNN, then deeper variants if needed).

Stopping when either (a) we beat AOM-PLS-best on ≥ 30 % of the cohort with
paired Wilcoxon p < 0.05, or (b) we exhaust the 8-phase / 12-GPU-hour budget
in `Prompt.md`.
