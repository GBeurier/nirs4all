# Round 17 candidate architectures — beyond the V2L ceiling

> **Status:** Reviewed by Codex (round-12 review). EVs and rankings
> have been recalibrated; several previously-ranked candidates were
> excluded because they violate the user's single-model / no-stacking /
> no-PLS-injection constraints. **Realistic NN-only ceiling on this
> cohort: ratio 1.25-1.30 vs AOM-PLS-best, NOT 1.05.** Matching
> AOM-PLS-best is empirically not achievable under the current
> constraints — be prepared for an honest negative outcome.

**Goal.** Find a neural-network architecture that gets as close as
possible to AOM-PLS-best on the representative cohort. User direction:

* **No pretraining** (LUCAS-V6b validated as marginal in R14/R16).
* **No stacking** (no `PLS(X) + CNN_residual(X)` two-component models;
  no PLS-derived auxiliary targets; no differentiable PLS layer).
* **Single-model predict path** (a single forward pass per sample at
  inference time — this excludes K-net deep ensembles unless the
  user explicitly relaxes the constraint).
* Multi-seed validation required (5 seeds × 10 datasets = 50 paired
  obs); R14 / R16 demonstrated that single-seed signals on this
  cohort are typically seed-noise.

**Honest realistic targets** (per Codex round-12 review):

| Target | Achievability with NN-only |
|--------|----------------------------|
| Match AOM-PLS-best (ratio ≤ 1.05) | **Unachievable** — gap is +29 %, best single-idea EV is 0-3 % closure |
| Beat Ridge multi-seed (Wilcoxon p < 0.05) | Possible but unlikely (Ridge wins 7/10 on representative, 34/38 on curated) |
| Match V2L within 1-3 % at lower variance | Most plausible outcome of any round-17 idea |
| Open the smoke gate by relaxing constraints | Yes — allowing deep ensembles or PLS-residual hybrid would close ~10-15 pp |

**Current state (multi-seed validated).** V2L-learnableRMS ≡
V6b-LucasPretrained-V2M (Wilcoxon p = 0.76 on 50 paired obs). Median
ratio vs AOM-PLS-best ≈ 1.39 (≈ +29 % gap to close, much further than
Codex's 5-pp threshold for "continue").

## What has been tried (rounds 5-16)

### Architecture changes already explored
| Idea | Round | Outcome |
|------|------:|---------|
| Multi-branch AOM-superblock (11 strict-linear ops + branch concat) | R5 | Production base (V2A) |
| Channel SE blocks (mid-trunk) | R5 | Default ON in V2A |
| Branch-level SE (input-level MKL analogue) | R7 | V2C — accepted (default ON) |
| Dilated conv super-block (dilations 1, 2, 4) | R7 | V2D — neutral, deferred |
| Wider trunk channels (32→64→128, 32→96→128) | R7 | Tiny gains, not worth the params |
| Deeper trunk (4-block 32→64→96→128) | R7, R11 | V2M-deeper — marginal +1 % vs V2L |
| Low-rank Detrend / Whittaker (rank 32 SVD) | R8 | V2H — accepted (default ON, params bounded) |
| Learnable per-branch RMS scale | R10 | V2L-learnableRMS — production |
| Multi-kernel parallel stem (3, 5, 7, 9) | R10 | V2O — neutral, deferred |
| Wavelength-attention head (post-trunk MHA) | R10 | V2P — neutral, deferred |
| AOM-Transformer trunk (1 conv + 2-layer encoder) | R12 | V3 — **hurts** (8/10 losses, p=0.16) |
| Tied global RMS (single shared scale) | R11 | Tied diagnostic — neutral |
| Per-branch RMS init=1 (no inverse-RMS data init) | R11 | Init1 diagnostic — neutral |
| Knowledge distillation from compact AOM-PLS | R12 | V6 — marginal (median −0.9 %, p=1.0) |
| Knowledge distillation from extended AOM-PLS | R13 | V6b — marginal (median −1.4 %, p=0.92) |
| Stochastic Weight Averaging | R13 | SWA — inactive (8/10 datasets unchanged) |
| Test-time Bjerrum augmentation (K=5) | R12 | V7-TTA — neutral (median +0.8 %) |
| LUCAS pretraining + V6b distill | R15-16 | V6b-LUCAS — neutral multi-seed (p=0.76) |

### Training-time changes already explored
* OneCycleLR with warmup, AdamW, weight decay 1e-4 — **production default**
* Bjerrum 2017 EMSC-parameter augmentation (offset, slope, multiplicative) — **production default**
* C-Mixup with default σ_y — rejected at smoke
* L2-from-init regularisation on AOM operator parameters — F1 fix (R6)
* AMP fp16 training — production default on CUDA
* Early stopping with patience 20

### Confirmed dead ends
1. **Architectural depth alone** plateaus around V2L/V2M (1 % delta range).
2. **Attention in the trunk** (V3) hurts on small-n NIR (likely overparam).
3. **Distillation from PLS-family teachers** is marginal (teacher accuracy ceiling).
4. **TTA with train-distribution augmentations** explores already-invariant neighbourhoods (no variance reduction).
5. **SWA with OneCycleLR + early stopping** averages near-stationary late weights.
6. **LUCAS supervised pretraining** has domain-conditional gains that wash out at the cohort level.

## What has NOT been tried — broad NN ecosystem sweep

This list is exhaustive (overlapping with `ROUND_12_ARCHITECTURES.md`
proposals plus brand-new ideas across the wider NN landscape). Each
entry is annotated with prior art on small spectroscopy / 1-D
regression where available.

### Tier 1 — high EV, low cost (best for round 17)

#### A. Deep ensemble (Lakshminarayanan 2017)
* **What.** Train 5 V2L instances with different random seeds; average
  predictions (or soft-voting) at inference. Single-model loss is
  unchanged — predict path is `mean over k forward passes`.
* **Why it might work.** R14 / R16 multi-seed analysis revealed
  per-seed std ≈ 21–37 %. An ensemble average should reduce the
  variance term of the bias-variance decomposition by ~k⁻⁰·⁵.
* **Prior art.** Sun et al. 2018 "Ensemble of CNN for soybean NIR"; the
  original Lakshminarayanan paper showed 1.5–4 % RMSE reduction on
  regression tasks.
* **Cost.** 5x training time per dataset; predict path 5x. Ablation:
  `n_ensemble = 5`. Expected GPU-h on representative cohort: 30 h.
* **EV.** Median Δ% closure of 5–15 %.
* **Risk.** Ensemble of biased models is still biased — if V2L
  systematically misses the same structure, averaging won't help.
* **Multi-seed compatible.** Yes; the *test* is itself multi-seed of
  the ensemble.

#### B. Heteroscedastic regression head
* **What.** Replace `Linear(96, 1)` with `Linear(96, 2)` predicting
  `(μ, log σ²)`. Train with negative log-likelihood loss
  `0.5·log σ² + (y - μ)²/(2σ²)`. At inference output μ.
* **Why it might work.** The NLL loss is robust to heteroscedastic
  noise (which NIR has at very small n). Datasets where Ridge
  catastrophically wins (Beer, Corn_Oil) have heavy-tailed residual
  distributions; an MSE-trained CNN can't down-weight them.
* **Prior art.** Lakshminarayanan 2017; Mishra 2021 "Heteroscedastic
  CNN for soil NIR".
* **Cost.** Negligible — 1 extra output unit. Same training time.
* **EV.** 3–8 % closure on the heavy-tailed datasets.
* **Risk.** Negative-log-likelihood is harder to optimise than MSE;
  may need lower lr.

#### C. PLS-residual CNN hybrid
* **What.** Train V2L to predict `y - y_PLS(X)` (residual), at inference
  predict `y_PLS(X) + V2L_residual(X)`. Forces the CNN to learn ONLY
  the non-linear residual signal that PLS can't capture.
* **Why it might work.** R12-16 evidence shows V2L beats Ridge on the
  Chla+b sets where the relationship is non-linear in the wavelength
  axis. Constraining the CNN to only model residuals removes its
  responsibility for the linear baseline (which it currently
  re-discovers from scratch with worse efficiency than PLS).
* **Prior art.** Wang & Trinkle 2014; "residual learning" in
  computer-vision sense; classic chemometrics "PLS plus non-linear
  correction".
* **Cost.** Per-fold PLS fit (~1 s) + same V2L training. Predict path:
  PLS forward + V2L forward (single pass each).
* **EV.** 5–15 % closure (depends on how much non-linear signal there is).
* **Risk.** If PLS already captures everything (linear-dominated
  dataset), the residual is noise and CNN over-fits. Mitigation:
  use PLS as data-dependent inductive bias only on n_train > 500.
* **Multi-seed compatible.** Yes.

#### D. Mixture-of-Experts (MoE) head
* **What.** Replace `Linear(96, 1)` with K=4 expert linear heads + a
  gating network that produces softmax weights from the GAP feature
  vector. Final prediction = `Σₖ gᵢ · headₖ(z)`.
* **Why it might work.** The cohort has 5+ distinct domains
  (soil, plant chemistry, food, beverage, inorganic). A single linear
  head must serve all; an MoE head can specialise per-domain implicitly.
* **Prior art.** Shazeer 2017; Switch Transformer; recent MoE-CNN for
  multi-task spectra (Wang 2023).
* **Cost.** 4× head params + small gate (96→4 + softmax). Same training time.
* **EV.** 4–10 % closure.
* **Risk.** MoE typically requires very large datasets to learn the
  routing. Top-K=1 hard routing might overfit on 80-sample datasets.
* **Multi-seed compatible.** Yes.

### Tier 2 — moderate EV, moderate cost

#### E. State-space model trunk (Mamba / S4)
* **What.** Replace the residual conv trunk with a 2-layer Mamba block
  (selective state-space model with linear-time complexity). Linear
  attention over wavelengths without the V3 transformer's quadratic
  cost.
* **Why it might work.** Mamba beats transformers on long-sequence
  benchmarks; spectra are exactly long sequences (4200 wavelengths).
  The state-space recurrence captures long-range wavelength
  dependencies that local convs cannot.
* **Prior art.** Gu & Dao 2023 "Mamba"; rapidly growing literature in
  signal processing. Less specific to spectra.
* **Cost.** Mamba kernel implementation needed (PyTorch impl available
  via `mamba_ssm`); ~50 % more training time than V2L. EV could be
  high but uncertain.
* **EV.** 5–15 % closure (speculative; prior art is sparse).
* **Risk.** Same V3 risk: too many free parameters for small n. But
  Mamba has stronger inductive bias than transformer.

#### F. Multi-task auxiliary loss (V2N-AuxYHead)
* **What.** Add an auxiliary head that predicts a derived target from
  the input — e.g. the per-fold PLS basis-projection coefficients
  (5 components). Auxiliary loss is `MSE(z_pls_proj, head_aux(features))`
  added with weight λ=0.3.
* **Why it might work.** The auxiliary target is *cheap to compute*
  on the train fold and provides a denser supervision signal that
  forces the trunk to learn PLS-friendly representations.
* **Prior art.** Padarian 2019 "multi-task soil model"; Mishra 2022
  "auxiliary chemistry targets".
* **Cost.** Per-fold 5-component PLS fit (~1 s) + extra head. Same
  base training time.
* **EV.** 3–8 % closure.

#### G. Heteroscedastic + Deep Ensemble (B + A combined)
* **What.** 5-net ensemble where each net has the heteroscedastic NLL
  head. Predict: `μ_combined = mean(μₖ)`, `σ²_combined = mean(σ²ₖ + μₖ²) − μ_combined²`.
* **Why it might work.** Lakshminarayanan's original recipe; gives
  both predictive uncertainty and point estimate.
* **Cost.** Same as A + B combined (5x training, negligible head).
* **EV.** 8–18 % closure (compounds A and B).

#### H. KAN — Kolmogorov-Arnold Networks
* **What.** Replace the trunk's linear projections with learnable
  univariate B-spline activations (KAN layer). Trunk: 11→32→64→96
  with KAN edges.
* **Why it might work.** KAN has been claimed to be more
  parameter-efficient on small data (Liu et al. 2024). NIR features
  are univariate (per-wavelength absorbance), so KAN's B-spline edges
  fit the inductive bias.
* **Prior art.** Liu et al. 2024; explosion of follow-ups in 2024-25.
* **Cost.** 2-5x slower than V2L (B-spline forward is heavier than
  matmul); ~30 GPU-h on the cohort.
* **EV.** 3–10 % closure (very speculative).
* **Risk.** KAN papers benchmark on synthetic / small datasets; real
  NIR is noisier.

#### I. Conformer / Squeezeformer trunk
* **What.** Conv + attention hybrid block: each block has
  `Conv1D + multi-head self-attention + Conv1D + macaron FFN`.
  Originally for ASR; converts well to 1-D spectra.
* **Why it might work.** Captures both local (conv) and global
  (attention) patterns in one block. The V3 transformer-only failed
  partly because it dropped local conv structure; Conformer keeps both.
* **Prior art.** Gulati 2020 (Conformer); Bertasius 2022; ASR
  benchmarks dominate.
* **Cost.** 2x slower than V2L per epoch.
* **EV.** 4–10 % closure.

### Tier 3 — speculative, higher cost

#### J. Differentiable PLS layer
* **What.** Replace the GAP-Linear head with a differentiable PLS
  block: SVD-derived scores + linear regression on scores, with
  n_components as a learnable smooth knob. The gradient propagates
  through the PLS decomposition.
* **Why it might work.** Combines deep features with PLS's small-n
  inductive bias. The classical chemometrics community has explored
  this (e.g., Auxiliary Network PLS, Næs 2002); modern differentiable
  versions are rare.
* **Cost.** Implementation effort high (custom autograd).
* **EV.** 5–15 % closure.

#### K. Implicit deep equilibrium model (DEQ)
* **What.** Replace finite-depth trunk with a fixed-point iteration:
  `z = f(z, x)` solved to convergence at each forward pass. Memory-
  efficient; enables effectively infinite depth.
* **Prior art.** Bai et al. 2019 (DEQ); not yet used on NIR.
* **Cost.** 3-5x slower forward (iterative solver); harder to train.
* **EV.** 3–8 % closure (speculative).

#### L. Test-time training (TTT)
* **What.** At inference, run a few gradient steps on a self-supervised
  loss (e.g. spectra reconstruction) per test sample before predicting.
* **Prior art.** Sun et al. 2020; works well on distribution shift.
* **Cost.** Per-sample gradient steps at predict time → 50-100x slower
  inference. Not viable for this benchmark.
* **EV.** 3–8 % closure.
* **Risk.** Predict path is no longer a single forward pass (violates
  user constraint).

#### M. Sharpness-Aware Minimization (SAM)
* **What.** Replaces standard SGD/Adam with SAM, which seeks flat
  minima by adding a perturbation step. Improves generalisation.
* **Prior art.** Foret 2021; widely used for robust ConvNets.
* **Cost.** 2x training time (SAM does 2 forward-backward per step).
* **EV.** 1–4 % closure.
* **Risk.** SAM gains shrink on noisy small-data regimes.

#### N. Spectrogram + 2-D CNN
* **What.** Apply STFT or wavelet transform to convert (1, L) spectra
  into a 2-D time-frequency representation (windowed first/second
  derivatives at multiple scales), then use a 2-D CNN.
* **Prior art.** Tsakiridis 2020; widely used in audio.
* **Cost.** Same as 1-D CNN at most.
* **EV.** 3–8 % closure.
* **Risk.** STFT loses localisation; the optimal window per dataset
  is a hyperparameter.

#### O. Stochastic depth + cutout regularisation
* **What.** Drop entire residual blocks with probability p during
  training (Huang 2016). Combine with cutout / band-mask of input
  spectra.
* **Cost.** Same training time.
* **EV.** 1–3 % closure.

### Tier 4 — likely not worth it (covered by previous failures)

* V2Q-DenseConnect (deferred from R10) — concat-based DenseNet variant.
  Mostly a width-vs-depth trade and GAP head averages out the gain.
* V2R-WaveletInit — initialise SG kernels with Daubechies wavelets.
  R12 V3 already showed alternative front-end inductive biases don't
  break the ceiling.
* V2T-DerivativeAware — explicit ∂x/∂λ Sobel channel. Already covered
  by the SG-d1 / FD branches in the AOM bank.
* Snapshot ensemble (cosine restart + ensemble). SWA was already
  inactive; snapshot ensemble has the same root cause (model is
  near-stationary at end of training).
* Larger backbone (more channels in trunk) — V2H already saturated
  param count; no signal at small n.

## Constraint-compliance audit (Codex round-12)

After review, several previously high-ranked candidates **violate the
constraints** the user explicitly stated and must be excluded unless
the user relaxes them:

| Idea | Violates | Why |
|------|----------|-----|
| **A** Deep ensemble (5-net) | single-model predict path | `mean(K forward passes)` is K trained models |
| **G** Heteroscedastic + ensemble | same as A | combines A with B |
| **C** PLS-residual hybrid | no-stacking / single-model | predict = `PLS(X) + CNN(X)` is a 2-component model |
| **F** Multi-task PLS-coef aux loss | borderline (PLS-injected supervision) | trains CNN with PLS-derived target; close to PLS distillation that R12 V6 already failed |
| **J** Differentiable PLS layer | borderline (PLS-injected) | imports PLS structure into model architecture |
| **L** Test-time training | single-model predict path + pretraining-like | gradient updates per test sample |

**Constraint-compliant candidates only.** These are the candidates
that genuinely respect "NN-only, no pretraining, no stacking, single
forward pass at inference":

* **B** — Heteroscedastic / robust-loss head (Gaussian NLL or Student-t / Huber)
* **D** — Mixture-of-Experts head
* **E** — Mamba / S4 trunk
* **H** — KAN trunk
* **I** — Conformer trunk
* **K** — Implicit deep equilibrium
* **M** — Sharpness-Aware Minimisation (SAM)
* **N** — Spectrogram + 2-D CNN
* **O** — Stochastic depth / cutout
* **P** — Per-dataset HPO (added per Codex review; ranked best
  remaining at +8-12 % closure in Codex round 11; not a new
  architecture but a search over the existing one)
* **Q** — Robust-loss only (Huber / Student-t NLL; no variance head;
  added per Codex review as the safer subset of B)

## Round 17 candidate ranking (Codex-recalibrated)

EVs are now expressed as median Δ% vs V2L on the representative cohort
(NOT vs AOM-PLS-best gap closure). The constraint-compliant ceiling
is **≈ 1-3 % improvement vs V2L**; nothing in this list realistically
clears the AOM-PLS smoke gate alone.

| Rank | Idea | EV (median Δ% vs V2L) | GPU-h | Multi-seed cost |
|-----:|------|---------------------:|------:|----------------:|
|  1   | **B** Heteroscedastic / Student-t head |  −0 to −3 % |   5  |   +6 |
|  2   | **Q** Robust loss (Huber, β=1.0) |  −0 to −2 % |   4  |   +5 |
|  3   | **D** MoE head (K=2 or 4) |  −0 to −2 % |   6  |   +8 |
|  4   | **P** Per-dataset HPO (LR / batch / depth) |  −0 to −3 %  |  60  |  +120 |
|  5   | **E** Mamba / S4 trunk |  −0 to −3 %  |  20  |  +25 |
|  6   | **I** Conformer trunk |  −0 to −2 %  |  15  |  +20 |
|  7   | **N** Spectrogram + 2-D CNN |  −0 to −2 % |  10  |  +12 |
|  8   | **M** SAM optimiser |  −0 to −1 %  |  10  |  +12 |
|  9   | **H** KAN trunk |  −0 to −2 %  |  35  |  +40 |
| 10   | **O** Stochastic depth |  −0 to −1 %  |   5  |   +6 |
| 11   | **K** DEQ trunk |  −0 to −2 %  |  15  |  +18 |

Per Codex: **R12 V3 attention trunk losing 8/10 vs V2L is the
strongest signal that pure-architectural changes plateau** at this
cohort size; therefore E / I / H / K (all variants of "different
trunk inductive bias") inherit V3's risk. EVs ranked accordingly.

## Codex-recommended round 17 plan (trimmed)

**Run AT MOST 2 ideas, gated.** The Codex review explicitly said
"trim to 2 ideas, not ~150 GPU-h. The 4-variant plan is mostly
invalid because A/G/C violate the stated predict-path or pure-NN
constraint."

### Step 1 (cheap, ~5-6 GPU-h): smoke test V2L-StudentT-NLL on the
representative cohort, single-seed.

```
V2L-StudentT-NLL: V2L architecture, Linear(96, 2) head predicting
                  (μ, log σ²); Student-t NLL loss with df=5; 
                  inference output μ.
```

**Gate.** If single-seed median Δ% ≤ −3 % vs V2L AND wins ≥ 6/10
(no catastrophic regression on any dataset > +20 %), promote to
multi-seed validation.

### Step 2 (only if step 1 passes): 5-seed validation of V2L-StudentT-NLL.

Costs ~25 GPU-h (5 seeds × 6 baseline × 0.85 because the Student-t
head is faster than V6b distill). Ship as round 17 verdict.

### Step 3 (only if step 2 passes the gate): try V2L-MoE-K2 next.

If V2L-StudentT-NLL truly improves ≤ -3 % vs V2L multi-seed, try
adding MoE-K2 head on top. Same gating (single-seed → multi-seed).

**Stopping criterion.** If two consecutive ideas miss the
single-seed gate, hard stop; declare "constraint-compliant NN
ceiling reached" and ship V2L as production.

## Realistic ceiling (Codex direct quote)

> "Best honest target is about 1.30-1.35 ratio vs AOM-PLS-best,
> maybe 1.25 with lucky per-dataset HPO. The smoke gate ratio 1.05
> is not realistic without changing constraints."

If the user wants to genuinely match AOM-PLS-best, the only realistic
paths are:

1. **Relax single-model constraint** → use deep ensemble (A) or
   PLS-residual hybrid (C). EV +5-15 % closure; could clear smoke gate.
2. **Relax no-pretraining constraint** → resume LUCAS pretraining
   experiments with more data / multi-task targets / contrastive.
   EV uncertain.
3. **Allow stacking** → user explicitly excluded but worth flagging.

Under the **current** strict constraints, NN-only ceiling is
~1.25-1.35 ratio vs AOM-PLS-best (i.e., ≈10-15 pp closure from
current V2L's +29 % gap, optimistically).

## Open questions for the user

1. **Confirm the constraints**: NN-only single forward pass at
   inference, no pretraining, no stacking. Y/N?
2. **If forced to pick a round-17 first variant**: Codex recommends
   **B** (heteroscedastic / Student-t NLL head) as the cheapest
   constraint-compliant idea. Single-seed gate first; ~5 GPU-h smoke,
   then ~25 GPU-h multi-seed if it passes. Acceptable budget?
3. **If you accept relaxing one constraint**, which? The single-model
   constraint is the easiest to relax (deep ensemble at predict time)
   and gives the highest EV. The no-pretraining constraint costs
   100-300 GPU-h to test multi-task LUCAS or contrastive. The
   no-stacking constraint is what we already had pre-round-9 (highest
   EV but you said no).
4. **Acknowledgement**: do you accept that **matching AOM-PLS-best
   exactly is empirically not achievable** under the current
   constraints, and that the best honest result we can ship is "V2L
   gets 65-75 % of the gap on Chla+b sets, ties Ridge multi-seed
   elsewhere"?
