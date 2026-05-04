# Pure-CNN architecture roadmap (rounds 10+)

User direction (2026-04-30): **stop stacking, focus on the deep model itself.**
"Le stacking je peux le faire tout seul" — the contribution we want from
nicon_v2 is a strong pure CNN, not an ensemble glue. Every new variant from
round 10 onwards is a stand-alone deep learner whose entire prediction comes
from a single forward pass through one neural network.

## Current production CNN (round 8)

`NiconV2A` with `bank=extended_lowrank, lowrank_rank=32, branch_se=True,
trainable_ops=True, operator_reg_lambda=0.0, bjerrum=True`. 11 strict-linear
AOM branches → RMSBranchNorm → input-level branch SE → 3 residual conv
blocks (kernels 7/5/3, channels 32/64/96) with per-block channel SE → GAP
→ Linear. ≈ 325 K params at p = 2151. Closes the AOM-PLS-best gap to +35.6 %
median on the representative cohort.

## Round 10 — single-model architectural extensions

Each ranked by expected value (best first). All keep the multi-branch AOM
front; we modify the trunk, normalisation, head, or branch fusion.

### Tier 1 — high EV, low cost

* **V2L-LearnableRMS.** `RMSBranchNorm` currently estimates RMS once on the
  first training batch. Replace with a learnable scalar gain `γ_b` per
  branch, initialised at `1 / RMS(b)` and updated by gradient. This lets
  the network refine the AOM-MKL branch weighting through training instead
  of fixing it at one snapshot.
* **V2O-MultiKernelStem.** Replace the first conv block (`Conv1D(11→32, k=7)`)
  with parallel branches at kernels {3, 5, 7, 9} (DeepSpectra-style
  Inception), summed or concatenated. Keeps mid-trunk and head identical.
  Targets datasets where features span multiple wavelength scales.
* **V2P-WavelengthAttention.** Replace the GAP head with a single-layer
  multi-head attention over the post-trunk wavelength tokens (8 heads,
  d_model = 96). Attention weights are dataset-level interpretable and
  can identify which wavelengths drive prediction.

### Tier 2 — moderate EV, moderate cost

* **V2M-DeeperConditional.** Add a 4th trunk block only when `p ≥ 1024
  and n_train ≥ 500` (Codex F4 conditional). Default 3 blocks for short /
  small-n datasets.
* **V2N-AuxYHead.** Multi-output head: predict y *and* an auxiliary
  target derived from the input (e.g. PLS-projection coefficients on
  the train fold's PLS basis). Auxiliary loss is a regulariser; main
  loss is unchanged. Inspired by Padarian 2019 multi-task soil model.
* **V2Q-DenseConnect.** DenseNet-style: each block's output is concatenated
  with all previous blocks' outputs before the next conv. Doubles or
  triples mid-trunk channels but keeps params bounded by the head's GAP
  averaging.

### Tier 3 — speculative, higher cost

* **V2R-WaveletInit.** Initialise the SG / FD kernels with Daubechies-4 or
  symlet wavelet coefficients instead of polynomial-fit SG. Stronger
  multi-resolution prior than SG.
* **V2S-AOMTransformer.** After branch concat, treat the 11 branches as
  a sequence of "tokens" and apply a 1-layer self-attention. Replaces the
  branch-SE with an attention-MKL.
* **V2T-DerivativeAware.** Add an explicit `∂x/∂λ` channel via a fixed
  Sobel-like 1D filter, separate from the SG d1 branch. Forces the network
  to attend to derivative magnitude.

## Method

Each round uses a single CNN variant on the user's 10-dataset representative
cohort. Compare against:

* `Ridge-baseline`, `PLS-baseline` — internal sanity controls.
* `V2H-lowrank-r32` — current best CNN, the variant to beat.
* paper Nicon, paper CatBoost, paper TabPFN-raw, paper Ridge — external
  references via the cohort-manifest CSVs.

Acceptance: a variant is accepted if it matches or beats V2H-lowrank-r32
on **median Δ% rmsep across the cohort** AND on at least 5/10 per-dataset
deltas. Codex review every round.

Stopping criterion: this is the deep learning headline. Stop when V2H+ 
clears the AOM-PLS-best smoke gate (≥ 10 % wins or median ratio ≤ 1.05) on
the representative cohort.

## Out of scope (explicit)

* OOF stacking (V2J / Stack-Ridge-PLS-V1c / Stack-AOMRidge-PLS-V1c) — user
  will handle separately.
* Bayesian / conformal UQ — orthogonal direction, deferred.
* Self-supervised pretraining on LUCAS — too expensive for the wall-clock
  budget.
