# AOM-PLS: Scientific Article Plan

**Scope.** This document is not the article — it is the dossier that a co-author
or reviewer needs in order to decide whether AOM-PLS (as currently deployed in
`nirs4all.operators.models.sklearn.aom_pls`) is ready to be written up as a
methodological contribution, and what would still have to be done first.

**Audience.** Chemometricians and ML-aware spectroscopists. The angle is
explicitly: *"a computer-scientist's take on an old chemometrics problem"* — the
problem being the manual, combinatorial, grid-search-driven choice of
preprocessing for PLS.

---

## 1. Current Deployed AOM-PLS: Architecture

### 1.1 High-level description

PLS in NIRS is never just PLS. The community spends most of its calibration
effort *before* the regressor, choosing among scatter corrections (SNV, MSC,
EMSC), smoothing (Savitzky–Golay with a window and polynomial order),
derivatives (first, second, Norris–Williams gap), detrending, wavelet
denoising, and compositions of the above. The standard workflow is an outer
cross-validated grid over these recipes with an inner grid over the number of
latent variables — a Cartesian product whose size explodes while the selection
signal per candidate shrinks.

AOM-PLS (Adaptive Operator-Mixture PLS) replaces this outer grid with a
*single training pass* in which the preprocessing choice is embedded inside the
NIPALS loop itself. The key idea is that most NIRS preprocessing pipelines can
be written — or well approximated — as **linear operators** `A : R^p -> R^p`
acting on each spectrum. Linearity is not a bug; it is what lets the choice be
absorbed into the NIPALS weight update without re-fitting anything.

The high-level architecture is:

1. **Operator bank.** A curated set `{A_b}, b=1..B` of cheap, linear,
   pre-initialised operators. Each `A_b` has an explicit apply (`x -> A_b x`)
   and an explicit adjoint (`c -> A_b^T c`) costing `O(p)` per sample (no
   `p x p` matrix is ever materialised for convolutional operators).
2. **Gate.** A selection mechanism that, given the current NIPALS state,
   picks either a single operator (*hard gate*) or a sparse convex combination
   of operators (*sparsemax gate*, experimental).
3. **NIPALS extraction through the chosen lens.** Components are extracted
   with the standard NIPALS deflation of X and Y, but with a weight update
   that uses the *adjoint trick*: the weight direction is the adjoint image
   of the cross-covariance, then re-normalised after the forward pass.
4. **Identity is always in the bank.** This guarantees
   `RMSE(AOM-PLS) ≤ RMSE(standard PLS)` in expectation on the selection
   criterion: if no operator helps, identity wins and AOM-PLS *is* PLS.
5. **Deterministic, single training run.** No outer grid, no Optuna, no CV
   folds for operator selection (though an Optuna-friendly hook,
   `operator_index`, exists for the rare case where one wants to tune it
   externally).

Relative to the manual grid search workflow, this moves preprocessing from a
*hyperparameter* to a *parameter*. It is learned on the same training set that
fits the PLS coefficients, with the same inductive bias (NIPALS projection),
and with a computational budget that is linear in `B`, not exponential.

### 1.2 Low-level description

Let `X in R^{n x p}` be centered (scale defaults to off — per-column scaling
destroys the spectral shape that convolutional operators exploit) and
`Y in R^{n x q}` be centered. Let `{A_b}_{b=1..B}` be the initialised bank and
let `K` be a cap on the number of latent variables.

#### 1.2.1 The operator bank

`default_operator_bank()` (`aom_pls.py:575`) ships ~60 operators composed from
six families, all `LinearOperator` subclasses:

| Family | Class | Apply cost (per sample) | Symmetric `A^T = A` ? |
|---|---|---|---|
| Identity | `IdentityOperator` | `O(1)` copy | yes |
| Savitzky–Golay (smoothing + derivatives) | `SavitzkyGolayOperator` | `O(p)` same-mode convolution, zero-padded for strict linearity | no |
| Detrend (polynomial projection complement) | `DetrendProjectionOperator` | `O(p * deg)` via QR basis | yes |
| Finite differences | `FiniteDifferenceOperator` | `O(p)` convolution | no |
| Norris–Williams gap derivative | `NorrisWilliamsOperator` | `O(p)` convolution | no |
| Composed operators (SG ∘ Detrend) | `ComposedOperator` | sum of parts | generally no |
| Wavelet projection (extended bank only) | `WaveletProjectionOperator` | `O(p^2)` initialisation, `O(p)` afterwards | yes |
| FFT bandpass (extended bank only) | `FFTBandpassOperator` | `O(p log p)` | yes |

Each operator also exposes `frobenius_norm_sq()` used (in principle) for
normalised block scoring. In the current deployment path it is computed but not
consumed by the selection criterion.

#### 1.2.2 The adjoint trick

At a given deflation step with current residuals `X_res`, `Y_res`, the
cross-covariance is `c_k = X_res^T Y_res` (collapsed to a vector via the top
left singular vector when `q > 1`). Classical PLS sets `w = c_k / ||c_k||`. The
question is: what is the right `w` if the spectra are viewed *through the
preprocessing lens* `A_b`?

Mathematically, if preprocessing is `X' = X A_b^T` (row-vectors transformed by
`A_b`), the lensed cross-covariance is `A_b c_k`. The lensed weight — the one
that, if used to score `X' w`, is proportional to `t = X A_b^T w`,
equivalently `t = X' w` — is obtained by first reading `c_k` *through the
adjoint* (`g_b = A_b^T c_k`), normalising, then forward-passing through `A_b`
and re-normalising:

```
g_b  = A_b^T c_k
w_hat = g_b / ||g_b||
a_w   = A_b w_hat
w_b   = a_w / ||a_w||
t_b   = X_res w_b
```

This is implemented in `_nipals_extract` (`aom_pls.py:846`) and is what makes
the whole scheme cheap: we never instantiate `X' = X A_b^T` for each operator;
we only do `O(p)` convolutions on the `p`-dimensional vector `c_k`. For `B`
operators this is `O(B p)` work per component, against `O(B n p)` if we were
preprocessing then projecting.

#### 1.2.3 The hard gate (default)

The hard gate does not score operators on a cheap proxy. It runs a *full*
K-component NIPALS (`_nipals_extract`) for every operator, yielding, per
operator `b`, a sequence of prefix regression matrices `{B_b^{(k)}}_{k=1..K}`.
It then selects the `(b*, k*)` pair with the lowest RMSE on a held-out signal:

- If `X_val, Y_val` are provided (the caller ran a split upstream — this is
  the nirs4all orchestrator's default): the external split is used directly.
- Otherwise: a deterministic 20% internal holdout is drawn with
  `np.random.RandomState(42)` (`aom_pls.py:1043`).
- If the caller has already made the operator decision (e.g. Optuna wrote
  `operator_index`), the scan is skipped and only the final fit runs.

After selection, a *single* K-component NIPALS is re-run on the full
centered data with operator `A_{b*}`, and the regression coefficients are
trimmed to the prefix `k*`. If `n_extracted == 0` for the winner (degenerate
operator), the code falls back to identity.

The `Gamma` attribute, shape `(n_extracted, B)`, is one-hot in the hard path:
every row equals `e_{b*}`. It exists to keep the public interface identical
between hard and sparsemax paths.

**`n_components` is not what the name suggests.** Today the constructor
argument `n_components=15` functions as a *ceiling*, not a target. In the
default hard-gate-without-`operator_index` path, the algorithm auto-selects
the winning prefix `k*` from the `(operator, k)` grid and trims to it;
`self.n_components_` becomes `k*`, not 15. In the Optuna path
(`operator_index` pre-set) it becomes the literal user value. In the
sparsemax path it depends on whether external validation data is provided.
This three-way inconsistency is the deployment bug that W3 of the
`PUBLICATION_BACKLOG.md` splits into two explicit parameters:
`max_components` (always a ceiling) and `n_components` (either `"auto"`
for algorithm selection, or an integer for user-forced).

#### 1.2.4 The sparsemax gate (experimental)

Here the selection is soft. Scores are computed on the *first* component only
via the squared-correlation proxy `s_b = cov(y, t_b)^2 / (||y||^2 ||t_b||^2)`.
A sparse weight vector `gamma = sparsemax(s / (tau * max(s)))` is produced
(`_sparsemax`, `aom_pls.py:742`). Sparsemax differs from softmax in that it
projects onto the probability simplex and returns *exact* zeros for weak
entries, which matches the semantics we want ("most operators should drop
out"). NIPALS then runs with a mixed weight
`w_k = sum_b gamma_b * (A_b w_hat_b / ||A_b w_hat_b||)`, renormalised at each
component. `Gamma[k]` stores the full row, not just the argmax.

Empirically (see §3.3) the sparsemax path has not demonstrated an advantage
over hard selection on the benchmark suite. It is retained for researchers
but is not what the library uses by default.

#### 1.2.5 OPLS pre-filter

Before the main loop the caller may request `n_orth` orthogonal components to
be removed from `X`. `_opls_prefilter` extracts components that have maximum
`X`-variance but are orthogonal to `y`, which is the direction OPLS removes.
This is a pragmatic concession to chemometrics: in cases where we *know* a
dominant systematic variation in `X` is unrelated to `y` (batch effects,
pathlength variation), removing it before the main loop improves the
conditioning of downstream component selection. Pre-filter loadings are stored
in `_P_orth` and replayed at predict time.

#### 1.2.6 What is *not* in the deployed path

- **Non-linear preprocessing.** SNV and MSC are non-linear (per-sample
  normalisation). They are out of scope for the native bank. A
  pseudo-linear SNV adjoint was prototyped (§2.2) but is not shipped. Users
  who need SNV place it upstream of AOM-PLS in the nirs4all pipeline.
- **The sparsemax path is the only place Frobenius norms could plausibly
  matter** (for scale invariance of scores). Today they are not used.
- **The Torch backend** exists (`operators/models/pytorch/aom_pls.py`, 366 LOC)
  and batches SG convolutions through `conv1d`, but it does not change the
  algorithm — it is a speed tweak for large `B` and large `p`.

### 1.3 Guarantees and failure modes

| Property | Holds? | Argument |
|---|---|---|
| `AOM-PLS ≥ PLS` on the selection criterion | Yes, by construction | Identity is in the bank; the argmin over operators at the chosen prefix dominates identity's RMSE |
| `AOM-PLS ≥ PLS` in *generalisation* | Not guaranteed | Selection is on an internal 20% holdout (or external val) — small-sample variance can flip the ordering |
| Deterministic | Yes, given fixed RNG seed | Bank is fixed; selection is an argmin; holdout permutation uses `RandomState(42)` |
| Interpretable | Yes | `get_preprocessing_report()` returns the winning operator per component (one winner for hard gate; a sparse mixture for sparsemax) |
| Prediction mode | Yes | Coefficients and loadings are stored; `predict` replays OPLS filter then applies `B_coefs[k_selected - 1]` |

The main residual risk is that the internal 20% holdout is small and fixed —
on datasets with `n < 50` it is both noisy and shared across all `B`
operators. This is the failure mode the publication plan needs to address
(§3, §4).

---

## 2. Prototypes Benchmarked in `bench/AOM/` (and one sibling in the library)

All the models discussed in this section are unpublished work by the same
author and are therefore competitors only in the sense of *prior
explorations* — none of them has a paper to cite yet. The five
`bench/AOM/` prototypes were implemented and benchmarked against the
deployed AOM-PLS on five regression datasets (Amylose, Beer extract,
Firmness, Milk lactose, Leaf phosphorus). A sixth model, FCK-PLS, lives in
the main library (`operators/models/sklearn/fckpls.py`) rather than in
`bench/AOM/`, and is covered in §2.6 because it is conceptually adjacent and
was an earlier attempt at the same problem.

Results from `report.md` (FCK-PLS row added from separate runs):

| Model | Rice_Amylose | Beer | Firmness | Milk_Lactose | Leaf P | Geo-mean time (s) |
|---|---|---|---|---|---|---|
| **Baseline AOM-PLS** | **1.887** | 0.204 | **0.377** | 0.058 | 0.173 | ~4 |
| Bandit AOM-PLS | 1.923 | **0.166** | 0.432 | 0.061 | 0.168 | ~4 |
| DARTS PLS | 1.860 | 0.195 | 0.358 | 0.059 | **0.163** | ~13 |
| Zero-Shot Router | 1.887 | 0.204 | 0.387 | 0.058 | 0.173 | ~27 |
| MoE PLS | **1.853** | 0.196 | 0.513 | **0.055** | 0.172 | ~270 |

The picture is consistent: the deployed baseline is *never catastrophically
beaten* and is the only model that is uniformly fast. Below, each prototype
is summarised together with the reason it did not displace the baseline.

### 2.1 Bandit AOM-PLS — `models.py:118` (`BanditAOMPLSRegressor`)

**Idea.** Split operator selection into two phases. Phase 1: run a small
`screen_components=3` NIPALS per operator and rank by cumulative R². Phase 2:
run full NIPALS + held-out RMSE only for the top-`K` operators.

**Why it doesn't win.** Rank-truncation on a 3-component R² proxy
systematically misranks operators whose advantage emerges only from component
4 onwards. This is exactly the case for sharp-derivative operators on
low-frequency-dominated spectra (Beer is the exception where it helps;
Firmness is where it hurts, by +14% RMSE). The speed-up over baseline is
marginal because full NIPALS is *already* cheap for the default bank.

**Verdict.** Useful if the bank grows to hundreds of operators. Not useful at
the current bank size. Keep as a research branch; do not deploy.

### 2.2 Pseudo-Linear SNV — `pseudo_linear_aom.py`, `pseudo_linear_snv_v1.py`

**Idea.** SNV is `(x - mean(x)) / std(x)`. The `std(x)` term makes it
non-linear per sample, so its exact adjoint depends on `x` itself. The v1
prototype uses the identity adjoint (wrong but cheap). The v2 prototype
computes a closed-form adjoint under the assumption that the divisor `std(x)`
is held constant (i.e., evaluated once on the mean spectrum). Under that
assumption,

```
A^T c = (c - mean(c) - <S_x, c>/p * S_x) / std_s
```

where `S_x = (mean_spectrum - mean(mean_spectrum)) / std(mean_spectrum)`.

**Why it doesn't ship.** It enters a grey zone mathematically: the
stop-gradient assumption on `std(x)` makes the adjoint correct *around* the
training mean spectrum but biased at prediction time on samples whose `std`
deviates. On the benchmark, enhanced_aom (= default + pseudo-linear SNV)
neither improves nor degrades the baseline meaningfully. Adding an operator
that is "almost linear, if you squint" to a bank whose whole discipline is
strict linearity introduces a subtle failure surface for a small expected
win. *Not deployed; the same effect is better obtained by a non-linear SNV
step upstream of AOM-PLS in the nirs4all pipeline.*

### 2.3 Zero-Shot Router — `zero_shot_router.py`

**Idea.** Compute ~10 spectral statistics (baseline-to-derivative variance
ratio, noise ratio, sample-to-sample correlation, Donoho–Johnstone noise
estimate, etc.), score nine candidate preprocessing recipes with a hand-built
scoring function, 3-fold cross-validate the top four, and route to the best
*only if* it beats "raw" on all folds *and* by ≥25% on average.

**Why it doesn't win.** The safety threshold is so conservative — it has to
be, because heuristic-scored preprocessing regressions are common — that on
four of the five benchmark datasets the router picks "raw" and *re-runs
AOM-PLS on raw data*. The measured cost of this is 5x the baseline time
(because it still cross-validates the top four candidates) for identical RMSE.
Only on Firmness does it pick anything different, and it loses slightly
(0.387 vs 0.377).

**Verdict.** It validates, negatively, a thesis worth stating in the paper:
*heuristic preprocessing routing cannot reliably beat an embedded
linear-operator search.* The router's value is diagnostic — its feature
outputs (`scatter_cv`, `pnr`, `smoothness`) are useful for reporting what
*kind* of spectra the user gave you, but not for deciding how to preprocess.

### 2.4 MoE PLS — `moe_pls.py`

**Idea.** Fit up to 94 preprocessing "experts" (nested chains of SNV, MSC,
EMSC, OSC, SG, wavelets, ArPLS, Kubelka–Munk, AreaNorm, etc.), screen them
with fast sklearn PLS OOF, promote 25 "core" experts to AOM-PLS OOF, remove
near-duplicates by correlation > 0.95, fit a RidgeCV meta-model on the OOF
predictions. Safety fallback to the best single expert if the meta-model does
not beat it in OOF RMSE.

**Why it doesn't win (in the general case).** It does marginally win on Rice
Amylose and Milk Lactose, where true non-linear preprocessing (SNV, MSC) is
genuinely helpful and which AOM-PLS's linear bank cannot match. It loses
badly on Firmness (+36% RMSE over baseline) because the safety fallback
picks a correlated-but-inferior single expert. It is 50–100x slower than the
baseline (130–715 s vs 1–11 s) due to the 94-expert OOF sweep.

**Verdict.** The scientifically honest way to frame this is: *MoE over
non-linear preprocessing is a strictly different modelling regime*. It is the
right tool when you know SNV/MSC are genuinely non-redundant with your linear
bank. In nirs4all, this capability already exists natively as a
`branch` + `merge: "predictions"` stacking pipeline, which composes better
with the rest of the system and is easier to audit. Keep MoE out of the
`aom_pls.py` codebase.

### 2.5 DARTS PLS — `darts_pls.py`

**Idea.** Differentiable architecture search (Liu et al., ICLR 2019) over
preprocessing. 18 operators, learnable `alpha` logits, Gumbel-Softmax with
temperature annealing from `tau=1` to `tau=0.1`, entropy regulariser to push
toward sparsity, differentiable PLS1 in PyTorch (`torch_pls1`) for the inner
loss. After training, pick between hard-snap (argmax operator) and top-3
weighted blend by validation RMSE, then refit with AOM-PLS.

**Why it is the most interesting prototype but still doesn't displace the
baseline.** It wins on two of five datasets (Amylose, Leaf phosphorus) by
~1%. It loses by less than 1% on Beer and Milk. The wins are within
measurement noise for these dataset sizes. But it is *consistently* slower
(13–24 s vs 1–11 s) because of the torch unrolling. And the `torch_pls1`
inner loop is a simplified PLS1 — it does not match the full NIPALS
factorisation used by AOM-PLS, which means the `alpha` gradient is computed
against a slightly biased model.

**Verdict.** DARTS is the only prototype whose *theory* would fit a paper's
"related work" section without special pleading. But the empirical gain
doesn't justify the complexity, and the biased inner PLS is a real
correctness concern. The architectural idea — viewing preprocessing as a
soft mixture during training and hardening at inference — is essentially
what the sparsemax gate *already* tries to do in AOM-PLS, only without the
surrogate PLS and without the inner optimisation. If anything, DARTS
motivates investing more in the sparsemax path rather than replacing it.

### 2.6 FCK-PLS — `nirs4all/operators/models/sklearn/fckpls.py`

**Idea.** Fractional Convolutional Kernel PLS. Instead of *selecting* a
preprocessing operator from a bank, FCK-PLS *concatenates* the outputs of a
whole bank of fractional-order convolutional filters and runs PLS on the
resulting wide feature matrix. A fractional filter is parameterised by
`(alpha, sigma, kernel_size)` where `alpha ∈ [0, 2]` smoothly interpolates
between smoothing (`alpha=0`), first derivative (`alpha=1`), and second
derivative (`alpha=2`), with a Gaussian envelope of width `sigma`. With the
default `alphas = (0, 0.5, 1, 1.5, 2)` and a single shared `sigma`, the
feature map blows `X ∈ R^{n×p}` up to `X_feat ∈ R^{n×5p}` and standard
sklearn `PLSRegression` is applied on top.

It is an honest alternative to AOM-PLS in the design space of "let PLS
choose which preprocessing to use": FCK-PLS exposes *all* preprocessings
simultaneously as features and lets PLS's own weight vector pick the useful
directions in the concatenated space. It supports NumPy and JAX backends.

**Why it was superseded by AOM-PLS.**

1. *Dimensionality and regularisation.* Stacking `L` filtered versions of a
   `p`-dimensional spectrum produces `L·p` features that are pairwise
   strongly correlated (they are filtered copies of the same signal).
   Standard PLS handles rank deficiency but tends to spread score mass
   across redundant features, which both dilutes component interpretability
   and makes the optimal number of components harder to pick. AOM-PLS's
   one-operator-per-model choice keeps feature dimensionality at `p`.
2. *Fractional kernels span a narrower family than the AOM bank.* The
   `(alpha, sigma)` parameterisation is elegant but covers only the
   Gaussian-envelope smooth-to-second-derivative axis. It cannot express
   detrending, Norris–Williams gap derivatives with non-unit segment, or
   compositions — all of which are first-class citizens of the AOM bank.
   In practice, for spectra where detrend or NW provides most of the lift
   (Amylose, Leaf phosphorus in the benchmark), FCK-PLS's feature space is
   structurally missing the right direction.
3. *No selection criterion, no identity-dominance guarantee.* FCK-PLS does
   not *select* — it always uses all filters. There is no analogue of
   "identity is in the bank, so I cannot be worse than PLS." Empirically,
   on spectra that already contain a dominant `alpha=0` direction, the
   extra filtered features are noise and degrade RMSE relative to vanilla
   PLS on raw spectra. Tuning is then shifted onto `n_components`, which
   is fragile.
4. *Interpretability.* For a chemometrics reader, "AOM-PLS picked SG
   (window=21, deriv=1) for all components" is a well-formed sentence.
   "FCK-PLS assigned weight vector entries 412..621 (in filter alpha=1,
   sigma=2) the largest PLS loading" is not.

**Verdict.** FCK-PLS is the "concatenate everything and let PLS sort it
out" baseline to AOM-PLS's "select the right operator per model." It is a
useful point of comparison in the paper — the experiment section should
include it as a baseline precisely because its weaker performance
*empirically motivates* AOM-PLS's selection-over-concatenation design
choice. It stays in the library as a sibling estimator, not a competitor
for the default recommendation.

### 2.7 BCD / DiffPLS / LatentPLS (sketches, `models.py`)

These three are early experiments kept only as waypoints:

- `BCDPLSRegressor` — learnable 5-tap smoothing kernel + scale + bias + PLS;
  it is "DARTS with a single operator". Useful for debugging the torch path,
  not for publication.
- `DiffPLSRegressor` — ResBlock CNN preprocessor + differentiable PLS1. By
  the time the CNN can express arbitrary preprocessing, you have left the
  chemometrics interpretability story entirely; AOM-PLS's selling point is
  that every component's preprocessing is a named operator.
- `LatentPLSRegressor` — conv autoencoder + AOM-PLS on latent. This is
  literally a different model; it belongs in a separate paper on
  autoencoder-regularised PLS.

### 2.8 Summary — why the deployed AOM-PLS won

Three reasons, in descending order of importance:

1. **Calibration-set efficiency.** The benchmark datasets have `n` between
   60 and 1224. The operator-selection signal is small per dataset; any
   method that spends that signal on *meta-learning a gating network*,
   *training DARTS alphas*, or *cross-validating 25 experts* has less signal
   left for the PLS fit itself. Baseline AOM-PLS spends it exactly once, on
   an internal holdout, and then commits.
2. **Linearity as a prior.** A linear operator bank is a tight, defensible
   inductive bias for physical spectra. Non-linear MoE and CNN preprocessors
   add capacity that generalisation gains do not pay for at these sample
   sizes. (They *would* pay for it at `n > 10^4`, which is the regime where
   a companion paper on `Torch` backend AOM-PLS starts making sense.)
3. **Identity in the bank.** This is the most underappreciated engineering
   choice. None of the alternative prototypes offer the same "at worst, I
   reduce to PLS" guarantee. Zero-Shot Router approximates it with a
   conservative threshold; MoE approximates it with a safety fallback; both
   approximations have failure modes on small data. AOM-PLS has no such
   failure mode on the *selection criterion* — its residual risk is purely
   the generalisation gap on the holdout.

---

## 3. What Must Be Done Before Publication

The method is, in my assessment, *mathematically complete* for a journal
paper. The publication gate is almost entirely empirical and statistical.

### 3.1 Benchmark extension

The current five-dataset benchmark is sufficient for internal development but
thin for publication. The author already maintains a **60-dataset benchmark
corpus with reference RMSE/R² scores** which should be the primary empirical
base of the paper. On top of that corpus:

- **Include well-known public NIRS corpora in the subset that is reportable
  without redistribution restrictions** (e.g. IDRC / Cargill Corn, Tecator,
  Shootout) so readers can reproduce at least part of the benchmark without
  access to the private collection.
- **Dataset diversity along three axes.** `n` (50, 200, 1000, 5000), `p`
  (100, 500, 2000), and response type (univariate regression, multivariate
  regression, classification via the AOM-PLS-classifier variant). Document
  which axes the 60-dataset corpus already covers and which bins are
  underpopulated — fill gaps with synthetic data from
  `nirs4all.generate.regression` if needed.
- **Hold-out protocol.** For each dataset, run (a) Kennard–Stone split, (b)
  SPXY split, (c) random 5×2 cross-validation. The last is what lets us
  compute corrected paired-t intervals.
- **Reference-score comparison.** The corpus already has reference scores
  from prior runs; report `ΔRMSE(AOM-PLS − reference)` per dataset as a
  headline figure, not just raw RMSE. This also acts as a regression test
  for every future modification of the bank or selection policy.

### 3.2 Statistical testing

This is currently missing everywhere and is what most chemometrics journals
will reject on. Required:

- **Nadeau–Bengio corrected resampled t-test** (JMLR 2003) across 5×2 CV
  folds to compare AOM-PLS against (i) standard PLS, (ii) grid-searched SG +
  PLS, (iii) POP-PLS, (iv) a stacking of SNV/MSC/SG variants via
  `{"branch": ..., "merge": "predictions"}`. Report p-values, not just
  means.
- **Friedman test + post-hoc Nemenyi** across all datasets for global ranking
  significance (Demšar, JMLR 2006). This is the chemometrics-community
  idiom.
- **Bootstrap confidence intervals** on RMSE for each (dataset, method)
  cell. Report CIs, not point estimates.

### 3.3 Ablation studies

At minimum three ablations:

1. **Bank composition.** Drop each family (SG-smoothing, SG-d1, SG-d2,
   detrend, composed) in turn. Quantify how much each family contributes
   and identify the minimum viable bank (the "POP-PLS bank" of ~9 operators
   is a natural comparison; see `pop_pls.py:68`).
2. **Hard vs sparsemax gate.** On which datasets does sparsemax help, if
   any? The current negative empirical evidence is from five datasets only.
   If sparsemax never helps after 20+ datasets, the honest move is to
   remove it from the deployed code and mention it only in the paper's
   "paths not taken" section.
3. **Holdout fraction and policy.** The deployed default is 20% internal
   holdout with a fixed seed. Compare against: 30%, 50%, 5×2 CV-based
   selection, and PRESS-based selection (as POP-PLS uses). Is 20% robust or
   lucky?

### 3.4 Formalisation

For a methodology paper, the `AOM-PLS >= PLS` guarantee needs to be stated
as a proposition:

> **Proposition (Identity-dominance).** *Let `X, Y` be centered and let `H`
> be a fixed holdout. For any operator bank `B` containing the identity
> operator, the hard-gate AOM-PLS regressor satisfies
> `RMSE_H(AOM-PLS) ≤ RMSE_H(PLS)` with equality when all non-identity
> operators are strictly dominated on `H`.*

The proof is trivial (argmin over a set containing the PLS coefficients).
The proposition is what reviewers will point to when asking "why should I
trust this not to overfit." It needs to be stated, not implied.

Separately, the *adjoint trick* deserves its own proposition showing that
the weight `w_b = A_b w_hat_b / ||A_b w_hat_b||` (with `w_hat_b` normalised
from `A_b^T c_k`) is proportional to the PLS weight on the lensed data
`X A_b^T`, up to the non-symmetry of `A_b`. This is standard material but
absent from the docstrings.

### 3.5 Reproducibility package

- Pin the operator bank contents with a version string (the current default
  bank has grown organically; reviewers will ask to reproduce *exactly*
  what you ran).
- Ship the benchmark script and the preprocessed public datasets.
- Report wall-clock with and without the Torch backend; the Torch backend
  needs its own small section showing that GPU batching does not change
  the operator selection distribution.

### 3.6 Comparisons with state-of-the-art chemometrics baselines

Must include, at minimum:

- **PLS with manually tuned SG + SNV/MSC** (the chemometrician's baseline).
- **VIP / iPLS / CARS-based variable selection + PLS.**
- **POP-PLS** (already in the same library; `operators/models/sklearn/pop_pls.py`).
  This is the closest methodological cousin and the most important
  head-to-head comparison.
- **Stacked / branched preprocessing via nirs4all's `branch`+`merge`.**
  This is the multi-preprocessing baseline AOM-PLS most directly competes
  with at the *library* level.

The comparison to POP-PLS is particularly important. POP-PLS selects a
*different* operator per component using PRESS (Allen 1974) on the whole
training set, whereas AOM-PLS selects *one* operator for all components using
a holdout. The two methods occupy different points on the
locality-vs-stability tradeoff and the paper should explicitly characterise
when one should be preferred over the other.

---

## 4. What Can Be Improved

The following are candidate improvements, roughly ordered by expected
impact-per-effort.

### 4.1 Replace the 20% internal holdout with PRESS

The PRESS criterion `PRESS = sum((y_i - y_hat_{i, -i})^2 / (1 - h_ii)^2)`
approximates LOO-CV without ever splitting the data. POP-PLS already uses
this path (`pop_pls.py`, the `auto_select=True` branch). Importing it into
AOM-PLS would:

- Remove the hardcoded seed and the small-`n` failure mode.
- Make operator and prefix selection *deterministic in data*, not in RNG.
- Shave the small but real overhead of running 1.25x more NIPALS (since
  today the operator is selected on 80% and then refit on 100%).

The small caveat is that `h_ii` requires the hat matrix for each prefix `k`;
for `K=15` and `B=60` this is 900 hat-matrix computations per fit.
Benchmarking is needed — PRESS might be slower in wall-clock even though it
is "free of holdout noise".

### 4.2 Operator-bank pruning

The default bank has ~60 operators, of which many are near-duplicates
(consecutive SG windows `{11, 15, 21, 31, 41}` × polynomials `{2, 3}` ×
derivative orders `{0, 1, 2}` × optional detrend composition). A principled
pruning based on *pairwise operator similarity on the training data* (say,
Procrustes distance between `A_b X` and `A_{b'} X`) would:

- Reduce variance in the selection signal.
- Make `Gamma` easier to interpret.
- Speed up the hard-gate scan.

A tempting design is a *training-time-adaptive* bank where redundant
operators are pruned after a cheap first-component screen. This would close
part of the gap to Bandit AOM-PLS without its rank-truncation pitfall.

### 4.3 Commit or remove the sparsemax gate

Today the sparsemax path exists, is not the default, and has not
demonstrated an advantage. The options are:

1. **Remove it** and cite DARTS / sparsemax in the paper as a path not
   taken, with the benchmark evidence.
2. **Commit to it** by (a) using the same full-NIPALS selection criterion
   instead of the first-component R² proxy, (b) adding temperature annealing
   (in the DARTS spirit), (c) making it the default when `B` is large.

The middle ground — shipping both — is the worst option for a publishable
artifact because it leaves two selection policies with different semantics
in the same codebase.

### 4.4 A PRESS-driven unification with POP-PLS, under the existing `AOMPLSRegressor` name

Both AOM-PLS and POP-PLS share all infrastructure (linear operator bank,
adjoint trick, NIPALS deflation). They differ only in the *granularity* of
the operator commitment: AOM is one-operator-per-model, POP is
one-operator-per-component. The unification (tracked as W5 in
`PUBLICATION_BACKLOG.md`) exposes this as a `selection` parameter:

- `AOMPLSRegressor(selection="global")` — classical AOM behaviour.
- `AOMPLSRegressor(selection="per_component")` — classical POP behaviour.
- `AOMPLSRegressor(selection="auto")` — data-driven pick, *conditional on
  empirical evidence that it works*.

The public class name stays as `AOMPLSRegressor`; `POPPLSRegressor` is
reduced to a convenience wrapper. The paper is titled and framed around
AOM-PLS, with the `selection` parameter presented as the axis of
generalisation. This is the stronger framing because (a) users do not have
to learn a new class name and (b) the paper's narrative — "PLS that picks
its own preprocessing" — is unchanged; we only add "and you can choose how
granular the picking is."

### 4.5 Non-linearity via a controlled fallback (SNV/MSC upstream), OSC inside the bank

**SNV and MSC are non-linear at apply time** (SNV divides by per-sample
`std(x)`; MSC divides by a per-sample OLS slope `a_i`). They cannot be
strict linear operators in the AOM-PLS bank. The pseudo-linear SNV
prototype (§2.2) was rejected for exactly this reason — the stop-gradient
approximation on `std(x)` introduces biased prediction-time behaviour.

**OSC is different.** DOSC (Direct OSC) stores a fixed projection matrix
`P_o` at fit time and applies `x → x (I − P_o P_o^T)` at predict time. That
is linear. The fit is supervised by `y`, so OSC is a *supervised linear
operator*. It can legitimately live in the bank, with the caveat that the
identity-dominance guarantee becomes conditional on the selection criterion
using data consistent with the operators (PRESS satisfies this; CV requires
refitting OSC per fold). W4 of the backlog quantifies whether OSC in the
bank measurably helps on the 60-dataset corpus; tentative recommendation
is yes.

**EPO** (External Parameter Orthogonalization) has the same structure as
OSC but uses an external reference block instead of y. It is also a
legitimate bank candidate on the subset of datasets where an EPO
reference block exists.

**For the genuinely non-linear SNV and MSC**, the honest recommendation is
the nirs4all composition pattern — either upstream placement or the
stacking pattern:

```python
pipeline = [
    {"branch": [
        [SNV(), AOMPLSRegressor()],          # non-linear fork
        [AOMPLSRegressor()],                  # linear fork (maybe with OSC in bank)
        [MSC(), AOMPLSRegressor()],           # non-linear fork
    ]},
    {"merge": "predictions"},
    Ridge(),
]
```

This is the pragmatic stacking baseline and it is already supported by the
library. It should be part of the paper's empirical comparisons and framed
as *complementary* to AOM-PLS, not competitive with it.

### 4.6 Multi-response handling

For `q > 1`, the cross-covariance `c_k` is collapsed to a vector via the top
left singular vector. This is reasonable for two or three correlated
responses but starts to bias operator selection when responses have
different preprocessing needs (e.g. predict moisture and protein
simultaneously). A per-response operator selection (or a block-SIMPLS
variant) would be a natural extension — not required for publication but
worth mentioning as future work.

### 4.7 Computational hygiene

- `frobenius_norm_sq` is computed by every operator at init and never
  consumed. Either wire it into the sparsemax score for scale invariance
  (`s_b <- s_b / nu_b`) or remove the attribute.
- `_nu` for composed operators is estimated with 50 Gaussian probes
  (`ComposedOperator._compute_nu_empirical`, `aom_pls.py:287`). This is
  non-deterministic across library imports; fix the RNG or compute
  analytically.
- The fallback path when `n_extracted == 0` (`aom_pls.py:1066`) searches
  linearly for the identity operator. Cache its index at bank init.

None of these affect results; they affect code hygiene and reviewer
perception.

### 4.8 Torch backend parity

The Torch backend (`operators/models/pytorch/aom_pls.py`) currently mirrors
the NumPy backend for hard and sparsemax gates. Before publication, a
parity test should show bitwise-equivalent operator selections on a fixed
seed for a standard dataset. This is a 50-line test that would catch the
most likely class of subtle bugs.

---

## 5. My Point of View

### 5.1 Is the contribution novel enough?

Yes, with the right framing. The three pieces that together constitute the
contribution are:

1. **Embedding preprocessing selection inside NIPALS via the adjoint trick.**
   This is the genuinely new piece. I am not aware of prior chemometrics
   work that treats preprocessing as a linear operator bank and uses the
   adjoint to update PLS weights in the same forward pass. Related work
   (multi-block PLS, on-the-fly variable selection) solves different
   problems.
2. **Identity-dominance as a deployment guarantee.** This is not novel
   *conceptually* — the "include the null option in your model class"
   pattern is ancient — but it is novel *in application* to preprocessing
   search, because the chemometrics grid-search workflow does not naturally
   produce this guarantee.
3. **A linear-operator-bank + single-training-run recipe that
   empirically matches or beats more sophisticated alternatives (MoE,
   DARTS, zero-shot routers) on standard NIRS datasets.** The paper's
   empirical argument is negative-result-driven: *you do not need
   differentiable architecture search or a mixture of experts to do this
   well*. This is a genuinely useful message to the field.

### 5.2 Is it mature?

Mostly. The code path is stable, interpretable, and has been used to train
hundreds of models in the nirs4all benchmark suite. The two remaining
maturity gaps are:

- The internal 20% holdout policy (§3.3, §4.1). This is a small but
  real source of small-sample variance that a reviewer will ask about.
  Replacing it with PRESS is the highest-leverage change.
- The sparsemax path's uncommitted status (§4.3). Ship it or cut it.

### 5.3 What is the one thing I would change before submitting?

Unify AOM-PLS and POP-PLS under the existing `AOMPLSRegressor` class with
a `selection ∈ {"global", "per_component", "auto"}` parameter and a shared
`selection_criterion` (winner of W1, most likely PRESS or a PRESS-CV
hybrid). Decision recorded in `PUBLICATION_BACKLOG.md` W5; the class name
stays as `AOMPLSRegressor`.

This:

- Strengthens the theoretical contribution (it becomes a family of
  selection policies, not a point method).
- Closes the one-vs-many-operators debate empirically and statistically,
  on the same datasets, with the same criterion. Without it the paper
  would have to awkwardly say "we recommend AOM for X, POP for Y" with
  no principled rule.
- Makes the software easier to maintain: one estimator, one set of
  tests, one docstring section.
- Keeps the public API user-friendly: existing `AOMPLSRegressor` code
  keeps working; new behaviour is unlocked by setting `selection` and
  `selection_criterion`.

The risk is scope creep on `selection="auto"`. The mitigation: ship
`"global"` and `"per_component"` under the unified path first (trivial
refactor), then bench `"auto"`. If `"auto"` does not clear the
acceptance bar, it is dropped from the shipped API and kept as a
"tried but did not generalise" note in the paper's discussion.

### 5.4 What is the paper's one-sentence pitch?

> *Adaptive Operator-Mixture PLS replaces the chemometrics workflow of
> exhaustive preprocessing grid-search with a single-training-run selection
> of preprocessing operators from a linear bank, using an adjoint-trick
> NIPALS update and an identity-dominance guarantee, and empirically
> matches or beats more elaborate Mixture-of-Experts and differentiable
> architecture search alternatives on standard NIRS datasets.*

That sentence would need to be tightened. The substance, I think, is
correct.

### 5.5 Risks the paper needs to name honestly

- **Non-linear preprocessing still matters on some datasets.** AOM-PLS is
  not a silver bullet; it is the right tool for the "well-behaved
  reflectance/absorbance spectrum" regime. Paper should say this and
  recommend stacking (§4.5) for the rest.
- **Selection on a 20% holdout can overfit on small-n datasets.** Paper
  should either use PRESS or report a sensitivity study.
- **The bank is curated, not learned.** Reviewers may push for a DARTS-
  style learned bank. The counter-argument is empirical (§2.5) and
  philosophical (interpretability of named operators matters in
  chemometrics), but both need to be stated, not assumed.

If those three caveats are in the discussion section, the paper is in good
shape.
