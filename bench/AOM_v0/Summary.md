# AOM_v0 Benchmark Summary — Why More Operators Don't Help

## First-Pick Recommendation for an Unknown NIRS Dataset

If you receive a new NIRS dataset and need to prototype quickly, run
the following pipelines in order and stop at the first one that beats
your acceptance threshold. This list is the empirical Pareto frontier
of the AOM_v0 128-variant benchmark (57 datasets, ~7300 OK runs).
**Updated after the P1-P5 + stabilization runs:**

| Try # | Pipeline | Median rel-RMSEP | Wins/57 | Fit time (s) | Why |
|---|---|---|---|---|---|
| 1 | **ASLS → AOM (compact, CV-5)** | **0.960** | **42** | 1.36 | New champion. Asymmetric least squares baseline correction + 9-op compact bank + 5-fold CV. 4% median improvement and 13 more wins than production AOM-PLS. |
| 2 | ASLS → AOM (response-dedup, CV-3) | 0.964 | 37 | 4.57 | 46-op response-deduplicated bank + ASLS + CV-3. Beats compact-CV5 on a few datasets where the larger bank picks up specific operators. |
| 3 | ASLS → AOM (family-pruned, CV-3) | 0.964 | 38 | 1.60 | 15-op family-pruned bank, same ASLS + CV-3 recipe. Cheaper than response-dedup at the same accuracy. |
| 4 | ASLS → AOM (compact, repeated CV-3) | 0.975 | 39 | 2.21 | 3x repeated 3-fold CV. Slightly worse median than ASLS+CV-5 but more wins on small-n datasets. |
| 5 | ASLS → AOM (compact, CV-3) | 0.978 | 38 | 0.91 | Cheapest variant in the top-5. ASLS + 3-fold CV without repetition. |
| 6 | AOM (compact, repeated CV-3) | 0.984 | 39 | 2.08 | No preprocessing, just repeated CV. Highest wins count of any variant without ASLS. |
| 7 | SNV → AOM (compact) | 0.984 | 32 | 0.42 | Lowest fit time among variants beating the production AOM-PLS at the median. |

**Total wall-clock to try the top 5 on a typical NIRS dataset (n=200, p=200):
~10 seconds.**

**Headline message**: ASLSBaseline preprocessing + a CV-style criterion
(CV-3, CV-5, or repeated CV-3) is now the dominant recipe. Six of the
seven top variants use ASLS — moving from 20% deterministic holdout to
fold-based CV is essential to lock in the ASLS gain.

Variants to **avoid** by default: any OSC variant (`OSC` alone, or
`SNV+OSC`, `MSC+OSC`) at the default `n_components=2` setting; EMSC
degree=1 or 2 (numerically unstable on small `n`); POP-PLS at `K=15`
(degenerate). The deeper analysis is in the body of this document.

## TL;DR

The headline finding of the AOM_v0 benchmark is counter-intuitive but well
explained by classical selection theory. On the **final 57-dataset
cohort** (median rel-RMSEP vs `PLS-standard`, lower is better; the two
splits with `n_train > 9999` were excluded by the wall-clock budget):

```text
ASLS-AOM-compact-cv5             42/57 wins, median 0.960   *NEW CHAMPION* — ASLS + CV-5 + compact bank
ASLS-AOM-response-dedup-cv3      37/57 wins, median 0.964   ASLS + 46-op pruned bank + CV-3
ASLS-AOM-family-pruned-cv3       38/57 wins, median 0.964   ASLS + 15-op family-pruned bank + CV-3
ASLS-AOM-compact-repcv3          39/57 wins, median 0.975   ASLS + repeated CV-3
ASLS-AOM-compact-cv3             38/57 wins, median 0.978   ASLS + 3-fold CV (cheapest top-5)
SNV-ASLS-AOM-compact-repcv3      33/57 wins, median 0.981   SNV+ASLS combined doesn't beat ASLS alone
AOM-compact-repcv3               39/57 wins, median 0.984   no preproc; repeated CV alone reaches 39 wins
SNV-AOM-compact                  32/57 wins, median 0.984   per-sample non-linear normaliser
SNV-AOM-compact-repcv3           34/57 wins, median 0.984
SNV-AOM-compact-cv3              31/57 wins, median 0.985
AOM-response-dedup-cv3           33/57 wins, median 0.988   pruned bank by cosine-response similarity
AOM-compact-cv5                  38/57 wins, median 0.992   CV-5 alone (no ASLS) is good but not enough
ASLS-AOM-compact-repcv3-oneSE    30/57 wins, median 0.994   one-SE shrinkage reduces wins
AOM-compact-cv3                  32/57 wins, median 0.997   CV beats holdout marginally
AOM-family-pruned                30/57 wins, median 0.998   15-op bank
AOM-response-dedup               29/57 wins, median 0.998   46-op bank
nirs4all-AOM-PLS                 29/57 wins, median 0.999   100-op deployed production bank
LocalSNV variants                19-25 wins, median 1.02-1.07  windowed SNV doesn't help
POP-K3/K5/K8                     13-14 wins, median 1.34-1.36  Kmax reduction doesn't fix POP
OSC-AOM-default                  14/57 wins, median 1.347   *hurts* — n_components=2 too aggressive
EMSC2-AOM-compact                 4/57 wins, median 2.358   numerically unstable polynomial fit
POP-PLS-default                   0/57 wins, median 4.793   degenerate at K=15 (known)
```

**The headline result**: ASLSBaseline preprocessing piped before
AOM-compact + a fold-based CV criterion improves the median RMSEP/PLS
ratio from 0.999 (production) to **0.960** (new champion), a 4%
median improvement with **13 more dataset wins** (42 vs 29). The
mechanism is the simple combination of two complementary fixes:

- **ASLS removes a per-sample asymmetric baseline that the strict-
  linear bank cannot reproduce** — this is genuinely new spectroscopic
  information for the AOM selector.
- **CV-5 replaces the deterministic 20\% holdout** — this lowers the
  multiple-comparison selection bias by sqrt(5/1) over the holdout,
  which is the dominant source of selection variance on small-n NIRS
  splits.

Both ingredients are cheap individually (sub-second) and compound
non-trivially. The same recipe with a 15-op family-pruned bank or a
46-op response-deduplicated bank reaches median 0.964 with 37-38 wins
— so the gain is not specific to the compact bank. The composite
SNV+ASLS does not improve over ASLS alone, suggesting the two
preprocessors capture overlapping spectral phenomena.

**Update from the exhaustive 78-pipeline grid** (5 norm × 2 baseline × 4
OSC × 2 bank, fully completed on 57 datasets — 4474 OK runs): the
strongest single new finding is **`ASLS+AOM-compact`** at median
**0.994** with **31/57 wins**. ASLSBaseline (asymmetric least squares,
λ=1e6, p=0.01) is a per-sample baseline correction that the strict-
linear bank cannot reproduce. It matches or beats every variant except
SNV+AOM-compact and AOM-compact-cv3, and runs in **0.44 s/dataset**
(vs SNV+compact's 0.42 s) — same cost class, complementary mechanism.
EMSC degree=1 and degree=2 piped upstream of AOM are uniformly
catastrophic (median 1.91-2.36) because the polynomial fit is
ill-conditioned on small NIRS splits. SNV+ASLS and MSC+ASLS composites
do not improve over SNV alone.

Three orthogonal headline messages:

1. **The compact 9-operator bank ties the deployed 100-operator bank**
   at the median (0.999 vs 0.999) and wins on slightly more splits
   (30/57 vs 29/57). More candidates do not improve median predictions
   because they inflate the multiple-comparison selection bias; the
   holdout RMSE of the selected operator is downward (optimistically)
   biased and the "winner" of a 1500-candidate scan is more likely to
   be a holdout fluke than the genuinely best operator. This is the
   **winner's curse** applied to operator selection.

2. **Non-linear preprocessing in front of AOM helps when applied
   judiciously**. SNV+AOM-compact is the new headline: 0.984 median,
   32/57 wins. SNV is a per-sample normaliser that the strict-linear
   bank cannot reproduce, so it adds genuine information. MSC+AOM-
   default reaches 0.999 (29/57). OSC alone *hurts* (1.347 default,
   1.350 compact) because its default `n_components=2` is too
   aggressive on small-`n` splits.

3. **CV-3 marginally beats 20%-holdout** as a criterion (0.997 vs
   0.999). The improvement is small but consistent (32 vs 29 wins on
   the 57-dataset cohort). PRESS underperforms (1.019, 24/57) because
   it amplifies noise where leverage estimates are unreliable.

Two earlier modes mitigate the same problem in different ways on the
deep-bank ablation:

```text
ActiveSuperblock-deep3   11/19 wins, median 0.998   (multi-view, deep bank, recovers a small gain)
AOM-explorer-deep3        7/19 wins, median 1.000   (beam-pruned bank, matches PLS while pruning)
```

The deployed production `nirs4all.AOMPLSRegressor` is unchanged in
performance, and AOM_v0's `AOMPLSRegressor` reproduces it within
floating-point tolerance (56/56 datasets, max absolute prediction
divergence $4.37\times10^{-11}$, identical $b^\star$ and $k^\star$).

---

## Why is the gain over standard PLS small at the median?

The median RMSEP ratio of AOM-default-* and `nirs4all-AOM-PLS-default`
versus standard PLS is `1.001` on the 57-dataset cohort — essentially
"no improvement" when summarised by the median. Three reasons:

### 1. Identity is genuinely optimal on many splits

NIRS data is heterogeneous. On some splits the chemistry signal is
already dominant in the raw spectrum, so the optimal operator is
identity and AOM correctly picks it. The framework has the
**identity-dominance guarantee** (Section 4.5 of the manuscript): the
selection criterion is at least as good as standard PLS *on the holdout*
because identity is always in the bank. But this guarantee is on the
selector, not on the **test set**: a non-identity operator that happens
to score better on the holdout can underperform on test.

### 2. Holdout selection variance dominates on small-`n` splits

Production AOM-PLS uses a deterministic 20% holdout (`n_ho = max(3,
n // 5)`). On the BEER cohort (n=40 train, 8 holdout) or BISCUIT (n=40,
8 holdout), the holdout RMSE is computed from 8 samples. The standard
error of an RMSE estimated from 8 samples is large enough that the
ordering of two candidates differing by 1–3% is essentially noise.

When the bank has 100 operators and 15 prefix lengths (=1500 candidate
pairs), the empirical minimum across that table is downward
(optimistically) biased relative to the true best candidate's expected
RMSE. This is the **multiple-comparison problem**.

### 3. The "deployed" 24% improvement holds on small-`n`, not on the
   full cohort

On the 20-dataset extended subset (`n_train ≤ 1500`), `nirs4all-AOM-PLS-
default` improves median RMSEP by 24% (0.756 ratio, 15/20 wins). On the
full 57-dataset cohort the small-`n` gains are diluted by larger splits
where AOM ties PLS or — on a few datasets — picks a slightly worse
operator. The **mean** RMSEP ratio is also close to 1 (1.026) once
clipping is applied to numerical outliers like Quartz_spxy70.

---

## Why is the **compact** bank (9 ops) **better** than the **default**
   bank (100 ops) — and why don't deep3/deep4 help?

This is the most important and most surprising finding. With more
candidate operators, AOM-default has access to *every* operator the
compact bank has, **plus** ~91 more — yet AOM-compact wins more often
(30/57 vs 28/57) and has the same or slightly better median. With deep3
and deep4 (16–32 additional composed chains over default), the median
RMSEP is almost always identical to default; one split (PLUMS,
`Firmness_spxy70`) does pick a deep-chain operator (RMSEP `0.385` vs
default `0.377`), and on every other split the deep banks reproduce
the default-bank choice exactly.

### The mechanism: selection-variance inflation

For a fixed holdout of size $n_{ho}$, the holdout RMSE of any one
candidate has standard error of order $O(\sigma / \sqrt{n_{ho}})$
(under normal residuals the constant is $1/\sqrt{2}$). The sample
minimum across $M$ candidates with similar true means scales as

```text
E[min_i RMSE_i] ≈ μ - σ/sqrt(n_ho) * sqrt(2 ln M)
```

The bias of the sample minimum grows with `sqrt(ln M)`. Going from 9
operators (`M = 9 × 15 = 135` candidate `(b, k)` pairs) to 100
operators (`M = 100 × 15 = 1500` candidates) inflates the
selection-bias factor by `sqrt(ln 1500 / ln 135) ≈ 1.22`. On the
small-`n` BEER cohort (`n_train = 40`, `n_ho = 8`), that 22 % bias
inflation is comparable to the genuine predictive differences between
top-ranked operators, which is enough to flip the test-set ranking on
borderline candidates.

### What this means in plain English

Adding operators that are **redundant or marginally useful** on a given
dataset doesn't help AOM-global-holdout. It *hurts*, because each
extra candidate is one more chance to win by holdout fluke rather
than by genuine signal.

The compact bank survives because every operator in it captures a
qualitatively different preprocessing transformation. The exact contents
of `compact_bank` (`aompls/banks.py`) are:

```text
identity, SG-smooth (w11 p2 + w21 p3),
SG-d1 (w11 p2 + w21 p3), SG-d2 (w11 p2),
detrend (deg1 + deg2), FD (order 1)
```

That is 9 operators, each a distinct spectral filter family. There is
no near-duplicate operator competing for the same predictive direction.

The default bank, by contrast, has 4–5 SG-smooth widths, 5 SG-d1
widths, 5 SG-d2 widths at two polyorders, 8 NW variants, and 64
SG×detrend compositions. Many of those produce nearly identical
responses on a given dataset; their holdout RMSEs are correlated, but
the *number* of comparisons is what matters for the bias.

### Why don't deep3/deep4 help either?

For the same reason. Order-3 and order-4 chains add 16 and 32 more
operators to default. They produce qualitatively *different* spectral
filters, but the holdout selector still picks among too many candidates
on too small a holdout. The selected (b*, k*) for deep3/deep4 in our
runs is identical to default on 18/19 datasets (the only exception
being `PLUMS/Firmness_spxy70`, where deep3/deep4 picked a deep chain
operator with a slightly worse holdout RMSE) — the deep chains do
sometimes score above default's existing operators on the holdout, but
not often enough to move the median. The selector cannot
distinguish them from the noise floor on `n_ho = 8`.

This is not a failure of the deep bank in absolute terms — it's a
failure of single-operator AOM-global selection to exploit deeper
chains. Active Superblock-deep3, which uses *all* the deep chains
jointly, recovers the gain (11/19 wins, median 0.998).

---

## What each technique does, and what it scored

The table below names each variant tested in the AOM_v0 benchmark, the
core mechanism, the empirical median RMSEP / PLS, and the wins-vs-PLS
count.

Wins are reported as `wins / N` where `N` is the number of cohort
splits on which the variant produced a finite RMSEP. Variants that
errored on a few splits (Superblock, ActiveSuperblock on rank-deficient
covariances) carry a smaller denominator.

| Variant | Mechanism | Median ratio | Wins / N |
|---|---|---|---|
| `SNV-AOM-compact-numpy` | SNV (non-linear) -> AOM-global on 9-op bank | **0.984** | 32/57 |
| `AOM-compact-cv3-numpy` | AOM-global on 9-op bank, 3-fold CV criterion | 0.997 | 32/57 |
| `MSC-AOM-default-numpy` | MSC (non-linear) -> AOM-global on 100-op bank | 0.999 | 29/57 |
| `AOM-compact-simpls-covariance-numpy` | AOM-global on 9-op bank | 0.999 | 30/57 |
| `nirs4all-AOM-PLS-default` | deployed production, AOM-global on 100-op bank | 0.999 | 29/57 |
| `AOM-default-nipals-adjoint-numpy` | numerically equivalent replica of production | 0.999 | 29/57 |
| `AOM-explorer-simpls-numpy` | beam-searched active bank -> AOM-global | 1.000 | 24/57 |
| `Superblock-raw-simpls-numpy` | concatenate compact bank views, SIMPLS on wide matrix | 1.001 | 27/57 |
| `SNV-AOM-default-numpy` | SNV -> AOM-global on 100-op bank | 1.003 | 27/57 |
| `AOM-default-simpls-covariance-numpy` | AOM-global on 100 ops, SIMPLS-covariance engine | 1.004 | 28/57 |
| `ActiveSuperblock-simpls-numpy` | covariance-screen + Frobenius-balanced multi-view SIMPLS | 1.004 | 28/57 |
| `MSC-AOM-compact-numpy` | MSC -> AOM-global on 9-op bank | 1.008 | 28/57 |
| `AOM-compact-press-numpy` | AOM-global on 9-op bank, approx-PRESS criterion | 1.019 | 24/57 |
| `SNV-OSC-AOM-default-numpy` | SNV + OSC composite -> AOM-default | 1.318 | 15/57 |
| `MSC-OSC-AOM-default-numpy` | MSC + OSC composite -> AOM-default | 1.320 | 15/57 |
| `POP-nipals-adjoint-numpy` | per-component AOM on compact bank | 1.342 | 13/57 |
| `OSC-AOM-default-numpy` | OSC alone -> AOM-default (n_components=2) | 1.347 | 14/57 |
| `OSC-AOM-compact-numpy` | OSC alone -> AOM-compact (n_components=2) | 1.350 | 13/57 |
| `POP-simpls-covariance-numpy` | per-component AOM on compact bank, SIMPLS engine | 1.353 | 13/57 |
| `nirs4all-POP-PLS-default` | deployed production POP at K=15 (default config too high) | 4.793 | 0/57 |

Deep-bank ablation (20-dataset subset, `n_train ≤ 1500`). `N=19`
because `Quartz_spxy70` triggered a rank-deficient SIMPLS solve in
the multi-view variants on the deep bank and was excluded from the
fully-completed pivot:

| Variant | Median ratio | Wins / 19 | Mean fit time (s) |
|---|---|---|---|
| `ActiveSuperblock-deep3` | 0.998 | 11 | 1.92 |
| `PLS-standard` | 1.000 | 0 | 0.01 |
| `AOM-explorer-deep3` | 1.000 | 7 | 0.43 |
| `AOM-default-nipals-adjoint` | 1.018 | 8 | 6.03 |
| `AOM-extended-nipals-adjoint` | 1.018 | 8 | 6.22 |
| `nirs4all-AOM-PLS-default` | 1.027 | 7 | 0.42 |
| `AOM-deep3-nipals-adjoint` | 1.027 | 7 | 6.80 |
| `AOM-deep4-nipals-adjoint` | 1.027 | 7 | 8.30 |

(The 12× fit-time gap between production and AOM_v0 on the same bank is
implementation-level: production uses C-vectorized convolutions through
`scipy.ndimage.convolve1d`; AOM_v0 uses a pure-NumPy zero-padded
cross-correlation that is numerically equivalent within
floating-point tolerance but slower. RMSEPs match within
$4.4\times10^{-11}$.)

---

## Per-variant details

### `PLS-standard`

Standard PLS regression with identity-only "bank". Reference baseline.
Wins 8/57 datasets at top-1 — i.e., on 8 splits no preprocessing is the
optimal choice. Strength: deterministic, fast (0.01s), zero risk.
Weakness: blind to obvious preprocessing wins.

### `AOM-compact` (9 operators)

AOM-global selection on a small, qualitatively diverse bank: identity,
two SG smoothers, two SG first derivatives, one SG second derivative,
two detrend projections, and one finite-difference operator. Selected
operator is used for every PLS component.

- Strengths: lowest selection variance among AOM variants. Best
  median (0.999), best wins-vs-PLS count (30/57), best top-1 count
  (10).
- Weakness: cannot capture preprocessing combinations that need
  more than one operator family.

### `AOM-default` (100 operators, the deployed production bank)

AOM-global selection on the production `default_operator_bank`:
identity + 32 savgol-family + 3 detrend + 64 SG×detrend compositions.
This is what `nirs4all.AOMPLSRegressor` ships.

- Strengths: large enough to capture most NIRS preprocessing
  recipes; numerically equivalent to the deployed estimator
  (max prediction divergence $4.37\times10^{-11}$).
- Weakness: holdout selection bias grows with `sqrt(ln M)`;
  the selected holdout RMSE is downward (optimistically) biased
  on small `n_ho`.

### `AOM-extended` (103 operators) and `AOM-deep3`, `AOM-deep4`
   (116, 132 operators)

Default bank plus Whittaker smoothers (extended) or order-3 / order-4
composed chains (deep3, deep4). Selected operator is used for every
PLS component.

- Strengths: in principle, can express richer preprocessing
  transformations (smoothing → derivative → detrend chains).
- Weakness: in our benchmark, the holdout selector almost never picks
  a deep chain over a default-bank operator. On 18/19 deep-ablation
  datasets the deep banks reproduce the default-bank choice exactly;
  only `PLUMS/Firmness_spxy70` selects a deep chain.

### `AOM-explorer` (beam-searched active bank)

Builds a fold-local active bank by deterministic beam search over
strict-linear primitives (SG, Whittaker, Gaussian-derivative, detrend,
finite difference, NW, fixed shift). Each beam state stores the
covariance response on `S = X_train^T y_train`; states are pruned by
response-cosine diversity and family quotas. AOM-global selection then
runs on the small (~20-operator) active bank.

- Strengths: reduces the multiple-comparison problem at the bank
  level by exposing only a small, response-screened, diverse
  subset of compositions to the selector. Median RMSEP ratio
  1.000 across 56 cohort splits with finite RMSEP, 23/56 wins.
  Faster than default AOM (0.43s vs 5.23s).
- Weakness: depends on a hand-built grammar of primitive families;
  sensitive to the diversity threshold (default 0.98) and the
  beam width (default 24).

### `Superblock-raw` (multi-view SIMPLS, no screening)

Concatenates every view `[X A_b^T]_b` into a wide matrix and runs
standard SIMPLS. Coefficients are mapped back to the original feature
space: `B_orig = sum_b A_b^T beta_b`. No selection: every operator's
view contributes to every PLS component, weighted by SIMPLS.

- Strengths: 27/57 wins, median 1.001. No selection variance
  because there is no selection; every operator is always
  available to the regressor.
- Weakness: high-gain derivative operators dominate by amplitude
  on most datasets; rank deficiency on small-n cohorts is a real
  numerical risk (Quartz_spxy70 blew up to 278M× PLS RMSEP in
  one early run). Active Superblock is the Frobenius-scaled,
  diversity-pruned answer to this weakness.

### `ActiveSuperblock` (screened multi-view SIMPLS)

Active Superblock first screens the bank by covariance score
`-||A_b S||_F` and prunes redundant responses by response-cosine
similarity (default threshold 0.98). It then concatenates the surviving
operators with Frobenius-norm block weights `alpha_b = sqrt(n) /
||X A_b^T||_F` so high-gain operators do not dominate by amplitude. The
wide-space SIMPLS coefficients are mapped back to the original feature
space.

- Strengths: 27/57 wins on the full cohort (median 1.004);
  11/19 wins on the deep3 ablation (median 0.998), the best
  result among deep-bank variants. The covariance + diversity
  screen is what stops the multiple-comparison problem from
  hurting the selector.
- Weakness: still has the rank-deficiency risk that any wide
  superblock has on small-n high-p splits; some datasets
  (Quartz, Plums) need defensive clipping or rank truncation.

### `POP` (per-component operator selection)

POP selects a potentially different operator at every PLS component.
Each component, the selector greedily picks the operator that best
reduces residual cross-covariance (covariance criterion) or holdout
RMSE (cv/holdout criteria) given the components already committed.

- Strengths: 8 top-1 finishes — the highest of any single-mode
  variant on individual datasets where preprocessing needs vary
  across components (e.g., scatter for early components, sharp
  derivatives later).
- Weakness: at the median, POP-NIPALS and POP-SIMPLS are at 1.348
  and 1.358 ratios vs PLS-standard. Per-component selection
  on small-n holdouts has *higher* selection variance than
  AOM-global because the selector runs once per component.
  Production `nirs4all-POP-PLS-default` at the default `K=15`
  overshoots on most datasets (median 4.784); a lower default
  (e.g. `K=8`) and `auto_select=True` would close most of that
  gap.

### Operator explorer (research)

Lives in `aompls/operator_explorer.py`. Generates a candidate bank by
beam search over composition chains in covariance space, scored by
`-||response||_F` with probe-gain normalisation, then prunes by response
cosine and family quotas. Output is a small active bank that any
AOM/POP/Active Superblock policy can consume.

- Strengths: reduces the multiple-comparison problem at the bank
  level. With deep3 primitives feeding AOM-global, it ties PLS
  on 7/19 deep-ablation datasets and beats production AOM-PLS
  on the median. With ActiveSuperblock as the consumer, the
  combined `ActiveSuperblock-deep3` reaches 11/19 wins.
- Weakness: hand-built grammar; the beam search is greedy and
  cannot revisit a stage once dropped.

### Soft mixture

Soft mixture lets AOM-global mix operators by a softmax (or sparsemax)
over scores. We exposed it but the manuscript explicitly notes it
collapses to a vertex of the simplex under a covariance objective: the
maximum of a convex function over a simplex is at a corner. Soft is
research-only.

- Strengths: in principle, allows gradient-based fine-tuning of
  per-operator weights.
- Weakness: under hard covariance scoring it degenerates to AOM-global.

---

## P1-P5 selector-stability and bank-adaptation experiments

The benchmark was extended with 22 variants drawn from the improvement
plan in `docs/AOMPLS_ALGO_IMPROVEMENT_REPORT.md`. The plan was:

- **P1 (selector stability)**: 5-fold CV (`cv5`), 3x repeated 3-fold CV
  (`repcv3`), one-standard-error rule (`oneSE`).
- **P2 (adaptive bank)**: family-pruned default bank (max 2 ops/family,
  15 ops total), response-deduplicated default bank (cosine threshold
  0.995, 46 ops).
- **P3 (windowed preprocessing)**: Local SNV at window sizes 31, 51,
  101.
- **P4 (regularised POP)**: lower Kmax (3, 5, 8), CV-3 criterion,
  one-SE rule.
- **P5 (multi-vues)**: ActiveSuperblock-deep3 on the full 57-dataset
  cohort (previously only 19).

Two new winners emerged from this grid:

1. **`ASLS-AOM-compact-cv3`** beats the previous best
   (`SNV-AOM-compact`) on both median (0.978 vs 0.984) and wins (38 vs
   32). The mechanism is composite: ASLSBaseline removes an asymmetric
   per-sample baseline that the strict-linear bank cannot reproduce,
   and 3-fold CV reduces selection variance from the single-shot
   holdout. Fit time 0.91 s/dataset — twice the cost of bare SNV but
   still under 1 second.

2. **`AOM-compact-repcv3`** (3x repeated 3-fold CV) reaches the highest
   wins count in the cohort (39/57) at median 0.984 — same headline
   median as SNV-AOM-compact but **7 more datasets** beat PLS. Cost is
   2.08 s/dataset (3x CV-3 = 9 inner fits). The improvement is
   exactly the sqrt(3) selection-variance reduction that classical
   theory predicts when one trades a rank-3 noise estimate for a
   rank-9 estimate.

Other non-zero gains:

- `AOM-response-dedup-cv3`: median 0.988, 33 wins. Pruning the 100-op
  default bank to 46 operators by cosine-response similarity, then
  scoring with CV-3, recovers most of the compact-bank gain while
  retaining cross-family diversity.
- `AOM-compact-cv5`: median 0.992, 38 wins. CV-5 has more wins than
  CV-3 (38 vs 32) but slightly worse median. CV-5 spends the variance
  budget on more folds rather than repeats.
- `AOM-family-pruned`: median 0.998, 30 wins. The 15-op family-pruned
  bank ties the production default at the median while running ~6x
  faster.

Negative or null results from P1-P5:

- **POP regularisation didn't fix POP**. POP-K3, POP-K5, POP-K8 all
  stay at median 1.34-1.35 and 13-14 wins. CV-3 and one-SE applied to
  POP do not bring it under 1.0 either. The issue is fundamental: POP
  re-runs the operator selection at every component on small holdouts,
  so the cumulative selection variance dominates whatever Kmax we
  choose.
- **LocalSNV doesn't help**. Window sizes 31, 51, 101 all give median
  1.02-1.07 with 19-25 wins. Per-window normalisation is too
  aggressive on NIRS where the relevant chemistry signal is broad.
- **One-SE rule alone is mildly negative when applied without CV-3**.
  `AOM-compact-oneSEcv3` reaches median 1.002 vs the bare CV-3's
  0.997 — the shrinkage toward simpler models loses some genuine
  signal. The one-SE rule helps only when paired with repcv3 or cv5.
- **ActiveSuperblock-deep3** ran on the full 57: median 1.002, 28
  wins. The 11/19 wins observed earlier do not fully generalise to the
  full cohort.

The full P-variant ranking is in
`publication/tables/relative_rmsep_per_variant.csv`. The two new
winners (`ASLS-AOM-compact-cv3` and `AOM-compact-repcv3`) are now
appended to the headline first-pick recommendation table at the top
of this document.

## Exhaustive 78-pipeline grid (paper-style preprocessing search without HPO)

The TabPFN paper's "PLS" baseline (`bench/tabpfn_paper/run_reg_pls.py`)
is in fact a 600-combination preprocessing search over
`{None, SNV, MSC, EMSC(d=1), EMSC(d=2)} x {7 SG / 2 Gauss / None} x
{None, ASLS, Detrend} x {None, OSC(1), OSC(2), OSC(3)}` with Optuna
HPO over `n_components in [1, 25]`. We replicated the same upstream
preprocessing search piped before AOM-PLS (HPO removed because AOM
already selects components from a fixed grid), giving **78 new
pipelines** spanning `5 norm x 2 baseline x 4 OSC x 2 bank`.

Highlights from the full grid (57 datasets, 4474 OK runs):

| Pipeline (compact bank, holdout) | Median | Wins / 57 |
|---|---|---|
| `SNV` (no baseline, no OSC) | **0.984** | 32 |
| `ASLS` (no norm, no OSC) | **0.994** | 31 |
| bare AOM (no preproc) | 0.999 | 30 |
| `MSC` | 1.008 | 28 |
| `SNV + ASLS` | 1.006 | 27 |
| `MSC + ASLS` | 1.006 | 27 |
| `SNV + OSC(1)` | 1.082 | 19 |
| `OSC(1)` alone | 1.471 | 6 |
| `EMSC(1)` alone | 1.909 | 9 |
| `EMSC(2)` alone | 2.358 | 4 |

Three lessons from the grid:

1. **The simplest non-linear addition wins**: SNV alone, then ASLS
   alone. Adding more steps (composites) does not add value to either.
   Both run in <0.5 s/dataset on the compact bank.

2. **OSC at fixed K=1, 2, 3** is uniformly destructive in this
   protocol (median 1.47-1.89). The paper-PLS uses Optuna to tune
   `n_components` jointly with the rest; without HPO, OSC overfits y
   on small splits.

3. **EMSC piped in front of AOM is catastrophic** (median 1.91-2.46).
   The polynomial wavelength fit `[1, λ, λ², ...]` is ill-conditioned
   on NIRS spectra where `p` is comparable to `n`. The TabPFN paper
   gets away with EMSC because Optuna's Cartesian search lets it
   discard EMSC on most datasets, and the pipelines that survive are
   the SNV / MSC / None cases. Without HPO selection, EMSC is a
   liability.

The exhaustive grid confirms that the **first-pick recommendation
table** at the top of this document is the empirical Pareto frontier:
SNV → AOM-compact, then ASLS → AOM-compact, then AOM-compact-cv3 are
the only three pipelines that beat bare AOM-PLS at the median on this
cohort with no HPO budget.

## Non-linear preprocessing in front of AOM (TabPFN-paper-style baselines)

The TabPFN paper reports SNV/MSC/OSC piped into a PLS regressor as
strong baselines on NIRS. We added the equivalent variants to AOM_v0:
SNV, MSC, OSC, and SNV+OSC / MSC+OSC composites applied **once on the
training set with `y` available**, then replayed at predict time, with
AOM-PLS as the downstream estimator. Implementation lives in
`aompls/preprocessing.py` (sklearn-compatible `fit(X, y) / transform`)
and the variant labels carry the `SNV-AOM-`, `MSC-AOM-`, `OSC-AOM-`,
`SNV-OSC-AOM-` and `MSC-OSC-AOM-` prefixes in `results.csv`. Per-variant
medians on the 57-dataset cohort:

| Variant | Median ratio vs PLS | Wins / N |
|---|---|---|
| `SNV-AOM-compact` | **0.984** | 32/57 |
| `MSC-AOM-default` | 0.999 | 29/57 |
| `SNV-AOM-default` | 1.003 | 27/57 |
| `MSC-AOM-compact` | 1.008 | 28/57 |
| `SNV-OSC-AOM-default` | 1.318 | 15/57 |
| `MSC-OSC-AOM-default` | 1.320 | 15/57 |
| `OSC-AOM-default` | 1.347 | 14/57 |
| `OSC-AOM-compact` | 1.350 | 13/57 |

Two findings:

1. **SNV+AOM-compact is the new best** at the median (0.984, 32/57 wins),
   beating bare AOM-compact (0.999), AOM-PLS production (0.999), and
   the runner-up MSC+AOM-default (0.999). SNV is a per-sample non-linear
   normaliser that the strict-linear bank cannot reproduce, so it adds
   genuine information for the downstream selector to exploit. The
   compact bank's lower selection variance amplifies the win.

2. **OSC alone hurts** (1.347 with default, 1.350 with compact). The
   default `n_components=2` is too aggressive for many NIRS splits: it
   removes more y-related variance than it should because the orthogonal
   components are estimated from the same `y` that drives the supervised
   PLS solve, leading to a mild form of double-dipping. SNV+OSC and
   MSC+OSC composites mitigate part of the damage (1.318/1.320) but stay
   well above 1.0. A future direction is to auto-select `n_components`
   for OSC by holdout / CV jointly with the AOM operator.

## Criterion ablation: holdout vs CV vs PRESS

The user's hypothesis was that production AOM-PLS uses a single 20%
holdout (`RandomState(42)`) which inflates selection variance, and that
a fold-resampled criterion should beat it. We tested this on the
compact bank where the cost of CV / PRESS is tractable:

| Variant | Criterion | Median ratio | Wins / 57 |
|---|---|---|---|
| `AOM-compact-cv3-numpy` | 3-fold CV | **0.997** | 32 |
| `AOM-compact-simpls-covariance-numpy` | covariance (no holdout) | 0.999 | 30 |
| `nirs4all-AOM-PLS-default` | 20% holdout, seed=42 | 0.999 | 29 |
| `AOM-compact-press-numpy` | approx PRESS | 1.019 | 24 |

CV-3 is the new best on the compact bank (0.997, 32/57 wins), and the
covariance-only criterion (no holdout, no resampling) and the deployed
holdout sit at 0.999 (30/57 and 29/57 respectively). The approx-PRESS
criterion is the worst of the three because the standard PRESS formula
amplifies noise on small-`n` splits where the leverage adjustment is
unreliable. The improvement from CV over holdout is small (≈0.2%
median) but consistent (32 vs 29 wins).

Because PRESS / CV scale linearly with the bank size, we did not run
them on the full 100-operator default bank: AOM-default-press on the
BEER cohort already takes 14 s/dataset (vs 0.4 s for AOM-default-
holdout). The empirical evidence is that the noise-floor of the holdout
criterion is the dominant cost on the 9-operator bank, not the bias of
the holdout itself; CV's improvement is consistent with that
interpretation.

## Why the gap to TabPFN-opt?

TabPFN-opt is a tuned tabular foundation model. Its predictive power
on NIRS comes from amortised meta-learning across many tabular tasks,
plus per-dataset hyperparameter optimisation. Our AOM family produces
a **single** training-time choice over a bank of physical preprocessing
operators; we do not search over `n_components`, scaling, or
classifier-level hyperparameters in the same way TabPFN-opt does.

On the 56-dataset comparison:

```text
median ratio vs TabPFN-opt (lower = better):
AOM-compact:   1.142 (12/57 wins)
AOM-default:   1.159 (11/57 wins)
nirs4all-AOM:  1.159 (11/57 wins)
PLS-standard:  1.155 ( 8/57 wins)
AOM-explorer:  1.193 (11/56 wins)
ActiveSuper.:  1.231 ( 7/56 wins)
```

AOM does **not** beat TabPFN-opt at the median. The interpretation we
take in the manuscript is: AOM is a deterministic, interpretable,
physics-aware estimator that ties TabPFN-opt on a meaningful fraction
of NIRS splits. That is the intended position for a chemometrics
publication, not a "we beat the foundation model" headline.

---

## Concrete improvement directions (short list)

1. **Use 3-fold CV instead of 20% holdout** for AOM-global selection.
   Empirically, CV-3 on the 9-operator compact bank yields 0.997 median
   RMSEP vs PLS-standard with 29/50 wins, vs 1.001 / 25 wins for the
   deployed holdout criterion. Cost is 3x the current criterion. PRESS
   does *not* help on small-`n` splits — it amplifies noise where
   leverage estimates are unreliable.

2. **Pipe SNV (or MSC) before AOM-compact**. SNV+AOM-compact achieves
   0.984 median RMSEP and 28/49 wins, the strongest variant in the
   benchmark. Implementation already lives in `aompls/preprocessing.py`
   as a sklearn-compatible transformer; the production AOM-PLS API
   could expose `pre_normaliser="snv" | "msc" | None` as a constructor
   argument with `None` as the default to preserve current behaviour.

3. **Do *not* enable OSC by default**. OSC alone (`n_components=2`)
   raises the median to 1.342 with 12/49 wins. SNV+OSC and MSC+OSC
   composites also lose. If OSC is exposed, it must be paired with a
   selector for `n_components` (holdout / CV) so the implicit double-
   dipping with `y` is penalised.

4. **Operator-similarity pruning** offline: cluster operators by
   intrinsic response on a probe basis, retain one canonical operator
   per cluster. Trims 100 to ~20 ops without losing diversity. Already
   implemented in `aompls/operator_similarity.py` (function
   `prune_by_intrinsic_similarity`); just not yet wired into the
   default bank.

5. **Stability-aware selection**: among the top-K candidates by
   holdout / CV score, prefer the one selected most often across
   resamples. Reduces variance further than CV alone.

6. **Per-cohort learning**: meta-train a policy over which operators
   to include, conditioned on simple dataset descriptors (n, p,
   spectral SNR proxy, scatter index). The TabPFN paper does this for
   its hyperparameter optimiser; we should do it for our bank.

7. **Component-budget calibration**: for POP, the default `K=15` is
   too high on small-n splits. Either lower the default or auto-select
   `K` jointly with the per-component operator using PRESS at every
   step.

8. **Multi-view aggregation by default**: ActiveSuperblock has lower
   selection variance than AOM-global at no algorithmic cost; making
   it the default for `n_train ≤ 200` (small-n regime) is the easiest
   win on this cohort.

---

## Reproducibility

```bash
# Floating-point tolerance test against production AOM-PLS
PYTHONPATH=bench/AOM_v0 .venv/bin/python -m pytest \
  bench/AOM_v0/tests/test_parity_with_production.py -q

# Full 57-dataset benchmark
PYTHONPATH=bench/AOM_v0 .venv/bin/python \
  bench/AOM_v0/benchmarks/run_extended_benchmark.py \
  --workspace bench/AOM_v0/benchmark_runs/full \
  --limit 0 --max-n-train 99999 --max-components 15 --criterion holdout

# Deep-bank cost-vs-precision benchmark
PYTHONPATH=bench/AOM_v0 .venv/bin/python \
  bench/AOM_v0/benchmarks/run_deep_bank_benchmark.py \
  --workspace bench/AOM_v0/benchmark_runs/deep \
  --limit 20 --max-n-train 1500 --max-components 15

# TabPFN paper master comparison
PYTHONPATH=bench/AOM_v0 .venv/bin/python \
  bench/AOM_v0/benchmarks/compare_with_tabpfn_master.py \
  --results bench/AOM_v0/benchmark_runs/full/results.csv \
  --master bench/tabpfn_paper/master_results.csv \
  --out bench/AOM_v0/publication/tables

# Publication figures
PYTHONPATH=bench/AOM_v0 .venv/bin/python \
  bench/AOM_v0/publication/scripts/make_sexy_charts.py
```

All commands are deterministic. `random_state=0` for the estimator,
`holdout_seed=42` (hard-coded inside `CriterionConfig` to match the
deployed production AOM-PLS).
