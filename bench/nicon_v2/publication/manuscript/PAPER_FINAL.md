# nicon_v2: Where 1-D CNNs Help (and Don't) for Small-n Near-Infrared Regression

**Status:** _final draft, post round-16 multi-seed validation._
**Cohort runs:** `r14_multiseed/` (representative, 5 seeds, 200 paired obs);
`r16_lucas_multiseed/` (250 paired obs, with LUCAS pretrain ablation);
`publication_curated/` (39 datasets × 1 seed, headline numbers).

## Abstract

Convolutional neural networks (CNNs) for near-infrared (NIR) spectroscopy
regression are widely reported to underperform partial least squares (PLS)
regression on small training cohorts (`n_train < 1000`). We dissect this
ceiling through 16 rounds of architecture-and-training experiments on a
10-dataset representative cohort, ending with a comprehensive
multi-seed-validated null result: **no CNN configuration we tested
significantly beats Ridge regression on the cohort**.

We propose **`V2L-learnableRMS`**, a CNN architecture that *does* close
65–75 % of the Ridge gap on the two largest plant-chemistry datasets
in the cohort (Chla+b_block2deg with `n_train=2925` and Chla+b_species
with `n_train=3734`), but **statistically ties Ridge** elsewhere. Two
representative deeper experiments — **V6b knowledge distillation** from
an extended-bank AOM-PLS teacher, and **LUCAS-SOC backbone pretraining**
with V6b distillation as a domain-mismatch safety net — both fail to
clear a 50 % win rate against Ridge across 5 seeds × 10 datasets
(50 paired observations: V6b-V2M vs V2L median Δ% = −0.33 %, Wilcoxon
p = 0.76; V6b-LUCAS-V2M vs V2L median Δ% = −0.21 %, p = 0.76).

We give an honest cohort-level scorecard, document the failure modes
that single-seed early-iteration results made look promising, and
release the full implementation, multi-seed CSVs, and 11 rounds of
Codex review log under `bench/nicon_v2/`.

The paper's main contribution is the **honest negative result** plus
**V2L-learnableRMS** as a strong-when-large-n CNN architecture: per-branch
strict-linear AOM operators (Identity, 5 SG variants, 2 Detrend, 1st
derivative, Norris-Williams, Whittaker), low-rank approximation for the
length-quadratic operators (rank 32), per-branch RMS normalisation with
a learnable scale, branch-level squeeze-and-excitation, residual conv
trunk and global-average-pool head. Code: `nicon_v2/models/v2_aom_cnn.py`.

## 1. Introduction

NIR spectroscopy regression is a long-standing chemometrics problem in
which classical linear models (PLS, Ridge) remain the workhorse. Recent
deep-learning work (DeepSpectra 2019, Cui & Fearn 2018, Mishra & Passos
2021–2023) shows that 1-D CNNs can match or beat PLS on large datasets
(`n > 5000`), and a growing body of LUCAS-pretrained literature
(Padarian 2019, Tsakiridis 2020, Mishra & Passos 2022) reports
foundation-model transfer gains on soil-NIR.

We test these claims rigorously on a 10-dataset representative cohort
spanning soil chemistry, plant chemistry, food science, and beverage
analysis (`n_train ∈ [40, 3734]`). We find that the published gains
**do not generalise beyond the 2 largest plant-chemistry datasets**:
classical Ridge regression beats every CNN configuration we tested on
7/10 datasets, multi-seed-validated.

## 2. Related Work

* DeepSpectra (Tsakiridis et al. 2020) — benchmark of CNNs on 30+ NIR
  datasets; reports PLS-comparable performance with 60-trial preprocessing
  search.
* Mishra & Passos (2021, 2022) — multi-block CNNs and LUCAS-pretrained
  transfer for soil NIR; reports 5-15 % improvement on soil cohorts.
* Padarian et al. (2019) — multi-task CNN on LUCAS-SOC + N + pH; reports
  10 000-sample heuristic for CNN to beat PLS.
* AOM-PLS (Beurier et al. 2023) — operator-mixture PLS with strict-linear
  spectroscopy operators (the design source of our AOM front-end).

Our work tests these architectures multi-seed on a cohort that
deliberately spans the small-n regime where the literature reports
mixed results.

## 3. Method

### 3.1 V2L-learnableRMS — production CNN

```
Input X ∈ ℝ^(N×1×L)
  ↓ AOM 11 branches (Identity, SG ×5, FD, NW, Detrend ×2, Whittaker)
        — strict-linear operators; matrix ops use rank-32 low-rank
          approximation A ≈ U V^T to keep params bounded for large L
  ↓ RMSBranchNorm with learnable scale γ_b (init: 1 / RMS_train(b))
  ↓ Branch-level Squeeze-and-Excitation (input-level MKL analogue)
  ↓ Channel concatenation: (N, 11, L)
  ↓ Residual Conv trunk: 3 blocks, channels 32→64→96, kernels 7→5→3
        each: Conv1D + GroupNorm + GELU + Dropout1d(0.2) + SE + identity skip
  ↓ MaxPool(2) between blocks
  ↓ AdaptiveAvgPool1d(1) + Flatten + Dropout(0.3)
  ↓ Linear(96 → 1)
```

Trained with AdamW (lr 1e-3, weight decay 1e-4), OneCycleLR,
batch size min(32, max(8, n_train // 8)), epochs 200 with patience 20,
Bjerrum 2017 EMSC-parameter augmentation (per-batch offset/slope/multiplicative
perturbations scaled to the per-dataset spectrum range).

### 3.2 V6b distillation ablation

`V6b-DistillExtended-V2M` adds a fold-local AOM-PLS teacher
(`AOMPLSRegressor`, extended bank with Whittaker variants, max
components 20, 5-fold inner CV) whose predictions on `X_train` are
standardised and added as a second target via
`L = MSE(y_pred, y_true) + 0.3 · MSE(y_pred, z_teacher)`. The student
remains a single CNN forward pass at inference time.

### 3.3 LUCAS pretraining ablation

`V6b-LucasPretrained-V2M` initialises the V2A backbone from a
checkpoint pretrained on 5000 LUCAS-SOC spectra (Nocita et al. 2014,
26 650-sample European soil library; 4200 wavelengths at 0.5 nm; SOC
target with `log1p` transform). The pretrained checkpoint reaches
R² = 0.76 on the held-out 5711-spectrum LUCAS validation set.

Length-invariant parameters (Conv kernels, branch SE, RMS norms, head
Linear) load via `_load_pretrained_compatible` with `strict=False`;
length-dependent operators (LowRank Detrend / Whittaker U/V matrices,
MSC means) at the target `L` are silently re-initialised. The
`RMSBranchNorm.fitted` buffer is reset at the start of `fit_branches`
so the target-domain RMS is recomputed (Codex round-11 M2 fix).

### 3.4 Reproducibility

* **Splits.** Predefined train/test splits from the TabPFN paper
  distribution (curated cohort) and the AOM-PLS bench manifests.
* **Inner CV.** sklearn KFold (5-fold) for hyper-parameter selection;
  never for evaluation.
* **Seeds.** Round 14 / 16 use seeds {0, 1, 2, 3, 4} = 5 seeds.
* **Outputs.** Aggregate CSV + per-sample predictions parquet for every
  `(variant, dataset, seed)`. Environment lockfile and git SHA stamped
  on every result row.

## 4. Experimental Setup

* **Representative cohort.** 10 datasets spanning soil (ALPINE_P,
  All_manure_MgO, All_manure_Total_N, grapevine_chloride, N_woOutlier),
  plant chemistry (An_NeoSpectra, Chla+b_block2deg, Chla+b_species),
  food (Beer_60_YbaseSplit), inorganic (TIC_spxy70).
  `n_train ∈ [40, 3734]`; `n_features ∈ [196, 2151]`.
* **Curated cohort.** 39 datasets from
  `bench/AOM_v0/Ridge/benchmark_runs/curated/`; used for the
  publication headline numbers.
* **Reference baselines** (from the cohort manifest CSVs):
  paper Ridge, paper PLS, paper TabPFN-raw, paper TabPFN-opt,
  paper CNN (NICON), paper CatBoost, AOM-PLS-best, AOM-Ridge-best.

## 5. Results

### 5.1 Representative cohort, 5 seeds (200 paired observations, V2L vs V6b-V2M)

The round-14 multi-seed validation of round-13's `V6b-DistillExtended-V2M`
signal:

| metric                   | V6b-V2M vs V2L-learnableRMS |
|--------------------------|----------------------------:|
| paired observations      | 50 (10 datasets × 5 seeds)  |
| median Δ% rmsep          | **−0.33 %**                 |
| mean Δ% rmsep            | +2.81 %                     |
| std Δ%                   | 21.40 %                     |
| wins / losses            | **25 / 25**                 |
| paired Wilcoxon p        | **0.7594**                  |

V6b-V2M and V2L are statistically indistinguishable. Earlier-iteration
single-seed results (round 13: median −1.17 %, 6 / 10 wins) were
seed-noise.

### 5.2 LUCAS pretraining ablation (250 paired observations, V6b-LUCAS-V2M)

The round-16 multi-seed validation of round-15's `V6b-LucasPretrained-V2M`
single-seed best-CNN result (4 / 10 wins as best CNN at seed 0):

| metric                   | V6b-LUCAS-V2M vs V2L | V6b-LUCAS-V2M vs V2M-deeper |
|--------------------------|---------------------:|----------------------------:|
| paired observations      |                  50  |                          50 |
| median Δ% rmsep          |             −0.21 %  |                  **−1.41 %** |
| mean Δ% rmsep            |             +6.92 %  |                     +4.81 % |
| wins / losses            |               26 / 24|                     30 / 20 |
| paired Wilcoxon p        |              0.7594  |                       0.215 |

LUCAS pretraining produces a directional but non-significant
improvement vs the matched-trunk V2M-deeper control (60 % wins,
p = 0.215). It ties V2L-learnableRMS exactly (52 % wins, p = 0.76).

The mean Δ% is positive (V6b-LUCAS slightly worse on average) because
the per-dataset distribution is bimodal: soil-domain datasets gain
5-30 %, non-soil datasets (Beer, TIC, N_woOutlier) lose 30-100 % on
some seeds. Distillation (V6b component) acts as a safety-net that
caps the worst-case but does not reverse the domain mismatch.

### 5.3 Curated cohort headline (39 datasets × 1 seed, V2L production CNN)

`publication_curated/results.csv`, 4 variants × 39 datasets = 155 OK
predefined-split observations (1 row failed: PLS could not fit on
`Firmness_spxy70` because n_components=25 > n_features−1=22 — affects
Ridge / PLS pairing only on that dataset). Pairing across all 4
variants is available on 38 / 39 datasets.

#### Headline (control = Ridge-baseline, 38 paired observations)

| variant                       | median rmsep | median Δ% vs Ridge | wins / 38 | Wilcoxon p |
|-------------------------------|-------------:|-------------------:|----------:|-----------:|
| Ridge-baseline                |       0.449  |        0.00 %      |    —      |     —      |
| PLS-baseline                  |       0.454  |       +2.93 %      |    7      |   1.6e-4   |
| **V6b-LucasPretrained-V2M**   |       0.750  |      **+23.5 %**   |    5      | **5.8e-7** |
| **V2L-learnableRMS**          |       0.846  |      **+29.5 %**   |    4      | **1.7e-6** |

**Both CNN variants are statistically significantly worse than Ridge
on the curated cohort** (p ≤ 1.7 × 10⁻⁶). V6b-LUCAS-V2M edges out V2L
on the curated cohort (+23.5 % vs +29.5 % vs Ridge; 5/38 vs 4/38 wins),
flipping the round-12-16 representative-cohort verdict. On the
curated cohort, LUCAS pretraining is therefore a net positive (+6 pp
gap reduction).

#### V2L vs paper / AOM-best baselines (descriptive)

| Comparison                              | Median Δ% rmsep | n_pairs |
|-----------------------------------------|----------------:|--------:|
| V2L vs paper Ridge                      |     +33.9 %     |  39     |
| V2L vs paper PLS                        |     +32.4 %     |  39     |
| V2L vs paper TabPFN-raw                 |     +14.97 %    |  39     |
| V2L vs paper TabPFN-opt                 |     +27.14 %    |  39     |
| V2L vs paper CNN (NICON)                |      +6.50 %    |  34     |
| V2L vs paper CatBoost                   |     +10.66 %    |  39     |
| V2L vs AOM-Ridge-best                   |     +40.46 %    |  39     |
| V6b-LUCAS vs paper CNN (NICON)          |      +6.66 %    |  34     |
| V6b-LUCAS vs AOM-Ridge-best             |     +39.79 %    |  39     |

**Smoke gate (Codex spec):** ratio ≤ 1.05 vs AOM-Ridge-best AND
≥ 50 % wins vs Ridge AND Wilcoxon p < 0.05 with V2L < Ridge.
**V2L: 1.40 ratio, 10.5 % win rate, p = 1.7e-6 in the WRONG direction.**
**Smoke gate not cleared in any of the three sub-criteria.**

#### CNN's 4 wins on the curated cohort (V2L < Ridge)

| dataset                                  | n_train | Ridge   | V2L     | Δ%       | domain |
|------------------------------------------|--------:|--------:|--------:|---------:|--------|
| An_spxyG70_30_byCultivar_ASD             |    82   |  4.055  |  3.560  | **−12.2 %** | plant chem |
| An_spxyG70_30_byCultivar_NeoSpectra      |    82   |  4.614  |  4.342  | **−5.9 %**  | plant chem |
| Pi_spxyG                                 |   ~80   |  0.171  |  0.146  | **−14.3 %** | plant chem |
| V25_spxyG                                |   ~80   |  0.341  |  0.250  | **−26.6 %** | plant chem |

V6b-LUCAS-V2M adds one more win:

| dataset                                  | n_train | Ridge   | V6b-LUCAS | Δ%       |
|------------------------------------------|--------:|--------:|----------:|---------:|
| MP_spxyG                                 |   ~80   |  0.0212 |   0.0207  | −2.4 %   |

All five CNN-wins are **AOM-PLS-bench plant-chemistry datasets**
(An, Pi, V25, MP — phenolics / sugars / pigments / other secondary
metabolites). They share two features: (1) plant-chemistry domain,
(2) `n_train ≈ 80–250` (small but not tiny). On every other dataset
(80 % of the cohort) CNN loses by 5-500 %.

#### CNN's worst losses on the curated cohort

| dataset                                  | V2L Δ% vs Ridge | V2L     | Ridge   |
|------------------------------------------|-------:|--------:|--------:|
| Quartz_spxy70 (target ≈ 0)               | +516 341 % | 0.0069 | ~0     | (degenerate target, excluded from headline)|
| Corn_Starch_80_ZhengChenPelegYbaseSplit  | +486.8 %    | 0.860 | 0.146  |
| Corn_Oil_80_ZhengChenPelegYbaseSplit     | +259.6 %    | 0.159 | 0.044  |
| Milk_Fat_1224_KS                         | +174.3 %    | 0.322 | 0.117  |
| Biscuit_Sucrose_40_RandomSplit           | +154.7 %    | 2.059 | 0.808  |

These are small-n high-dimensional datasets where the CNN
catastrophically over-fits. The dataset-by-dataset variance dominates
the cohort-level Wilcoxon.

### 5.4 Per-dataset structure

The two Chla+b plant-chemistry datasets (`block2deg`, `species`,
both `n_train ≥ 2900`) are the only datasets where any CNN closes
≥ 50 % of the Ridge gap. On the remaining 8/10 representative datasets
(or 37/39 curated) Ridge wins or ties.

| dataset                    | n_train | best variant   | Δ% vs Ridge |
|----------------------------|--------:|-----------------|-----------:|
| Chla+b_block2deg           |   2925  | V2L (or V6b-LUCAS) | **−75 %**  |
| Chla+b_species             |   3734  | V2L              | **−68 %**  |
| An_NeoSpectra              |     82  | V2L              | **−7 %**   |
| N_woOutlier                |   1205  | V2L              | **−2 %**   |
| ALPINE_P_291_KS            |    247  | Ridge            | +11 %      |
| All_manure_MgO             |    343  | Ridge            | +26 %      |
| All_manure_Total_N         |    343  | Ridge            | +15 %      |
| Beer_60_YbaseSplit         |     40  | Ridge            | +18 %      |
| TIC_spxy70                 |     43  | Ridge            | +28 %      |
| grapevine_chloride_556_KS  |    388  | Ridge            | +1 %       |

The pattern is consistent with Padarian's 2019 "10000-sample heuristic"
extrapolated downward: CNN closes the Ridge gap progressively as
`n_train` grows; the threshold on this cohort is approximately
`n_train ≈ 2900`.

## 6. Ablations

### 6.1 The 16-round ceiling experiment

We ran 16 rounds of architecture-and-training experiments to test
whether a CNN configuration can clear the Ridge ceiling on small-n
NIR. The full chronology is in `docs/IMPLEMENTATION_LOG.md`. Headlines:

| Round | Variant | Result on representative cohort |
|------:|---------|-----------------------|
| 5-8   | V2A → V2H (low-rank Detrend / Whittaker, rank 32) | +30 % vs AOM-Ridge-best (single seed) |
| 10    | V2L-learnableRMS (production) | +18 % vs paper Ridge, 4/10 wins |
| 11    | V2L-perbranchInit1 + V2M-deeper (Codex round-9 priorities) | +35 % vs AOM-PLS-best (single seed) |
| 12    | V3-AOMTransformer / V6-Distill / V7-TTA | V3 hurts (+6 %); V7 marginal; V6 directional (−0.9 %, p=1.0) |
| 13    | V6b-DistillExtended (extended bank teacher) + SWA | V6b 6/10 wins, 6.5 pp closure (single seed) |
| 14    | **5-seed validation of V6b** | **V6b ≡ V2L (25/25, p = 0.76). The round-13 signal was seed-0 noise.** |
| 15    | V6b-LucasPretrained-V2M (foundation transfer) | 4/10 best CNN (single seed) |
| 16    | **5-seed validation of V6b-LUCAS** | **V6b-LUCAS ≡ V2L (26/24, p = 0.76); −1.41 % vs V2M but p = 0.21.** |

After 16 rounds and ~250 GPU-hours, no CNN variant (V2L, V2M, V6, V6b,
V6b-LUCAS, V3-Transformer, V7-TTA, SWA, branch SE, dilated trunk,
multi-kernel stem, learnable / tied / unit RMS) significantly beats Ridge
across the cohort.

### 6.2 V3 transformer trunk hurts

A 2-layer transformer encoder over post-conv wavelength tokens
(replacing the second + third ResConvBlocks of V2L) lost on 8 of 10
datasets (median Δ% +6.27 % vs V2L, p = 0.16). The wavelength-axis
self-attention has too many free parameters for the median
`n_train ≈ 340` and likely destroys the local convolutional inductive
bias that V2L exploits.

### 6.3 SWA was effectively inactive

Stochastic Weight Averaging (averaging the last 25 % of training epochs)
on V2L produced a strict val-loss improvement on only 2/10 datasets;
on the remaining 8 the early-stop checkpoint was already at or below
the SWA average. The OneCycleLR schedule + 20-epoch patience leaves
the model nearly stationary in the late phase, so SWA averages
near-identical weights.

### 6.4 LUCAS pretraining: domain-conditional gain, cohort-level wash

V6b-LUCAS-V2M wins 1/10 datasets (Chla+b_block2deg) outright and 3/10
as best-CNN (Chla+b_block2deg, All_manure_MgO, TIC), all soil-related
or plant-chemistry. On non-soil targets (Beer beverage chemistry,
TIC inorganic) the LUCAS-domain RMS scales hurt by 30-100 %; the AOM-PLS
distillation safety-net partially recovers but cannot reverse the
domain shift. Net cohort effect: tied with V2L (p = 0.76).

## 7. Discussion

### 7.1 The CNN ceiling on small-n NIR — confirmed empirically

Our 16-round multi-seed-validated search converges on the same
conclusion the literature has discussed for years (Cui & Fearn 2018,
Mishra & Passos 2021, Padarian 2019): **for small-n NIR regression
(`n_train ≤ 1200`), no current CNN architecture or training recipe
significantly beats Ridge regression**. The deep-learning gains
reported in the literature do generalise to large-n datasets (our
Chla+b_species at `n_train = 3734` shows a 68 % Ridge gap closure)
but not to the typical NIR cohort.

### 7.2 Single-seed traps and multi-seed validation

**Round 13's "6/10 wins, 6.5 pp closure" turned into round 14's "25/25,
p = 0.76" once seeds 1-4 were added.** The same pattern repeated in
rounds 15 → 16. This is a cautionary tale about reporting CNN-vs-Ridge
results on small NIR cohorts at single-seed: per-dataset standard
deviations of the order of 20–30 % at `n_train < 100` make any
10-paired observation conclusion essentially noise.

### 7.3 Why LUCAS pretraining underdelivers

The published LUCAS-pretrained gains in the soil-NIR literature
(Mishra & Passos 2022, Tsakiridis 2020) are typically reported on
soil-only target cohorts. Our representative cohort mixes soil,
plant chemistry, food, and inorganic targets — a deliberate breadth
test. LUCAS pretraining helps soil-domain targets (ALPINE,
All_manure_MgO, grapevine) but hurts non-soil targets where the
LUCAS-learned RMS scales and conv kernels do not match the target
distribution. The cohort-level effect washes out.

A future study restricted to soil-only targets would likely show
clear LUCAS gains; we make no claim against that finding.

### 7.4 What we got wrong

* **Round 13's GO verdict** was based on Codex's interpretation of a
  single-seed 6.5 pp closure as a "5-pp continuation threshold met".
  In retrospect, single-seed at 10 datasets gives ±21 % per-observation
  noise, so a 6.5 pp closure is well within noise. The GO was
  technically correct but the underlying signal was not real.
* **V3-Transformer** was tried with the expectation that wavelength-axis
  attention would help. It did not — the inductive bias of local
  convolutions matters more than long-range dependencies on small NIR
  cohorts.
* **SWA** was tried without first checking whether the OneCycleLR
  schedule leaves the model stationary at late epochs. It does, so
  SWA averages near-identical weights and provides no signal.

### 7.5 Limitations

* **Codex round-11 M1 (theoretical impurity).** Distillation teachers
  are fit on the full outer `(X_train, y_train)` before the internal
  validation split is carved. This leaks internal-validation labels
  into the training algorithm but never touches test data. We do not
  expect this to overstate distillation's measured benefit (which is
  null), but a future revision should split first then fit teacher.
* **Codex round-11 M3.** Branch operators (MSC reference, RMSBranchNorm)
  are also fit on the full outer training set rather than only the
  inner-train subset, for the same reason.
* **Codex round-11 M2 (fixed).** LUCAS-pretrained `RMSBranchNorm.fitted=1`
  buffer was loaded from the LUCAS checkpoint, preventing target-domain
  RMS recomputation. Fixed in `fit_branches` by resetting `fitted=0`
  before re-fitting.
* **Codex round-11 M4.** Some result CSV rows record a "dirty" git SHA
  during ad-hoc seed re-runs; tracked diffs were empty between SHAs.
* The cohort is 10 representative datasets + 39 curated datasets;
  this is small enough that domain-specific architectures (e.g.
  soil-only with LUCAS pretrain) might still show wins not visible
  in our cohort-level numbers.
* Each variant is single-seed inside the test split (predefined
  train/test); the multi-seed analysis varies model init + internal
  val split but not the outer test split.
* All experiments use a fixed hyperparameter set across datasets;
  per-dataset HPO (Codex round-11's top-ranked remaining idea) was not
  attempted but was estimated to give 8-12 pp closure — still below
  the smoke gate.

## 8. Conclusion

We dissect the CNN-vs-Ridge ceiling for small-n NIR spectroscopy
through 16 rounds of architecture-and-training experiments and a
multi-seed validation protocol. We propose `V2L-learnableRMS`, a
multi-branch AOM-superblock CNN with low-rank length-quadratic
operators and a learnable per-branch RMS scale, that closes 65-75 %
of the Ridge gap on plant-chemistry datasets with `n_train ≥ 2900`.

We show that **on the representative cohort, no CNN configuration we
tested significantly beats Ridge across 5 seeds** (V6b-V2M vs V2L
median Δ% = −0.33 %, p = 0.76; V6b-LUCAS-V2M vs V2L median Δ% =
−0.21 %, p = 0.76). Earlier single-seed signals (rounds 13 and 15)
were noise and were correctly killed by the multi-seed validation
protocol of rounds 14 and 16.

The honest negative result + the V2L architecture together comprise
the paper's contribution. Code, multi-seed CSVs, and 11 rounds of
Codex review log are released under `bench/nicon_v2/`.

## Appendix A — Reproducibility

* `nicon_v2/` — source.
* `tests/` — 88 + 4 round-12 + … (see `tests/`) tests covering metrics,
  datasets, baselines, training-loop reproducibility, length-robustness,
  geometry parity vs upstream NICON, no-leak invariants, preprocessing
  parity vs scipy, augmentation determinism, fold isolation in stacking,
  V3 / V6 / V7 / SWA round-12 features.
* `benchmarks/run_baseline_benchmark.py` — resumable runner, predictions
  parquet output.
* `publication/scripts/{cohort_summary,representative_table,make_figures,
  make_tables}.py` — paired Wilcoxon, figures, LaTeX tables, two-tier
  success check.
* Git SHA, environment lockfile, dataset hashes recorded per benchmark
  run.

## Appendix B — Codex review log

11 rounds of Codex review applied:

* Rounds 1-4 — early-iteration reviews (plan, V1a, V1c + stacking,
  pre-curated state). All Critical/High applied.
* Round 5 — V2A AOM-superblock CNN design.
* Round 6 — V2A / V2B implementation review (F1 L2-from-init reg loss
  fix wired).
* Round 7 — V2C / V2D / V2E ablation review.
* Round 8 — V2H low-rank review.
* Round 9 — V2L / V2M ceiling diagnosis (architecture-only ceiling
  ≈ 1.20-1.30 vs AOM-PLS).
* Round 10 — V3 / V6 / V7 round-12 review (NO-GO pure architecture;
  GO V6b + SWA).
* **Round 11 — final review (rounds 12-16)**: 4 medium issues
  flagged (M2 RMS reset bug fixed in production; M1/M3 documented as
  theoretical impurities; M4 git hygiene). Verdict: **NO_GO on further
  rounds; start publication pass.**

Full chronology in `docs/IMPLEMENTATION_LOG.md`.

## Appendix C — V2L hyperparameters

```python
NiconV2A(
    input_shape=(1, n_features),
    bank="extended_lowrank",
    lowrank_rank=32,
    trainable_ops=True,
    matrix_trainable_ops=True,
    branch_se=True,
    learnable_rms=True,
    rms_init_mode="inverse_rms",
    trunk_channels=(32, 64, 96),
    trunk_kernels=(7, 5, 3),
    spatial_dropout=0.2,
    head_dropout=0.3,
    se_reduction=4,
    block_type="res",
    head_type="gap_linear",
    trunk_type="conv",
)

# Training:
TrainConfig(
    epochs=200,
    patience=20,
    batch_size=min(32, max(8, n_train // 8)),
    lr=1e-3,
    weight_decay=1e-4,
    val_fraction=0.2,
    use_amp=(device == "cuda"),
    one_cycle=True,
    augmenter=BjerrumAugmenter(BjerrumConfig(enabled=True), ...),
)
```
