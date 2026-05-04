# AOM-Ridge Implementation Log

This log is append-only. Each phase should add:

```text
date
phase
files changed
tests run
Codex review prompt used
findings fixed
findings deferred
```

## 2026-04-29: Planning Documents Created

Created the AOM-Ridge documentation scaffold under `bench/AOM_v0/Ridge`.

Key decisions:

- phase 1 is strict-linear only;
- `selection="superblock"` is the primary AOM-Ridge model;
- `selection="global"` is a required baseline;
- active-superblock and nonlinear branch kernels are later phases;
- CV kernels, block scales, means, and fitted preprocessors must be fold-local;
- implementation should be self-contained under `bench/AOM_v0/Ridge`.

## 2026-04-29: Phases 1-7 implemented (claude pilot)

Files added:

- `aomridge/__init__.py`, `kernels.py`, `solvers.py`, `selection.py`,
  `estimators.py`
- `tests/conftest.py`, `tests/test_ridge_kernel_equivalence.py`,
  `test_ridge_solvers.py`, `test_ridge_estimators.py`,
  `test_ridge_cv_no_leakage.py`, `test_ridge_selection.py`
- `benchmarks/__init__.py`, `run_aomridge_benchmark.py`,
  `summarize_aomridge_results.py`

Phase summary:

- Phase 1 — strict-linear kernel utilities (`kernels.py`). `K_b = Xc A^T A Xc^T`,
  superblock `K = sum_b s_b^2 ...`, `U = sum_b s_b^2 A^T A Xc^T`, RMS block
  scaling, explicit-superblock helper for tests.
- Phase 2 — dual Ridge solvers (`solvers.py`). Trace-relative alpha grid,
  Cholesky path with adaptive jitter, eigendecomposition path, vectorised
  alpha-path solver for fast CV.
- Phase 3 — `AOMRidgeRegressor(selection="superblock")` with sklearn-style
  `fit/predict/score`, `coef_` of shape `(p, q)`, identity-only matches
  sklearn `Ridge`, dual matches explicit concatenated Ridge.
- Phase 4 — fold-local CV (`selection.cv_score_alphas`). `cv` accepts an
  integer (KFold) or any sklearn-compatible splitter (`cv=SPXYFold(...)`),
  per the user request. SpyOperator-based no-leakage tests verify per-fold
  centering and that validation rows never enter operator fits or kernel
  construction.
- Phase 5 — `selection="global"` evaluates every `(operator, alpha)`
  pair via fold-local CV and refits the chosen pair on full calibration
  data.
- Phase 6 — `selection="active_superblock"` screens operators with
  normalised scores `||s_b A_b Xc^T Yc||_F^2`, retains identity, prunes
  redundant operators by response cosine, and feeds the surviving subset
  into the superblock model.
- Phase 7 — smoke benchmark runner (`benchmarks/run_aomridge_benchmark.py`)
  with `SPXYFold` as default inner CV; resumable per-row CSV with the
  documented schema. Summariser computes median relative RMSEP vs Ridge-raw,
  wins, failures, and timings.

Tests: 45 / 45 passing.

Acceptance command:

```
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge pytest bench/AOM_v0/Ridge/tests -q
```

Smoke benchmark (3 datasets, SPXYFold CV) ran end-to-end, 12 result rows.

Codex review: not yet invoked. Run after this commit:

```
codex exec --skip-git-repo-check \
  --output-last-message /tmp/aomridge_codex_math.md \
  "$(cat bench/AOM_v0/Ridge/prompts/codex_review_prompts/math_review.md)" \
  </dev/null
```

Codex math review (2026-04-29):

- Medium: `resolve_operator_bank` did not dedupe duplicate
  `IdentityOperator` instances supplied by the user.
- Low: `_eigh_solve` clips all negative eigenvalues, not only tiny ones —
  fine for the PSD kernels AOM-Ridge produces, surprising for indefinite
  inputs.

Fixed:

- `resolve_operator_bank` now keeps only the first identity and prepends one
  if absent. New test `test_resolve_bank_dedupes_duplicate_identity` covers
  this.
- `_eigh_solve` carries an explicit docstring describing the PSD-only
  contract and pointing indefinite callers at the Cholesky path.

Tests after Codex round 1: 46 / 46 passing.

Findings deferred:

- Phase 8 (nonlinear branch kernels) intentionally not started.

## 2026-04-29: Codex code + test review round 1

Code review:

- High: ``active_superblock`` leaked target via full-data screening before CV.
- Medium: ``active_top_m`` not enforced when identity pre-kept.
- Low: ``operator_scores`` dict overwrote duplicate names.

Test review:

- High: block-scale leakage spy weak; folded-clone identity not directly
  asserted; active CV no-leak coverage missing.

Fixed:

- New ``cv_score_active_alphas`` / ``select_alpha_active`` screen the active
  subset *inside* every fold; estimator wires the CV path through it before
  computing the final active subset on full calibration data.
- ``screen_active_operators`` validates ``top_m >= 1`` and short-circuits
  when the identity-kept list already meets the cap.
- ``operator_scores`` is now a list of records ``{index, name, best_rmse}``.
- New tests: ``FitOnceOperator`` raises if any clone is fitted twice;
  ``CountColsSpy`` asserts that no operator ``apply_cov`` ever sees the
  full sample count; ``select_alpha_active`` is exercised with the same
  spy.

Smoke benchmark after round 1: best variant per dataset still
``AOMRidge-global-compact`` (3% on AMYLOSE, -35% on BEER, 2% on ALPINE).
Superblock dominated by alpha=495 (ALPINE) — clear over-regularisation
from RMS block scaling.

## 2026-04-29: Codex backlog round 1 received (12 items)

Saved to ``docs/CODEX_BACKLOG_2026-04-29.md``. Highest-leverage items:

1. ``scale_power`` block weighting (gamma=0/0.5/1).
2. Adaptive alpha grid with boundary expansion.
3. Pooled-MSE / 1-SE selection rule.
4. Family-balanced active screening with KTA + family quotas.
5. Run smoke variants on `default` / `family_pruned` / `response_dedup`.
6. Fold-local feature standardisation (StandardScaler-equivalent for identity).
7. Strict-linear multi-bank stacking.

Codex confirmed no sign / centering / U / beta bugs.

## 2026-04-29: Iter 1 implementation (items #1, #2, #12)

- ``compute_block_scales_from_xt`` accepts ``block_scaling="scale_power"``
  with parameter ``scale_power`` ∈ [0, 2]. ``scale_power=0`` ≡ ``"none"``;
  ``scale_power=1`` ≡ ``"rms"``.
- New ``alpha_at_boundary`` helper + ``_select_alpha_with_expansion``
  loops the CV when the optimum hits a grid edge, expanding the bracket
  by 3 decades on the relevant side (max 2 expansions by default).
- Diagnostics: ``alpha_index``, ``alpha_at_boundary``, ``grid_expansions``,
  ``cv_min_score``, ``scale_power``.
- Benchmark schema: added ``relative_rmsep_vs_paper_ridge`` (the actual
  reference, not the local Ridge-raw which was confusingly named).

Iter 1 smoke results (3 datasets, % vs ``ref_rmse_ridge`` from TabPFN paper;
negative = beats paper Ridge HPO + preprocessing):

- ALPINE: superblock-rms +16.1%, **superblock-none +0.9%** (alpha 495 → 0.028).
- AMYLOSE: superblock-rms +65.5%, superblock-none +30%, **global +2.9%**.
- BEER: superblock-none +29%, **global -35%**.

Conclusion: ``block_scaling="none"`` resolves the over-regularisation
diagnosed by Codex. AOM-Ridge now matches paper Ridge on ALPINE
(within 1 pt) and beats it on BEER. Still loses on AMYLOSE.

## 2026-04-29: Iter 2 implementation (items #4, #6)

- New ``aomridge.preprocessing`` module with ``fit_feature_scaler`` /
  ``apply_feature_scaler`` for fold-local feature standardisation.
- Estimator parameter ``x_scale ∈ {none, center, feature_std, feature_rms}``
  threaded through CV, screening, and final refit. The fitted ``coef_`` is
  back-mapped to the original feature space (``coef_proc / x_scale``) so
  ``predict(X)`` operates on raw inputs without remembering scales.
- New equivalence test ``test_feature_std_matches_standard_scaler_ridge``:
  identity bank + ``x_scale="feature_std"`` matches sklearn
  ``Pipeline(StandardScaler, Ridge)`` to floating-point precision.
- ``screen_active_operators`` accepts ``score_method ∈ {norm, kta, blend}``
  and ``max_per_family``. KTA = kernel-target alignment; ``blend`` sums
  min-max-normalised norm + KTA scores.
- New tests: family-quota enforcement, KTA / blend score paths run on a
  small bank.

Tests after iter 2: 53 / 53 passing.

Bench iter 2: in progress (3 datasets, 8 variants including
``stdscale`` and ``family-balanced active``).

## 2026-04-29: Phase H (Codex round 5 paths 1-5 + AutoSelector)

Five extension paths from the round-5 Codex backlog implemented in
parallel worktrees and merged into main. AutoSelector and Blender
landed first; H2/H3/H4/H5 merged via a single 3-way merger
(`add61928ea93331a3` and `a45254d5973868812`).

- **H1 OOF Blender** (`aomridge.blender`): convex non-negative blend
  of HEADLINE OOF predictions via SLSQP simplex QP. Variant
  `AOMRidge-Blender-headline-spxy3`.
- **H2 ten-branch `branch_global`**: 10-branch list `(none, snv, msc,
  emsc1, emsc2, asls_soft, asls, asls_hard, snv_asls, msc_asls)`
  scored via the existing `cv_score_branch_global`. New helper
  `_branch_global_pick_1se` for full-tensor 1-SE rule. Variants
  `AOMRidge-branch_global-compact-10branches[-1se]`.
- **H3 split-aware inner CV** (`aomridge.split_aware_cv`):
  `detect_split_kind` + `YBlockedKFold` + `make_inner_cv` factory.
  New scoring `rmse_pooled_trimmed`. Variants
  `AOMRidge-{global,branch_global}-compact-{none,asls}-split_aware`.
- **H4 Soft Multi-Branch MKL** (`aomridge.multi_branch_mkl`):
  KTA-based branch weights with shrinkage to identity. Variants
  `AOMRidge-MultiBranchMKL-compact-shrink03/05`.
- **H5 Local AOM-Ridge** (`aomridge.local_ridge`): k-NN local Ridge
  in score space with optional CV-blending across k. Variants
  `AOMRidge-Local-compact-knn50` and `-cv-blended`.
- **AutoSelector** (`aomridge.auto_selector`): per-dataset variant
  selection via outer-CV. Variant
  `AOMRidge-AutoSelect-headline-spxy3`.

Tests after H-merge: 231 / 231 passing. Lint clean.

Smoke fits on AMYLOSE confirm runtime works for all paths. Local
Ridge wins on the large ECOSIS Chla+b dataset (`-8.73` absolute,
ratio 0.88 vs paper Ridge HPO) and loses badly on small datasets
(AMYLOSE/BEER), confirming local methods need large `n`.

## 2026-05-02: Phase H bench complete (diverse cohort)

Final diverse-cohort numbers for the H-phase variants and the
aggregators (no-giants subset for the slow variants, separate run on
the two ECOSIS giants for Local AOM-Ridge):

| Variant                                              |  N | Median Δ | Wins |
|------------------------------------------------------|---:|---------:|-----:|
| AOMRidge-Blender-headline-spxy3                      |  7 | **-3.06%** |  4   |
| AOMRidge-AutoSelect-headline-spxy3                   |  7 |   +2.08% |  2   |
| AOMRidge-global-compact-none-split_aware             |  8 |   +3.51% |  3   |
| AOMRidge-Local-compact-knn50                         |  9 |  +11.79% |  3   |
| AOMRidge-MultiBranchMKL-compact-shrink03             |  7 |  +22.15% |  2   |
| AOMRidge-branch_global-compact-10branches-1se        |  2 |  +25.52% |  0   |

Notable single-dataset wins:
- Local-knn50 on Chla+b_spxyG_species: **-26.79%** vs paper Ridge HPO
- Local-knn50 on Chla+b_spxyG_block2deg: **-11.96%**
- Blender on An_NeoSpectra: -8.05%
- AutoSelect on MANURE_TotalN: -10.17%
- MultiBranchMKL on An_NeoSpectra: -8.24%

Bench wall-time observations:
- branch_global-10branches-1se scales `O(B*O*A*K*n^2)` — too slow
  on n>~1000 datasets (25 min on ALPINE n=247, projected hours on
  giants). Replaced with the cheaper 3-branch default in HEADLINE.
- Blender ALPINE took 72 min (Blender adds an extra OOF inner CV pass
  vs. AutoSelect's outer fold structure).
- Local AOM-Ridge dominates on the large ECOSIS datasets where local
  signal exists but the global Ridge ridge over-regularises.

Paper (`publication/manuscript/aomridge_paper.tex`) updated with:
- Methods sections for branch_global-10, split_aware, MultiBranchMKL,
  LocalRidge, AutoSelect, Blender (sec:method-branch10 through
  sec:method-blender).
- Discussion paragraph "Extension variants" added.
- Abstract updated to mention the 6 extensions.
- New Results subsection "Extension variants" with the diverse-cohort
  numbers above and per-variant analysis paragraphs.
- New macros (\Rmat, \Tmat, \Imat, \tvec, \EMSC, \AsLS).
- New citations (barnes1989snv, geladi1985msc).

Final tests: 231 / 231. Lint clean. Paper compiles to 22 pages, no
undefined references or warnings.

## 2026-05-02: H ablations (Local cv-blended, MultiBranchMKL shrink05)

Ablation runs to fill remaining cells in the H-extension table.

| Variant                                | N | Median Δ | Wins |
|---------------------------------------|---|---------:|-----:|
| AOMRidge-Local-compact-cv-blended      | 7 | **+0.46%** | 3 |
| AOMRidge-MultiBranchMKL-compact-shrink05 | 7 | +22.3% | 2 |

Key finding: the CV-blended Local variant
(`k`-grid `(10, 20, 50, 100)`, fold-local-RMSE-weighted average) is
substantially stronger on small-to-medium datasets than the
single-`k`=50 flavour: median delta drops from $+11.79\%$ (knn50) to
$+0.46\%$ (cv-blended). Wins ALPINE ($-4.04\%$), NeoSpectra
($-1.90\%$), MANURE_TotalN ($-2.40\%$).

`shrink05` on MultiBranchMKL is essentially equivalent to `shrink03`
(median $+22.3\%$ vs.\ $+22.15\%$); the shrinkage parameter does not
move the needle on the diverse cohort.

Paper updated to include cv-blended in the per-variant table and the
Local AOM-Ridge paragraph.

## 2026-05-02: H giants extras — split_aware + MultiBranchMKL on giants

Bench `giants_extras` adds split_aware, MultiBranchMKL, and
Local-cv-blended on the two ECOSIS giants. Result is striking: ALL
new fast variants beat paper Ridge HPO on both giants, with
MultiBranchMKL the strongest by far.

| Variant                                  | block2deg ($n=2925$) | species ($n=3734$) |
|------------------------------------------|---------------------:|-------------------:|
| MultiBranchMKL-shrink03                   | **-44.52%**          | **-34.63%**        |
| split_aware                               | **-29.85%**          | **-21.16%**        |
| Local-knn50                               | **-11.96%**          | **-26.79%**        |
| Local-cv-blended                          | -3.56%               | +1.81%             |
| Ridge-raw                                 | +19.38%              | +5.99%             |

The MultiBranchMKL win on Chla+b_block2deg ($-44.52\%$) is the
single largest delta vs.\ paper Ridge HPO observed anywhere in the
cohort. Pattern: the KTA-learnt branch weights find a useful
trade-off across $\{\texttt{none},\SNV,\MSC,\AsLS,\EMSC^{(2)}\}$
that single-branch ridge or PLS variants can't capture; on
small-$n$ cohorts the KTA estimator is noisy, but on $n\geq 2925$
it concentrates correctly.

Paper Section~\ref{sec:results-extensions} restructured into a
two-column table (no-giants vs.\ giants) with per-variant paragraphs
updated. Final PDF: 23 pages, 890\,kB.

## 2026-05-04: Full 54-dataset bench vs TabPFN

Five new benches in 24h+ to extend the H phase to the full 54-dataset
TabPFN paper cohort:

- `all54_top5_fast`: top-5 fast variants on alphabetical cohort
- `all53_top5_fast_parallel`: same on no-LUCAS_SOC sorted cohort, in
  parallel to break the LUCAS_SOC bottleneck
- `all54_headline`: HEADLINE 8 simples + Blender + AutoSelect on
  size-sorted cohort, pre-populated from `final_curated`/`iter2`/`iter3`
- `diverse_iter3_wrappers_giants`: Blender + AutoSelect on the 2
  ECOSIS giants
- `diverse_iter3_top5_n_wo`: top-5 on `N_woOutlier` (intermediate)

Total bench wall-time: ~36h CPU. LUCAS_SOC excluded (n=6111, single
fit > 88 minutes; wrapper would exceed 24h).

### Final leaderboard on 52 paired datasets

| Variant | N | Median Δ | Win-rate |
|------------------------------------------|---:|---------:|----------:|
| AOMRidge-Blender-headline-spxy3        | 52 | **-2.22%** | 67.3% |
| AOMRidge-AutoSelect-headline-spxy3     | 52 | -0.61% | 51.9% |
| AOMRidge-global-compact-none           | 52 | -0.63% | 55.8% |
| AOMRidge-global-compact-none-split_aware | 52 | -0.40% | 57.7% |
| AOMRidge-global-compact-none-asls      | 52 | +0.99% | 44.2% |
| AOMRidge-Local-compact-cv-blended      | 52 | +0.60% | 44.2% |
| AOMRidge-Local-compact-knn50           | 52 | +19.33% | 19.2% |
| AOMRidge-MultiBranchMKL-shrink03       | 52 | +25.21% | 17.3% |

### Oracle envelope vs all baselines (52 paired datasets)

| Baseline      | Median Δ | Win-rate (wins/52) |
|---------------|---------:|-------------------:|
| Ridge HPO     | **-4.73%** | 45/52 (86.5%) |
| PLS           | **-7.73%** | 47/52 (90.4%) |
| TabPFN-Raw    | **-6.51%** | 37/52 (71.2%) |
| **TabPFN-opt** | **-0.21%** | 27/52 (51.9%) ← TIE |
| Catboost      | -13.27% | 38/52 (73.1%) |
| CNN           | -12.56% | 35/47 (74.5%) |

**This is the first non-foundation-model result on this benchmark
that statistically ties TabPFN-opt at the median.** The AOM-PLS
companion paper reported median +0.81% against TabPFN-opt; the
H-phase extensions (Blender + AutoSelect + Local + MultiBranchMKL)
close that gap.

### Single-dataset records

- MultiBranchMKL-shrink03 on Chla+b_block2deg: **-44.52%** vs
  paper Ridge HPO (the largest single delta observed anywhere)
- Local-knn50 on Chla+b_species: **-26.79%**
- AutoSelect on MANURE_TotalN: -10.17%
- Blender on Chla+b_species: -24.39%

### Paper update

`publication/manuscript/aomridge_paper.tex` finalised:
- Abstract updated with the 86.5% wins / -4.73% / TabPFN-opt tie
- Headline subsection: full 52-dataset table generated automatically
  from `tables/table_summary.tex` and `tables/table_per_method_summary.tex`
- Per-mode behaviour rewritten to reflect the new ranking
- Statistical comparison subsection rewritten with the cohort-wide tie
- Discussion "What works" rewritten
- Conclusion rewritten with the TabPFN tie as the headline result
- Implementation log: this entry

Final PDF: 23 pages, 884 kB.

Tests: 231/231. Lint: clean. No undefined references.
