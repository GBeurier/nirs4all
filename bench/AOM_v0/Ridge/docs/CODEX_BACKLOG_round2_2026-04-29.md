**Code Review Findings**

- **Medium:** adaptive alpha expansion has an off-by-one in diagnostics/control. In [estimators.py](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Ridge/aomridge/estimators.py:234), the last allowed grid can still increment `info["expansions"]` without evaluating that newly expanded grid. With default `max_grid_expansions=2`, [results.csv](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Ridge/benchmark_runs/smoke/results.csv:33) shows `grid_expansions=3`. Fix by breaking before expansion when the max is reached, and count only evaluated expansions.

- **Medium:** MSC is fold-safe as a transformer, but the benchmark wrapper leaks inner-CV validation rows into the MSC reference. [run_aomridge_benchmark.py](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py:286) calls `preproc.fit_transform(Xtr)` before `est.fit`, so AOM-Ridge’s internal CV sees MSC data transformed with a full-calibration reference. SNV is stateless, so it is fine. External test rows are not leaked, but MSC alpha/operator selection is not paper-HPO comparable.

- **Low:** global selection leaves `cv_min_score` blank. The global path stores `_selection_rmse_table`, but diagnostics only read `_selection_rmse_per_alpha` in [estimators.py](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Ridge/aomridge/estimators.py:509). Use `np.min(self._selection_rmse_table)` for global.

No functional bug found in the scale-power formula, coefficient back-mapping, per-operator alpha return, family quota/identity interaction, or pooled-MSE multi-output weighting. One validation gap: [compute_block_scales_from_xt](/home/delete/nirs4all/nirs4all/bench/AOM_v0/Ridge/aomridge/kernels.py:141) does not itself enforce `scale_power in [0, 2]`; only the estimator does.

Verification: `pytest bench/AOM_v0/Ridge/tests -q` passed, 55 tests.

**Empirical Commentary**

- **ALPINE:** best is `AOMRidge-superblock-compact-none-stdscale`, `relative_rmsep_vs_paper_ridge=0.9848` / **-1.52%**. The stdscale cluster is best, but margins are small.
- **AMYLOSE:** best is `AOMRidge-global-compact-none`, `1.0290` / **+2.90%**. This is the only remaining miss; absolute gap is `1.9363 - 1.8817 = 0.0546` RMSEP.
- **BEER:** best is `AOMRidge-global-compact-none-snv`, `0.4160` / **-58.40%**. MSC is essentially tied, but discount its CV-selection evidence until fold-local branch CV is implemented.

`mse_pooled` did not change selection in this smoke run: same operator, alpha, and RMSEP as `AOMRidge-global-compact-none` on all three datasets.

SNV/MSC are not generally helpful. They hurt ALPINE and AMYLOSE; they massively help BEER when paired with global AOM, selecting the same `sg_smooth_w21_p3` operator.

**Round-2 Backlog**

1. **AMYLOSE Strict-Linear Bank Search**  
   Severity: High. Where: `aompls/banks.py`, benchmark variants. Expected delta: AMYLOSE `-0.015` to `-0.040`. Risk: default bank already worsened AMYLOSE, so avoid broad 100-op multiple-comparison search; use family-pruned / response-dedup around smoothers and detrend compositions.

2. **MKL-Light Supervised Block Weights**  
   Severity: High. Where: `selection.py`, `kernels.py`, estimator diagnostics. Expected delta: AMYLOSE `-0.015` to `-0.035`. Risk: overfit on small n; constrain to top 3-6 blocks, nonnegative weights, fold-local only.

3. **Fold-Local Multi-Branch Kernel Stacking**  
   Severity: High. Where: `branches.py`, `selection.py`, estimator branch mode. Expected delta: AMYLOSE `-0.005` to `-0.020`; BEER already shows large branch upside. Risk: raw+SNV+MSC can hurt AMYLOSE unless weights can suppress bad branches; MSC must be fitted inside CV.

4. **Repeated / 5-Fold SPXY Stability Sweep**  
   Severity: Medium/High. Where: benchmark CV builder, optional `cv_repeats`. Expected delta: AMYLOSE `-0.005` to `-0.020`. Risk: runtime; if `sg_smooth_w21_p3` and alpha remain stable, this is diagnostic only.

5. **Alpha Grid Refinement + 1-SE Rule**  
   Severity: Medium. Where: `solvers.py`, `selection.py`, `estimators.py`. Expected delta: AMYLOSE `-0.005` to `+0.010`; 100-point grid already did not help. Risk: 1-SE may over-regularize; fix expansion accounting first.

6. **A2: SPXY-Derived Alpha Schedule**  
   Severity: Medium/Low. Where: SPXYFold diagnostics plus selection alpha scaling. Expected delta: AMYLOSE `0.000` to `-0.015`. Risk: easy to encode split geometry quirks rather than true generalization; only pursue if repeated CV shows alpha tracks fold diversity.