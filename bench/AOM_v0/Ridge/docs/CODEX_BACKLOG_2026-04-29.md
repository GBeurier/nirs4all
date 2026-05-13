**Current Issues Spotted**
- No sign, centering, `U`, `beta`, or fold-reuse bug stood out.
- `block_scaling="rms"` is equal-trace scaling: each scaled block gets `trace(s_b^2 K_b)/n ~= p`, so noisy derivative blocks get the same kernel mass as useful smooth/raw blocks.
- `scoring` is accepted but ignored; CV always averages fold RMSE.
- `global` builds one identity-scaled alpha grid. Harmless under current RMS single-block scaling, wrong once `block_scaling="none"` or supervised weights are added.
- `scale=True` is reserved but unimplemented.

**1. [High] Add Scale-Power Block Weighting; Benchmark `superblock` With `none` First**
- Change: implement `s_b = (T / (trace(K_b) + eps)) ** (gamma / 2)`, with `gamma=0` = none, `gamma=1` = equal-trace/RMS, `gamma in {0,.25,.5,.75,1}`.
- Where: `kernels.compute_block_scales_from_xt`, estimator param validation, benchmark variants.
- Test: invariant that `gamma=0` equals unit scales; `gamma=1` equalizes block traces; explicit-superblock equivalence still passes. Benchmark: `rmsep/ref_rmse_ridge`, `alpha`, `block_scaling`.
- Risk: `none` may let raw/smooth blocks dominate and suppress derivatives; tune only by CV, not test RMSEP.
- Default call: for benchmark, yes, try `selection="superblock"` and `active_superblock` with `block_scaling="none"`/`gamma=0` before keeping RMS as default.

**2. [High] Use Fold-Local Lambda Grids With Adaptive Boundary Expansion**
- Change: select dimensionless `lambda`; per fold use `alpha_f = lambda * trace(K_f)/n_f`, final refit uses `alpha = lambda * trace(K_full)/n`. Start `10**linspace(-10, 8, 80)`; if best is endpoint, expand and rerun; then refine within +/-1 decade.
- Where: `solvers.make_alpha_grid`, `selection.cv_score_alphas`, `selection.cv_score_active_alphas`, `estimators._build_alpha_grid_from_data`.
- Test: identity-only still matches sklearn Ridge over explicit alphas; boundary test forces expansion. Benchmark: add `lambda`, `alpha_index`, `alpha_at_boundary`.
- Risk: more CV time; if CV is noisy, wider grids can select tiny alphas and overfit.

**3. [High] Implement Real CV Scoring: Pooled MSE, RMSE, Repeats, Selection Rule**
- Change: honor `scoring`; add `scoring="mse_pooled"` default candidate, `rmse_mean` current behavior, `cv_repeats`, and `selection_rule={"min","one_se_simpler","one_se_less_regularized"}`. Use pooled squared errors before sqrt.
- Where: `selection.cv_score_alphas`, `selection.cv_score_active_alphas`, `estimators.__init__`.
- Test: unequal fold-size fixture where mean-RMSE and pooled-MSE choose different alphas. Benchmark: compare `alpha`, `cv_score`, `rmsep/ref_rmse_ridge`.
- Risk: standard 1-SE chooses larger alpha and may worsen the current mean-collapse symptom; do not make it default until curves are inspected.

**4. [High] Family-Balanced Active Screening**
- Change: replace pure top-score active selection with top-k per family plus kernel-target alignment: `align_b = <K_b, YY^T> / (||K_b||_F ||YY^T||_F)`. Keep at least one smoother, one detrend, one derivative, identity optional.
- Where: `selection.screen_active_operators`.
- Test: fold-local no-leakage spy test; fixture where a useful smoother survives despite derivative scores. Benchmark: `active_operator_names`, `rmsep/ref_rmse_ridge`.
- Risk: family quotas can keep weak blocks; use small quotas and diversity pruning.

**5. [High] Add `default`, `family_pruned`, and `response_dedup` Smoke Variants**
- Change: run compact plus `family_pruned` and `response_dedup`; use `global` and active first, superblock only after scale-power is added.
- Where: `aompls.banks`, `benchmarks/run_aomridge_benchmark.py`.
- Test: bank size/identity dedupe tests; benchmark schema includes bank name. Benchmark: wins vs `ref_rmse_ridge`, selected operator names.
- Risk: larger banks increase multiple-comparison bias and runtime; prefer family-pruned/dedup before full default.

**6. [High] Fold-Local Feature Standardisation**
- Change: implement `x_scale={"none","feature_std","feature_rms"}`. For identity, match `StandardScaler + Ridge`; for coef, map back with `beta_raw = beta_scaled / sigma`.
- Where: `estimators.fit`, `_fold_local_kernels`, likely a small preprocessing helper module.
- Test: identity-only equivalence to sklearn pipeline; fold-local scaler no-leak test. Benchmark: `Ridge-raw-feature_std`, `AOMRidge-*-feature_std`.
- Risk: scaling before derivatives changes derivative meaning; evaluate both pre-operator and post-block scaling.

**7. [High] Strict-Linear Multi-Bank Stacking**
- Change: support branch list like raw, row-detrended, derivative bank: `K = sum_m sum_b s_mb^2 Z_mb Z_mb^T`, where every `T_m` is fixed linear in Phase 1.
- Where: new branch kernel helpers beside `kernels.py`, estimator branch option, benchmark variants.
- Test: explicit concatenation equivalence; branch labels preserved. Benchmark: `AOMRidge-linear-branches-compact`.
- Risk: duplicate correlated blocks can recreate superblock over-regularisation unless paired with scale-power or supervised weights.

**8. [Medium/High] Supervised Block Weights / MKL-Light**
- Change: after normalizing base kernels, learn `w_b >= 0`, `sum w_b = 1`; start with alignment weights, then coordinate-search CV over top blocks. Kernel: `K = sum_b w_b K_b`.
- Where: `selection.py` new `select_supervised_weights`; `kernels.linear_operator_kernel_train` accepts weights.
- Test: weights learned inside each fold only; synthetic fixture selects the informative block. Benchmark: `block_weights`, `rmsep/ref_rmse_ridge`.
- Risk: high overfit risk on n=40-200; constrain top_m and add identity floor.

**9. [Medium/High] Per-Operator Alpha Grids for `global`**
- Change: for `global`, each operator should score its own lambda/grid from its own fold kernel, not a shared identity absolute-alpha grid.
- Where: `estimators.fit` global branch, `selection.select_global`.
- Test: high-gain operator fixture where shared grid fails but per-operator lambda succeeds. Benchmark: `AOMRidge-global-compact-none`, selected alpha/operator.
- Risk: more computation and slightly more selection flexibility.

**10. [Medium] OOF Ridge Expert Stacking**
- Change: train per-operator Ridge experts with own alpha, collect OOF predictions, fit nonnegative meta weights, refit experts on full train.
- Where: new estimator/selection mode, benchmark runner.
- Test: OOF predictions never include validation rows; meta weights sum/constraints. Benchmark: `AOMRidge-oof-experts-*`.
- Risk: meta-learner overfits small datasets; require NNLS or ridge with very few experts.

**11. [Medium] Phase-2 Nonlinear Branch Preprocessing**
- Change: add true SNV, MSC, EMSC/log-reflectance branches: `Z_b = T_b.fit(train).transform(X)`, with fold-local fitted references. This is outside strict-linear coef guarantees.
- Where: new branch transformer module; benchmark variant `AOMRidge-branches-raw-snv-msc`.
- Test: no-leakage tests for fitted MSC reference; prediction works without `coef_` for nonlinear branches. Benchmark: direct `rmsep/ref_rmse_ridge`.
- Risk: violates Phase-1 strict-linear scope and original-space `coef_`; expose as branch-kernel mode only.

**12. [Medium] Add Diagnostics Required To Debug Over-Regularisation**
- Change: write `relative_rmsep_vs_ref_ridge`, `alpha_index`, `alpha_boundary`, `cv_min_score`, per-block traces, scale summary, selected lambda, and fold score variance.
- Where: `estimators._build_diagnostics`, `benchmarks/run_aomridge_benchmark.py`, summarizer.
- Test: CSV schema test; diagnostic values finite. Benchmark: use these columns to rank backlog experiments.
- Risk: schema churn; keep old columns for compatibility.