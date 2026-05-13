# AOM paper review: experiments and reviewer risks

Date: 2026-05-13.

Scope reviewed: `bench/AOM`, `bench/AOM_v0`, `bench/AOM_v0/Ridge`, and
`bench/tabpfn_paper`, excluding files whose path contains `synthetic`.
No benchmark was launched for this review.

## Talanta repositioning update

The target journal is now **Talanta**. This changes the bar: the manuscript
must read less like an algorithmic chemometrics note and more like an advance
in analytical NIRS calibration development. The central story should be:

> preprocessing selection is a critical part of analytical method development,
> and AOM converts it from an external trial-and-error pipeline search into a
> model-internal, auditable calibration component.

For Talanta, the paper should foreground:

- analytical calibration, not only PLS/Ridge implementation;
- external validation and leakage-safe preprocessing;
- robustness and reproducibility across heterogeneous matrices and traits;
- development cost and refit cost;
- traceability of selected operators and original-wavelength coefficients;
- deployability through `nirs4all`, `nirs4all Studio`, and planned standalone
  AOM-PLS/AOM-Ridge packages.

The manuscript has been updated in this direction, but the experiments below
remain necessary before a defensible Talanta submission.

## Talanta P0 experiments and tables

These should be completed before submission to Talanta.

1. **Freeze a single Talanta cohort manifest.** Include dataset group, dataset,
   analytical domain, trait/response type, split type, `n_train`, `n_test`,
   `p`, inclusion/exclusion reason, and source baseline file. This is the most
   important anti-reviewer-risk item.

2. **Regenerate all headline comparisons from that manifest.** Required rows:
   PLS-HPO, Ridge-HPO, CatBoost, CNN-1D, AOM-PLS compact CV-5, AOM-Ridge
   Blender, and clearly separated AOM-Ridge oracle envelope.

3. **Add paired statistics on the common subset.** Minimum Talanta table:
   `Comparison`, `N`, `median ratio/delta`, `bootstrap 95% CI`, `wins`,
   `Wilcoxon p`, `Holm-adjusted p`, and an effect-size column. Current draft
   values exist for AOM-PLS compact vs PLS and AOM-Ridge Blender vs Ridge, but
   they must be regenerated after final cohort freeze.

4. **Add benchmark diversity table in final form.** Current manuscript has a
   first version. Final table should include analytical domains, response
   families, min/median/max sample size, min/median/max wavelength variables,
   split strategies, and number of external-test datasets per comparison.

5. **Add operator-selection patterns.** Talanta reviewers will expect evidence
   that AOM is interpretable in practice. Provide selected operator frequencies
   for AOM-PLS compact, selected component counts, and Ridge selected operator
   or kernel-family frequencies.

6. **Add robustness-to-search-complexity results.** Formalize compact vs
   family-pruned vs response-dedup vs default/deep-bank behavior. The story is
   important analytically: larger preprocessing spaces can worsen external
   robustness through selection variance.

7. **Add failure-case table.** Include QUARTZ exclusion, y-based splits,
   outlier-sensitive cases, missing CNN rows, large-N compute limitations, and
   datasets where ASLS/SNV branches harm performance.

8. **Add software validation plan.** The future clean-room C++ implementation
   for R and Python package `AOM-PLS`, plus planned pip AOM-PLS/AOM-Ridge
   packages, should be validated against the reference NumPy/PyTorch
   implementation on identity, fixed-operator, covariance/SIMPLS, and
   prediction-equivalence tests.

## Talanta P1 experiments

These are strongly recommended for revision robustness.

1. **Multi-seed full-cohort rerun.** At least seeds 0/1/2 for final deployable
   AOM-PLS and AOM-Ridge variants and the key linear baselines.

2. **Strict-linear vs branch ablation.** Report strict AOM only, ASLS branch,
   SNV branch, MSC/EMSC branches where stable, and no-ASLS variants. This
   prevents branch preprocessing gains from being attributed solely to strict
   operator algebra.

3. **Runtime and memory table.** Median/q90 fit time, prediction time, OOM or
   skipped count, and hardware. Talanta will care because routine analytical
   recalibration cost is part of the claim.

4. **Chemometric metrics table.** In addition to RMSEP ratios, report RMSECV,
   MAE, R2, bias and, where response scale permits, RPD/RPIQ or justify their
   exclusion across heterogeneous response scales.

5. **Prediction artifacts.** Save fold predictions and final test predictions
   for all final rows to enable residual plots and independent audit.

6. **Deployment examples.** Add a short, reproducible use case showing how
   selected operators, coefficients and prediction artifacts are exported from
   `nirs4all`/Studio or the future `AOM-PLS` package.

## Executive verdict

The current material can support a Talanta-oriented analytical calibration
manuscript if the claims are frozen around deployable variants and documented
search budgets. The main blocker is not implementation volume; it is claim
control. The local sources mix at least four different empirical stories:

- early AOM-PLS prototypes on 5 datasets in `bench/AOM/report.md`;
- AOM_v0/AOM-PLS selection-bank experiments on roughly 57 datasets in
  `bench/AOM_v0/Summary.md`;
- a curated 39-dataset AOM-Ridge manuscript pipeline in
  `bench/AOM_v0/Ridge/REPRODUCIBILITY.md` and
  `bench/AOM_v0/Ridge/benchmark_runs/curated_v2/results.csv`;
- a later 52-paired-dataset AOM-Ridge combined analysis in
  `bench/AOM_v0/Ridge/benchmark_runs/all54_combined/results.csv` and
  `bench/AOM_v0/Ridge/publication/tables/`.

For Talanta, the safest path is to present one frozen analytical-calibration
story: compact AOM-PLS as the primary deployable result, AOM-Ridge Blender as a
complementary deployable result, and Ridge oracle envelopes only as secondary
headroom analyses. A general AOM paper that merges AOM-PLS, AOM_v0, AOM-Ridge,
foundation-model comparisons, oracles, branch preprocessors, and classification
without strict cohort control would invite reviewer objections about multiple
testing, cohort drift, and retrospective selection.

## Local evidence map

| Path | What it contains | Review implication |
| --- | --- | --- |
| `bench/AOM/report.md` | 5-dataset comparison of baseline AOM-PLS vs Bandit, DARTS, Zero-Shot Router, and MoE. | Useful background only; too small and prototype-heavy for headline evidence. |
| `bench/AOM/PUBLICATION_PLAN.md` | Architecture argument for deployed AOM-PLS, including hard-gate holdout selection and known `n_components` inconsistency. | Good methods narrative, but it documents unresolved API and validation issues. |
| `bench/AOM/PUBLICATION_BACKLOG.md` | Workstreams W1-W8 requiring selection-criterion head-to-heads, holdout removal, API split, custom banks, stats, and drafting. | Many "must land before paper" items are still listed as required. |
| `bench/AOM_v0/Summary.md` | 57-dataset AOM_v0 conclusion: ASLS + compact bank + CV5 is the current champion; compact banks beat larger banks due to selection variance. | Strongest AOM-PLS empirical story, but it blends strict-linear AOM with non-linear upstream preprocessing. |
| `bench/AOM_v0/benchmark_runs/full/results.csv` | 7,888 rows, 59 datasets, 1 seed (`seed=0`, `run_seed=0`), statuses `ok` and `skipped`. Prediction path columns are empty. | Broad coverage but insufficient uncertainty and incomplete artifacts for audit. |
| `bench/AOM_v0/Multi-kernel/docs/BENCHMARK_PROTOCOL.md` | Planned 5-seed regression/classification protocol, master-result joins, Friedman/Nemenyi, Wilcoxon, bootstrap CIs. | The intended statistical protocol is stricter than most completed local outputs. |
| `bench/AOM_v0/Multi-kernel/docs/CODEX_REVIEWS.md` | Prior high-severity fixes and remaining deferred issues: classification CV criterion, supervised-operator CV leakage, ignored `scale`, sklearn validation. | Reviewers may ask whether these deferred issues affect current claims. |
| `bench/AOM_v0/Ridge/docs/BENCHMARK_PROTOCOL.md` | AOM-Ridge benchmark protocol and required schema: RMSEP, MAE, R2, relative RMSEP, time, status, alpha, operator names. | Good protocol skeleton; final claims need to align exactly with it. |
| `bench/AOM_v0/Ridge/docs/AOM_RIDGE_MATH_SPEC.md` | Strict linear Ridge math, fold-local block scaling, alpha grid, global/superblock/active variants, branch warning. | Useful for methods rigor; must separate strict-linear claims from branch preprocessors. |
| `bench/AOM_v0/Ridge/docs/HEADLINE_SPXY3_NESTED_AUDIT.md` | Anti-leakage audit for AutoSelect and Blender; verdict nested, but caveats include single seed, 53/57 coverage, duplicate source runs, promotion gate pending. | Strong anti-leakage evidence, but explicitly labels the headline variants exploratory until multi-seed/full evidence lands. |
| `bench/AOM_v0/Ridge/docs/D_A_001_FAST12_PAIRED_STATS.md` | 12-dataset, 3-seed stats for Blender/AutoSelect; several comparisons fail Holm-adjusted thresholds. | Good pilot, not enough for broad headline claims. |
| `bench/AOM_v0/Ridge/docs/D_A_001_AUDIT20_PAIRED_STATS.md` | 20-dataset, 3-seed stats; AutoSelect vs Ridge strong, but comparisons vs ASLS-AOM often no-win after Holm; Blender has extreme Quartz ratios. | Useful audit-tier evidence; still not full-cohort. |
| `bench/AOM_v0/Ridge/REPRODUCIBILITY.md` | Reproducibility guide claims 39-dataset curated cohort, 96 tests, 1.5h benchmark, QUARTZ dropped. | Inconsistent with later manuscript/tables using 52 paired datasets and 231 tests. |
| `bench/AOM_v0/Ridge/publication/manuscript/aomridge_paper.tex` | Manuscript draft: intro says 39 datasets and 32/38 wins; results later say 52 paired datasets, Blender 35/52, oracle 45/52. | Needs consistency pass before Talanta submission. |
| `bench/AOM_v0/Ridge/publication/tables/table_summary.tex` | Oracle envelope vs baselines: 45/52 vs Ridge, median -4.73%; 27/52 vs TabPFN-opt, median -0.21%. | This is oracle, not deployable; cannot be the primary practitioner claim. |
| `bench/AOM_v0/Ridge/publication/tables/table_per_method_summary.tex` | Deployable variant ranking: Blender 35/52, median -2.22%; AutoSelect 27/52, median -0.61%. | Best current deployable Ridge claim if cohort and stats are cleaned. |
| `bench/tabpfn_paper/scan_results.json` | 65 loaded regression dataset cards, 0 failures; paths are Windows-style from a prior machine. | Useful dataset metadata, but path provenance should be normalized for reproducibility. |
| `bench/tabpfn_paper/run_reg_pls.py` | Full TabPFN-paper-like pipeline with SPXYFold(3), 25 PLS trials, 60 Ridge trials, 50 CNN trials, 20 TabPFN estimators; dataset list partly commented. | Confirms reference protocol complexity; not itself a frozen result source. |
| `bench/tabpfn_paper/run_reg_aom.py` | AOM-PLS script with 200 Optuna trials over `n_components`, `n_orth`, and `operator_index`; current active list includes only two LUCAS datasets marked skipped too big. | Not suitable as current full AOM evidence unless rerun/frozen. |
| `bench/1_master_results.csv` | 335 rows, 61 regression splits, 6 baselines, seed 42. | Actual local TabPFN baseline table; note docs sometimes cite a missing `bench/tabpfn_paper/master_results.csv`. |
| `bench/benchmark_master_results.csv` | 25,936 unified records across regression/classification, 446 variants, mixed statuses/seeds/maturity. | Good registry, but too heterogeneous for direct paper statistics without filtering rules. |

## Blocking uncertainties

1. **Paper object is not yet uniquely defined.** `bench/AOM/PUBLICATION_PLAN.md`
   argues for deployed AOM-PLS. `bench/AOM_v0/Summary.md` argues for ASLS +
   compact AOM-PLS with CV5. `bench/AOM_v0/Ridge/publication/manuscript/`
   argues for AOM-Ridge. A reviewer will ask what the contribution is:
   operator-adaptive PLS, operator-adaptive Ridge, a meta-selector, or a
   benchmark of preprocessing strategies.

2. **Cohort denominators drift across files.** Local sources mention 5, 20, 39,
   52, 54, 57, 59, 61, and 65 datasets. This is explainable by filters, but
   the manuscript cannot leave it implicit. The largest visible contradiction
   is in `bench/AOM_v0/Ridge/publication/manuscript/aomridge_paper.tex`: the
   introduction reports 39 datasets and 32/38 wins, while the results/tables
   report 52 paired datasets and 35/52 or 45/52 wins.

3. **Oracle and deployable claims are too easy to conflate.** The Ridge paper
   explicitly reports an oracle envelope in
   `publication/tables/table_summary.tex` and a deployable Blender in
   `publication/tables/table_per_method_summary.tex`. The oracle uses
   retrospective test-set knowledge and must not be presented as a method
   result. The primary claim should be Blender 35/52, median -2.22% vs paper
   Ridge HPO, not oracle 45/52, median -4.73%.

4. **Full-cohort seed uncertainty is missing.** The broad AOM_v0 result CSV has
   a single seed. Ridge `all54_combined/results.csv`, `curated_v2/results.csv`,
   and `final_curated/results.csv` also show one `random_state=0`. Multi-seed
   evidence exists only for subsets such as `da001_partial_fast12_seeds012` and
   `da001_audit20_seeds012`.

5. **Some published-style claims already have non-winning paired tests.** In
   `D_A_001_FAST12_PAIRED_STATS.md`, Blender vs Ridge-tuned-cv5 is `NO_WIN`
   after Holm despite median -13.55%; AutoSelect vs Ridge-tuned-cv5 is also
   `NO_WIN`. In `D_A_001_AUDIT20_PAIRED_STATS.md`, Blender and AutoSelect often
   fail vs `ASLS-AOM-compact-cv5-numpy` after Holm. This does not kill the
   method, but it blocks overbroad "beats everything" wording.

6. **Selection protocol is a central vulnerability.** AOM-PLS sources document
   that the old 20% holdout is arbitrary and high variance
   (`bench/AOM/PUBLICATION_BACKLOG.md`). AOM_v0 says CV5 is essential for the
   ASLS gain (`bench/AOM_v0/Summary.md`). Ridge says SPXY3 nested selection is
   leakage-safe but high variance (`HEADLINE_SPXY3_NESTED_AUDIT.md`). The paper
   needs one final selection policy and a clear reason it is not tuned to the
   test set.

7. **Strict-linear vs non-linear preprocessing boundaries are blurred.** AOM's
   clean math is for fixed strict-linear operators. But the strongest AOM_v0
   story uses ASLS and sometimes SNV/MSC/EMSC/OSC branches. Ridge math
   correctly says these are branch transformers, not fixed `A_b` operators
   (`docs/AOM_RIDGE_MATH_SPEC.md`). Claims must state which gains are strict
   AOM and which are branch/preprocessing ensemble gains.

8. **Artifact-level reproducibility is incomplete.** `bench/AOM_v0/benchmark_runs/full/results.csv`
   has empty prediction-path columns despite the Multi-kernel protocol requiring
   fold and final predictions. The Ridge runs have CSVs and tables, but selector
   diagnostics are explicitly not surfaced in the results schema according to
   `D_A_001_FAST12_PAIRED_STATS.md` and `D_A_001_AUDIT20_PAIRED_STATS.md`.

9. **Known failure cases require quantified handling.** The Ridge discussion
   already identifies y-based split miscalibration and SNV-on-outlier failures.
   A reviewer will expect per-failure tables: AMYLOSE y-based splits, BEER
   `YbaseSplit`, TIC, Quartz degeneracy, LUCAS compute exclusion, and giant
   ECOSIS behavior.

10. **Reference baseline provenance is fragile.** Protocol docs cite
    `bench/tabpfn_paper/master_results.csv`, but the local baseline file is
    `bench/1_master_results.csv`; the unified registry is
    `bench/benchmark_master_results.csv`. The paper should name exactly which
    file generated every table.

## Metrics missing before a defensible Talanta submission

- **Full-cohort uncertainty:** bootstrap 95% confidence intervals for median
  delta RMSEP, win-rate intervals, and per-dataset paired seed distributions.
- **Full-cohort significance:** Friedman/Nemenyi ranks and paired Wilcoxon or
  Nadeau-Bengio corrected tests on the same final cohort used in the headline.
- **Multi-seed stability:** at least seeds 0/1/2 for the final deployable
  variants and baselines on the full paired cohort, not only fast12/audit20.
- **Selector diagnostics:** AutoSelect chosen-candidate counts, Blender weights
  mean/std, OOF fold RMSE variance, and cases where the selected candidate
  differs across seeds.
- **Failure-case metrics:** separate table for y-based splits, outlier splits,
  tiny-`n` splits, giant-`n` splits, and degenerate-reference exclusions.
- **No-harm metrics:** q75/q90/worst relative RMSEP for each headline variant,
  with named worst datasets and explicit exclusion policy for Quartz-like
  denominators.
- **Compute metrics:** median and q90 fit time, predict time, and memory/OOM
  counts per variant; the Ridge manuscript currently mentions 46-72 minutes on
  ALPINE and 4-4.5 hours on giant datasets, but this needs tabular support.
- **Calibration-style chemometrics metrics:** RMSEC, RMSECV, RMSEP, SEP, MAE,
  R2, bias, RPD/RPIQ where available. `early_results.txt` shows the TabPFN
  source reports these; the AOM paper should either align or justify not doing
  so.
- **Alpha/grid diagnostics for Ridge:** alpha boundary rate, grid expansions,
  one-SE selection frequency, and effective component distributions. Columns
  exist in Ridge CSVs but are not summarized in the paper tables.
- **Artifact integrity:** links or paths to final predictions, fold predictions,
  configs, exact cohort manifests, and generated figures/tables. Empty
  prediction path columns in AOM_v0 full results need a note or regeneration.

## Experiments to redo or add

### P0: required before posting today

1. **Freeze one paper scope.** Choose either AOM-PLS/AOM_v0 or AOM-Ridge. Do
   not submit a broad "general AOM" paper unless all mixed claims are
   explicitly downgraded to related work.

2. **Create one final cohort manifest.** For the chosen paper, produce a CSV
   with dataset group, dataset, task, n_train, n_test, n_features, split type,
   inclusion/exclusion reason, and source baseline file. Use one denominator
   throughout.

3. **Regenerate paper tables from one results file.** For Ridge, use
   `bench/AOM_v0/Ridge/benchmark_runs/all54_combined/results.csv` if the
   52-paired-dataset story is final, or revert the manuscript to the 39-dataset
   `curated_v2` story. Do not mix them.

4. **Separate deployable vs oracle tables.** The deployable table must lead.
   The oracle table should be labeled "upper bound, retrospective test-set
   selection" in caption and text.

5. **Add a claim ledger.** For every abstract/introduction claim, record:
   result file, cohort size, variant, baseline, metric, seed count, and whether
   it is deployable or oracle.

6. **Add an explicit exclusion table.** At minimum: Quartz due to degenerate
   paper Ridge RMSEP; LUCAS Cropland due to compute for wrappers; any missing
   CNN baselines; any skipped/failed rows in AOM_v0 full results.

7. **Aggregate available subset stats into the manuscript.** Use the existing
   fast12/audit20 3-seed results as "audit subsets", not as full-cohort proof.
   State where Holm-adjusted tests are no-win.

8. **Normalize TabPFN baseline provenance.** Replace references to missing
   `bench/tabpfn_paper/master_results.csv` with the actual source used:
   `bench/1_master_results.csv` or `bench/benchmark_master_results.csv`.

9. **Downgrade unsupported language.** Avoid "best current", "dominates",
   "state of the art", and "beats TabPFN-opt" unless the exact deployable
   paired result supports it. For Ridge, current deployable Blender loses to
   TabPFN-opt at the median in the manuscript text (+4.72%).

10. **Insert failure-mode table.** The discussion already names y-based splits
    and SNV-on-outliers; add rows with exact datasets, selected variant/branch,
    baseline winner, and delta.

### P1: should be done for a robust revision

1. **Full paired multi-seed rerun for the final deployable variants.** Minimum:
   seeds 0/1/2 on the final 52/57/61 cohort for Blender, AutoSelect, strongest
   single AOM variant, raw/tuned Ridge, PLS, ASLS-AOM compact CV5, TabPFN-Raw,
   and TabPFN-opt where feasible.

2. **Selection-criterion ablation.** For AOM-PLS/AOM_v0: holdout20 vs CV3 vs
   CV5 vs repeated CV3 vs PRESS on the same cohort. `bench/AOM_v0/Summary.md`
   says PRESS underperforms and CV5 is important; make that a formal table.

3. **Bank-size ablation with uncertainty.** Compact vs family-pruned vs
   response-dedup vs default vs deep banks, with the multiple-comparison
   explanation tested by seed stability and selected-operator entropy.

4. **Non-linear branch ablation.** Strict-linear AOM only, SNV upstream, ASLS
   upstream, MSC, EMSC, OSC, and branch ensembles. Report gains separately so
   strict AOM claims are not inflated by branch preprocessing.

5. **Leakage audit for every final variant.** Ridge has a strong audit for
   AutoSelect/Blender; the final AOM-PLS/AOM_v0 path needs a similarly concise
   fold-local audit, especially for ASLS/SNV/MSC/EMSC/OSC branches.

6. **Prediction artifact export.** Save final and fold predictions for the
   final runs. This enables residual plots, paired tests, bias/RPD tables, and
   independent audit.

7. **Runtime scaling experiment.** Small representative benchmark over
   increasing `n`, `p`, bank size, and branch count. AOM-Ridge wrappers are
   expensive enough that reviewers will ask about practical limits.

8. **Robustness to split type.** Stratify results by KS, SPXY, random,
   group/cultivar, block2deg, species, y-based, and random split. The known
   y-based failure mode should become an analysis, not a caveat only.

9. **Reference-baseline verification.** Recompute at least PLS and Ridge on the
   frozen cohort under the same split and scaling rules. Treat TabPFN/CNN/CatBoost
   as imported baselines only if their provenance is fully documented.

10. **Classification either removed or finished.** Multi-kernel docs include a
    classification plan and deferred classification CV issues. If the paper is
    regression-only, remove classification claims. If classification remains,
    rerun with balanced accuracy, macro-F1, log loss, ECE, label encoding, and
    stratified fold-local calibration.

## Reviewer risk register

| Severity | Likely reviewer concern | Local evidence | Required mitigation |
| --- | --- | --- | --- |
| High | "The headline result is retrospective/oracle, not deployable." | Oracle table in `table_summary.tex`; deployable table in `table_per_method_summary.tex`. | Lead with deployable Blender/AutoSelect; label oracle as upper bound only. |
| High | "The paper changes dataset denominator midstream." | `aomridge_paper.tex` intro 39/38; results 52; `REPRODUCIBILITY.md` 39; `all54_combined` 54 raw datasets. | One cohort manifest and consistency pass across abstract, intro, captions, results. |
| High | "Single-seed full-cohort results are not enough." | `all54_combined`, `curated_v2`, `final_curated`, and AOM_v0 full all show one random state/seed. | Full multi-seed or downgrade to exploratory with subset audit evidence. |
| High | "Claims are overfit to a local benchmark through many variants." | `benchmark_master_results.csv` has 446 variants; Ridge oracle uses 29 benched variants. | Pre-register final variants; report all tried families; use held-out/frozen claim ledger. |
| High | "Strict-linear math does not cover ASLS/SNV/MSC/EMSC/OSC gains." | `AOM_RIDGE_MATH_SPEC.md` says these are branch transformers, not fixed `A_b`. | Split strict-linear AOM claims from branch ensemble claims. |
| High | "Known no-win statistical tests contradict headline wording." | `D_A_001_FAST12_PAIRED_STATS.md` and `D_A_001_AUDIT20_PAIRED_STATS.md`. | Include subset stats honestly; avoid universal superiority language. |
| Medium | "Selection CV is miscalibrated on y-based splits." | Ridge discussion and AOM_v0 Summary both identify this. | Add split-type stratified results and failure table. |
| Medium | "Quartz and degenerate denominators distort relative metrics." | `REPRODUCIBILITY.md`, audit20 worst ratios, manuscript table captions. | Exclusion policy plus absolute RMSEP deltas for degenerate cases. |
| Medium | "Compute cost makes the method impractical." | Manuscript says Blender can take 72 min on ALPINE and 4.5h on giants. | Runtime table, recommended fast variant, and compute-budget guidance. |
| Medium | "Reference baselines are imported, not rerun under identical conditions." | TabPFN scripts and `bench/1_master_results.csv` use seed 42; Ridge uses local random_state 0. | Clarify imported baselines; rerun PLS/Ridge locally for sanity. |
| Medium | "Prediction artifacts and configs are missing." | Empty path columns in `bench/AOM_v0/benchmark_runs/full/results.csv`. | Export artifacts or document why unavailable. |
| Medium | "Deferred code-review issues may affect results." | `CODEX_REVIEWS.md` lists deferred classification CV, supervised-operator leakage, ignored scale. | State which final variants are unaffected; fix or remove affected scopes. |
| Low | "Windows paths in dataset cards hurt reproducibility." | `bench/tabpfn_paper/scan_results.json` paths start with `D:\nirs4all`. | Regenerate cards on current workspace or store relative paths. |

## Minimum acceptable Talanta framing

If submitting the current draft trajectory, use conservative wording:

- "We propose operator-adaptive analytical calibration and evaluate frozen
  deployable PLS and Ridge variants on a heterogeneous NIRS benchmark."
- "The best deployable Ridge aggregator, Blender, improves over paper Ridge HPO
  on 35/52 paired datasets with median -2.22% RMSEP; an oracle envelope shows
  additional headroom but is not deployable."
- "Multi-seed audit subsets support the trend but full-cohort multi-seed
  confirmation is ongoing."
- "Gains from ASLS/SNV/MSC/EMSC/OSC branches are reported separately from the
  strict-linear operator contribution."

Avoid:

- "AOM beats TabPFN-opt" for deployable Ridge; the manuscript text reports a
  median loss vs TabPFN-opt for Blender.
- "State of the art" unless the final deployable result beats all imported and
  rerun baselines under the same frozen protocol.
- "Leakage-free" globally unless every final branch and selector path has the
  same audit quality as `HEADLINE_SPXY3_NESTED_AUDIT.md`.
- Any headline based on the oracle envelope without saying "retrospective upper
  bound requiring test-set knowledge."

## Files read as primary evidence

- `bench/AOM/report.md`
- `bench/AOM/PUBLICATION_PLAN.md`
- `bench/AOM/PUBLICATION_BACKLOG.md`
- `bench/AOM_v0/README.md`
- `bench/AOM_v0/Summary.md`
- `bench/AOM_v0/benchmark_runs/full/results.csv`
- `bench/AOM_v0/Multi-kernel/docs/BENCHMARK_PROTOCOL.md`
- `bench/AOM_v0/Multi-kernel/docs/CODEX_REVIEWS.md`
- `bench/AOM_v0/Ridge/README.md`
- `bench/AOM_v0/Ridge/REPRODUCIBILITY.md`
- `bench/AOM_v0/Ridge/docs/BENCHMARK_PROTOCOL.md`
- `bench/AOM_v0/Ridge/docs/AOM_RIDGE_MATH_SPEC.md`
- `bench/AOM_v0/Ridge/docs/HEADLINE_SPXY3_NESTED_AUDIT.md`
- `bench/AOM_v0/Ridge/docs/D_A_001_FAST12_PAIRED_STATS.md`
- `bench/AOM_v0/Ridge/docs/D_A_001_AUDIT20_PAIRED_STATS.md`
- `bench/AOM_v0/Ridge/benchmark_runs/all54_combined/results.csv`
- `bench/AOM_v0/Ridge/benchmark_runs/curated_v2/results.csv`
- `bench/AOM_v0/Ridge/benchmark_runs/final_curated/results.csv`
- `bench/AOM_v0/Ridge/benchmark_runs/da001_partial_fast12_seeds012/results.csv`
- `bench/AOM_v0/Ridge/benchmark_runs/da001_audit20_seeds012/results.csv`
- `bench/AOM_v0/Ridge/publication/manuscript/aomridge_paper.tex`
- `bench/AOM_v0/Ridge/publication/tables/table_summary.tex`
- `bench/AOM_v0/Ridge/publication/tables/table_per_method_summary.tex`
- `bench/tabpfn_paper/early_results.txt`
- `bench/tabpfn_paper/scan_results.json`
- `bench/tabpfn_paper/scan_datasets.py`
- `bench/tabpfn_paper/run_reg_pls.py`
- `bench/tabpfn_paper/run_reg_aom.py`
- `bench/1_master_results.csv`
- `bench/benchmark_master_results.csv`
- `bench/master_results_classif.csv`
