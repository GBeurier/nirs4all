# Transfer Validation

## Status

`BLOCKED_BY_REALISM_GATE`

Phase B4 transfer validation is gate-first. The B2/B3 realism evidence was read before any synthetic build, real-only fit, TSTR route, or RTSR diagnostic.

## Command

`PYTHONPATH=bench/nirs_synthetic_pfn/src python bench/nirs_synthetic_pfn/experiments/exp03_transfer_validation.py --max-real-datasets 3 --max-samples 180 --n-splits 2 --test-fraction 0.25 --n-synthetic-samples 80 --seed 20260429`

## Outputs

- Markdown: `bench/nirs_synthetic_pfn/reports/transfer_validation.md`
- CSV metrics summary: `bench/nirs_synthetic_pfn/reports/transfer_validation.csv`

## B2/B3 Provenance

- B2 scorecards CSV inspected: `bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv`
- B2 exists: `True`
- B2 raw compared rows: 71
- B2 raw smoke failures: 71
- B2 raw blocked rows: 6
- B2 raw missing AUC rows: 0
- B3 adversarial AUC CSV inspected: `bench/nirs_synthetic_pfn/reports/adversarial_auc.csv`
- B3 exists: `True`
- B3 raw gate status: `NO-GO`
- B3 raw compared rows: 71
- B3 raw smoke failures: 71
- B3 raw blocked evidence gaps: 6
- B3 raw missing AUC rows: 0

## Raw Authoritative Gate

- Raw evidence is authoritative for this gate.
- SNV evidence is diagnostic only and cannot pass or override the gate.
- CSV row raw_compared: 71
- CSV row raw_smoke_failures: 71
- CSV row raw_blocked: 6
- CSV row raw_missing_auc: 0
- Blocking reasons: `adversarial_auc_raw_gate_NO-GO;B2_raw_realism_gate_failed`

## No Integration Or Transfer Claim

- No integration readiness is claimed.
- No transfer claim is made.
- Synthetic generation count: 0.
- Fitted model count: 0.
- Real-only baseline fit count: 0.
- TSTR/RTSR route count: 0.
- exp02 was not launched by this experiment.

## Next Actions

- Remediate B2 raw failures and evidence gaps before rerunning transfer validation.
- Prioritize named B2 gaps: BEER, DIESEL, and CORN.
- Re-run the B2 scorecards and B3 adversarial AUC audit after remediation.
- Only revisit transfer validation after the raw authoritative realism gate passes.

## Git Status

- `git status --short` lines: 67
- First entries:
  - ` M bench/AOM_v0/Ridge/aomridge/estimators.py`
  - ` M bench/AOM_v0/Ridge/aomridge/kernels.py`
  - ` M bench/AOM_v0/Ridge/aomridge/selection.py`
  - ` M bench/AOM_v0/Ridge/benchmark_runs/smoke/results.csv`
  - ` M bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py`
  - ` M bench/AOM_v0/Ridge/docs/IMPLEMENTATION_LOG.md`
  - ` M bench/AOM_v0/Ridge/tests/test_ridge_cv_no_leakage.py`
  - ` M bench/AOM_v0/aompls/estimators.py`
  - ` M bench/AOM_v0/aompls/preprocessing.py`
  - ` M bench/AOM_v0/aompls/scorers.py`
  - ` M bench/AOM_v0/aompls/selection.py`
  - ` M bench/AOM_v0/benchmark_runs/full/results.csv`
  - ` M bench/AOM_v0/benchmarks/run_aompls_benchmark.py`
  - ` M bench/AOM_v0/benchmarks/run_extended_benchmark.py`
  - ` M bench/AOM_v0/publication/tables/relative_rmsep_per_variant.csv`
  - ` M bench/nirs_synthetic_pfn/experiments/exp00_smoke_prior_dataset.py`
  - ` M bench/nirs_synthetic_pfn/experiments/exp02_real_synthetic_scorecards.py`
  - ` M bench/nirs_synthetic_pfn/experiments/exp03_transfer_validation.py`
  - ` M bench/nirs_synthetic_pfn/reports/integration_gate_status.md`
  - ` M bench/nirs_synthetic_pfn/reports/nirs_context_query_sampler_contract.md`
  - ` M bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv`
  - ` M bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.md`
  - ` M bench/nirs_synthetic_pfn/reports/transfer_validation.csv`
  - ` M bench/nirs_synthetic_pfn/reports/transfer_validation.md`
  - ` M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/adapters/prior_adapter.py`
  - ` M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/evaluation/realism.py`
  - ` M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/evaluation/transfer.py`
  - ` M bench/nirs_synthetic_pfn/tests/test_prior_adapter.py`
  - ` M bench/nirs_synthetic_pfn/tests/test_realism_scorecards.py`
  - ` M bench/nirs_synthetic_pfn/tests/test_transfer_validation.py`
  - `?? .claude/`
  - `?? .codex`
  - `?? bench/AOM_v0/Ridge/Makefile`
  - `?? bench/AOM_v0/Ridge/REPRODUCIBILITY.md`
  - `?? bench/AOM_v0/Ridge/aomridge/branches.py`
  - `?? bench/AOM_v0/Ridge/aomridge/cv.py`
  - `?? bench/AOM_v0/Ridge/aomridge/mkl.py`
  - `?? bench/AOM_v0/Ridge/benchmark_runs/curated/`
  - `?? bench/AOM_v0/Ridge/benchmark_runs/curated_cohort.csv`
  - `?? bench/AOM_v0/Ridge/benchmark_runs/curated_v2/`
  - `?? bench/AOM_v0/Ridge/benchmark_runs/smoke6/`
  - `?? bench/AOM_v0/Ridge/benchmark_runs/smoke_cv5/`
  - `?? bench/AOM_v0/Ridge/docs/CODEX_BACKLOG_round2_2026-04-29.md`
  - `?? bench/AOM_v0/Ridge/publication/`
  - `?? bench/AOM_v0/Ridge/tests/test_ridge_branch_global.py`
  - `?? bench/AOM_v0/Ridge/tests/test_ridge_mkl.py`
  - `?? bench/AOM_v0/Ridge/tests/test_ridge_one_se_and_repeated_cv.py`
  - `?? bench/AOM_v0/Ridge/tests/test_ridge_round3_fixes.py`
  - `?? bench/AOM_v0/docs/CV_SPLITTER_DESIGN.md`
  - `?? bench/AOM_v0/publication/tables/table_top15_score_time.tex`
  - `?? bench/nirs_synthetic_pfn/docs/06_SYNTHETIC_REALISM_REMEDIATION_ROADMAP.md`
  - `?? bench/nirs_synthetic_pfn/experiments/exp04_adversarial_auc.py`
  - `?? bench/nirs_synthetic_pfn/experiments/exp05_minimal_ablation_attribution.py`
  - `?? bench/nirs_synthetic_pfn/experiments/exp06_encoder_tabpfn_gate_precheck.py`
  - `?? bench/nirs_synthetic_pfn/experiments/exp07_nirs_icl_gate_precheck.py`
  - `?? bench/nirs_synthetic_pfn/reports/adversarial_auc.csv`
  - `?? bench/nirs_synthetic_pfn/reports/adversarial_auc.md`
  - `?? bench/nirs_synthetic_pfn/reports/encoder_tabpfn_gate.csv`
  - `?? bench/nirs_synthetic_pfn/reports/encoder_tabpfn_gate.md`
  - `?? bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.csv`
  - `?? bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.md`
  - `?? bench/nirs_synthetic_pfn/reports/nirs_icl_gate_precheck.csv`
  - `?? bench/nirs_synthetic_pfn/reports/nirs_icl_gate_precheck.md`
  - `?? bench/nirs_synthetic_pfn/tests/test_adversarial_auc_report.py`
  - `?? bench/nirs_synthetic_pfn/tests/test_encoder_tabpfn_gate_precheck.py`
  - `?? bench/nirs_synthetic_pfn/tests/test_minimal_ablation_attribution.py`
  - `?? bench/nirs_synthetic_pfn/tests/test_nirs_icl_gate_precheck.py`

## Raw Summary JSON

```json
{
  "gate_summary": {
    "adversarial_summary": {
      "exists": true,
      "gate_status": "NO-GO",
      "path": "/home/delete/nirs4all/nirs4all/bench/nirs_synthetic_pfn/reports/adversarial_auc.csv",
      "raw_blocked": 6,
      "raw_compared": 71,
      "raw_missing_auc": 0,
      "raw_rows": 77,
      "raw_smoke_failures": 71,
      "reason": "raw_smoke_failures;raw_blocked_evidence_gaps",
      "row_count": 219
    },
    "b2_summary": {
      "adversarial_auc_failures": 71,
      "b2_realism_failed": true,
      "compared_rows": 71,
      "exists": true,
      "path": "/home/delete/nirs4all/nirs4all/bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv",
      "pca_overlap_failures": 68,
      "raw_blocked": 6,
      "raw_compared": 71,
      "raw_missing_auc": 0,
      "raw_rows": 77,
      "raw_smoke_failures": 71,
      "reason": "B2_raw_realism_gate_failed",
      "row_count": 219
    },
    "blocked": true,
    "blocking_reasons": [
      "adversarial_auc_raw_gate_NO-GO",
      "B2_raw_realism_gate_failed"
    ],
    "raw_authoritative": true,
    "snv_can_pass_gate": false,
    "status": "BLOCKED_BY_REALISM_GATE"
  },
  "git_status": {
    "line_count": 67,
    "lines": [
      " M bench/AOM_v0/Ridge/aomridge/estimators.py",
      " M bench/AOM_v0/Ridge/aomridge/kernels.py",
      " M bench/AOM_v0/Ridge/aomridge/selection.py",
      " M bench/AOM_v0/Ridge/benchmark_runs/smoke/results.csv",
      " M bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py",
      " M bench/AOM_v0/Ridge/docs/IMPLEMENTATION_LOG.md",
      " M bench/AOM_v0/Ridge/tests/test_ridge_cv_no_leakage.py",
      " M bench/AOM_v0/aompls/estimators.py",
      " M bench/AOM_v0/aompls/preprocessing.py",
      " M bench/AOM_v0/aompls/scorers.py",
      " M bench/AOM_v0/aompls/selection.py",
      " M bench/AOM_v0/benchmark_runs/full/results.csv",
      " M bench/AOM_v0/benchmarks/run_aompls_benchmark.py",
      " M bench/AOM_v0/benchmarks/run_extended_benchmark.py",
      " M bench/AOM_v0/publication/tables/relative_rmsep_per_variant.csv",
      " M bench/nirs_synthetic_pfn/experiments/exp00_smoke_prior_dataset.py",
      " M bench/nirs_synthetic_pfn/experiments/exp02_real_synthetic_scorecards.py",
      " M bench/nirs_synthetic_pfn/experiments/exp03_transfer_validation.py",
      " M bench/nirs_synthetic_pfn/reports/integration_gate_status.md",
      " M bench/nirs_synthetic_pfn/reports/nirs_context_query_sampler_contract.md",
      " M bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv",
      " M bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.md",
      " M bench/nirs_synthetic_pfn/reports/transfer_validation.csv",
      " M bench/nirs_synthetic_pfn/reports/transfer_validation.md",
      " M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/adapters/prior_adapter.py",
      " M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/evaluation/realism.py",
      " M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/evaluation/transfer.py",
      " M bench/nirs_synthetic_pfn/tests/test_prior_adapter.py",
      " M bench/nirs_synthetic_pfn/tests/test_realism_scorecards.py",
      " M bench/nirs_synthetic_pfn/tests/test_transfer_validation.py",
      "?? .claude/",
      "?? .codex",
      "?? bench/AOM_v0/Ridge/Makefile",
      "?? bench/AOM_v0/Ridge/REPRODUCIBILITY.md",
      "?? bench/AOM_v0/Ridge/aomridge/branches.py",
      "?? bench/AOM_v0/Ridge/aomridge/cv.py",
      "?? bench/AOM_v0/Ridge/aomridge/mkl.py",
      "?? bench/AOM_v0/Ridge/benchmark_runs/curated/",
      "?? bench/AOM_v0/Ridge/benchmark_runs/curated_cohort.csv",
      "?? bench/AOM_v0/Ridge/benchmark_runs/curated_v2/",
      "?? bench/AOM_v0/Ridge/benchmark_runs/smoke6/",
      "?? bench/AOM_v0/Ridge/benchmark_runs/smoke_cv5/",
      "?? bench/AOM_v0/Ridge/docs/CODEX_BACKLOG_round2_2026-04-29.md",
      "?? bench/AOM_v0/Ridge/publication/",
      "?? bench/AOM_v0/Ridge/tests/test_ridge_branch_global.py",
      "?? bench/AOM_v0/Ridge/tests/test_ridge_mkl.py",
      "?? bench/AOM_v0/Ridge/tests/test_ridge_one_se_and_repeated_cv.py",
      "?? bench/AOM_v0/Ridge/tests/test_ridge_round3_fixes.py",
      "?? bench/AOM_v0/docs/CV_SPLITTER_DESIGN.md",
      "?? bench/AOM_v0/publication/tables/table_top15_score_time.tex",
      "?? bench/nirs_synthetic_pfn/docs/06_SYNTHETIC_REALISM_REMEDIATION_ROADMAP.md",
      "?? bench/nirs_synthetic_pfn/experiments/exp04_adversarial_auc.py",
      "?? bench/nirs_synthetic_pfn/experiments/exp05_minimal_ablation_attribution.py",
      "?? bench/nirs_synthetic_pfn/experiments/exp06_encoder_tabpfn_gate_precheck.py",
      "?? bench/nirs_synthetic_pfn/experiments/exp07_nirs_icl_gate_precheck.py",
      "?? bench/nirs_synthetic_pfn/reports/adversarial_auc.csv",
      "?? bench/nirs_synthetic_pfn/reports/adversarial_auc.md",
      "?? bench/nirs_synthetic_pfn/reports/encoder_tabpfn_gate.csv",
      "?? bench/nirs_synthetic_pfn/reports/encoder_tabpfn_gate.md",
      "?? bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.csv",
      "?? bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.md",
      "?? bench/nirs_synthetic_pfn/reports/nirs_icl_gate_precheck.csv",
      "?? bench/nirs_synthetic_pfn/reports/nirs_icl_gate_precheck.md",
      "?? bench/nirs_synthetic_pfn/tests/test_adversarial_auc_report.py",
      "?? bench/nirs_synthetic_pfn/tests/test_encoder_tabpfn_gate_precheck.py",
      "?? bench/nirs_synthetic_pfn/tests/test_minimal_ablation_attribution.py",
      "?? bench/nirs_synthetic_pfn/tests/test_nirs_icl_gate_precheck.py"
    ],
    "returncode": 0,
    "truncated": false
  },
  "row_count": 1,
  "status": "BLOCKED_BY_REALISM_GATE"
}
```
