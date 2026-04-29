# Transfer Validation

## Objective

Implement a minimal B3 transfer-validation route on local regression cohorts with real-only baselines and a synthetic PCA diagnostic.

## Command

`PYTHONPATH=bench/nirs_synthetic_pfn/src python bench/nirs_synthetic_pfn/experiments/exp03_transfer_validation.py --max-real-datasets 3 --max-samples 180 --n-splits 2 --test-fraction 0.25 --n-synthetic-samples 80 --seed 20260429`

## Outputs

- Markdown: `bench/nirs_synthetic_pfn/reports/transfer_validation.md`
- CSV metrics summary: `bench/nirs_synthetic_pfn/reports/transfer_validation.csv`

## Gate Flags

- `phase_a_gate_override`: `A3_failed_documented`
- `b2_realism_risk`: `B2_realism_failed`
- B2 CSV inspected: `bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv`
- B2 compared rows: 60
- B2 adversarial AUC failures: 60
- B2 PCA overlap failures: 60
- Synthetic transfer usefulness claims are blocked unless real-data baselines and realism evidence support them.

## Config

- Seed: 20260429
- Selected real regression rows: 3
- Repeated splits per selected row: 2
- Max real samples per row: 180
- Test fraction: 0.25
- Synthetic preset: `grain`
- Synthetic samples: 80
- B1 downstream training status: `allowed`

## Seed Policy

- Primary seed: 20260429
- Real-row downsampling uses `seed + dataset_index`.
- Repeated splits use `seed + dataset_index * 100 + split_index`.
- The A2 synthetic diagnostic run uses the primary seed.

## Git Status

- `git status --short` lines: 13
- First entries:
  - `?? .claude/`
  - `?? .codex`
  - `?? bench/AOM/PLAN_V1.md`
  - `?? bench/AOM/PUBLICATION_BACKLOG.md`
  - `?? bench/AOM/PUBLICATION_PLAN.md`
  - `?? bench/AOM/ROADMAP.md`
  - `?? bench/AOM_v0/`
  - `?? bench/nirs_synthetic_pfn/`
  - `?? bench/tabpfn_paper/Robin_s_article-1.pdf`
  - `?? bench/tabpfn_paper/Robin_s_article-1.pdf:Zone.Identifier`
  - `?? bench/tabpfn_paper/master_results.csv`
  - `?? bench/tabpfn_paper/master_results.csv:Zone.Identifier`
  - `?? docs/_internal/synthetic/spectral_synthesis_inventory_and_pfn_prior_plan.md`

## Real Cohort Inventory

| source | cohort path | exists | total rows | status ok rows | runnable rows | rows with missing paths |
|---|---|---|---:|---:|---:|---:|
| `AOM_regression` | `/home/delete/nirs4all/nirs4all/bench/AOM_v0/benchmarks/cohort_regression.csv` | `True` | 61 | 61 | 61 | 0 |
| `AOM_classification` | `/home/delete/nirs4all/nirs4all/bench/AOM_v0/benchmarks/cohort_classification.csv` | `True` | 17 | 16 | 16 | 0 |

## Runnable Counts

- Runnable local real rows discovered: 77
- Runnable regression rows discovered: 61
- Selected/runnable regression rows attempted: 3/61
- Completed metric rows: 22
- Blocked/unsupported rows: 8
- Dataset load/evaluation failures before row emission: 0

## Contract Checks

- A2 synthetic validation status: `passed`
- Downstream training status: `allowed`
- Hard checks passed: 9/9
- Failed hard checks: none.

## Repeated Split Metrics

| route | model | datasets | splits | RMSE mean | RMSE std | MAE mean | R2 mean |
|---|---|---:|---:|---:|---:|---:|---:|
| `RTSR_diagnostic` | `synthetic_pca_ridge` | 2 | 4 | 2.253 | 2.172 | 1.681 | 0.4392 |
| `real_only` | `pca_ridge` | 3 | 6 | 28.88 | 37.94 | 19.75 | 0.3644 |
| `real_only` | `pls` | 3 | 6 | 25.82 | 33.76 | 19.15 | 0.473 |
| `real_only` | `ridge` | 3 | 6 | 28.63 | 38.06 | 19.53 | 0.4951 |

## Blocked Transfer Routes

- `ALPINE/ALPINE_P_291_KS` split `0` `TSTR/synthetic_only_supervised`: blocked_target_domain_mismatch: A2 synthetic targets are preset latent targets, not calibrated to the selected real analyte labels; B2 realism failure blocks usefulness claims.
- `ALPINE/ALPINE_P_291_KS` split `1` `TSTR/synthetic_only_supervised`: blocked_target_domain_mismatch: A2 synthetic targets are preset latent targets, not calibrated to the selected real analyte labels; B2 realism failure blocks usefulness claims.
- `AMYLOSE/Rice_Amylose_313_YbasedSplit` split `0` `TSTR/synthetic_only_supervised`: blocked_target_domain_mismatch: A2 synthetic targets are preset latent targets, not calibrated to the selected real analyte labels; B2 realism failure blocks usefulness claims.
- `AMYLOSE/Rice_Amylose_313_YbasedSplit` split `1` `TSTR/synthetic_only_supervised`: blocked_target_domain_mismatch: A2 synthetic targets are preset latent targets, not calibrated to the selected real analyte labels; B2 realism failure blocks usefulness claims.
- `BEEFMARBLING/Beef_Marbling_RandomSplit` split `0` `RTSR_diagnostic/synthetic_pca_ridge`: ValueError: real/synthetic wavelength grids have fewer than three overlapping points
- `BEEFMARBLING/Beef_Marbling_RandomSplit` split `0` `TSTR/synthetic_only_supervised`: blocked_target_domain_mismatch: A2 synthetic targets are preset latent targets, not calibrated to the selected real analyte labels; B2 realism failure blocks usefulness claims.
- `BEEFMARBLING/Beef_Marbling_RandomSplit` split `1` `RTSR_diagnostic/synthetic_pca_ridge`: ValueError: real/synthetic wavelength grids have fewer than three overlapping points
- `BEEFMARBLING/Beef_Marbling_RandomSplit` split `1` `TSTR/synthetic_only_supervised`: blocked_target_domain_mismatch: A2 synthetic targets are preset latent targets, not calibrated to the selected real analyte labels; B2 realism failure blocks usefulness claims.

## Ablation Report

- `without_instruments`: deferred; no defensible synthetic-transfer claim while B2 realism is failed.
- `without_scatter`: deferred; no defensible synthetic-transfer claim while B2 realism is failed.
- `without_products_or_aggregates`: deferred; no matched real/synthetic target route in this smoke.
- `without_procedural_diversity`: deferred; needs a successful realism/domain-matching gate first.

## Scientific Decision

Route runnable: real-only baselines and a synthetic PCA diagnostic were produced. Synthetic usefulness remains blocked because B2 realism failed and supervised TSTR target matching is not available.

## Load Failures

None.

## Raw Summary JSON

```json
{
  "b1_validation": {
    "checks": [
      {
        "details": {
          "validation_summary_status": "passed"
        },
        "message": "A2 dataset contract is passed.",
        "metrics": {},
        "name": "a2_contract",
        "severity": "hard",
        "status": "passed",
        "thresholds": {}
      },
      {
        "details": {
          "row_normalized": true,
          "shape": [
            80,
            3
          ]
        },
        "message": "Concentrations are finite, normalized, and within [0, 1].",
        "metrics": {
          "finite": true,
          "max": 0.8349016398796842,
          "max_row_sum_error": 2.220446049250313e-16,
          "min": 0.06827086386863203,
          "row_sum_max": 1.0000000000000002,
          "row_sum_min": 0.9999999999999999
        },
        "name": "concentration_sums_and_ranges",
        "severity": "hard",
        "status": "passed",
        "thresholds": {
          "max": 1.0,
          "min": 0.0,
          "row_sum_tolerance": 1e-06
        }
      },
      {
        "details": {
          "invalid": []
        },
        "message": "Declared component concentration prior ranges are valid probabilities.",
        "metrics": {
          "component_count": 3,
          "invalid_count": 0
        },
        "name": "concentration_prior_ranges_declared",
        "severity": "hard",
        "status": "passed",
        "thresholds": {}
      },
      {
        "details": {},
        "message": "Regression target is finite, variable, and inside the declared range.",
        "metrics": {
          "y_max": 0.8,
          "y_min": 0.3,
          "y_std": 0.0987532115119232
        },
        "name": "target_distribution",
        "severity": "hard",
        "status": "passed",
        "thresholds": {
          "min_std": 1e-08,
          "range": [
            0.3,
            0.8
          ]
        }
      },
      {
        "details": {
          "declared_nonlinearity": "none"
        },
        "message": "Regression target declares no nonlinearity.",
        "metrics": {},
        "name": "nonlinear_target_behavior",
        "severity": "informational",
        "status": "not_applicable",
        "thresholds": {}
      },
      {
        "details": {
          "failures": [],
          "instrument": "foss_xds",
          "instrument_category": "benchtop",
          "measurement_mode": "reflectance"
        },
        "message": "Wavelength grid and mode are compatible with the selected instrument.",
        "metrics": {
          "n_wavelengths": 351,
          "wavelength_max": 2500.0,
          "wavelength_min": 1100.0
        },
        "name": "wavelengths_and_mode",
        "severity": "hard",
        "status": "passed",
        "thresholds": {}
      },
      {
        "details": {},
        "message": "Spectral SNR is finite and inside the B1 smoke range.",
        "metrics": {
          "finite": true,
          "median_snr": 25.40129330348541,
          "q05": 22.55438125869666,
          "q95": 29.449251696173228
        },
        "name": "spectral_snr",
        "severity": "hard",
        "status": "passed",
        "thresholds": {
          "median_snr": [
            3.0,
            100000.0
          ]
        }
      },
      {
        "details": {},
        "message": "First-derivative variability is inside the B1 smoke range.",
        "metrics": {
          "finite": true,
          "median_first_derivative_std": 0.005102375767231683,
          "q05": 0.003441762138480673,
          "q95": 0.00669462364104837
        },
        "name": "derivative_statistics",
        "severity": "hard",
        "status": "passed",
        "thresholds": {
          "median_first_derivative_std": [
            1e-07,
            0.05
          ]
        }
      },
      {
        "details": {},
        "message": "Baseline curvature is inside the B1 smoke range.",
        "metrics": {
          "finite": true,
          "median_baseline_curvature": 0.17707993680479903,
          "q05": 0.12144759195591129,
          "q95": 0.23425603143042126
        },
        "name": "baseline_curvature",
        "severity": "hard",
        "status": "passed",
        "thresholds": {
          "median_baseline_curvature": [
            1e-05,
            1.0
          ]
        }
      },
      {
        "details": {},
        "message": "Peak density is inside the B1 smoke range.",
        "metrics": {
          "finite": true,
          "median_peak_density_per_100nm": 2.571428571428571,
          "q05": 1.9964285714285714,
          "q95": 3.0749999999999997
        },
        "name": "peak_density",
        "severity": "hard",
        "status": "passed",
        "thresholds": {
          "median_peak_density_per_100nm": [
            0.05,
            12.0
          ]
        }
      }
    ],
    "downstream_training_status": "allowed",
    "phase_a_gate_override": "A3_failed_documented",
    "preset": "grain",
    "summary": {
      "X_shape": [
        80,
        351
      ],
      "blocking_checks": [],
      "domain": "agriculture_grain",
      "instrument": "foss_xds",
      "measurement_mode": "reflectance",
      "target_type": "regression",
      "unsupported_checks": [],
      "wavelength_range_nm": [
        1100.0,
        2500.0
      ],
      "y_shape": [
        80
      ]
    },
    "validation_status": "passed"
  },
  "b2_summary": {
    "adversarial_auc_failures": 60,
    "b2_realism_failed": true,
    "compared_rows": 60,
    "exists": true,
    "path": "/home/delete/nirs4all/nirs4all/bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv",
    "pca_overlap_failures": 60,
    "reason": "B2_realism_failed",
    "row_count": 77
  },
  "git_status": {
    "line_count": 13,
    "lines": [
      "?? .claude/",
      "?? .codex",
      "?? bench/AOM/PLAN_V1.md",
      "?? bench/AOM/PUBLICATION_BACKLOG.md",
      "?? bench/AOM/PUBLICATION_PLAN.md",
      "?? bench/AOM/ROADMAP.md",
      "?? bench/AOM_v0/",
      "?? bench/nirs_synthetic_pfn/",
      "?? bench/tabpfn_paper/Robin_s_article-1.pdf",
      "?? bench/tabpfn_paper/Robin_s_article-1.pdf:Zone.Identifier",
      "?? bench/tabpfn_paper/master_results.csv",
      "?? bench/tabpfn_paper/master_results.csv:Zone.Identifier",
      "?? docs/_internal/synthetic/spectral_synthesis_inventory_and_pfn_prior_plan.md"
    ],
    "returncode": 0,
    "truncated": false
  },
  "inventories": [
    {
      "exists": true,
      "missing_paths": [],
      "missing_rows": 0,
      "ok_rows": 61,
      "path": "/home/delete/nirs4all/nirs4all/bench/AOM_v0/benchmarks/cohort_regression.csv",
      "runnable_rows": 61,
      "source": "AOM_regression",
      "total_rows": 61
    },
    {
      "exists": true,
      "missing_paths": [],
      "missing_rows": 0,
      "ok_rows": 16,
      "path": "/home/delete/nirs4all/nirs4all/bench/AOM_v0/benchmarks/cohort_classification.csv",
      "runnable_rows": 16,
      "source": "AOM_classification",
      "total_rows": 17
    }
  ],
  "load_failures": [],
  "row_count": 30,
  "status": "done"
}
```
