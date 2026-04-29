# Prior Predictive Checks

## Objective

Run B1 prior predictive checks on the 10 A2 smoke preset datasets and emit an explicit downstream training status.

## Phase A Gate Override

- `phase_a_gate_override`: `A3_failed_documented`
- A3 fitted-only real-fit gate remains scientifically failed/blocked as documented in `real_fit_adapter_smoke.md`; B1 is continued by explicit user instruction.

## Command

`PYTHONPATH=bench/nirs_synthetic_pfn/src python bench/nirs_synthetic_pfn/experiments/exp01_prior_predictive_checks.py --n-samples 40 --seed 20260429`

## Config

- Seed base: 20260429
- Samples per dataset: 40
- Presets: A2 smoke presets from `exp00_smoke_prior_dataset.py`

## Git Status

- Return code: 0
- Entries: 13
- Status counts: `{'??': 13}`
- Sample:
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

## Preset Report Table

| preset | domain | target | X shape | B1 status | downstream training | blocking checks | key spectral metrics |
|---|---|---|---:|---|---|---|---|
| `grain` | `agriculture_grain` | `regression` | `[40, 351]` | `passed` | `allowed` | `none` | SNR=25.35; d1=0.005007; curv=0.1798; peaks=2.714 |
| `forage` | `agriculture_forage` | `classification` | `[40, 351]` | `passed` | `allowed` | `none` | SNR=24.63; d1=0.004698; curv=0.1438; peaks=2.536 |
| `oilseeds` | `agriculture_oilseeds` | `regression` | `[40, 351]` | `passed` | `allowed` | `none` | SNR=12.83; d1=0.004856; curv=0.13; peaks=2.571 |
| `fruit` | `agriculture_fruit` | `classification` | `[40, 101]` | `passed` | `allowed` | `none` | SNR=9.169; d1=0.001387; curv=0.02746; peaks=2 |
| `dairy` | `food_dairy` | `regression` | `[40, 351]` | `passed` | `allowed` | `none` | SNR=27.83; d1=0.004037; curv=0.1415; peaks=2.857 |
| `meat` | `food_meat` | `classification` | `[40, 201]` | `passed` | `allowed` | `none` | SNR=29.04; d1=0.003347; curv=0.1077; peaks=2.375 |
| `baking` | `food_bakery` | `regression` | `[40, 351]` | `passed` | `allowed` | `none` | SNR=28.17; d1=0.004141; curv=0.1441; peaks=3 |
| `tablets` | `pharma_tablets` | `classification` | `[40, 351]` | `passed` | `allowed` | `none` | SNR=19.02; d1=0.006152; curv=0.1889; peaks=1.857 |
| `powders` | `pharma_powder_blends` | `regression` | `[40, 351]` | `passed` | `allowed` | `none` | SNR=19.84; d1=0.005701; curv=0.1726; peaks=1.964 |
| `fuel` | `petrochem_fuels` | `classification` | `[40, 201]` | `passed` | `allowed` | `none` | SNR=15.96; d1=0.004104; curv=0.1049; peaks=2.375 |

## Check Coverage

| preset | concentrations | target | nonlinear | wavelengths/mode | SNR | derivative | baseline | peaks | unsupported |
|---|---|---|---|---|---|---|---|---|---|
| `grain` | `passed` | `passed` | `not_applicable` | `passed` | `passed` | `passed` | `passed` | `passed` | `none` |
| `forage` | `passed` | `passed` | `not_applicable` | `passed` | `passed` | `passed` | `passed` | `passed` | `none` |
| `oilseeds` | `passed` | `passed` | `not_applicable` | `passed` | `passed` | `passed` | `passed` | `passed` | `none` |
| `fruit` | `passed` | `passed` | `not_applicable` | `passed` | `passed` | `passed` | `passed` | `passed` | `none` |
| `dairy` | `passed` | `passed` | `not_applicable` | `passed` | `passed` | `passed` | `passed` | `passed` | `none` |
| `meat` | `passed` | `passed` | `not_applicable` | `passed` | `passed` | `passed` | `passed` | `passed` | `none` |
| `baking` | `passed` | `passed` | `not_applicable` | `passed` | `passed` | `passed` | `passed` | `passed` | `none` |
| `tablets` | `passed` | `passed` | `not_applicable` | `passed` | `passed` | `passed` | `passed` | `passed` | `none` |
| `powders` | `passed` | `passed` | `not_applicable` | `passed` | `passed` | `passed` | `passed` | `passed` | `none` |
| `fuel` | `passed` | `passed` | `not_applicable` | `passed` | `passed` | `passed` | `passed` | `passed` | `none` |

## Summary

- Passed: 10 / 10
- Blocked: 0 / 10
- Training rule: any hard `failed` or `unsupported` check sets `downstream_training_status=blocked`.

## Failing-Case Examples

- None in this 10-preset smoke run.
- Synthetic blocked examples are covered by tests: bad concentration sums, imbalanced or non-integer classification labels, missing concentration metadata, and invalid empty wavelength grids.

## Unsupported Checks

No hard unsupported checks in this smoke run. Nonlinear target behavior is marked `not_applicable` for classification and linear-regression presets.

## Residual Risks

- Thresholds are B1 smoke guardrails, not calibrated real/synthetic realism thresholds.
- A2 row-normalizes concentrations; B1 checks normalized mixture support and declared prior ranges separately.
- `measurement_mode` compatibility is checked against prior weights and instrument category, but mode-specific optical physics remains a documented A2/A3 risk.
- A3 fitted-only real-fit scientific gate is still failed/blocked; this report intentionally carries the override note above.

## Raw Summary JSON

```json
{
  "git_status": {
    "by_status": {
      "??": 13
    },
    "entry_count": 13,
    "returncode": 0,
    "sample": [
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
    ]
  },
  "results": [
    {
      "X_shape": [
        40,
        351
      ],
      "blocking_checks": [],
      "check_statuses": {
        "a2_contract": "passed",
        "baseline_curvature": "passed",
        "concentration_prior_ranges_declared": "passed",
        "concentration_sums_and_ranges": "passed",
        "derivative_statistics": "passed",
        "nonlinear_target_behavior": "not_applicable",
        "peak_density": "passed",
        "spectral_snr": "passed",
        "target_distribution": "passed",
        "wavelengths_and_mode": "passed"
      },
      "domain": "agriculture_grain",
      "downstream_training_status": "allowed",
      "instrument": "foss_xds",
      "key_metrics": {
        "component_count": 3,
        "finite": true,
        "invalid_count": 0,
        "max": 0.8243428636091362,
        "max_row_sum_error": 2.220446049250313e-16,
        "median_baseline_curvature": 0.17978778687072172,
        "median_first_derivative_std": 0.0050070151823420695,
        "median_peak_density_per_100nm": 2.714285714285714,
        "median_snr": 25.347907880905375,
        "min": 0.05512642415139776,
        "n_wavelengths": 351,
        "q05": 2.0678571428571426,
        "q95": 3.2857142857142856,
        "row_sum_max": 1.0,
        "row_sum_min": 0.9999999999999998,
        "wavelength_max": 2500.0,
        "wavelength_min": 1100.0,
        "y_max": 0.8,
        "y_min": 0.3,
        "y_std": 0.1086062921112861
      },
      "mode": "reflectance",
      "phase_a_gate_override": "A3_failed_documented",
      "preset": "grain",
      "target_type": "regression",
      "unsupported_checks": [],
      "validation": {
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
                40,
                3
              ]
            },
            "message": "Concentrations are finite, normalized, and within [0, 1].",
            "metrics": {
              "finite": true,
              "max": 0.8243428636091362,
              "max_row_sum_error": 2.220446049250313e-16,
              "min": 0.05512642415139776,
              "row_sum_max": 1.0,
              "row_sum_min": 0.9999999999999998
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
              "y_std": 0.1086062921112861
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
              "median_snr": 25.347907880905375,
              "q05": 21.44644836643822,
              "q95": 30.377002708780257
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
              "median_first_derivative_std": 0.0050070151823420695,
              "q05": 0.003503346790922068,
              "q95": 0.006418922843419235
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
              "median_baseline_curvature": 0.17978778687072172,
              "q05": 0.12233384982310028,
              "q95": 0.22098060448671591
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
              "median_peak_density_per_100nm": 2.714285714285714,
              "q05": 2.0678571428571426,
              "q95": 3.2857142857142856
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
            40,
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
            40
          ]
        },
        "validation_status": "passed"
      },
      "validation_status": "passed",
      "y_shape": [
        40
      ]
    },
    {
      "X_shape": [
        40,
        351
      ],
      "blocking_checks": [],
      "check_statuses": {
        "a2_contract": "passed",
        "baseline_curvature": "passed",
        "concentration_prior_ranges_declared": "passed",
        "concentration_sums_and_ranges": "passed",
        "derivative_statistics": "passed",
        "nonlinear_target_behavior": "not_applicable",
        "peak_density": "passed",
        "spectral_snr": "passed",
        "target_distribution": "passed",
        "wavelengths_and_mode": "passed"
      },
      "domain": "agriculture_forage",
      "downstream_training_status": "allowed",
      "instrument": "foss_xds",
      "key_metrics": {
        "class_counts": {
          "0": 14,
          "1": 13,
          "2": 13
        },
        "component_count": 3,
        "expected_classes": [
          0,
          1,
          2
        ],
        "finite": true,
        "invalid_count": 0,
        "max": 0.7112247227767874,
        "max_row_sum_error": 2.220446049250313e-16,
        "median_baseline_curvature": 0.14376436399440806,
        "median_first_derivative_std": 0.004698356979412538,
        "median_peak_density_per_100nm": 2.5357142857142856,
        "median_snr": 24.628969160154305,
        "min": 0.09335070334335552,
        "min_class_fraction": 0.325,
        "n_wavelengths": 351,
        "observed_classes": [
          0,
          1,
          2
        ],
        "q05": 1.9249999999999998,
        "q95": 3.0749999999999997,
        "row_sum_max": 1.0000000000000002,
        "row_sum_min": 0.9999999999999999,
        "wavelength_max": 2500.0,
        "wavelength_min": 1100.0
      },
      "mode": "reflectance",
      "phase_a_gate_override": "A3_failed_documented",
      "preset": "forage",
      "target_type": "classification",
      "unsupported_checks": [],
      "validation": {
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
                40,
                3
              ]
            },
            "message": "Concentrations are finite, normalized, and within [0, 1].",
            "metrics": {
              "finite": true,
              "max": 0.7112247227767874,
              "max_row_sum_error": 2.220446049250313e-16,
              "min": 0.09335070334335552,
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
            "message": "Classification target contains all declared classes with acceptable balance.",
            "metrics": {
              "class_counts": {
                "0": 14,
                "1": 13,
                "2": 13
              },
              "expected_classes": [
                0,
                1,
                2
              ],
              "min_class_fraction": 0.325,
              "observed_classes": [
                0,
                1,
                2
              ]
            },
            "name": "target_distribution",
            "severity": "hard",
            "status": "passed",
            "thresholds": {
              "min_class_fraction": 0.1
            }
          },
          {
            "details": {},
            "message": "Nonlinear target behavior is only defined for regression targets.",
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
              "median_snr": 24.628969160154305,
              "q05": 20.89248344143323,
              "q95": 32.9886830817739
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
              "median_first_derivative_std": 0.004698356979412538,
              "q05": 0.0036007198730341346,
              "q95": 0.006460779968754789
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
              "median_baseline_curvature": 0.14376436399440806,
              "q05": 0.11031892759244924,
              "q95": 0.19895010142396777
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
              "median_peak_density_per_100nm": 2.5357142857142856,
              "q05": 1.9249999999999998,
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
        "preset": "forage",
        "summary": {
          "X_shape": [
            40,
            351
          ],
          "blocking_checks": [],
          "domain": "agriculture_forage",
          "instrument": "foss_xds",
          "measurement_mode": "reflectance",
          "target_type": "classification",
          "unsupported_checks": [],
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_shape": [
            40
          ]
        },
        "validation_status": "passed"
      },
      "validation_status": "passed",
      "y_shape": [
        40
      ]
    },
    {
      "X_shape": [
        40,
        351
      ],
      "blocking_checks": [],
      "check_statuses": {
        "a2_contract": "passed",
        "baseline_curvature": "passed",
        "concentration_prior_ranges_declared": "passed",
        "concentration_sums_and_ranges": "passed",
        "derivative_statistics": "passed",
        "nonlinear_target_behavior": "not_applicable",
        "peak_density": "passed",
        "spectral_snr": "passed",
        "target_distribution": "passed",
        "wavelengths_and_mode": "passed"
      },
      "domain": "agriculture_oilseeds",
      "downstream_training_status": "allowed",
      "instrument": "foss_xds",
      "key_metrics": {
        "component_count": 3,
        "finite": true,
        "invalid_count": 0,
        "max": 0.6574659106756585,
        "max_row_sum_error": 2.220446049250313e-16,
        "median_baseline_curvature": 0.13001652300440203,
        "median_first_derivative_std": 0.004855849841821774,
        "median_peak_density_per_100nm": 2.571428571428571,
        "median_snr": 12.830055376761145,
        "min": 0.07456660151481379,
        "n_wavelengths": 351,
        "q05": 2.2785714285714285,
        "q95": 2.932142857142857,
        "row_sum_max": 1.0000000000000002,
        "row_sum_min": 0.9999999999999998,
        "wavelength_max": 2500.0,
        "wavelength_min": 1100.0,
        "y_max": 0.44999999999999996,
        "y_min": 0.1,
        "y_std": 0.07532835210203309
      },
      "mode": "reflectance",
      "phase_a_gate_override": "A3_failed_documented",
      "preset": "oilseeds",
      "target_type": "regression",
      "unsupported_checks": [],
      "validation": {
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
                40,
                3
              ]
            },
            "message": "Concentrations are finite, normalized, and within [0, 1].",
            "metrics": {
              "finite": true,
              "max": 0.6574659106756585,
              "max_row_sum_error": 2.220446049250313e-16,
              "min": 0.07456660151481379,
              "row_sum_max": 1.0000000000000002,
              "row_sum_min": 0.9999999999999998
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
              "y_max": 0.44999999999999996,
              "y_min": 0.1,
              "y_std": 0.07532835210203309
            },
            "name": "target_distribution",
            "severity": "hard",
            "status": "passed",
            "thresholds": {
              "min_std": 1e-08,
              "range": [
                0.1,
                0.45
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
              "median_snr": 12.830055376761145,
              "q05": 11.876563623916251,
              "q95": 14.739105258630259
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
              "median_first_derivative_std": 0.004855849841821774,
              "q05": 0.0038123295284246553,
              "q95": 0.006319769660655966
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
              "median_baseline_curvature": 0.13001652300440203,
              "q05": 0.10138234419431952,
              "q95": 0.16901100869500751
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
              "q05": 2.2785714285714285,
              "q95": 2.932142857142857
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
        "preset": "oilseeds",
        "summary": {
          "X_shape": [
            40,
            351
          ],
          "blocking_checks": [],
          "domain": "agriculture_oilseeds",
          "instrument": "foss_xds",
          "measurement_mode": "reflectance",
          "target_type": "regression",
          "unsupported_checks": [],
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_shape": [
            40
          ]
        },
        "validation_status": "passed"
      },
      "validation_status": "passed",
      "y_shape": [
        40
      ]
    },
    {
      "X_shape": [
        40,
        101
      ],
      "blocking_checks": [],
      "check_statuses": {
        "a2_contract": "passed",
        "baseline_curvature": "passed",
        "concentration_prior_ranges_declared": "passed",
        "concentration_sums_and_ranges": "passed",
        "derivative_statistics": "passed",
        "nonlinear_target_behavior": "not_applicable",
        "peak_density": "passed",
        "spectral_snr": "passed",
        "target_distribution": "passed",
        "wavelengths_and_mode": "passed"
      },
      "domain": "agriculture_fruit",
      "downstream_training_status": "allowed",
      "instrument": "foss_xds",
      "key_metrics": {
        "class_counts": {
          "0": 20,
          "1": 20
        },
        "component_count": 3,
        "expected_classes": [
          0,
          1
        ],
        "finite": true,
        "invalid_count": 0,
        "max": 0.904051820703642,
        "max_row_sum_error": 2.220446049250313e-16,
        "median_baseline_curvature": 0.027464107061400102,
        "median_first_derivative_std": 0.0013874328647256857,
        "median_peak_density_per_100nm": 2.0,
        "median_snr": 9.169182651268368,
        "min": 0.026545900219573804,
        "min_class_fraction": 0.5,
        "n_wavelengths": 101,
        "observed_classes": [
          0,
          1
        ],
        "q05": 1.25,
        "q95": 3.0,
        "row_sum_max": 1.0000000000000002,
        "row_sum_min": 0.9999999999999998,
        "wavelength_max": 1100.0,
        "wavelength_min": 700.0
      },
      "mode": "reflectance",
      "phase_a_gate_override": "A3_failed_documented",
      "preset": "fruit",
      "target_type": "classification",
      "unsupported_checks": [],
      "validation": {
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
                40,
                3
              ]
            },
            "message": "Concentrations are finite, normalized, and within [0, 1].",
            "metrics": {
              "finite": true,
              "max": 0.904051820703642,
              "max_row_sum_error": 2.220446049250313e-16,
              "min": 0.026545900219573804,
              "row_sum_max": 1.0000000000000002,
              "row_sum_min": 0.9999999999999998
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
            "message": "Classification target contains all declared classes with acceptable balance.",
            "metrics": {
              "class_counts": {
                "0": 20,
                "1": 20
              },
              "expected_classes": [
                0,
                1
              ],
              "min_class_fraction": 0.5,
              "observed_classes": [
                0,
                1
              ]
            },
            "name": "target_distribution",
            "severity": "hard",
            "status": "passed",
            "thresholds": {
              "min_class_fraction": 0.1
            }
          },
          {
            "details": {},
            "message": "Nonlinear target behavior is only defined for regression targets.",
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
              "n_wavelengths": 101,
              "wavelength_max": 1100.0,
              "wavelength_min": 700.0
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
              "median_snr": 9.169182651268368,
              "q05": 7.938174033605716,
              "q95": 15.585778725691222
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
              "median_first_derivative_std": 0.0013874328647256857,
              "q05": 0.0009717188381093396,
              "q95": 0.0018161875502654275
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
              "median_baseline_curvature": 0.027464107061400102,
              "q05": 0.01934639971492765,
              "q95": 0.035858372815333224
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
              "median_peak_density_per_100nm": 2.0,
              "q05": 1.25,
              "q95": 3.0
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
        "preset": "fruit",
        "summary": {
          "X_shape": [
            40,
            101
          ],
          "blocking_checks": [],
          "domain": "agriculture_fruit",
          "instrument": "foss_xds",
          "measurement_mode": "reflectance",
          "target_type": "classification",
          "unsupported_checks": [],
          "wavelength_range_nm": [
            700.0,
            1100.0
          ],
          "y_shape": [
            40
          ]
        },
        "validation_status": "passed"
      },
      "validation_status": "passed",
      "y_shape": [
        40
      ]
    },
    {
      "X_shape": [
        40,
        351
      ],
      "blocking_checks": [],
      "check_statuses": {
        "a2_contract": "passed",
        "baseline_curvature": "passed",
        "concentration_prior_ranges_declared": "passed",
        "concentration_sums_and_ranges": "passed",
        "derivative_statistics": "passed",
        "nonlinear_target_behavior": "not_applicable",
        "peak_density": "passed",
        "spectral_snr": "passed",
        "target_distribution": "passed",
        "wavelengths_and_mode": "passed"
      },
      "domain": "food_dairy",
      "downstream_training_status": "allowed",
      "instrument": "foss_xds",
      "key_metrics": {
        "component_count": 3,
        "finite": true,
        "invalid_count": 0,
        "max": 0.8924928577708051,
        "max_row_sum_error": 2.220446049250313e-16,
        "median_baseline_curvature": 0.141540005342731,
        "median_first_derivative_std": 0.004037189030499943,
        "median_peak_density_per_100nm": 2.8571428571428568,
        "median_snr": 27.82537135838602,
        "min": 0.027496435693496846,
        "n_wavelengths": 351,
        "q05": 2.2714285714285714,
        "q95": 3.6499999999999995,
        "row_sum_max": 1.0000000000000002,
        "row_sum_min": 0.9999999999999998,
        "wavelength_max": 2500.0,
        "wavelength_min": 1100.0,
        "y_max": 0.92,
        "y_min": 0.8,
        "y_std": 0.030303890470758443
      },
      "mode": "reflectance",
      "phase_a_gate_override": "A3_failed_documented",
      "preset": "dairy",
      "target_type": "regression",
      "unsupported_checks": [],
      "validation": {
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
                40,
                3
              ]
            },
            "message": "Concentrations are finite, normalized, and within [0, 1].",
            "metrics": {
              "finite": true,
              "max": 0.8924928577708051,
              "max_row_sum_error": 2.220446049250313e-16,
              "min": 0.027496435693496846,
              "row_sum_max": 1.0000000000000002,
              "row_sum_min": 0.9999999999999998
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
              "y_max": 0.92,
              "y_min": 0.8,
              "y_std": 0.030303890470758443
            },
            "name": "target_distribution",
            "severity": "hard",
            "status": "passed",
            "thresholds": {
              "min_std": 1e-08,
              "range": [
                0.8,
                0.92
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
              "median_snr": 27.82537135838602,
              "q05": 23.682143033973773,
              "q95": 32.72607873089794
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
              "median_first_derivative_std": 0.004037189030499943,
              "q05": 0.0029268707744891033,
              "q95": 0.0059470001835726215
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
              "median_baseline_curvature": 0.141540005342731,
              "q05": 0.10210524180344394,
              "q95": 0.21563105897278412
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
              "median_peak_density_per_100nm": 2.8571428571428568,
              "q05": 2.2714285714285714,
              "q95": 3.6499999999999995
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
        "preset": "dairy",
        "summary": {
          "X_shape": [
            40,
            351
          ],
          "blocking_checks": [],
          "domain": "food_dairy",
          "instrument": "foss_xds",
          "measurement_mode": "reflectance",
          "target_type": "regression",
          "unsupported_checks": [],
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_shape": [
            40
          ]
        },
        "validation_status": "passed"
      },
      "validation_status": "passed",
      "y_shape": [
        40
      ]
    },
    {
      "X_shape": [
        40,
        201
      ],
      "blocking_checks": [],
      "check_statuses": {
        "a2_contract": "passed",
        "baseline_curvature": "passed",
        "concentration_prior_ranges_declared": "passed",
        "concentration_sums_and_ranges": "passed",
        "derivative_statistics": "passed",
        "nonlinear_target_behavior": "not_applicable",
        "peak_density": "passed",
        "spectral_snr": "passed",
        "target_distribution": "passed",
        "wavelengths_and_mode": "passed"
      },
      "domain": "food_meat",
      "downstream_training_status": "allowed",
      "instrument": "foss_xds",
      "key_metrics": {
        "class_counts": {
          "0": 14,
          "1": 13,
          "2": 13
        },
        "component_count": 3,
        "expected_classes": [
          0,
          1,
          2
        ],
        "finite": true,
        "invalid_count": 0,
        "max": 0.7607629114052353,
        "max_row_sum_error": 2.220446049250313e-16,
        "median_baseline_curvature": 0.10765835781672808,
        "median_first_derivative_std": 0.0033466858558147726,
        "median_peak_density_per_100nm": 2.375,
        "median_snr": 29.035058219736943,
        "min": 0.058816858015539986,
        "min_class_fraction": 0.325,
        "n_wavelengths": 201,
        "observed_classes": [
          0,
          1,
          2
        ],
        "q05": 1.86875,
        "q95": 3.0062499999999996,
        "row_sum_max": 1.0000000000000002,
        "row_sum_min": 0.9999999999999998,
        "wavelength_max": 1700.0,
        "wavelength_min": 900.0
      },
      "mode": "reflectance",
      "phase_a_gate_override": "A3_failed_documented",
      "preset": "meat",
      "target_type": "classification",
      "unsupported_checks": [],
      "validation": {
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
                40,
                3
              ]
            },
            "message": "Concentrations are finite, normalized, and within [0, 1].",
            "metrics": {
              "finite": true,
              "max": 0.7607629114052353,
              "max_row_sum_error": 2.220446049250313e-16,
              "min": 0.058816858015539986,
              "row_sum_max": 1.0000000000000002,
              "row_sum_min": 0.9999999999999998
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
            "message": "Classification target contains all declared classes with acceptable balance.",
            "metrics": {
              "class_counts": {
                "0": 14,
                "1": 13,
                "2": 13
              },
              "expected_classes": [
                0,
                1,
                2
              ],
              "min_class_fraction": 0.325,
              "observed_classes": [
                0,
                1,
                2
              ]
            },
            "name": "target_distribution",
            "severity": "hard",
            "status": "passed",
            "thresholds": {
              "min_class_fraction": 0.1
            }
          },
          {
            "details": {},
            "message": "Nonlinear target behavior is only defined for regression targets.",
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
              "n_wavelengths": 201,
              "wavelength_max": 1700.0,
              "wavelength_min": 900.0
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
              "median_snr": 29.035058219736943,
              "q05": 22.2609939245761,
              "q95": 35.926698367499675
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
              "median_first_derivative_std": 0.0033466858558147726,
              "q05": 0.0022934079065935597,
              "q95": 0.004110829556586787
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
              "median_baseline_curvature": 0.10765835781672808,
              "q05": 0.07337571857535491,
              "q95": 0.13410687288506243
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
              "median_peak_density_per_100nm": 2.375,
              "q05": 1.86875,
              "q95": 3.0062499999999996
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
        "preset": "meat",
        "summary": {
          "X_shape": [
            40,
            201
          ],
          "blocking_checks": [],
          "domain": "food_meat",
          "instrument": "foss_xds",
          "measurement_mode": "reflectance",
          "target_type": "classification",
          "unsupported_checks": [],
          "wavelength_range_nm": [
            900.0,
            1700.0
          ],
          "y_shape": [
            40
          ]
        },
        "validation_status": "passed"
      },
      "validation_status": "passed",
      "y_shape": [
        40
      ]
    },
    {
      "X_shape": [
        40,
        351
      ],
      "blocking_checks": [],
      "check_statuses": {
        "a2_contract": "passed",
        "baseline_curvature": "passed",
        "concentration_prior_ranges_declared": "passed",
        "concentration_sums_and_ranges": "passed",
        "derivative_statistics": "passed",
        "nonlinear_target_behavior": "not_applicable",
        "peak_density": "passed",
        "spectral_snr": "passed",
        "target_distribution": "passed",
        "wavelengths_and_mode": "passed"
      },
      "domain": "food_bakery",
      "downstream_training_status": "allowed",
      "instrument": "foss_xds",
      "key_metrics": {
        "component_count": 3,
        "finite": true,
        "invalid_count": 0,
        "max": 0.8474536702143612,
        "max_row_sum_error": 2.220446049250313e-16,
        "median_baseline_curvature": 0.14412326314056118,
        "median_first_derivative_std": 0.00414067943363564,
        "median_peak_density_per_100nm": 3.0,
        "median_snr": 28.16857317238963,
        "min": 0.03802437648652834,
        "n_wavelengths": 351,
        "q05": 2.357142857142857,
        "q95": 3.714285714285714,
        "row_sum_max": 1.0000000000000002,
        "row_sum_min": 0.9999999999999999,
        "wavelength_max": 2500.0,
        "wavelength_min": 1100.0,
        "y_max": 1.0,
        "y_min": 0.0,
        "y_std": 0.21194666798216125
      },
      "mode": "reflectance",
      "phase_a_gate_override": "A3_failed_documented",
      "preset": "baking",
      "target_type": "regression",
      "unsupported_checks": [],
      "validation": {
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
                40,
                3
              ]
            },
            "message": "Concentrations are finite, normalized, and within [0, 1].",
            "metrics": {
              "finite": true,
              "max": 0.8474536702143612,
              "max_row_sum_error": 2.220446049250313e-16,
              "min": 0.03802437648652834,
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
              "y_max": 1.0,
              "y_min": 0.0,
              "y_std": 0.21194666798216125
            },
            "name": "target_distribution",
            "severity": "hard",
            "status": "passed",
            "thresholds": {
              "min_std": 1e-08,
              "range": [
                0.0,
                1.0
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
              "median_snr": 28.16857317238963,
              "q05": 21.060249805899577,
              "q95": 34.05764736436891
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
              "median_first_derivative_std": 0.00414067943363564,
              "q05": 0.003050086518518313,
              "q95": 0.005598731500299879
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
              "median_baseline_curvature": 0.14412326314056118,
              "q05": 0.1051938857024139,
              "q95": 0.18529484333711088
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
              "median_peak_density_per_100nm": 3.0,
              "q05": 2.357142857142857,
              "q95": 3.714285714285714
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
        "preset": "baking",
        "summary": {
          "X_shape": [
            40,
            351
          ],
          "blocking_checks": [],
          "domain": "food_bakery",
          "instrument": "foss_xds",
          "measurement_mode": "reflectance",
          "target_type": "regression",
          "unsupported_checks": [],
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_shape": [
            40
          ]
        },
        "validation_status": "passed"
      },
      "validation_status": "passed",
      "y_shape": [
        40
      ]
    },
    {
      "X_shape": [
        40,
        351
      ],
      "blocking_checks": [],
      "check_statuses": {
        "a2_contract": "passed",
        "baseline_curvature": "passed",
        "concentration_prior_ranges_declared": "passed",
        "concentration_sums_and_ranges": "passed",
        "derivative_statistics": "passed",
        "nonlinear_target_behavior": "not_applicable",
        "peak_density": "passed",
        "spectral_snr": "passed",
        "target_distribution": "passed",
        "wavelengths_and_mode": "passed"
      },
      "domain": "pharma_tablets",
      "downstream_training_status": "allowed",
      "instrument": "foss_xds",
      "key_metrics": {
        "class_counts": {
          "0": 10,
          "1": 10,
          "2": 10,
          "3": 10
        },
        "component_count": 4,
        "expected_classes": [
          0,
          1,
          2,
          3
        ],
        "finite": true,
        "invalid_count": 0,
        "max": 0.7428868958120932,
        "max_row_sum_error": 2.220446049250313e-16,
        "median_baseline_curvature": 0.18893370304147666,
        "median_first_derivative_std": 0.006152177830443308,
        "median_peak_density_per_100nm": 1.857142857142857,
        "median_snr": 19.016109817734943,
        "min": 0.010329983955470967,
        "min_class_fraction": 0.25,
        "n_wavelengths": 351,
        "observed_classes": [
          0,
          1,
          2,
          3
        ],
        "q05": 1.4285714285714284,
        "q95": 2.214285714285714,
        "row_sum_max": 1.0000000000000002,
        "row_sum_min": 0.9999999999999999,
        "wavelength_max": 2500.0,
        "wavelength_min": 1100.0
      },
      "mode": "reflectance",
      "phase_a_gate_override": "A3_failed_documented",
      "preset": "tablets",
      "target_type": "classification",
      "unsupported_checks": [],
      "validation": {
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
                40,
                4
              ]
            },
            "message": "Concentrations are finite, normalized, and within [0, 1].",
            "metrics": {
              "finite": true,
              "max": 0.7428868958120932,
              "max_row_sum_error": 2.220446049250313e-16,
              "min": 0.010329983955470967,
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
              "component_count": 4,
              "invalid_count": 0
            },
            "name": "concentration_prior_ranges_declared",
            "severity": "hard",
            "status": "passed",
            "thresholds": {}
          },
          {
            "details": {},
            "message": "Classification target contains all declared classes with acceptable balance.",
            "metrics": {
              "class_counts": {
                "0": 10,
                "1": 10,
                "2": 10,
                "3": 10
              },
              "expected_classes": [
                0,
                1,
                2,
                3
              ],
              "min_class_fraction": 0.25,
              "observed_classes": [
                0,
                1,
                2,
                3
              ]
            },
            "name": "target_distribution",
            "severity": "hard",
            "status": "passed",
            "thresholds": {
              "min_class_fraction": 0.1
            }
          },
          {
            "details": {},
            "message": "Nonlinear target behavior is only defined for regression targets.",
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
              "median_snr": 19.016109817734943,
              "q05": 17.007062769580692,
              "q95": 20.617776540050844
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
              "median_first_derivative_std": 0.006152177830443308,
              "q05": 0.004845833378197763,
              "q95": 0.008038655030102752
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
              "median_baseline_curvature": 0.18893370304147666,
              "q05": 0.1460157767512129,
              "q95": 0.24502579311493847
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
              "median_peak_density_per_100nm": 1.857142857142857,
              "q05": 1.4285714285714284,
              "q95": 2.214285714285714
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
        "preset": "tablets",
        "summary": {
          "X_shape": [
            40,
            351
          ],
          "blocking_checks": [],
          "domain": "pharma_tablets",
          "instrument": "foss_xds",
          "measurement_mode": "reflectance",
          "target_type": "classification",
          "unsupported_checks": [],
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_shape": [
            40
          ]
        },
        "validation_status": "passed"
      },
      "validation_status": "passed",
      "y_shape": [
        40
      ]
    },
    {
      "X_shape": [
        40,
        351
      ],
      "blocking_checks": [],
      "check_statuses": {
        "a2_contract": "passed",
        "baseline_curvature": "passed",
        "concentration_prior_ranges_declared": "passed",
        "concentration_sums_and_ranges": "passed",
        "derivative_statistics": "passed",
        "nonlinear_target_behavior": "not_applicable",
        "peak_density": "passed",
        "spectral_snr": "passed",
        "target_distribution": "passed",
        "wavelengths_and_mode": "passed"
      },
      "domain": "pharma_powder_blends",
      "downstream_training_status": "allowed",
      "instrument": "foss_xds",
      "key_metrics": {
        "component_count": 3,
        "finite": true,
        "invalid_count": 0,
        "max": 0.8070321186992282,
        "max_row_sum_error": 2.220446049250313e-16,
        "median_baseline_curvature": 0.17260303896774476,
        "median_first_derivative_std": 0.005701400088324971,
        "median_peak_density_per_100nm": 1.9642857142857142,
        "median_snr": 19.84344636853669,
        "min": 0.032190875851008396,
        "n_wavelengths": 351,
        "q05": 1.5678571428571428,
        "q95": 2.432142857142857,
        "row_sum_max": 1.0000000000000002,
        "row_sum_min": 0.9999999999999998,
        "wavelength_max": 2500.0,
        "wavelength_min": 1100.0,
        "y_max": 1.0,
        "y_min": 0.0,
        "y_std": 0.24015867298744825
      },
      "mode": "reflectance",
      "phase_a_gate_override": "A3_failed_documented",
      "preset": "powders",
      "target_type": "regression",
      "unsupported_checks": [],
      "validation": {
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
                40,
                3
              ]
            },
            "message": "Concentrations are finite, normalized, and within [0, 1].",
            "metrics": {
              "finite": true,
              "max": 0.8070321186992282,
              "max_row_sum_error": 2.220446049250313e-16,
              "min": 0.032190875851008396,
              "row_sum_max": 1.0000000000000002,
              "row_sum_min": 0.9999999999999998
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
              "y_max": 1.0,
              "y_min": 0.0,
              "y_std": 0.24015867298744825
            },
            "name": "target_distribution",
            "severity": "hard",
            "status": "passed",
            "thresholds": {
              "min_std": 1e-08,
              "range": [
                0.0,
                1.0
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
              "median_snr": 19.84344636853669,
              "q05": 17.45122411344914,
              "q95": 23.667902759968513
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
              "median_first_derivative_std": 0.005701400088324971,
              "q05": 0.004272387859196118,
              "q95": 0.007838713557809263
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
              "median_baseline_curvature": 0.17260303896774476,
              "q05": 0.12693218353483746,
              "q95": 0.24335178179540917
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
              "median_peak_density_per_100nm": 1.9642857142857142,
              "q05": 1.5678571428571428,
              "q95": 2.432142857142857
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
        "preset": "powders",
        "summary": {
          "X_shape": [
            40,
            351
          ],
          "blocking_checks": [],
          "domain": "pharma_powder_blends",
          "instrument": "foss_xds",
          "measurement_mode": "reflectance",
          "target_type": "regression",
          "unsupported_checks": [],
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_shape": [
            40
          ]
        },
        "validation_status": "passed"
      },
      "validation_status": "passed",
      "y_shape": [
        40
      ]
    },
    {
      "X_shape": [
        40,
        201
      ],
      "blocking_checks": [],
      "check_statuses": {
        "a2_contract": "passed",
        "baseline_curvature": "passed",
        "concentration_prior_ranges_declared": "passed",
        "concentration_sums_and_ranges": "passed",
        "derivative_statistics": "passed",
        "nonlinear_target_behavior": "not_applicable",
        "peak_density": "passed",
        "spectral_snr": "passed",
        "target_distribution": "passed",
        "wavelengths_and_mode": "passed"
      },
      "domain": "petrochem_fuels",
      "downstream_training_status": "allowed",
      "instrument": "foss_xds",
      "key_metrics": {
        "class_counts": {
          "0": 20,
          "1": 20
        },
        "component_count": 3,
        "expected_classes": [
          0,
          1
        ],
        "finite": true,
        "invalid_count": 0,
        "max": 0.7433560661254626,
        "max_row_sum_error": 2.220446049250313e-16,
        "median_baseline_curvature": 0.1049406015973598,
        "median_first_derivative_std": 0.004103665665593942,
        "median_peak_density_per_100nm": 2.375,
        "median_snr": 15.96298358905475,
        "min": 0.005576445212293542,
        "min_class_fraction": 0.5,
        "n_wavelengths": 201,
        "observed_classes": [
          0,
          1
        ],
        "q05": 1.74375,
        "q95": 2.875,
        "row_sum_max": 1.0000000000000002,
        "row_sum_min": 0.9999999999999998,
        "wavelength_max": 1700.0,
        "wavelength_min": 900.0
      },
      "mode": "reflectance",
      "phase_a_gate_override": "A3_failed_documented",
      "preset": "fuel",
      "target_type": "classification",
      "unsupported_checks": [],
      "validation": {
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
                40,
                3
              ]
            },
            "message": "Concentrations are finite, normalized, and within [0, 1].",
            "metrics": {
              "finite": true,
              "max": 0.7433560661254626,
              "max_row_sum_error": 2.220446049250313e-16,
              "min": 0.005576445212293542,
              "row_sum_max": 1.0000000000000002,
              "row_sum_min": 0.9999999999999998
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
            "message": "Classification target contains all declared classes with acceptable balance.",
            "metrics": {
              "class_counts": {
                "0": 20,
                "1": 20
              },
              "expected_classes": [
                0,
                1
              ],
              "min_class_fraction": 0.5,
              "observed_classes": [
                0,
                1
              ]
            },
            "name": "target_distribution",
            "severity": "hard",
            "status": "passed",
            "thresholds": {
              "min_class_fraction": 0.1
            }
          },
          {
            "details": {},
            "message": "Nonlinear target behavior is only defined for regression targets.",
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
              "n_wavelengths": 201,
              "wavelength_max": 1700.0,
              "wavelength_min": 900.0
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
              "median_snr": 15.96298358905475,
              "q05": 12.72203391239212,
              "q95": 21.894619120583044
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
              "median_first_derivative_std": 0.004103665665593942,
              "q05": 0.003331962794535303,
              "q95": 0.0053200700599404915
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
              "median_baseline_curvature": 0.1049406015973598,
              "q05": 0.08545408300493298,
              "q95": 0.13269005209170784
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
              "median_peak_density_per_100nm": 2.375,
              "q05": 1.74375,
              "q95": 2.875
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
        "preset": "fuel",
        "summary": {
          "X_shape": [
            40,
            201
          ],
          "blocking_checks": [],
          "domain": "petrochem_fuels",
          "instrument": "foss_xds",
          "measurement_mode": "reflectance",
          "target_type": "classification",
          "unsupported_checks": [],
          "wavelength_range_nm": [
            900.0,
            1700.0
          ],
          "y_shape": [
            40
          ]
        },
        "validation_status": "passed"
      },
      "validation_status": "passed",
      "y_shape": [
        40
      ]
    }
  ]
}
```

## Decision

Pass B1 prior predictive smoke gate for these presets; downstream training is allowed for all rows.
