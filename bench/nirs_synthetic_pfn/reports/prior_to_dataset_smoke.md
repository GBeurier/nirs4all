# Prior-to-Dataset Smoke

## Objective

Generate 10 finite synthetic datasets from 10 canonical A1-style records.

## Command

`PYTHONPATH=bench/nirs_synthetic_pfn/src python bench/nirs_synthetic_pfn/experiments/exp00_smoke_prior_dataset.py --n-samples 40 --seed 20260429`

## Summary

- Seed base: 20260429
- Samples per dataset: 40
- Passed: 10 / 10

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

## Dataset Summary

| preset | domain | instrument | mode | target | X shape | y shape | y min | y max | status |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `grain` | `agriculture_grain` | `foss_xds` | `reflectance` | `regression` | `[40, 351]` | `[40]` | 0.3 | 0.8 | `passed` |
| `forage` | `agriculture_forage` | `foss_xds` | `reflectance` | `classification` | `[40, 351]` | `[40]` | 0 | 2 | `passed` |
| `oilseeds` | `agriculture_oilseeds` | `foss_xds` | `reflectance` | `regression` | `[40, 351]` | `[40]` | 0.1 | 0.45 | `passed` |
| `fruit` | `agriculture_fruit` | `foss_xds` | `reflectance` | `classification` | `[40, 101]` | `[40]` | 0 | 1 | `passed` |
| `dairy` | `food_dairy` | `foss_xds` | `reflectance` | `regression` | `[40, 351]` | `[40]` | 0.8 | 0.92 | `passed` |
| `meat` | `food_meat` | `foss_xds` | `reflectance` | `classification` | `[40, 201]` | `[40]` | 0 | 2 | `passed` |
| `baking` | `food_bakery` | `foss_xds` | `reflectance` | `regression` | `[40, 351]` | `[40]` | 0 | 1 | `passed` |
| `tablets` | `pharma_tablets` | `foss_xds` | `reflectance` | `classification` | `[40, 351]` | `[40]` | 0 | 3 | `passed` |
| `powders` | `pharma_powder_blends` | `foss_xds` | `reflectance` | `regression` | `[40, 351]` | `[40]` | 0 | 1 | `passed` |
| `fuel` | `petrochem_fuels` | `foss_xds` | `reflectance` | `classification` | `[40, 201]` | `[40]` | 0 | 1 | `passed` |

## Contract Checks

| preset | shape | finite | wavelengths | target | concentrations | seed |
|---|---|---|---|---|---|---:|
| `grain` | `True` | `True` | `True` | `True` | `True` | 20260429 |
| `forage` | `True` | `True` | `True` | `True` | `True` | 20260430 |
| `oilseeds` | `True` | `True` | `True` | `True` | `True` | 20260431 |
| `fruit` | `True` | `True` | `True` | `True` | `True` | 20260432 |
| `dairy` | `True` | `True` | `True` | `True` | `True` | 20260433 |
| `meat` | `True` | `True` | `True` | `True` | `True` | 20260434 |
| `baking` | `True` | `True` | `True` | `True` | `True` | 20260435 |
| `tablets` | `True` | `True` | `True` | `True` | `True` | 20260436 |
| `powders` | `True` | `True` | `True` | `True` | `True` | 20260437 |
| `fuel` | `True` | `True` | `True` | `True` | `True` | 20260438 |

## Unsupported Fields

None for this smoke set. A2 maps target, row-normalized concentration mixtures, temperature, particle-size scatter, edge roll-off, batch, instrument, and mode fields explicitly.

Note: `measurement_mode` is passed to `SyntheticNIRSGenerator` and preserved in metadata; this smoke validates the executable dataset contract, not mode-specific optical physics.

## Residual Risks

- `measurement_mode` is passed through and preserved, but A2 does not validate mode-specific optical physics.
- Concentrations are row-normalized and should be interpreted as normalized latent fractions, not raw domain-prior magnitudes.
- Smoke presets are curated; the repaired-prior sweep reduces risk but does not replace B1 prior predictive checks.
- A2 target mapping is smoke-level; B1/B2 must validate target distributions and realism.

## Provenance

Each run metadata stores `prior_config`, `builder_config`, `validation_summary`, and A1 provenance fields `source_prior_config`, `_raw_prior_config`, and `_canonical_repairs` when present.

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
      "component_keys": [
        "starch",
        "protein",
        "moisture"
      ],
      "domain": "agriculture_grain",
      "instrument": "foss_xds",
      "metadata_keys": [
        "builder_config",
        "domain",
        "generation_metadata",
        "instrument",
        "mode",
        "nuisance",
        "prior_config",
        "provenance_a1",
        "target",
        "validation_summary"
      ],
      "mode": "reflectance",
      "preset": "grain",
      "provenance_a1": {
        "_canonical_repairs": null,
        "_raw_prior_config": null,
        "source_prior_config": {
          "components": [
            "starch",
            "protein",
            "moisture"
          ],
          "domain": "grain",
          "domain_category": "research",
          "instrument": "foss_xds",
          "instrument_category": "benchtop",
          "matrix_type": "solid",
          "measurement_mode": "reflectance",
          "n_samples": 100,
          "noise_level": 1.0,
          "particle_size": 150.0,
          "random_state": 20260429,
          "spectral_resolution": 4.0,
          "target_config": {
            "n_targets": 1,
            "nonlinearity": "none",
            "type": "regression"
          },
          "temperature": 25.0,
          "wavelength_range": [
            400,
            2500
          ]
        }
      },
      "status": "passed",
      "target_type": "regression",
      "validation_summary": {
        "adapter_notes": [
          "measurement_mode is passed to SyntheticNIRSGenerator and preserved in metadata; A2 contract checks do not validate mode-specific optical physics."
        ],
        "checks": {
          "concentrations_row_normalized": true,
          "finite": true,
          "seed": 20260429,
          "shape": true,
          "target_contract": true,
          "wavelengths_monotonic": true
        },
        "failures": [],
        "status": "passed",
        "summary": {
          "X_max": 1.0946534192771353,
          "X_min": -0.1348225556822579,
          "X_shape": [
            40,
            351
          ],
          "concentration_row_sum_max": 1.0,
          "concentration_row_sum_min": 0.9999999999999998,
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_max": 0.8,
          "y_min": 0.3,
          "y_shape": [
            40
          ]
        },
        "unsupported_fields": []
      }
    },
    {
      "component_keys": [
        "protein",
        "moisture",
        "cellulose"
      ],
      "domain": "agriculture_forage",
      "instrument": "foss_xds",
      "metadata_keys": [
        "builder_config",
        "domain",
        "generation_metadata",
        "instrument",
        "mode",
        "nuisance",
        "prior_config",
        "provenance_a1",
        "target",
        "validation_summary"
      ],
      "mode": "reflectance",
      "preset": "forage",
      "provenance_a1": {
        "_canonical_repairs": null,
        "_raw_prior_config": null,
        "source_prior_config": {
          "components": [
            "protein",
            "moisture",
            "cellulose"
          ],
          "domain": "forage",
          "domain_category": "research",
          "instrument": "foss_xds",
          "instrument_category": "benchtop",
          "matrix_type": "solid",
          "measurement_mode": "reflectance",
          "n_samples": 100,
          "noise_level": 1.0,
          "particle_size": 150.0,
          "random_state": 20260430,
          "spectral_resolution": 4.0,
          "target_config": {
            "n_classes": 3,
            "separation": "moderate",
            "type": "classification"
          },
          "temperature": 25.0,
          "wavelength_range": [
            400,
            2500
          ]
        }
      },
      "status": "passed",
      "target_type": "classification",
      "validation_summary": {
        "adapter_notes": [
          "measurement_mode is passed to SyntheticNIRSGenerator and preserved in metadata; A2 contract checks do not validate mode-specific optical physics."
        ],
        "checks": {
          "concentrations_row_normalized": true,
          "finite": true,
          "seed": 20260430,
          "shape": true,
          "target_contract": true,
          "wavelengths_monotonic": true
        },
        "failures": [],
        "status": "passed",
        "summary": {
          "X_max": 1.1944615408997412,
          "X_min": -0.11354353070130471,
          "X_shape": [
            40,
            351
          ],
          "concentration_row_sum_max": 1.0000000000000002,
          "concentration_row_sum_min": 0.9999999999999999,
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_max": 2.0,
          "y_min": 0.0,
          "y_shape": [
            40
          ]
        },
        "unsupported_fields": []
      }
    },
    {
      "component_keys": [
        "oil",
        "protein",
        "moisture"
      ],
      "domain": "agriculture_oilseeds",
      "instrument": "foss_xds",
      "metadata_keys": [
        "builder_config",
        "domain",
        "generation_metadata",
        "instrument",
        "mode",
        "nuisance",
        "prior_config",
        "provenance_a1",
        "target",
        "validation_summary"
      ],
      "mode": "reflectance",
      "preset": "oilseeds",
      "provenance_a1": {
        "_canonical_repairs": null,
        "_raw_prior_config": null,
        "source_prior_config": {
          "components": [
            "oil",
            "protein",
            "moisture"
          ],
          "domain": "oilseeds",
          "domain_category": "research",
          "instrument": "foss_xds",
          "instrument_category": "benchtop",
          "matrix_type": "solid",
          "measurement_mode": "reflectance",
          "n_samples": 100,
          "noise_level": 1.0,
          "particle_size": 150.0,
          "random_state": 20260431,
          "spectral_resolution": 4.0,
          "target_config": {
            "n_targets": 1,
            "nonlinearity": "none",
            "type": "regression"
          },
          "temperature": 25.0,
          "wavelength_range": [
            400,
            2500
          ]
        }
      },
      "status": "passed",
      "target_type": "regression",
      "validation_summary": {
        "adapter_notes": [
          "measurement_mode is passed to SyntheticNIRSGenerator and preserved in metadata; A2 contract checks do not validate mode-specific optical physics."
        ],
        "checks": {
          "concentrations_row_normalized": true,
          "finite": true,
          "seed": 20260431,
          "shape": true,
          "target_contract": true,
          "wavelengths_monotonic": true
        },
        "failures": [],
        "status": "passed",
        "summary": {
          "X_max": 0.8229624361865868,
          "X_min": -0.13581832534265736,
          "X_shape": [
            40,
            351
          ],
          "concentration_row_sum_max": 1.0000000000000002,
          "concentration_row_sum_min": 0.9999999999999998,
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_max": 0.44999999999999996,
          "y_min": 0.1,
          "y_shape": [
            40
          ]
        },
        "unsupported_fields": []
      }
    },
    {
      "component_keys": [
        "water",
        "glucose",
        "fructose"
      ],
      "domain": "agriculture_fruit",
      "instrument": "foss_xds",
      "metadata_keys": [
        "builder_config",
        "domain",
        "generation_metadata",
        "instrument",
        "mode",
        "nuisance",
        "prior_config",
        "provenance_a1",
        "target",
        "validation_summary"
      ],
      "mode": "reflectance",
      "preset": "fruit",
      "provenance_a1": {
        "_canonical_repairs": null,
        "_raw_prior_config": null,
        "source_prior_config": {
          "components": [
            "water",
            "glucose",
            "fructose"
          ],
          "domain": "fruit",
          "domain_category": "research",
          "instrument": "foss_xds",
          "instrument_category": "benchtop",
          "matrix_type": "solid",
          "measurement_mode": "reflectance",
          "n_samples": 100,
          "noise_level": 1.0,
          "particle_size": 150.0,
          "random_state": 20260432,
          "spectral_resolution": 4.0,
          "target_config": {
            "n_classes": 2,
            "separation": "moderate",
            "type": "classification"
          },
          "temperature": 25.0,
          "wavelength_range": [
            400,
            2500
          ]
        }
      },
      "status": "passed",
      "target_type": "classification",
      "validation_summary": {
        "adapter_notes": [
          "measurement_mode is passed to SyntheticNIRSGenerator and preserved in metadata; A2 contract checks do not validate mode-specific optical physics."
        ],
        "checks": {
          "concentrations_row_normalized": true,
          "finite": true,
          "seed": 20260432,
          "shape": true,
          "target_contract": true,
          "wavelengths_monotonic": true
        },
        "failures": [],
        "status": "passed",
        "summary": {
          "X_max": 0.261900127679786,
          "X_min": -0.1265880353435388,
          "X_shape": [
            40,
            101
          ],
          "concentration_row_sum_max": 1.0000000000000002,
          "concentration_row_sum_min": 0.9999999999999998,
          "wavelength_range_nm": [
            700.0,
            1100.0
          ],
          "y_max": 1.0,
          "y_min": 0.0,
          "y_shape": [
            40
          ]
        },
        "unsupported_fields": []
      }
    },
    {
      "component_keys": [
        "water",
        "lactose",
        "casein"
      ],
      "domain": "food_dairy",
      "instrument": "foss_xds",
      "metadata_keys": [
        "builder_config",
        "domain",
        "generation_metadata",
        "instrument",
        "mode",
        "nuisance",
        "prior_config",
        "provenance_a1",
        "target",
        "validation_summary"
      ],
      "mode": "reflectance",
      "preset": "dairy",
      "provenance_a1": {
        "_canonical_repairs": null,
        "_raw_prior_config": null,
        "source_prior_config": {
          "components": [
            "water",
            "lactose",
            "casein"
          ],
          "domain": "dairy",
          "domain_category": "research",
          "instrument": "foss_xds",
          "instrument_category": "benchtop",
          "matrix_type": "solid",
          "measurement_mode": "reflectance",
          "n_samples": 100,
          "noise_level": 1.0,
          "particle_size": 150.0,
          "random_state": 20260433,
          "spectral_resolution": 4.0,
          "target_config": {
            "n_targets": 1,
            "nonlinearity": "none",
            "type": "regression"
          },
          "temperature": 25.0,
          "wavelength_range": [
            400,
            2500
          ]
        }
      },
      "status": "passed",
      "target_type": "regression",
      "validation_summary": {
        "adapter_notes": [
          "measurement_mode is passed to SyntheticNIRSGenerator and preserved in metadata; A2 contract checks do not validate mode-specific optical physics."
        ],
        "checks": {
          "concentrations_row_normalized": true,
          "finite": true,
          "seed": 20260433,
          "shape": true,
          "target_contract": true,
          "wavelengths_monotonic": true
        },
        "failures": [],
        "status": "passed",
        "summary": {
          "X_max": 1.1631685229398987,
          "X_min": -0.12159852732963461,
          "X_shape": [
            40,
            351
          ],
          "concentration_row_sum_max": 1.0000000000000002,
          "concentration_row_sum_min": 0.9999999999999998,
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_max": 0.92,
          "y_min": 0.8,
          "y_shape": [
            40
          ]
        },
        "unsupported_fields": []
      }
    },
    {
      "component_keys": [
        "water",
        "protein",
        "lipid"
      ],
      "domain": "food_meat",
      "instrument": "foss_xds",
      "metadata_keys": [
        "builder_config",
        "domain",
        "generation_metadata",
        "instrument",
        "mode",
        "nuisance",
        "prior_config",
        "provenance_a1",
        "target",
        "validation_summary"
      ],
      "mode": "reflectance",
      "preset": "meat",
      "provenance_a1": {
        "_canonical_repairs": null,
        "_raw_prior_config": null,
        "source_prior_config": {
          "components": [
            "water",
            "protein",
            "lipid"
          ],
          "domain": "meat",
          "domain_category": "research",
          "instrument": "foss_xds",
          "instrument_category": "benchtop",
          "matrix_type": "solid",
          "measurement_mode": "reflectance",
          "n_samples": 100,
          "noise_level": 1.0,
          "particle_size": 150.0,
          "random_state": 20260434,
          "spectral_resolution": 4.0,
          "target_config": {
            "n_classes": 3,
            "separation": "moderate",
            "type": "classification"
          },
          "temperature": 25.0,
          "wavelength_range": [
            400,
            2500
          ]
        }
      },
      "status": "passed",
      "target_type": "classification",
      "validation_summary": {
        "adapter_notes": [
          "measurement_mode is passed to SyntheticNIRSGenerator and preserved in metadata; A2 contract checks do not validate mode-specific optical physics."
        ],
        "checks": {
          "concentrations_row_normalized": true,
          "finite": true,
          "seed": 20260434,
          "shape": true,
          "target_contract": true,
          "wavelengths_monotonic": true
        },
        "failures": [],
        "status": "passed",
        "summary": {
          "X_max": 0.8401579853972733,
          "X_min": -0.10252183305288022,
          "X_shape": [
            40,
            201
          ],
          "concentration_row_sum_max": 1.0000000000000002,
          "concentration_row_sum_min": 0.9999999999999998,
          "wavelength_range_nm": [
            900.0,
            1700.0
          ],
          "y_max": 2.0,
          "y_min": 0.0,
          "y_shape": [
            40
          ]
        },
        "unsupported_fields": []
      }
    },
    {
      "component_keys": [
        "starch",
        "gluten",
        "moisture"
      ],
      "domain": "food_bakery",
      "instrument": "foss_xds",
      "metadata_keys": [
        "builder_config",
        "domain",
        "generation_metadata",
        "instrument",
        "mode",
        "nuisance",
        "prior_config",
        "provenance_a1",
        "target",
        "validation_summary"
      ],
      "mode": "reflectance",
      "preset": "baking",
      "provenance_a1": {
        "_canonical_repairs": null,
        "_raw_prior_config": null,
        "source_prior_config": {
          "components": [
            "starch",
            "gluten",
            "moisture"
          ],
          "domain": "baking",
          "domain_category": "research",
          "instrument": "foss_xds",
          "instrument_category": "benchtop",
          "matrix_type": "solid",
          "measurement_mode": "reflectance",
          "n_samples": 100,
          "noise_level": 1.0,
          "particle_size": 150.0,
          "random_state": 20260435,
          "spectral_resolution": 4.0,
          "target_config": {
            "n_targets": 1,
            "nonlinearity": "none",
            "type": "regression"
          },
          "temperature": 25.0,
          "wavelength_range": [
            400,
            2500
          ]
        }
      },
      "status": "passed",
      "target_type": "regression",
      "validation_summary": {
        "adapter_notes": [
          "measurement_mode is passed to SyntheticNIRSGenerator and preserved in metadata; A2 contract checks do not validate mode-specific optical physics."
        ],
        "checks": {
          "concentrations_row_normalized": true,
          "finite": true,
          "seed": 20260435,
          "shape": true,
          "target_contract": true,
          "wavelengths_monotonic": true
        },
        "failures": [],
        "status": "passed",
        "summary": {
          "X_max": 0.8935261487536198,
          "X_min": -0.14370669557307358,
          "X_shape": [
            40,
            351
          ],
          "concentration_row_sum_max": 1.0000000000000002,
          "concentration_row_sum_min": 0.9999999999999999,
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_max": 1.0,
          "y_min": 0.0,
          "y_shape": [
            40
          ]
        },
        "unsupported_fields": []
      }
    },
    {
      "component_keys": [
        "starch",
        "cellulose",
        "lactose",
        "moisture"
      ],
      "domain": "pharma_tablets",
      "instrument": "foss_xds",
      "metadata_keys": [
        "builder_config",
        "domain",
        "generation_metadata",
        "instrument",
        "mode",
        "nuisance",
        "prior_config",
        "provenance_a1",
        "target",
        "validation_summary"
      ],
      "mode": "reflectance",
      "preset": "tablets",
      "provenance_a1": {
        "_canonical_repairs": null,
        "_raw_prior_config": null,
        "source_prior_config": {
          "components": [
            "starch",
            "cellulose",
            "lactose",
            "moisture"
          ],
          "domain": "tablets",
          "domain_category": "research",
          "instrument": "foss_xds",
          "instrument_category": "benchtop",
          "matrix_type": "solid",
          "measurement_mode": "reflectance",
          "n_samples": 100,
          "noise_level": 1.0,
          "particle_size": 150.0,
          "random_state": 20260436,
          "spectral_resolution": 4.0,
          "target_config": {
            "n_classes": 4,
            "separation": "moderate",
            "type": "classification"
          },
          "temperature": 25.0,
          "wavelength_range": [
            400,
            2500
          ]
        }
      },
      "status": "passed",
      "target_type": "classification",
      "validation_summary": {
        "adapter_notes": [
          "measurement_mode is passed to SyntheticNIRSGenerator and preserved in metadata; A2 contract checks do not validate mode-specific optical physics."
        ],
        "checks": {
          "concentrations_row_normalized": true,
          "finite": true,
          "seed": 20260436,
          "shape": true,
          "target_contract": true,
          "wavelengths_monotonic": true
        },
        "failures": [],
        "status": "passed",
        "summary": {
          "X_max": 1.2603348160174639,
          "X_min": -0.1573831929200846,
          "X_shape": [
            40,
            351
          ],
          "concentration_row_sum_max": 1.0000000000000002,
          "concentration_row_sum_min": 0.9999999999999999,
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_max": 3.0,
          "y_min": 0.0,
          "y_shape": [
            40
          ]
        },
        "unsupported_fields": []
      }
    },
    {
      "component_keys": [
        "starch",
        "cellulose",
        "lactose"
      ],
      "domain": "pharma_powder_blends",
      "instrument": "foss_xds",
      "metadata_keys": [
        "builder_config",
        "domain",
        "generation_metadata",
        "instrument",
        "mode",
        "nuisance",
        "prior_config",
        "provenance_a1",
        "target",
        "validation_summary"
      ],
      "mode": "reflectance",
      "preset": "powders",
      "provenance_a1": {
        "_canonical_repairs": null,
        "_raw_prior_config": null,
        "source_prior_config": {
          "components": [
            "starch",
            "cellulose",
            "lactose"
          ],
          "domain": "powders",
          "domain_category": "research",
          "instrument": "foss_xds",
          "instrument_category": "benchtop",
          "matrix_type": "solid",
          "measurement_mode": "reflectance",
          "n_samples": 100,
          "noise_level": 1.0,
          "particle_size": 150.0,
          "random_state": 20260437,
          "spectral_resolution": 4.0,
          "target_config": {
            "n_targets": 1,
            "nonlinearity": "none",
            "type": "regression"
          },
          "temperature": 25.0,
          "wavelength_range": [
            400,
            2500
          ]
        }
      },
      "status": "passed",
      "target_type": "regression",
      "validation_summary": {
        "adapter_notes": [
          "measurement_mode is passed to SyntheticNIRSGenerator and preserved in metadata; A2 contract checks do not validate mode-specific optical physics."
        ],
        "checks": {
          "concentrations_row_normalized": true,
          "finite": true,
          "seed": 20260437,
          "shape": true,
          "target_contract": true,
          "wavelengths_monotonic": true
        },
        "failures": [],
        "status": "passed",
        "summary": {
          "X_max": 1.334294688770629,
          "X_min": -0.10779886844550071,
          "X_shape": [
            40,
            351
          ],
          "concentration_row_sum_max": 1.0000000000000002,
          "concentration_row_sum_min": 0.9999999999999998,
          "wavelength_range_nm": [
            1100.0,
            2500.0
          ],
          "y_max": 1.0,
          "y_min": 0.0,
          "y_shape": [
            40
          ]
        },
        "unsupported_fields": []
      }
    },
    {
      "component_keys": [
        "alkane",
        "aromatic",
        "oil"
      ],
      "domain": "petrochem_fuels",
      "instrument": "foss_xds",
      "metadata_keys": [
        "builder_config",
        "domain",
        "generation_metadata",
        "instrument",
        "mode",
        "nuisance",
        "prior_config",
        "provenance_a1",
        "target",
        "validation_summary"
      ],
      "mode": "reflectance",
      "preset": "fuel",
      "provenance_a1": {
        "_canonical_repairs": null,
        "_raw_prior_config": null,
        "source_prior_config": {
          "components": [
            "alkane",
            "aromatic",
            "oil"
          ],
          "domain": "fuel",
          "domain_category": "research",
          "instrument": "foss_xds",
          "instrument_category": "benchtop",
          "matrix_type": "solid",
          "measurement_mode": "reflectance",
          "n_samples": 100,
          "noise_level": 1.0,
          "particle_size": 150.0,
          "random_state": 20260438,
          "spectral_resolution": 4.0,
          "target_config": {
            "n_classes": 2,
            "separation": "moderate",
            "type": "classification"
          },
          "temperature": 25.0,
          "wavelength_range": [
            400,
            2500
          ]
        }
      },
      "status": "passed",
      "target_type": "classification",
      "validation_summary": {
        "adapter_notes": [
          "measurement_mode is passed to SyntheticNIRSGenerator and preserved in metadata; A2 contract checks do not validate mode-specific optical physics."
        ],
        "checks": {
          "concentrations_row_normalized": true,
          "finite": true,
          "seed": 20260438,
          "shape": true,
          "target_contract": true,
          "wavelengths_monotonic": true
        },
        "failures": [],
        "status": "passed",
        "summary": {
          "X_max": 0.9901776215331857,
          "X_min": -0.1394058866268075,
          "X_shape": [
            40,
            201
          ],
          "concentration_row_sum_max": 1.0000000000000002,
          "concentration_row_sum_min": 0.9999999999999998,
          "wavelength_range_nm": [
            900.0,
            1700.0
          ],
          "y_max": 1.0,
          "y_min": 0.0,
          "y_shape": [
            40
          ]
        },
        "unsupported_fields": []
      }
    }
  ]
}
```

## Decision

Pass A2 smoke gate.
