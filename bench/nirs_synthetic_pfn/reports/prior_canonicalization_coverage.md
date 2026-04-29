# Prior Canonicalization Coverage

## Objective

Validate Phase A1 canonicalization coverage for `PriorSampler` samples in two regimes:
raw (no repair) and canonical (components re-sampled from canonical domain).

## Command

`PYTHONPATH=bench/nirs_synthetic_pfn/src python bench/nirs_synthetic_pfn/experiments/prior_coverage.py --n-samples 1000 --seed 20260428`

## Summary

| regime | samples | valid | invalid |
|---|---:|---:|---:|
| raw | 1000 | 6 | 994 |
| canonical (repaired) | 1000 | 918 | 82 |

- Seed: 20260428

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

## Raw Coverage (no repair)

### Invalid Reasons

| reason | count |
|---|---:|
| `domain_component_mismatch` | 893 |
| `invalid_component` | 824 |
| `invalid_target_prior` | 273 |
| `wavelength_domain_mismatch` | 66 |

### Invalid Fields

| field | count |
|---|---:|
| `components` | 1717 |
| `target_config.n_targets` | 273 |
| `wavelength_range` | 66 |

### Source Domains

| source_domain | count |
|---|---:|
| `grain` | 143 |
| `dairy` | 105 |
| `tablets` | 101 |
| `oilseeds` | 73 |
| `forage` | 71 |
| `powders` | 57 |
| `fruit` | 53 |
| `liquids` | 49 |
| `meat` | 47 |
| `beverages` | 46 |
| `fuel` | 44 |
| `water_quality` | 42 |
| `polymers` | 41 |
| `baking` | 34 |
| `soil` | 27 |
| `tissue` | 23 |
| `lubricants` | 21 |
| `blood` | 15 |
| `textiles` | 8 |

### Validated Domains

| domain | count |
|---|---:|
| `biomedical_tissue` | 3 |
| `agriculture_forage` | 2 |
| `agriculture_grain` | 1 |

### Validated Components

| component | count |
|---|---:|
| `lipid` | 6 |
| `protein` | 6 |
| `cellulose` | 3 |
| `water` | 3 |

## Canonical Coverage (components re-sampled from canonical domain)

### Invalid Reasons

| reason | count |
|---|---:|
| `wavelength_domain_mismatch` | 82 |

### Invalid Fields

| field | count |
|---|---:|
| `wavelength_range` | 82 |

### Repairs Applied

| repair | count |
|---|---:|
| `components_resampled_from_canonical_domain` | 1000 |
| `target_n_targets_clipped_to_component_count` | 8 |

### Validated Domains

| domain | count |
|---|---:|
| `agriculture_grain` | 122 |
| `pharma_tablets` | 101 |
| `food_dairy` | 99 |
| `agriculture_forage` | 74 |
| `agriculture_oilseeds` | 70 |
| `petrochem_fuels` | 61 |
| `pharma_raw_materials` | 60 |
| `food_meat` | 48 |
| `beverage_juice` | 46 |
| `biomedical_tissue` | 39 |
| `pharma_powder_blends` | 38 |
| `agriculture_fruit` | 36 |
| `petrochem_polymers` | 35 |
| `environmental_water` | 34 |
| `environmental_soil` | 28 |
| `food_bakery` | 20 |
| `textile_natural` | 7 |

### Validated Instruments

| instrument | count |
|---|---:|
| `metrohm_ds2500` | 155 |
| `unity_spectrastar` | 133 |
| `foss_xds` | 119 |
| `buchi_nirmaster` | 89 |
| `nir_o_process` | 73 |
| `asd_fieldspec` | 69 |
| `viavi_micronir` | 45 |
| `thermo_antaris` | 41 |
| `perkin_spectrum_two` | 33 |
| `tellspec` | 33 |
| `bruker_mpa` | 30 |
| `siware_neoscanner` | 20 |
| `linksquare` | 17 |
| `abb_mb3600` | 15 |
| `innospectra` | 13 |
| `scio` | 10 |
| `neospectra_micro` | 9 |
| `foss_infratec` | 8 |
| `perten_da7200` | 6 |

### Validated Measurement Modes

| mode | count |
|---|---:|
| `reflectance` | 465 |
| `transmittance` | 248 |
| `transflectance` | 157 |
| `atr` | 48 |

### Validated Components

| component | count |
|---|---:|
| `moisture` | 372 |
| `protein` | 342 |
| `starch` | 309 |
| `cellulose` | 291 |
| `lipid` | 237 |
| `water` | 225 |
| `lactose` | 173 |
| `glucose` | 152 |
| `paracetamol` | 117 |
| `caffeine` | 116 |
| `hemicellulose` | 107 |
| `aspirin` | 105 |
| `unsaturated_fat` | 86 |
| `gluten` | 84 |
| `sucrose` | 84 |
| `oil` | 82 |
| `dietary_fiber` | 81 |
| `casein` | 71 |
| `lignin` | 58 |
| `urea` | 58 |
| `fructose` | 55 |
| `ethanol` | 49 |
| `malic_acid` | 46 |
| `saturated_fat` | 44 |
| `alkane` | 43 |
| `aromatic` | 42 |
| `citric_acid` | 40 |
| `methanol` | 39 |
| `collagen` | 38 |
| `carotenoid` | 37 |
| `polyethylene` | 32 |
| `polyester` | 30 |
| `acetic_acid` | 29 |
| `nitrogen_compound` | 29 |
| `nylon` | 29 |
| `natural_rubber` | 28 |
| `polystyrene` | 27 |
| `carbonates` | 17 |
| `gypsum` | 14 |
| `kaolinite` | 14 |

## Raw Summary JSON

```json
{
  "canonical": {
    "component_counts": {
      "acetic_acid": 29,
      "alkane": 43,
      "aromatic": 42,
      "aspirin": 105,
      "caffeine": 116,
      "carbonates": 17,
      "carotenoid": 37,
      "casein": 71,
      "cellulose": 291,
      "citric_acid": 40,
      "collagen": 38,
      "cotton": 3,
      "dietary_fiber": 81,
      "ethanol": 49,
      "fructose": 55,
      "glucose": 152,
      "gluten": 84,
      "gypsum": 14,
      "hemicellulose": 107,
      "kaolinite": 14,
      "lactose": 173,
      "lignin": 58,
      "lipid": 237,
      "malic_acid": 46,
      "methanol": 39,
      "moisture": 372,
      "natural_rubber": 28,
      "nitrogen_compound": 29,
      "nylon": 29,
      "oil": 82,
      "paracetamol": 117,
      "polyester": 30,
      "polyethylene": 32,
      "polystyrene": 27,
      "protein": 342,
      "saturated_fat": 44,
      "starch": 309,
      "sucrose": 84,
      "unsaturated_fat": 86,
      "urea": 58,
      "water": 225,
      "waxes": 1
    },
    "domain_counts": {
      "agriculture_forage": 74,
      "agriculture_fruit": 36,
      "agriculture_grain": 122,
      "agriculture_oilseeds": 70,
      "beverage_juice": 46,
      "biomedical_tissue": 39,
      "environmental_soil": 28,
      "environmental_water": 34,
      "food_bakery": 20,
      "food_dairy": 99,
      "food_meat": 48,
      "petrochem_fuels": 61,
      "petrochem_polymers": 35,
      "pharma_powder_blends": 38,
      "pharma_raw_materials": 60,
      "pharma_tablets": 101,
      "textile_natural": 7
    },
    "instrument_counts": {
      "abb_mb3600": 15,
      "asd_fieldspec": 69,
      "bruker_mpa": 30,
      "buchi_nirmaster": 89,
      "foss_infratec": 8,
      "foss_xds": 119,
      "innospectra": 13,
      "linksquare": 17,
      "metrohm_ds2500": 155,
      "neospectra_micro": 9,
      "nir_o_process": 73,
      "perkin_spectrum_two": 33,
      "perten_da7200": 6,
      "scio": 10,
      "siware_neoscanner": 20,
      "tellspec": 33,
      "thermo_antaris": 41,
      "unity_spectrastar": 133,
      "viavi_micronir": 45
    },
    "invalid_count": 82,
    "invalid_field_counts": {
      "wavelength_range": 82
    },
    "invalid_reason_counts": {
      "wavelength_domain_mismatch": 82
    },
    "measurement_mode_counts": {
      "atr": 48,
      "reflectance": 465,
      "transflectance": 157,
      "transmittance": 248
    },
    "n_samples": 1000,
    "random_state": 20260428,
    "repair_counts": {
      "components_resampled_from_canonical_domain": 1000,
      "target_n_targets_clipped_to_component_count": 8
    },
    "repair_domain_components": true,
    "source_domain_counts": {
      "baking": 22,
      "beverages": 47,
      "blood": 15,
      "dairy": 103,
      "forage": 96,
      "fruit": 39,
      "fuel": 47,
      "grain": 140,
      "liquids": 60,
      "lubricants": 14,
      "meat": 48,
      "oilseeds": 78,
      "polymers": 36,
      "powders": 44,
      "soil": 38,
      "tablets": 104,
      "textiles": 8,
      "tissue": 24,
      "water_quality": 37
    },
    "valid_count": 918
  },
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
  "raw": {
    "component_counts": {
      "cellulose": 3,
      "lipid": 6,
      "protein": 6,
      "water": 3
    },
    "domain_counts": {
      "agriculture_forage": 2,
      "agriculture_grain": 1,
      "biomedical_tissue": 3
    },
    "instrument_counts": {
      "foss_xds": 2,
      "thermo_antaris": 1,
      "unity_spectrastar": 2,
      "viavi_micronir": 1
    },
    "invalid_count": 994,
    "invalid_field_counts": {
      "components": 1717,
      "target_config.n_targets": 273,
      "wavelength_range": 66
    },
    "invalid_reason_counts": {
      "domain_component_mismatch": 893,
      "invalid_component": 824,
      "invalid_target_prior": 273,
      "wavelength_domain_mismatch": 66
    },
    "measurement_mode_counts": {
      "reflectance": 4,
      "transflectance": 1,
      "transmittance": 1
    },
    "n_samples": 1000,
    "random_state": 20260428,
    "repair_counts": {},
    "repair_domain_components": false,
    "source_domain_counts": {
      "baking": 34,
      "beverages": 46,
      "blood": 15,
      "dairy": 105,
      "forage": 71,
      "fruit": 53,
      "fuel": 44,
      "grain": 143,
      "liquids": 49,
      "lubricants": 21,
      "meat": 47,
      "oilseeds": 73,
      "polymers": 41,
      "powders": 57,
      "soil": 27,
      "tablets": 101,
      "textiles": 8,
      "tissue": 23,
      "water_quality": 42
    },
    "valid_count": 6
  }
}
```

## Decision

Raw validation rate 0.6% confirms production `PriorSampler` falls back to generic components for unknown domain aliases. Canonical sampling (repair_domain_components=True) raises validation to 91.8%, with the remaining failures attributable to wavelength/domain overlap rather than component identity. Target-count clipping occurred in 8 samples and is recorded in `_canonical_repairs` alongside the original raw config under `_raw_prior_config`.
