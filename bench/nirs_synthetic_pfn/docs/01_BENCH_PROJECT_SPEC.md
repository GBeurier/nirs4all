# Bench Project Specification

## Scope Boundary

Implementation lives under `bench/nirs_synthetic_pfn` until the method is
validated. The bench project may import from `nirs4all`, but it must not require
production package changes for ordinary experiments.

Production integration is a later phase governed by `05_INTEGRATION_GATE.md`.

## Objectives

The bench project must answer four scientific questions:

1. Can the existing synthetic prior be made executable as datasets and tasks?
2. Does the synthetic distribution pass basic realism and prior predictive
   checks?
3. Can multi-view synthetic spectra train a useful spectral representation?
4. Does that representation improve real NIRS prediction, especially few-shot or
   cross-instrument settings?

## Non-Goals

The first bench milestone does not include:

- replacing the public `nirs4all.generate` API;
- training a full NIRS-PFN from scratch;
- packaging pretrained weights into the library;
- promising one universal latent for all domains before validation;
- copying `nirs4all.synthesis` into bench as a fork.

## Planned Package Layout

```text
bench/nirs_synthetic_pfn/
  src/nirsyntheticpfn/
    __init__.py
    adapters/
      prior_adapter.py
      builder_adapter.py
      fitted_config_adapter.py
    data/
      tasks.py
      latents.py
      views.py
      metadata.py
    generation/
      dataset_factory.py
      multiview_factory.py
      shard_writer.py
    encoders/
      cnn.py
      patch_transformer.py
      sklearn_wrapper.py
      losses.py
      train.py
    evaluation/
      prior_checks.py
      realism.py
      transfer.py
      invariance.py
      baselines.py
    reports/
      report_schema.py
      markdown.py
    utils/
      config.py
      seeds.py
      provenance.py
  experiments/
    exp00_smoke_prior_dataset.py
    exp01_prior_scorecards.py
    exp02_multiview_pairs.py
    exp03_encoder_pretraining.py
    exp04_tabpfn_transfer.py
    exp05_icl_task_sampler.py
  configs/
    presets/
    experiments/
  tests/
    test_prior_adapter.py
    test_task_contract.py
    test_multiview_factory.py
    test_validation_reports.py
```

## Core Contracts

### PriorConfigRecord

Canonicalized, validated output from the current `PriorSampler`.

Required fields:

- `domain_key`: canonical `APPLICATION_DOMAINS` key;
- `product_key` or `aggregate_key`, optional;
- `instrument_key`;
- `measurement_mode`;
- `wavelength_policy`;
- `component_keys`;
- `concentration_prior`;
- `nuisance_prior`;
- `target_prior`;
- `task_prior`;
- `random_seed`;
- `source_prior_config`.

Validation:

- domain key exists;
- every component key resolves in the component library or procedural source;
- wavelength range is compatible with instrument and mode;
- target prior has an executable mapping;
- nuisance values are in configured physical ranges.

### SyntheticDatasetRun

Executable output for classic X/y workflows.

Required fields:

- `X`;
- `y`;
- `wavelengths`;
- `metadata`;
- `latent_metadata`, when available;
- `prior_config`;
- `builder_config`;
- `validation_summary`.

### CanonicalLatentBatch

Latent state before rendering instrument or measurement views.

Required fields:

- concentrations;
- component identities and spectra references;
- path length or optical geometry parameters;
- baseline parameters;
- scatter parameters;
- environment parameters;
- particle/sample presentation parameters;
- batch/group ids;
- clean target and noisy target;
- provenance and seed.

This object may be approximate in early phases. The rule is that fields must be
explicit and testable, even if some are initially sampled by adapters.

### SpectralViewBatch

Rendered spectra from one latent batch under one view configuration.

Required fields:

- `X`;
- `wavelengths`;
- `view_config`;
- `instrument_key`;
- `measurement_mode`;
- `preprocessing_state`;
- `noise_state`;
- `latent_ids`;
- `metadata`.

### NIRSPriorTask

Context/query task for ICL or PFN experiments.

Required fields:

- `X_context`, `y_context`;
- `X_query`, `y_query`;
- `wavelengths_context`, `wavelengths_query`;
- `metadata_context`, `metadata_query`;
- `domain_key`;
- `instrument_context`, `instrument_query`;
- `measurement_mode`;
- `target_name`;
- `target_type`;
- `latent_params`;
- `prior_config`;
- `split_policy`;
- `task_seed`.

## Experiment Requirements

Every experiment must write a report under `reports/` with:

- command and git status summary;
- config path and resolved config;
- seeds;
- dataset names and split policy;
- metrics with repeated runs when feasible;
- failure cases;
- decision: pass, iterate, or stop.

## Benchmark Data Sources

The bench project may use the local TabPFN paper and AOM-PLS benchmark data as
real-data evaluation sources:

- `bench/tabpfn_paper/master_results.csv`: TabPFN paper regression reference
  results and model baselines.
- `bench/tabpfn_paper/data/regression/`: train/test regression splits referenced
  by the TabPFN paper benchmark.
- `bench/tabpfn_paper/data/classification/`: classification splits scanned by
  the AOM benchmark tooling.
- `bench/AOM_v0/benchmarks/cohort_regression.csv`: AOM-PLS regression cohort
  with resolved paths and reference RMSE columns.
- `bench/AOM_v0/benchmarks/cohort_classification.csv`: AOM-PLS classification
  cohort with resolved paths.

Use the consolidated 57-dataset regression/classification reference set from
the TabPFN paper and AOM-PLS work when available. Do not hard-code the number in
loaders: derive the exact runnable cohort from the cohort files, rows where
`status == "ok"`, and local file availability, then report the final count in
every benchmark report.

These datasets are valid for:

- real/synthetic scorecards;
- TSTR and RTSR;
- encoder plus TabPFN evaluation;
- few-shot curves;
- NIRS-ICL task sanity checks;
- final go/no-go evidence before production integration.

## Planned Smoke Commands

These commands define the intended developer loop once implementation starts:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src pytest bench/nirs_synthetic_pfn/tests -q

PYTHONPATH=bench/nirs_synthetic_pfn/src \
  python bench/nirs_synthetic_pfn/experiments/exp00_smoke_prior_dataset.py \
  --config bench/nirs_synthetic_pfn/configs/experiments/smoke_prior.yaml

PYTHONPATH=bench/nirs_synthetic_pfn/src \
  python bench/nirs_synthetic_pfn/experiments/exp04_tabpfn_transfer.py \
  --config bench/nirs_synthetic_pfn/configs/experiments/encoder_tabpfn.yaml
```

## Library Interaction

Bench code should use existing package APIs where possible:

- `nirs4all.synthesis.PriorSampler`;
- `nirs4all.synthesis.SyntheticDatasetBuilder`;
- `nirs4all.synthesis.SyntheticNIRSGenerator`;
- `nirs4all.synthesis.RealDataFitter`;
- `nirs4all.synthesis.validation`;
- existing pipeline and TabPFN bench utilities.

When an existing package limitation blocks a bench experiment, implement a bench
adapter first and document the production change proposed by the adapter.
