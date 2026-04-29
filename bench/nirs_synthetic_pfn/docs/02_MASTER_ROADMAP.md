# Master Roadmap

## North Star

Build a scientific bench project that proves whether synthetic NIRS priors can
improve real NIRS modeling. The immediate target is executable synthetic tasks
and a validated spectral encoder plus TabPFN workflow. Production integration
happens only after bench validation.

## Roadmap A: Executable Synthetic Prior

Goal: make the prior produce valid datasets, metadata, and reports.

Duration: 3 to 4 weeks.

### A0. Project Setup

Status: document phase.

Deliverables:

- bench directory and driver docs;
- project scope;
- validation protocol;
- agent allocation;
- integration gate.

Acceptance:

- docs identify ownership, contracts, gates, and non-goals;
- no production package changes are required.

### A1. Prior Canonicalization

Goal: convert current prior samples into canonical, validated records.

Tasks:

- map short prior domains to `APPLICATION_DOMAINS` keys;
- validate component keys against the component library;
- validate instrument, wavelength, mode, matrix, and target choices;
- add explicit failure messages for impossible prior samples;
- snapshot 1000 prior samples and summarize coverage.

Deliverables:

- `PriorConfigRecord` implementation;
- coverage report by domain, component, instrument, and measurement mode;
- tests for invalid domains/components and reproducibility.

Gate:

- 1000 sampled configs either validate or fail with classified reasons;
- at least 10 canonical domain presets build without fallback components.

### A2. Prior-to-Dataset Adapter

Goal: turn canonical prior records into `SyntheticDatasetRun`.

Tasks:

- map target priors to builder calls;
- map nuisance priors to generator/builder configs or bench-side adapters;
- expose environment, scatter, edge, batch, and measurement fields in run
  metadata;
- produce regression and classification datasets;
- record builder config and provenance.

Deliverables:

- smoke experiment generating 10 datasets from 10 presets;
- shape/finite/range checks;
- report with target distributions and spectral summaries.

Gate:

- every preset builds a finite dataset with stable seeds;
- target distributions match the declared target type;
- metadata contains domain, instrument, mode, target, and nuisance state.

### A3. Real-Fit Adapter

Goal: use `RealDataFitter.to_full_config()` as an executable bench contract.

Tasks:

- fit synthetic-real-like datasets;
- convert full fitted configs into dataset runs;
- compare source and regenerated scorecards;
- log unsupported fields as explicit TODOs.

Deliverables:

- `fitted_config_adapter`;
- report: fit synthetic -> regenerate -> compare.

Gate:

- no silent dropping of fitted fields;
- unsupported fields are listed in reports;
- regenerated spectra pass basic similarity checks on synthetic-real-like data.

## Roadmap B: Scientific Validation and Presets

Goal: make synthetic data scientifically auditable.

Duration: 2 to 3 weeks, partly parallel with Roadmap A.

### B1. Prior Predictive Checks

Tasks:

- validate concentration ranges and sums;
- validate target ranges, class balance, nonlinear target behavior;
- validate spectral SNR, derivative statistics, baseline curvature, peak density;
- flag mode/instrument conflicts.

Deliverables:

- `prior_checks.py`;
- preset report table;
- failing-case examples.

Gate:

- every experiment starts with prior checks;
- failed checks stop downstream model training.

### B2. Real/Synthetic Scorecards

Tasks:

- standardize scorecard output;
- adversarial AUC real vs synthetic;
- PCA/UMAP overlap report;
- derivative and noise distribution report;
- domain-specific thresholds marked as provisional.

Deliverables:

- markdown report generator;
- CSV metrics summary;
- figures saved under `reports/`.

Gate:

- the TabPFN paper and AOM-PLS benchmark cohorts have a documented comparison
  route, including the consolidated 57-dataset regression/classification
  reference set when locally available;
- if the full cohort is unavailable, at least 4 public or local benchmark-like
  datasets have a documented comparison route and the missing datasets are
  reported explicitly;
- thresholds are explicit, even when provisional.

### B3. Transfer Validation

Tasks:

- TSTR: train on synthetic, test on real;
- RTSR: pretrain or select on synthetic, finetune/evaluate on real;
- real-only baselines with PLS, ASLSBaseline/PCA/TabPFN, and spectral latent
  features where available;
- use `bench/tabpfn_paper/master_results.csv` and the AOM-PLS cohort CSV files
  as primary real-data benchmark references when available;
- ablations: without instruments, without scatter, without products/aggregates,
  without procedural diversity.

Deliverables:

- transfer benchmark script;
- repeated split report;
- ablation report.

Gate:

- no claim of synthetic usefulness without a real-data baseline;
- improvements include confidence intervals or repeated splits.

## Roadmap C: Multi-View Encoder and TabPFN

Goal: prove that synthetic priors can train a useful representation.

Duration: 4 to 6 weeks after A1/A2 smoke gates.

### C1. Canonical Latents and Multi-View Rendering

Tasks:

- define `CanonicalLatentBatch`;
- sample common compositions and nuisance variables;
- render same latent through multiple instruments/modes/grids;
- create positive/negative pairs for contrastive learning;
- test same-latent alignment and different-latent separation.

Deliverables:

- multi-view factory;
- pair dataloader;
- report with examples and invariance metrics.

Gate:

- same latent can render at least 2 instruments or views with traceable metadata;
- latent ids survive all rendering and split operations.

### C2. Encoder Prototypes

Tasks:

- implement lightweight CNN encoder;
- implement patch transformer encoder;
- implement sklearn-compatible transform wrapper;
- implement InfoNCE and optional reconstruction/target auxiliary losses;
- log training curves and checkpoints.

Deliverables:

- encoder smoke tests;
- training script;
- first checkpoint;
- embedding diagnostic report.

Gate:

- training loss converges on held-out synthetic batches;
- embeddings preserve analyte signal in a linear probe;
- instrument/batch predictability from embeddings is lower than from raw spectra
  when invariance is intended.

### C3. Encoder plus TabPFN Evaluation

Tasks:

- compare raw/PCA/latent-feature/encoder inputs to TabPFN;
- include PLS and current best pipeline baselines;
- evaluate few-shot curves at n=10, 25, 50, 100 when dataset size allows;
- test with and without ASLSBaseline before encoder;
- ablate view diversity and augmentation families.

Deliverables:

- benchmark report;
- model card for checkpoint;
- go/no-go recommendation.

Gate:

- target success: average RMSE improvement greater than 5 percent or clear
  few-shot improvement greater than 10 percent on repeated real benchmarks;
- minimum useful result: competitive with PCA while improving at least one
  difficult few-shot or transfer setting;
- if worse than baseline, stop full-PFN work and diagnose the prior.

## Roadmap D: NIRS-ICL/PFN Task Prior

Goal: generate context/query episodes for ICL or eventual PFN training.

Duration: 4 to 8 weeks after Roadmap A and initial Roadmap C validation.

### D1. Task Object and Sampler

Tasks:

- implement `NIRSPriorTask`;
- sample context/query sizes;
- support regression, classification, multi-target, nonlinear targets;
- support group/batch/instrument/domain shifts;
- support variable wavelength policies.

Deliverables:

- task sampler;
- task validation tests;
- task visualization report.

Gate:

- 1000 sampled tasks pass shape, target, split, and metadata contracts;
- leakage checks prove query labels are not used in context preprocessing.

### D2. ICL Baselines

Tasks:

- evaluate TabPFN on PCA, spectral latent features, and learned encoder;
- evaluate calibration-transfer tasks;
- evaluate same-analyte cross-domain tasks;
- compare to standard supervised baselines.

Deliverables:

- ICL benchmark report;
- curriculum recommendations.

Gate:

- ICL tasks produce stable, non-trivial baselines;
- synthetic task difficulty is calibrated, not saturated.

### D3. Full PFN Feasibility Review

Tasks:

- estimate training data volume and compute;
- choose architecture only if encoder/ICL results justify it;
- define uncertainty/calibration objectives;
- define real-data validation plan before training.

Deliverables:

- full PFN feasibility memo;
- go/no-go decision.

Gate:

- no full PFN training before a documented positive signal from Roadmap C or D2.

## Roadmap E: Production Integration

Goal: move validated pieces into `nirs4all` with minimal API surface.

Duration: 2 to 4 weeks after success gates.

Candidate ports:

- domain alias and prior validation;
- `prior_to_builder_config` or public equivalent;
- task dataclasses if stable;
- builder methods for environment/scatter/edge effects;
- measurement mode wiring if validated;
- label-preserving augmentation contracts;
- encoder transformer only if model benefits are stable and weights policy is
  decided.

Gate:

- see `05_INTEGRATION_GATE.md`.

## Suggested Timeline

| Week | Main Focus | Parallel Focus | Decision |
|---|---|---|---|
| 1 | A1 prior canonicalization | B1 checks | Can the prior validate? |
| 2 | A2 prior-to-dataset | B2 scorecards | Can presets build? |
| 3 | A3 real-fit adapter | B3 transfer baselines | Is validation runnable? |
| 4 | C1 multi-view rendering | B3 ablations | Can same latent make views? |
| 5 | C2 encoder training | reports | Does encoder learn invariances? |
| 6 | C3 real evaluation | ablations | Continue, iterate, or stop? |
| 7 | D1 task sampler | C3 repeats | Are ICL tasks valid? |
| 8 | D2 ICL baselines | integration memo | Is PFN justified? |
