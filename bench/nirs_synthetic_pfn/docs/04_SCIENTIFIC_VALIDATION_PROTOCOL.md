# Scientific Validation Protocol

## Principle

The synthetic project succeeds only if synthetic data improves real modeling or
reveals useful failure modes. Visual plausibility is not enough.

Validation is layered:

1. contract validation;
2. prior predictive checks;
3. real/synthetic realism checks;
4. downstream transfer checks;
5. model-specific invariance and ablation checks.

## Layer 1: Contract Validation

Required for every generated dataset or task:

- arrays are finite;
- shapes match metadata;
- wavelength grids are monotonic and compatible with the selected instrument;
- component keys resolve;
- target type matches target values;
- class labels and regression ranges are valid;
- seeds reproduce identical outputs;
- train/query/test split metadata is present;
- no context/query leakage is possible through shared preprocessing state.

Failure policy:

- contract failure stops the experiment;
- the report must classify the failure as prior, adapter, generator, target, or
  validation issue.

## Layer 2: Prior Predictive Checks

For each preset and broad prior sample:

- domain coverage;
- component coverage;
- instrument and detector coverage;
- measurement mode coverage;
- concentration distributions;
- target distributions;
- batch/group distribution;
- SNR distribution;
- derivative statistics;
- baseline curvature;
- peak density;
- wavelength range distribution;
- missing or unsupported field count.

Minimum smoke target:

- 1000 prior samples summarized;
- 10 named presets validated;
- no silent fallback to invalid generic components.

## Layer 3: Real/Synthetic Realism

Metrics:

- spectral mean and variance profiles;
- derivative mean and variance profiles;
- correlation length;
- SNR;
- baseline curvature;
- peak density;
- PCA overlap;
- adversarial AUC real vs synthetic;
- nearest-neighbor distance ratio;
- domain-specific scorecard.

Initial provisional thresholds:

| Metric | Smoke Threshold | Interpretation |
|---|---:|---|
| finite spectra | 100 percent | hard contract |
| adversarial AUC | below 0.85 | synthetic not trivially separable |
| adversarial AUC stretch | below 0.70 | stronger realism target |
| SNR range | overlaps real IQR | domain-specific |
| derivative stats | no order-of-magnitude gap | smoke realism |
| PCA overlap | non-empty overlap | smoke realism |

These thresholds are provisional. They should be tightened per domain after
observing real benchmark distributions.

## Layer 4: Downstream Transfer

Required baselines:

- PLS with standard preprocessing;
- ASLSBaseline/PCA/TabPFN where available;
- existing `bench/_tabpfn` spectral latent features where practical;
- real-only training baseline;
- synthetic-only TSTR;
- synthetic pretraining or feature learning plus real evaluation.

Primary local benchmark sources:

- TabPFN paper regression data and reference metrics from
  `bench/tabpfn_paper/master_results.csv` and
  `bench/tabpfn_paper/data/regression/`.
- AOM-PLS regression and classification cohorts from
  `bench/AOM_v0/benchmarks/cohort_regression.csv` and
  `bench/AOM_v0/benchmarks/cohort_classification.csv`.
- Classification data from `bench/tabpfn_paper/data/classification/`.

The preferred full validation set is the consolidated 57-dataset
regression/classification reference set used by the TabPFN paper and AOM-PLS
work, when it is present locally. Loaders must not assume the count blindly:
filter for runnable rows, validate paths, and report the exact number of
datasets, regression tasks, and classification tasks used.

Core experiments:

- TSTR: train synthetic, test real;
- RTSR: use synthetic to pretrain/select features, train/evaluate on real;
- few-shot curves: n=10, 25, 50, 100 where available;
- cross-instrument transfer;
- cross-domain same-analyte transfer when data exists.

Reporting:

- repeated splits;
- mean and standard deviation;
- RMSE, MAE, R2 for regression;
- balanced accuracy, F1, ROC AUC for classification;
- inference time where model changes affect deployment.

Success signal:

- target: greater than 5 percent average RMSE improvement over the current best
  PCA/TabPFN-style baseline, or greater than 10 percent few-shot improvement;
- acceptable research signal: one difficult transfer or few-shot setting improves
  consistently while average performance is competitive;
- stop signal: synthetic-trained features are consistently worse and ablations
  do not identify a fixable prior issue.

## Layer 5: Encoder Invariance Validation

For multi-view encoder work:

- same latent, different views -> embeddings close;
- different latent, same view -> embeddings separated;
- analyte signal remains predictable from embeddings;
- instrument and batch identity become less predictable when invariance is
  intended;
- nearest-neighbor retrieval returns same latent more often than chance;
- downstream TabPFN/PLS/RF metrics improve or remain competitive.

Required ablations:

- no multi-view training;
- no instrument variation;
- no scatter variation;
- no baseline variation;
- no measurement mode variation;
- products/aggregates off;
- procedural components off;
- contrastive-only vs contrastive plus auxiliary target loss, if implemented.

## NIRS-ICL/PFN Task Validation

Every sampled task must report:

- context size;
- query size;
- target type and target function;
- domain and instrument split;
- wavelength policy;
- label noise policy;
- batch/group shift;
- class balance or regression range;
- whether context and query share instrument, domain, or analyte.

Task difficulty checks:

- constant baseline score;
- PLS score;
- TabPFN/PCA score;
- oracle-like synthetic latent score when available.

Tasks are invalid if:

- all models saturate trivially;
- no model beats a constant baseline;
- target is leaked by metadata;
- train/query split violates the declared shift.

## Required Report Sections

Each report should contain:

1. objective;
2. command;
3. config;
4. git status summary;
5. seed policy;
6. dataset/task summary;
7. contract checks;
8. metrics;
9. ablations or missing ablations;
10. failures and unsupported fields;
11. decision.

## Publication-Quality Evidence

A result is publication-quality only if it includes:

- multiple real datasets, preferably the TabPFN/AOM-PLS 57-dataset
  regression/classification reference set when locally available, or one deeply
  analyzed domain with a clear reason for not using the full cohort;
- repeated split statistics;
- ablations showing which synthetic effects matter;
- negative results for failed priors;
- clear comparison to standard chemometric baselines;
- reproducible config and command;
- no reliance on a private undocumented dataset without a public substitute or
  clear data access note.
