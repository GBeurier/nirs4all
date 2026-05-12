# Agent Roadmap

This document describes the work distribution for a multi-agent development
cycle. The agents are roles, not production modules. They can be humans or
coding agents, but each role has a clear ownership boundary.

## Agent Roles

| Agent | Role | Primary Ownership | Main Outputs |
|---|---|---|---|
| A0 | Scientific lead | hypotheses, gates, final decisions | decision log, go/no-go calls |
| A1 | Prior and ontology agent | prior canonicalization, domain/component validation | `PriorConfigRecord`, coverage reports |
| A2 | Generator adapter agent | prior-to-builder, dataset runs, fitted config adapter | `SyntheticDatasetRun`, smoke datasets |
| A3 | Latent and multi-view agent | canonical latents, view rendering, pair sampler | `CanonicalLatentBatch`, `SpectralViewBatch` |
| A4 | Encoder and ML agent | CNN/transformer encoders, losses, training scripts | checkpoints, embedding reports |
| A5 | Validation agent | scorecards, transfer benchmarks, ablations | validation scripts, report tables |
| A6 | QA and reproducibility agent | tests, config schema, seeds, provenance | test suite, CI-style commands |
| A7 | Integration agent | later production porting plan | integration proposal and migration PR plan |

## Ownership Rules

- A1 owns prior records but not dataset rendering.
- A2 owns dataset generation adapters but not model training.
- A3 owns latent/view contracts but not downstream TabPFN evaluation.
- A4 owns encoder code and training but not scientific pass/fail thresholds.
- A5 owns metrics and reports but does not change generation behavior without a
  prior issue from A1/A2/A3.
- A6 can touch tests and config plumbing across the bench project.
- A7 stays inactive until bench gates show a stable candidate for production.

## Phase Allocation

### Phase 0: Bench Setup

Lead: A0

Supporting agents: A6

Tasks:

- establish project structure;
- freeze scope boundary;
- write initial docs;
- define smoke commands and report expectations.

Exit criteria:

- project has driver docs;
- no ambiguity about bench vs production.

### Phase 1: Executable Prior

Lead: A1

Supporting agents: A2, A6

A1 tasks:

- build canonical domain alias table;
- align prior domain names with application domains;
- validate component identities;
- classify invalid prior samples;
- produce coverage summaries.

A2 tasks:

- map valid prior records to builder/generator calls;
- implement target mapping for regression and classification;
- record builder config and metadata;
- expose unsupported fields in reports.

A6 tasks:

- tests for seed stability;
- tests for invalid prior failure modes;
- tests for finite generated arrays.

Exit criteria:

- 10 presets generate valid datasets;
- 1000 prior samples have a coverage report;
- failures are explicit and classified.

### Phase 2: Validation Suite

Lead: A5

Supporting agents: A1, A2, A6

A5 tasks:

- implement prior predictive checks;
- standardize scorecard metrics;
- implement adversarial real/synthetic AUC report;
- implement transfer baseline runner;
- define provisional thresholds per domain.

A1/A2 tasks:

- provide preset configs and metadata fields needed by validation;
- fix generation issues uncovered by validation reports.

A6 tasks:

- make reports deterministic;
- assert required report sections exist.

Exit criteria:

- every dataset run writes a validation summary;
- failed prior checks block downstream experiments;
- at least one real/synthetic comparison can be reproduced end to end.

### Phase 3: Multi-View Latents

Lead: A3

Supporting agents: A1, A2, A5, A6

A3 tasks:

- define latent and view dataclasses;
- render same latent through several instrument/mode/view configs;
- create positive and negative pair batches;
- preserve latent ids through all transforms.

A1/A2 tasks:

- provide prior records and dataset adapters compatible with multi-view fields.

A5 tasks:

- define invariance metrics;
- measure same-latent vs different-latent distances.

A6 tasks:

- tests for shape, ids, finite values, and split safety.

Exit criteria:

- same-latent multi-view samples are reproducible;
- pair batches pass invariance smoke checks;
- metadata can trace every view to its latent source.

### Phase 4: Encoder plus TabPFN

Lead: A4

Supporting agents: A3, A5, A6

A4 tasks:

- implement CNN and patch transformer prototypes;
- implement losses and training script;
- implement sklearn-compatible transform wrapper;
- train initial checkpoint;
- summarize embedding diagnostics.

A3 tasks:

- keep pair sampler stable for training;
- add view diversity knobs requested by A4.

A5 tasks:

- compare encoder plus TabPFN against current baselines;
- run few-shot and ablation studies;
- generate final model evaluation report.

A6 tasks:

- smoke tests for forward pass, serialization, config loading, and deterministic
  CPU fallback.

Exit criteria:

- encoder training converges on synthetic validation batches;
- real-data benchmark report includes baselines and repeated splits;
- A0 can make a go/no-go decision.

### Phase 5: NIRS-ICL Task Prior

Lead: A1 and A3 jointly

Supporting agents: A4, A5, A6

A1 tasks:

- define task-level prior dimensions;
- sample task domains, analytes, target functions, and split policies.

A3 tasks:

- create context/query tasks from latent/view batches;
- support instrument and domain shifts.

A4 tasks:

- run ICL baselines with TabPFN and encoder features.

A5 tasks:

- assess task difficulty, leakage, calibration, and real transfer.

A6 tasks:

- contract tests for `NIRSPriorTask`.

Exit criteria:

- 1000 tasks pass contract checks;
- baselines are non-trivial and not saturated;
- task reports show difficulty gradients.

### Phase 6: Integration Proposal

Lead: A7

Supporting agents: A0, A1, A2, A5, A6

Tasks:

- identify minimal production APIs to port;
- separate stable contracts from research-only utilities;
- write migration plan;
- list tests needed in production;
- list docs and examples needed for users.

Exit criteria:

- A0 accepts integration proposal;
- production changes are minimal and backed by bench reports.

## Parallel Execution Plan

| Timebox | Parallel Work | Blocking Dependency |
|---|---|---|
| Week 1 | A1 prior aliases, A6 config/test skeleton, A5 report schema | none |
| Week 2 | A2 dataset adapter, A5 prior checks, A6 smoke tests | A1 canonical fields |
| Week 3 | A3 latent draft, A5 transfer baselines, A2 fitted adapter | A2 smoke datasets |
| Week 4 | A3 multi-view batches, A4 encoder skeleton, A5 invariance metrics | A3 latent ids |
| Week 5 | A4 training, A5 scorecards, A6 deterministic tests | C1 pair sampler |
| Week 6 | A5 real evaluation, A4 ablations, A0 decision memo | trained checkpoint |
| Week 7 | A1/A3 task sampler, A4 ICL baseline, A6 task tests | positive or neutral C3 result |
| Week 8 | A7 integration proposal, A5 final reports, A0 go/no-go | bench evidence |

## Decision Meetings

Each phase ends with a short decision memo:

- what was built;
- what passed;
- what failed;
- what was learned scientifically;
- whether to continue, iterate, stop, or integrate.

No phase should advance on code completeness alone. The required evidence is a
reproducible report with metrics and failure analysis.

