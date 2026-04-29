# NIRS Synthetic PFN Bench Project

Bench project for validating synthetic NIRS priors, executable synthetic tasks,
multi-view spectral encoders, and eventual NIRS-ICL or NIRS-PFN experiments.

This directory is intentionally a bench workspace, not production library code.
The production package may be read as a dependency and reference, but methods are
ported to `nirs4all` only after the bench gates are passed.

## Source Material

Primary review input:

- `docs/_internal/synthetic/spectral_synthesis_inventory_and_pfn_prior_plan.md`

Related historical material:

- `docs/_internal/nirsPFN/02_nirspfn_development_plan.md`
- `docs/_internal/nirsPFN/03_poc_roadmap.md`
- `bench/synthetic/synthesis_summary.md`
- `bench/_tabpfn/spectral_latent_features.md`
- `bench/tabpfn_paper/master_results.csv`
- `bench/AOM_v0/benchmarks/cohort_regression.csv`
- `bench/AOM_v0/benchmarks/cohort_classification.csv`

Benchmark data may use the TabPFN paper test data and the AOM-PLS benchmark
cohorts, including the consolidated 57-dataset regression/classification
reference set when available locally. Experiments must report the exact dataset
count used after filtering `status == "ok"` rows and missing local files.

## Project Boundary

Work here first:

- executable prior-to-dataset adapters;
- context/query task samplers;
- multi-view synthetic rendering from common latents;
- contrastive spectral encoder experiments;
- validation reports and ablations;
- prototype APIs and scientific contracts.

Do not port to `nirs4all` until:

- the method is reproducible in bench;
- the synthetic prior passes contract and realism checks;
- downstream real-data metrics justify integration;
- an integration proposal identifies the minimal stable API.

## Directory Layout

```text
bench/nirs_synthetic_pfn/
  README.md
  configs/            # YAML/JSON experiment configs once implementation starts
  docs/               # Driver documents and roadmaps
  experiments/        # Executable experiments and run scripts
  reports/            # Generated markdown/CSV/figures from experiments
  src/                # Self-contained bench package code
  tests/              # Bench-only tests
  artifacts/          # Local checkpoints/shards, normally not committed
  source_materials/   # Optional copied extracts, never a second source of truth
```

Planned internal package:

```text
src/nirsyntheticpfn/
  adapters/           # prior -> builder/task adapters
  data/               # task dataclasses, latents, view batches
  encoders/           # spectral encoder prototypes
  evaluation/         # metrics, scorecards, benchmark runners
  reports/            # report builders
  utils/              # seeds, config loading, provenance
```

## Driver Documents

- `docs/00_CONTEXT_REVIEW.md`: review of the source document and strategic decisions.
- `docs/01_BENCH_PROJECT_SPEC.md`: project architecture, contracts, and scope.
- `docs/02_MASTER_ROADMAP.md`: phased roadmap from executable prior to integration.
- `docs/03_AGENT_ROADMAP.md`: agent ownership, parallel workstreams, and handoffs.
- `docs/04_SCIENTIFIC_VALIDATION_PROTOCOL.md`: required validation protocol.
- `docs/05_INTEGRATION_GATE.md`: criteria for moving bench work into `nirs4all`.

## Default Strategy

The first validated objective is not a full NIRS-PFN. The lower-risk route is:

1. make the synthetic prior executable and testable;
2. prove it can generate credible datasets and tasks;
3. train or evaluate a spectral encoder from synthetic multi-views;
4. compare encoder plus TabPFN against ASLSBaseline/PCA/TabPFN;
5. consider full NIRS-ICL/PFN only if the prior proves downstream value.

## Working Rules

- Keep all experimental code under this directory until integration gates pass.
- Use existing `nirs4all.synthesis` as the engine, not a copied fork.
- Every experiment must have a config, seed, command, output report, and summary.
- Every prior change must run prior predictive checks before downstream training.
- Every claimed gain must include a baseline, split policy, confidence interval or
  repeated run, and an ablation when feasible.
