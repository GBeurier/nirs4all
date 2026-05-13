# Master Prompt

You are working on `bench/nirs_synthetic_pfn`, a bench-only research project for
synthetic NIRS priors, multi-view encoders, and NIRS-ICL/PFN feasibility.

## Boundary

Do not modify production `nirs4all` code unless a later task explicitly asks for
integration. Use production modules as dependencies and references.

Primary source review:

- `docs/_internal/synthetic/spectral_synthesis_inventory_and_pfn_prior_plan.md`

Driver docs in this project:

- `docs/00_CONTEXT_REVIEW.md`
- `docs/01_BENCH_PROJECT_SPEC.md`
- `docs/02_MASTER_ROADMAP.md`
- `docs/03_AGENT_ROADMAP.md`
- `docs/04_SCIENTIFIC_VALIDATION_PROTOCOL.md`
- `docs/05_INTEGRATION_GATE.md`

## Default First Work Order

Start with Roadmap A:

1. implement prior canonicalization;
2. implement prior validation tests;
3. generate a 1000-sample prior coverage report;
4. implement a smoke prior-to-dataset experiment;
5. write a report before moving to model training.

## Agent Selection

Choose the agent role from `docs/03_AGENT_ROADMAP.md` that matches the task.
Keep ownership boundaries clear:

- A1: prior and ontology;
- A2: generator adapters;
- A3: latents and multi-view;
- A4: encoder and ML;
- A5: validation;
- A6: QA and reproducibility;
- A7: integration, inactive until gates pass.

## Evidence Standard

Every phase needs code, tests, and a report. A model result without validation is
not a pass. A synthetic dataset that looks plausible but fails transfer checks is
not a success.

