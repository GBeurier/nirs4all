# Integration Gate

This document defines when bench work may move into the production `nirs4all`
package.

## Rule

No experimental API is ported just because it works on synthetic smoke tests.
Port only stable pieces that passed bench validation and have a clear user-facing
purpose.

## Gate G0: Bench Reproducibility

Required:

- experiment config committed;
- command documented;
- seeds documented;
- report generated under `bench/nirs_synthetic_pfn/reports`;
- tests pass for the relevant bench modules;
- failures and unsupported fields documented.

Decision:

- if G0 fails, keep work in bench.

## Gate G1: Contract Stability

Required:

- dataclasses or config schemas are stable across at least two experiments;
- prior validation catches invalid domains, components, targets, and wavelength
  policies;
- generated datasets/tasks include complete metadata;
- no silent fallback behavior hides invalid prior samples.

Decision:

- if contracts are still changing every experiment, keep work in bench.

## Gate G2: Scientific Value

At least one of the following must be true:

- synthetic prior improves a real-data downstream metric under repeated splits;
- encoder trained from synthetic multi-views improves few-shot or transfer
  performance;
- validation reveals a production bug or missing contract whose fix is clearly
  useful independent of model gains;
- prior-to-dataset generation unlocks a documented user workflow not currently
  possible in `nirs4all`.

Decision:

- if there is no real-data value or clear production bug, keep work in bench.

## Gate G3: Minimal API

Required:

- the proposed public API is smaller than the bench implementation;
- research-only reports, training scripts, and ablations stay in bench;
- production API has clear names, parameters, defaults, and error modes;
- backward compatibility impact is reviewed.

Candidate production APIs:

- `generate.prior(...)` only after prior-to-builder behavior stabilizes;
- `generate.task(...)` only after task contract stabilizes;
- builder methods for environment/scatter/edge only if they map cleanly to the
  generator;
- measurement mode wiring only with tests for each mode;
- augmentation label-preserving tags and validators;
- encoder operator only if checkpoint policy and dependency impact are accepted.

## Gate G4: Production Test Plan

Required:

- unit tests for schema and validation;
- integration tests for dataset generation;
- regression tests for deterministic seeds;
- tests for measurement modes if touched;
- tests for augmentation label preservation if touched;
- docs or examples for public APIs.

Decision:

- no port without a production test plan.

## Porting Strategy

Use small PRs in this order:

1. pure validation fixes and aliases;
2. builder/generator contract fixes;
3. task or prior adapters;
4. public API wrappers;
5. model/encoder integration, only if justified.

Avoid porting:

- training loops;
- exploratory ablation scripts;
- benchmark-specific code;
- heavy checkpoint artifacts;
- unstable dataclasses used by only one experiment;
- full PFN prototypes before feasibility review.

## Integration Memo Template

Before a production change, write:

```text
Title:
Bench evidence:
Problem solved:
Minimal API:
Files proposed for production:
Tests required:
Docs required:
Backward compatibility:
Risks:
Decision:
```

## Stop Conditions

Keep the work in bench if:

- real-data validation is absent;
- gains are not reproducible;
- synthetic tasks are too easy or saturated;
- metadata contracts are incomplete;
- model checkpoints are required but not governed;
- the production API would expose research assumptions as stable behavior.

