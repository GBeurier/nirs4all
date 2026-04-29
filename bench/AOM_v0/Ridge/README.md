# AOM-Ridge Workspace

This directory contains the implementation brief for the AOM-Ridge extension
of `bench/AOM_v0`. It is separate from the validated AOM-PLS implementation so
Ridge can be implemented and reviewed without changing the PLS reference path.

## Documents

- `docs/PLAN_REVIEW_CORRECTIONS.md`: corrected review of the proposed plan.
- `docs/AOM_RIDGE_MATH_SPEC.md`: mathematical specification.
- `docs/IMPLEMENTATION_PLAN.md`: module layout and phase gates.
- `docs/AOM_RIDGE_API.md`: estimator API draft.
- `docs/TEST_PLAN.md`: required equivalence and leakage tests.
- `docs/BENCHMARK_PROTOCOL.md`: benchmark variants and reporting protocol.
- `docs/CODEX_REVIEW_WORKFLOW.md`: Codex review loop.
- `docs/IMPLEMENTATION_LOG.md`: append-only implementation log.
- `prompts/CLAUDE_CODE_DRIVER_PROMPT.md`: Claude Code implementation prompt.
- `prompts/codex_review_prompts/`: independent Codex review prompts.

## Intended Code Layout

```text
bench/AOM_v0/Ridge/
  aomridge/
    __init__.py
    kernels.py
    solvers.py
    selection.py
    estimators.py
    branches.py
    diagnostics.py
  tests/
  benchmarks/
```

The Ridge package may import strict linear operators and bank presets from
`bench/AOM_v0/aompls`. It must not modify production `nirs4all` code.

Recommended test command after implementation:

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge pytest bench/AOM_v0/Ridge/tests -q
```

