# nirs4all parity oracle

This directory holds the **frozen pipeline contract** that the legacy
`PipelineRunner` / `SpectroDataset` backend implements today, and that the
future `dag-ml` + `dag-ml-data` backend must reproduce without regression.

Every case here is a real-world pipeline shape (branches, merges, multi-source,
aggregation, repetition, tags/exclude, augmentation, generators, refit/predict)
expressed in the public nirs4all 0.9.x DSL. Each case is registered with the
keywords it exercises so the parity runner can prove coverage of every
load-bearing public feature.

## Why it exists

The integration roadmap (`/home/delete/.claude/plans/use-4-agents-et-immutable-axolotl.md`)
makes a thin vertical slice + a gold-standard parity suite the gate before any
backend cutover. This directory is the seed of that suite. Each case must:

- run today on the legacy backend (or carry an explicit `skip_reason`);
- have a stable name, stable keywords, and a stable expected shape;
- exercise public nirs4all API only — never internal symbols.

Once the dag-ml bridge lands, the bridge runs every case here through both
backends and compares results within the tolerance declared in ADR-01.

## Layout

```
tests/integration/parity/
├── __init__.py
├── README.md                ← you are here
├── _registry.py             ← PipelineCase + the case registry
├── _datasets.py             ← dataset-path helpers (sample_data/*)
├── cases_baseline.py        ← simple regression / classification baselines
├── cases_branches_merges.py ← duplication + separation branches, all merge modes
├── cases_multi_source.py    ← multi-source pipelines (NIR + markers)
├── cases_aggregation_reps.py← repetitions + sample-level aggregation
├── cases_augmentation.py    ← sample/feature augmentation + concat_transform
├── cases_generators.py      ← _or_ / _grid_ / _range_ / _cartesian_ / _zip_ / _chain_ / _sample_
├── cases_tags_exclude.py    ← tag, exclude (single + mode any/all)
├── cases_refit_predict.py   ← refit → predict → retrain → session → .n4a round-trip
├── conftest.py              ← pytest fixtures (auto-loads all cases)
├── test_parity_compiles.py  ← fast: every case constructs PipelineConfigs cleanly
└── test_parity_smoke.py     ← slow: every case actually runs and returns predictions
```

## Coverage matrix

`test_parity_compiles.py` asserts that every canonical CLAUDE.md DSL
keyword has at least one parity case — uncovered keywords go through an
explicit `intentionally_uncovered` allowlist with a written justification
(currently: `auto_transfer_preproc`, `fill_value`, `na_policy` —
follow-up work). To print the matrix on demand, import
`_registry.keyword_coverage()` from a shell:

```bash
python -c "from tests.integration.parity._registry import keyword_coverage; \
  import json; print(json.dumps(keyword_coverage(), indent=2))"
```

This is **keyword coverage, not feature completeness**. One case with
`sample_augmentation` does not cover `augmentation → fitted transform →
augmentation`, `augmentation → exclude`, or the same workflow through a CLI
worker. A test that expects DAG-ML to refuse a legacy-supported pipeline proves
the refusal is clear; it does not establish parity.

For a feature-completeness audit before a release:

1. Inventory the **legacy-successful** pipeline grammar from the parser and
   controllers, including step options, ordering constraints, data layouts,
   cross-validation/no-splitter execution, and train/predict/export operations.
   Do not derive the inventory from DAG-ML's current supported-shape detector.
2. Track each feature alone, valid **ordered pairs**, and interactions that
   change scope or state (at least augmentation, fitted preprocessing, exclusion,
   branches, splitters, and refit). Cross these with single/multi-source data,
   regression/classification, and both in-process and CLI execution. Record an
   owner and a runnable fixture for every valid row; invalid combinations need
   an explicit parser/legacy reason, not a missing test.
3. For every legacy-successful row, require `engine="dag-ml"` to execute natively
   without fallback or an expected refusal. Compare the public result contract,
   final refit/test predictions by sample ID, and exported-model predictions.
   Compare CV scores only when both engines use the same split and fit scope.
   Where DAG-ML deliberately removes legacy leakage, pin that semantic difference
   with a separate fold-level oracle and still require full refit parity.
4. When a row diverges, compare the **first differing boundary**: parsed steps,
   selected train/validation IDs, exclusion masks, synthetic origins, per-fold X/y
   shapes and hashes, fitted transform state, model inputs, OOF predictions,
   refit predictions, then archive replay. This localizes a bridge bug without
   treating a final RMSE difference as the diagnosis.
5. The release gate is zero untriaged rows and zero legacy-successful expected
   refusals. A documented refusal remains a capability gap; a passing test for
   it must not count as feature coverage. Run the same cases through both DAG-ML
   mechanisms and keep the feature ledger in CI, not only in a manual checklist.

`coverage_meter.py` currently counts registered cases and declared keywords; it
does **not** generate this ordered-composition matrix. Its `expected_refusal`
bucket must therefore be reviewed separately before claiming feature parity.
`concat_transform_pca_svd_plsr` now runs natively with fold-local PCA/SVD;
the registered-case refusal bucket is empty. The broader feature-completeness
gate remains open while other compositions in `feature_gaps.json` remain.
Run `python -m tests.integration.parity.coverage_meter --require-zero-refusals`
to make the registered-case refusal check fail in a release job. This check is
necessary but insufficient until the ordered-composition inventory above is
complete.
Use the repository's `.venv/bin/python` and `.venv/bin/pytest` for local runs;
the system interpreter can import an older DAG-ML wheel and produce unrelated
API failures.

`feature_gaps.json` records legacy-successful compositions reproduced by the
audit, including gaps the case registry has not yet captured, and tracks
plausible gaps that still need a public legacy reproduction under `unverified`.
Run
`.venv/bin/python -m tests.integration.parity.coverage_meter
--require-feature-complete` before a release: it fails while the inventory is
unfinished, any confirmed gap or unverified probe remains open, or a conformance case still
refuses DAG-ML. Add a failing composition to the ledger immediately, then add
its dual-engine regression test when fixing it; remove the gap only when that
test passes through the relevant runtime mechanisms. `inventory_complete` may
become true only after the legacy parser/controller inventory and ordered
composition matrix above have been reviewed.

## Skip vs. xfail policy

Each skipped case carries a `skip_kind` so the runner picks the right
pytest disposition:

- `skip_kind="fixture"` — missing column or wrong-typed corpus → `pytest.skip`.
- `skip_kind="unknown_semantics"` — exact DSL contract not yet pinned →
  `pytest.skip`. Promote to runnable once the contract is confirmed.
- `skip_kind="legacy_bug"` — known nirs4all 0.9.x bug → `pytest.xfail(strict=True)`.
  The case stays loud and XPASS-flips the day the legacy bug is fixed,
  so a fix never silently shrinks coverage.

## Running

```bash
# Fast: only check every case compiles to a PipelineConfigs
pytest tests/integration/parity/test_parity_compiles.py -v

# Slow: run every case end-to-end on the legacy backend
pytest tests/integration/parity/test_parity_smoke.py -v

# Subset by tag (registered on each case)
pytest tests/integration/parity/ -v -k "branches or multi_source"

# Single case by name
pytest tests/integration/parity/ -v -k "branch_separation_by_source"
```

## Adding a case

1. Pick (or create) the `cases_*.py` file that fits the keyword family.
2. Write a `pipeline_factory()` returning a fresh pipeline list (factories
   exist because some operators are mutable and must not be shared across cases).
3. Register via `@register` decorator with: name, description, keywords,
   dataset_key, task, expected_min_predictions (lower bound), tags.
4. If the case can't run yet on the legacy backend, set `skip_reason` to the
   GitHub issue or ADR explaining why.
5. `ruff check .` and `mypy nirs4all/` must stay green; the docstring rule
   (lead with invariant + failure mode + example) applies to every helper.

## Promotion to the dag-ml bridge

The bridge under construction (workstream E in the roadmap) reads this exact
registry. Cases here are the source of truth for the keywords the bridge
must translate. **Do not delete a case to make the bridge easier** — open an
ADR (managed-debt policy, ADR-14) instead.
