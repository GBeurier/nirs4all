# Documentation Style: Writing Pipelines

This page is the house style for **how pipelines are written in the docs and
examples**. It exists so every snippet a newcomer copies looks the same and reads
cleanly. It is enforced by a lightweight, advisory CI lint (not a hard wall).

## The rule: inline by default, not tyrannically

Write a pipeline **inline, in one shot, inside `run()`** — no external variable —
**when the pipeline is simple**:

```python
result = nirs4all.run(
    pipeline=[MinMaxScaler(), ShuffleSplit(n_splits=3), {"model": PLSRegression(n_components=10)}],
    dataset=nirs4all.generate.regression(n_samples=500),
)
```

Do **not** do this for a simple pipeline:

```python
pipeline = [MinMaxScaler(), {"model": PLSRegression(n_components=10)}]   # ✗ avoid for simple pipelines
result = nirs4all.run(pipeline, dataset)
```

### When a named variable IS fine

Readability wins. A **genuinely complex** pipeline may be assigned to a
descriptive variable — cramming a deeply nested structure into the call args is
*less* readable, which defeats the purpose. A named variable is acceptable when:

- the pipeline has **nested `branch`/`merge`** trees or large generator sweeps
  (`_cartesian_`, `_or_`, `_grid_`, …);
- it has **more than ~6 top-level steps**;
- the **same pipeline is reused** across several `run()` calls, or you build a
  **list of pipelines** (`run(pipeline=[pipeline_a, pipeline_b, pipeline_c])`).

```python
branching_pipeline = [                       # ✓ fine: complex, named for clarity
    {"branch": {"by_source": [...]}},
    {"merge": "predictions"},
    {"_cartesian_": [...]},
]
result = nirs4all.run(pipeline=branching_pipeline, dataset=...)
```

## The lint

`scripts/lint_inline_pipeline.py` is **advisory** (a warning, never a blocking
gate). It uses an AST analysis and flags **only** the clear-cut simple case: a
`pipeline`-named variable bound to a *simple* list literal that is then passed —
positionally or by keyword — as the `pipeline` argument of `run`/`session`/
`Session`/`PipelineRunner`. Complex/nested pipelines, reuse, and list-of-pipelines
are auto-exempt.

```bash
python scripts/lint_inline_pipeline.py            # advisory report
python scripts/lint_inline_pipeline.py --list     # one line per suggestion
```

To suppress a specific site (e.g. a deliberate teaching example that keeps the
variable), add an auditable marker on the assignment line:

```python
pipeline = [MinMaxScaler(), PLSRegression(10)]   # noqa: inline-pipeline  (kept for the walkthrough)
```

The companion codemod inlines the simple cases automatically:

```bash
python scripts/codemod_inline_pipeline.py            # dry-run diff
python scripts/codemod_inline_pipeline.py --apply    # apply
```

## Related contributor checks

Two metadata/reference gates run in CI and are **blocking**:

- `scripts/check_doc_metadata.py` — every version string must equal
  `nirs4all.__version__`, and first-contact pages must state the dual license
  (default AGPL-3.0-or-later) rather than a single contradicting license.
- `scripts/check_example_refs.py` — every `examples/…py` file named in the docs
  must exist on disk (catches dangling example references).
