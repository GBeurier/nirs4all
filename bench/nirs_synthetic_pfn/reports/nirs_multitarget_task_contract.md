# NIRSPriorTask Multi-Target Contract (Phase D3)

## Objective

Lift the D1 / D2 single-output restriction so that `NIRSPriorTask` and
the D2 context/query sampler accept multi-output regression and
multi-output classification tasks. D3 widens the contract surface
without introducing any realism, transfer, or downstream training
claim. The A3 / B2 risk gates remain explicitly negative on every
task; no other invariant of D1 / D2 is relaxed.

## Scope

In scope (D3):

- Accept 1D `y` (single output) and 2D `y` (multi output) on both
  splits; cross-split dimensionality and number of outputs must be
  consistent.
- Normalise 2D `(n_rows, 1)` targets back to 1D so the single-output
  convention stays unique and stable.
- Extend `target_semantics` with `n_outputs`, `output_names`, and
  `multi_output_supported=True`. Single-output tasks still expose
  `n_outputs=1` and `output_names=[target_name]`.
- Validate classification labels as integer-like and finite for every
  output column (single-output and multi-output).
- D2 sampler: drop the `multi_output_unsupported` rejection in
  `_select_target_array`. Multi-output regression diagnostics aggregate
  per-output ranges; multi-output classification stratifies on the
  joint label tuple. `split_policy.diagnostics_summary` continues to
  exclude target ranges and any per-sample target value.
- Tasks built either through `NIRSPriorTask.from_batches` or through
  `sample_nirs_prior_task` carry `provenance.limitations
  .multi_output_supported = True` and explicit `n_outputs` in
  `provenance`.

Out of scope (still deferred):

- A3 / B2 risk gates: remain `True` and explicitly documented.
- No realism evaluation, no transfer evaluation, no encoder training,
  no PFN ingestion, no cross-domain / cross-instrument shifts.

## Files changed

- `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/tasks.py`
  (D3: multi-output `y`, new `target_semantics` keys, cross-split
  consistency check, contract-level documentation refresh).
- `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/task_sampling.py`
  (D3: multi-output target selection, joint-label stratification,
  per-output regression diagnostics, extended non-leaky summary keys).
- `bench/nirs_synthetic_pfn/tests/test_prior_task.py`
  (replace D1 multi-output rejection test with positive D3 multi-output
  regression / classification tasks; add 2D single-column normalisation
  tests, task-id output-structure collision coverage, target semantics
  type validation, and cross-split y dimensionality check).
- `bench/nirs_synthetic_pfn/tests/test_task_sampling.py`
  (add D3 multi-output regression diagnostics test, joint-label
  stratified classification, infeasible singleton joint label
  rejection, non-integer multi-output rejection, JSON serialisation
  round-trip with `n_outputs`).
- `bench/nirs_synthetic_pfn/reports/nirs_multitarget_task_contract.md`
  (this report, new).

No file inside `nirs4all/` is modified. D3 is implemented entirely
under `bench/nirs_synthetic_pfn/`.

## Commands

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src \
    pytest bench/nirs_synthetic_pfn/tests/test_prior_task.py \
           bench/nirs_synthetic_pfn/tests/test_task_sampling.py -q

PYTHONPATH=bench/nirs_synthetic_pfn/src \
    pytest bench/nirs_synthetic_pfn/tests -q

ruff check bench/nirs_synthetic_pfn

PYTHONPATH=bench/nirs_synthetic_pfn/src \
    mypy \
    bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/tasks.py \
    bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/task_sampling.py \
    bench/nirs_synthetic_pfn/tests/test_prior_task.py \
    bench/nirs_synthetic_pfn/tests/test_task_sampling.py
```

## Contract changes (D3)

`NIRSPriorTask.__post_init__`:

- `y_context` / `y_query` accept 1D **and** 2D shapes. 2D arrays must
  have at least one output column (`shape[1] >= 1`) and matching row
  counts (`shape[0] == n_rows`). 1D arrays continue to require
  `shape[0] == n_rows`.
- New cross-split check: `y_context.ndim == y_query.ndim`, and when
  both are 2D, `y_context.shape[1] == y_query.shape[1]`. Mismatches
  are reported as `shape_mismatch` on `y_query`.
- Classification: every output column of `y_context` / `y_query` must
  be integer-like and finite. Single-output and multi-output tasks
  share the same check.
- `target_semantics` now requires `n_outputs` (positive int),
  `output_names` (non-empty list of non-empty strings, length
  `n_outputs`), and `multi_output_supported=True`. The
  `n_outputs` value must agree with the actual `y` array shape.

`NIRSPriorTask.from_batches`:

- Single-column 2D targets are normalised back to 1D.
- `n_outputs` is derived from the resolved target array.
- `output_names` resolution order:
  1. `target_metadata["output_names"]` if present and length matches.
  2. For single-output, `[resolved_target_name]`.
  3. For multi-output regression, when
     `len(component_keys) == n_outputs`, `[f"target__{key}"]` per
     component key.
  4. Otherwise, `[f"{resolved_target_name}_{i}"]` for `i` in range.
- Default `target_name` for multi-output is `"multi_target"`
  (regression) or `"multi_label"` (classification); single-output
  defaults are unchanged.
- `provenance.phase` becomes `"D3"` when `n_outputs > 1`, otherwise
  stays `"D1"`. `provenance.n_outputs` is added.
- `provenance.limitations.multi_output_supported = True` for every
  task.
- `task_id` includes structural output identity (`target_type`,
  `n_outputs`, `output_names`) and split/id metadata, but not `X` or `y`
  values.

`task_sampling._select_target_array`:

- Returns 1D for single-output (also for `(n, 1)` 2D inputs) and 2D
  `(n_rows, n_outputs)` for multi-output.
- Drops the `multi_output_unsupported` failure.

`_stratified_classification_split`:

- Uses `_encode_joint_labels(target)` to turn 1D / 2D integer-like
  targets into a 1D joint-label id array. The existing largest
  remainder allocator runs unchanged on those ids.
- Single-output integer labels behave exactly as before (joint class
  is the original integer wrapped in a 1-tuple).
- Multi-output: every joint label class needs at least 2 samples;
  otherwise `infeasible_stratification` is raised.

`_build_diagnostics`:

- Always reports `n_outputs` for both regression and classification.
- Regression: `target_range_context` / `target_range_query` are 1D
  pairs for single-output and lists of per-output pairs for
  multi-output. They remain *out* of the non-leaky
  `diagnostics_summary` and never reach `split_policy`.
- Classification single-output: existing `class_counts_context`,
  `class_counts_query`, `n_classes_*`, `shared_label_count` keys are
  preserved.
- Classification multi-output: aggregate `joint_label_count_context`,
  `joint_label_count_query`, `n_joint_labels`,
  `shared_joint_label_count`, and `n_classes_*` (joint class counts)
  are reported. Joint label keys are stringified tuples (e.g. `"0,1"`)
  – aggregate, not per-sample values.

`_NON_LEAKY_DIAGNOSTIC_KEYS` adds `n_outputs`, `n_joint_labels`, and
`shared_joint_label_count`. Per-output target ranges and joint label
counts are deliberately excluded from the summary.

## Test coverage (D3)

`bench/nirs_synthetic_pfn/tests/test_prior_task.py` (47 tests):

- All D1 single-output checks remain green.
- `test_from_batches_populates_split_arrays_ids_and_semantics` updated
  to assert `multi_output_supported=True`, `output_names`, and
  `provenance.n_outputs` on single-output tasks.
- New: `test_from_batches_accepts_multi_output_regression_target`
  (2D `y`, `n_outputs=3`, light/dict serialisation, false claims).
- New: `test_from_batches_accepts_multi_output_classification_integer_labels`
  (2D integer-like labels, `n_outputs=2`).
- New: `test_from_batches_rejects_multi_output_classification_non_integer_labels`
  (`invalid_classification_labels` reason).
- New: `test_from_batches_normalises_2d_single_column_target_to_1d`
  (stable single-output convention).
- New: `test_validation_normalises_2d_single_column_y_via_replace`
  (direct dataclass validation keeps the same convention).
- New: `test_task_id_changes_with_output_structure_not_values`
  prevents collision between single/multi-output tasks and different
  output-name structures with the same ids, split, target name, and
  seed.
- New:
  `test_validation_rejects_non_bool_target_clean_equals_target_noisy_semantics`
  checks the required boolean semantics field.
- New: `test_validation_rejects_inconsistent_y_dimensionality_via_replace`
  (cross-split shape mismatch).

`bench/nirs_synthetic_pfn/tests/test_task_sampling.py` (38 tests):

- All D2 single-output checks remain green.
- New: `test_random_split_supports_multi_output_regression_diagnostics`
  validates per-output `target_range_*` lists, summary excludes them,
  `provenance.risk_gates` and false realism / transfer claims survive
  the end-to-end task build.
- New:
  `test_stratified_classification_supports_multi_output_via_joint_labels`
  asserts joint label coverage in both splits and aggregate diagnostics
  with stringified tuple keys.
- New:
  `test_stratified_multi_output_classification_rejects_singleton_joint_label`
  triggers `infeasible_stratification` when a joint label has count 1.
- New: `test_stratified_classification_rejects_multi_output_non_integer_labels`
  triggers `invalid_classification_labels`.
- New: `test_split_policy_serialisable_for_multi_output_regression`
  round-trips through JSON with `diagnostics_summary.n_outputs == 2`.

## Results

- `pytest bench/nirs_synthetic_pfn/tests/test_prior_task.py
  bench/nirs_synthetic_pfn/tests/test_task_sampling.py -q`: **85 passed**.
- `pytest bench/nirs_synthetic_pfn/tests -q`: **206 passed**, 4 inherited
  sklearn `y residual` warnings on the C-suite transfer tests; no
  regression.
- `ruff check bench/nirs_synthetic_pfn`: **All checks passed**.
- `mypy tasks.py + task_sampling.py + tests`: **Success, no issues found**
  across the four targeted source files.

## Risk gates (negative, documented)

D3 keeps the two upstream risk gates explicitly **negative** on every
task it produces:

- `A3_failed_documented = True` — fitted-only real-fit adapter remains
  failing; D3 does not depend on A3 and does not lift this gate.
- `B2_realism_failed = True` — synthetic vs real realism scorecards
  remain blocked; D3 makes no realism claim.

`provenance.claims = {"realism": False, "transfer": False}` is
preserved verbatim. The contract refuses to construct or replace a
task whose provenance is missing either gate.

## Known limitations

- `target_clean == target_noisy` is still inherited from the source
  `CanonicalLatentBatch`. D3 does not introduce a separate noiseless
  target.
- `output_names` falls back to `f"{target_name}_{i}"` when
  `target_metadata` does not provide explicit names and component keys
  do not match `n_outputs`. This is documented at construction time.
- Multi-output regression `target_range_*` diagnostics expose per-
  output min/max; they remain out of `diagnostics_summary` so callers
  that only consume the summary cannot leak them through the sampler.
- D3 does **not** implement training, evaluation, transfer, encoder,
  PFN ingestion, or cross-domain / cross-instrument shifts.

## Decision

D3 contract implemented. `NIRSPriorTask` and the D2 context/query
sampler now support both single-output and multi-output regression /
classification with explicit `target_semantics` (`n_outputs`,
`output_names`, `multi_output_supported=True`) and aggregated, non-
leaky diagnostics. A3 and B2 remain negative and documented; no
realism / transfer / training claim is derived.
