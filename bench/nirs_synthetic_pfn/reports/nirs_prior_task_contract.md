# NIRSPriorTask Contract (Phase D1)

## Objective

Define the bench-side `NIRSPriorTask` contract that turns one
`CanonicalLatentBatch` (C1) and one aligned `SpectralViewBatch` (C2/C3)
into a frozen, validated PFN-style `(context, query)` task carrying:

- per-split spectra `X_context` / `X_query` and labels `y_context` /
  `y_query`,
- per-split wavelength axes (`wavelengths_context` /
  `wavelengths_query`),
- per-split deterministic ids (`context_latent_ids` /
  `query_latent_ids`, `context_view_ids` / `query_view_ids`),
- structured per-split metadata namespaces (`metadata_context` /
  `metadata_query`),
- the optical configuration (`domain_key`, `instrument_context`,
  `instrument_query`, `measurement_mode`),
- explicit target semantics (`target_name`, `target_type`,
  `target_semantics`),
- structural latent and prior descriptors (`latent_params`,
  `prior_config`, `split_policy`, `task_seed`),
- a provenance namespace with the inherited A3 / B2 risk gates and
  explicit non-realism / non-transfer claims.

D1 only declares this contract surface and the `from_batches`
constructor. It does **not** introduce a context/query sampler, a
multi-output target path, or any realism / transfer claim.

## Scope

In scope:

- `NIRSPriorTask` frozen dataclass with `__post_init__` validation.
- `NIRSPriorTaskError(ValueError)` carrying a structured `failures` list
  in the same shape as `CanonicalLatentBatchError` /
  `SpectralViewBatchError`.
- `NIRSPriorTask.from_batches(latent_batch, spectral_view,
  context_indices, query_indices, *, target_source="target_noisy",
  target_name=None, split_policy=None, task_seed=None)` constructor.
- `to_dict()` and `to_light_dict()` serialisation helpers; the light
  variant omits `X_context`, `X_query`, `y_context`, `y_query`.
- Deterministic `task_id` derived via `hashlib.sha256` from
  `(builder_config_name, random_state, task_seed, target_source,
  target_name, target_type, context_latent_ids, query_latent_ids,
  context_view_ids, query_view_ids, split_policy)`.
- Tests in `bench/nirs_synthetic_pfn/tests/test_prior_task.py` covering
  construction, determinism, index validation, misalignment, dataclass
  replace validation paths, leakage, risk gates, multi-output rejection,
  and serialisation.

Out of scope (deferred to later phases):

- A context/query sampler (D2). D1 only consumes integer index splits.
- Multi-output targets (D3). Tasks with `y` having more than one column
  are rejected with reason `multi_output_unsupported`.
- Lifting the inherited A3 / B2 risk gates.
- Encoders, training, ingestion, real-fit transfer, or realism claims.

## Files changed

- `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/tasks.py` (new).
- `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/__init__.py`
  (re-export `NIRSPriorTask`, `NIRSPriorTaskError`).
- `bench/nirs_synthetic_pfn/tests/test_prior_task.py` (new).
- `bench/nirs_synthetic_pfn/reports/nirs_prior_task_contract.md`
  (this report, new).

No file inside `nirs4all/` is modified. D1 is implemented entirely
under `bench/nirs_synthetic_pfn/`.

## Commands

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src \
    pytest bench/nirs_synthetic_pfn/tests/test_prior_task.py -q

PYTHONPATH=bench/nirs_synthetic_pfn/src \
    pytest bench/nirs_synthetic_pfn/tests -q

ruff check bench/nirs_synthetic_pfn

PYTHONPATH=bench/nirs_synthetic_pfn/src \
    mypy \
    bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/tasks.py \
    bench/nirs_synthetic_pfn/tests/test_prior_task.py
```

## Seed policy

D1 keeps every randomness source explicit and deterministic at the
contract level:

- `task_seed` defaults to the canonical `provenance.seed` carried by
  the source `CanonicalLatentBatch` (which equals the A2 builder
  `random_state`). Callers can override with an explicit integer.
- `task_id` is a SHA-256 digest of a canonical JSON payload that mixes
  `(builder_config_name, random_state, task_seed, target_source,
  target_name, target_type, context_latent_ids, query_latent_ids,
  context_view_ids, query_view_ids, split_policy)`. Identical inputs produce identical
  ids; changing the split, the task seed, the target source, or the
  view ids changes the id. Numerical `X` / `y` values are deliberately
  excluded from the signature so the id remains a structural task id.
- D1 itself does not draw any random number. Numerical reproducibility
  upstream of `from_batches` is the responsibility of A2 / C1 / C2 / C3.

## Contract checks

`NIRSPriorTask.__post_init__` enforces:

- `task_id`: non-empty string.
- `X_context` / `X_query`: 2D, numeric, finite arrays with at least one
  row each.
- `y_context` / `y_query`: 1D, numeric, finite arrays. Classification
  labels must be integer-like. Single-output only; multi-output `y` is
  rejected by `from_batches` before construction with reason
  `multi_output_unsupported`.
- `wavelengths_context` / `wavelengths_query`: 1D, numeric, finite,
  strictly increasing arrays whose size matches the corresponding
  `X.shape[1]`.
- `context_latent_ids`, `query_latent_ids`, `context_view_ids`,
  `query_view_ids`: tuples of non-empty strings, each of length equal
  to the corresponding number of rows, with no duplicates inside a
  split. Context and query splits must not share any latent id or any
  view id (`overlapping_split_ids`).
- `domain_key`, `instrument_context`, `instrument_query`,
  `measurement_mode`, `target_name`: non-empty strings.
- `target_type`: must be `"regression"` or `"classification"`.
- `task_seed`: must be an `int` (booleans and floats are rejected,
  including explicit `from_batches(..., task_seed=True)`).
- `metadata_context`, `metadata_query`, `target_semantics`,
  `latent_params`, `prior_config`, `split_policy`, `provenance`:
  non-empty `dict` instances.
- `metadata_context` / `metadata_query`: must not contain target or
  latent leakage at any nested path. Forbidden keys (and their
  prefixed `<key>_*` forms) are: `y`, `target`, `targets`,
  `concentration`, `concentrations`, `target_clean`, `target_noisy`,
  `latent_feature`, `latent_features`.
- `prior_config` / `provenance`: must not embed per-sample target or
  latent values. Forbidden keys (exact match, no prefix) are: `y`,
  `target_clean`, `target_noisy`, `concentrations`, `latent_features`.
  Configuration descriptors such as `target_prior` /
  `concentration_prior` are explicitly allowed.
- `provenance.risk_gates` must be a dict equal to at least
  `{"A3_failed_documented": True, "B2_realism_failed": True}`.
- `target_semantics` must include `target_source`, `target_name`,
  `target_type`, `target_clean_equals_target_noisy`, and
  `target_source` must be `"target_clean"` or `"target_noisy"`.
  `target_semantics.target_name` / `target_semantics.target_type` must
  match the top-level `target_name` / `target_type`.

`from_batches` additionally:

- Calls `spectral_view.assert_aligned_to(latent_batch)` before any
  work; cross-batch latent id mismatches raise
  `SpectralViewBatchError(alignment_mismatch)`.
- Validates `context_indices` / `query_indices`: 1D integer arrays,
  non-empty, in range `[0, n_total)`, unique within each split, and
  disjoint between splits (`overlapping_indices`,
  `duplicate_indices`, `invalid_indices`, `empty_split`).
- Selects `target_source` from `latent_batch.target_clean` or
  `latent_batch.target_noisy`; rejects target arrays whose flattened
  shape does not match `n_total`, and rejects multi-output targets
  (`y.ndim > 1` with more than one column) with reason
  `multi_output_unsupported`.
- Rejects classification targets whose labels are not integer-like with
  reason `invalid_classification_labels`.
- Builds `target_semantics` declaring the source, name, type,
  `target_clean_equals_target_noisy` flag (inherited from C1),
  `n_outputs=1`, and `multi_output_supported=False`.
- Builds `metadata_context` / `metadata_query` from the spectral view
  state only (wavelength summary, instrument, mode, view phase /
  kind, preprocessing flag, noise flag, `n_rows`, split label). No
  target or latent value is referenced.
- Builds `prior_config` from
  `latent_batch.provenance.a1_provenance.source_prior_config` when
  available, or a documented stub otherwise.
- Builds `latent_params` from structural latent metadata
  (`component_keys`, `latent_feature_names`, counts, transform note).
- Builds `split_policy` declaring the split kind
  (`explicit_indices`), per-split sizes, total size, and the
  `indices_disjoint=True` flag. A caller-provided dict is accepted
  and augmented with the same defaults.
- Resolves `task_seed` to the explicit argument or to
  `provenance.seed` of the canonical latent batch.
- Records `provenance.claims = {"realism": False, "transfer": False}`,
  `provenance.risk_gates = {"A3_failed_documented": True,
  "B2_realism_failed": True}`, and
  `provenance.limitations.context_query_sampler_implemented = False`,
  `provenance.limitations.multi_output_supported = False`.

`to_light_dict` omits `X_context`, `X_query`, `y_context`, `y_query`,
adds `n_context`, `n_query`, `n_wavelengths_context`,
`n_wavelengths_query`, and serialises every other field through
`_to_builtin`. `to_dict` returns the full payload with arrays
materialised as Python lists.

## Test coverage (D1)

`bench/nirs_synthetic_pfn/tests/test_prior_task.py` (40 tests):

- Constructor populates per-split arrays, ids, wavelengths, target
  semantics, risk gates, and false realism / transfer claims.
- `target_clean` source path produces matching labels and updates
  semantics / provenance accordingly.
- `task_id` is deterministic for identical inputs, ignores numerical
  `X` / `y` values, and changes when the split, `task_seed`, or
  `target_type` changes.
- Index validation rejects overlapping context/query indices,
  duplicates within a split, empty splits, out-of-range indices,
  non-integer indices, and non-1D indices.
- Misaligned `(latent_batch, spectral_view)` pairs raise
  `SpectralViewBatchError(alignment_mismatch)`.
- `dataclasses.replace` paths reject non-finite `X_context`,
  non-finite `y_query`, X/y shape mismatches, non-monotonic
  wavelengths, and overlapping split latent ids.
- Per-split metadata leakage is rejected for all canonical leaky keys
  (top-level, nested, prefixed, and suffixed), while `prior_config` accepts legitimate
  `target_prior` / `concentration_prior` configuration keys but
  rejects per-sample `target_clean`, `target_noisy`, `concentrations`,
  and `latent_features` arrays.
- Missing or partial `provenance.risk_gates` (and a missing
  `risk_gates` key entirely) are rejected with
  `missing_risk_gates`.
- Explicit non-integer `task_seed` values are rejected with reason
  `invalid_task_seed`.
- Classification labels are accepted when integer-like and rejected
  with reason `invalid_classification_labels` when non-integer.
- Multi-output `y` is rejected with reason
  `multi_output_unsupported`.
- `to_light_dict` omits `X_*` and `y_*`; `to_dict` materialises
  arrays as lists.

## Results

- `pytest bench/nirs_synthetic_pfn/tests/test_prior_task.py -q`:
  **40 passed**.
- `pytest bench/nirs_synthetic_pfn/tests -q`:
  **161 passed**, 4 pre-existing sklearn `y residual` warnings
  inherited from the C-suite transfer tests; no D1 regression.
- `ruff check bench/nirs_synthetic_pfn`: **All checks passed**.
- `mypy tasks.py + test_prior_task.py`: **Success, no issues**.

## Known limitations

- D1 only consumes pre-computed integer index splits. There is no
  context/query sampling policy here; D2 will introduce that surface.
- Multi-output targets are rejected, not handled. D3 will lift this
  restriction.
- `target_clean == target_noisy` is inherited from C1 because A2
  `SyntheticDatasetRun` exposes a single target array. D1 records this
  fact in `target_semantics.target_clean_equals_target_noisy` and in
  `provenance.target_clean_equals_target_noisy`.
- `instrument_context` and `instrument_query` always equal the
  canonical instrument carried by the source latent batch. Cross-domain
  / cross-instrument tasks are not supported in D1; the field surface
  is reserved for later phases.
- `wavelengths_context` and `wavelengths_query` are identical (both
  copied from the spectral view); per-split wavelength axes are part
  of the contract surface for later phases but D1 does not vary them.
- D1 performs no realism evaluation, no transfer evaluation, no
  encoder training, and no PFN ingestion.

## Risk gates (negative, documented)

D1 inherits and explicitly documents the two upstream risk gates that
remain **negative**:

- `A3_failed_documented = True` — the fitted-only real-fit adapter
  (Phase A3) is still failing and is documented as such. D1 does not
  depend on A3 and does not lift this gate.
- `B2_realism_failed = True` — synthetic vs real realism scorecards
  (Phase B2) remain blocked. D1 makes no realism claim.

Both flags are recorded inside every task's `provenance.risk_gates`,
together with `provenance.claims = {"realism": False,
"transfer": False}`. The contract refuses to construct or replace a
task whose provenance is missing either gate.

## Decision

D1 contract implemented. `NIRSPriorTask` is the bench-side single-output,
explicit-index PFN-style task surface. No realism, transfer, or
multi-output claim is derived; A3 and B2 remain negative and
documented. D2 (sampler) and D3 (multi-output) are out of scope and
not implemented here.
