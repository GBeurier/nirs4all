# Spectral View Batch (Phase C2)

## Objective

Define the bench-side `SpectralViewBatch` contract that aligns one A2
`SyntheticDatasetRun` to a `CanonicalLatentBatch` (C1) as a frozen,
validated container of:

- the rendered spectra `X` and the wavelength axis,
- per-row `latent_ids` (mirrored from C1) and deterministic `view_ids`,
- a structured `view_config`, `preprocessing_state`, `noise_state`, and
  generic `metadata` namespace,
- the optical configuration (`instrument_key`, `measurement_mode`).

C2 only declares this contract surface. It does **not** implement a
multi-view factory, an encoder, a preprocessing pipeline, or a noise
augmentation layer; it does not introduce any realism or transfer claim.

## Scope

In scope:

- `SpectralViewBatch` frozen dataclass with `__post_init__` validation.
- `SpectralViewBatchError(ValueError)` carrying a structured failure list.
- `SpectralViewBatch.from_synthetic_dataset_run(run, latent_batch, *, view_config=None, view_id_prefix=None)`
  constructor that consumes the existing A2 `SyntheticDatasetRun` and a
  `CanonicalLatentBatch`.
- `subset(indices)` returning a new validated batch.
- `assert_aligned_to(latent_batch)` for explicit alignment checks.
- `to_dict()` and `to_light_dict()` serialisation helpers.
- Deterministic `view_ids` derived via `hashlib.sha256` from
  `(prefix, latent_id, instrument_key, measurement_mode,
  wavelength_summary, view_config, row_index)`.
- New `tests/test_spectral_views.py` covering construction, alignment,
  determinism, subset semantics, validation failures (shape,
  finiteness, monotonicity, id length, duplication), metadata leakage,
  risk-gate enforcement, and light/full serialisation.

Out of scope (deferred):

- Multi-view factories, augmentations, preprocessing pipelines (left to
  later phases).
- Latent or spectral encoders, PFN ingestion (C3).
- Any new realism or transfer evaluation.
- Lifting the inherited A3 / B2 risk gates.

## Files changed

- `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/views.py` (new)
- `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/__init__.py`
  (re-export `SpectralViewBatch`, `SpectralViewBatchError`)
- `bench/nirs_synthetic_pfn/tests/test_spectral_views.py` (new)
- `bench/nirs_synthetic_pfn/reports/spectral_view_batch.md` (new, this
  report)

No file inside `nirs4all/` is modified. C2 is implemented entirely under
`bench/nirs_synthetic_pfn/`.

## Commands

Targeted suite:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src \
    pytest bench/nirs_synthetic_pfn/tests/test_spectral_views.py -q
```

Full bench suite:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src \
    pytest bench/nirs_synthetic_pfn/tests -q
```

Static checks on the new files:

```bash
ruff check bench/nirs_synthetic_pfn

PYTHONPATH=bench/nirs_synthetic_pfn/src \
    mypy \
    bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/views.py \
    bench/nirs_synthetic_pfn/tests/test_spectral_views.py
```

## Results

- `test_spectral_views.py`: all tests pass.
- Full bench suite passes; pre-existing sklearn `y residual` warnings
  inherited from the transfer tests are unchanged by C2.
- `ruff check bench/nirs_synthetic_pfn`: all checks passed.
- `mypy` on `views.py` and `test_spectral_views.py`: success, no issues
  found.

## Contract checks

`SpectralViewBatch.__post_init__` enforces:

- `X`: 2D numeric and finite array with at least one row.
- `wavelengths`: non-empty 1D numeric and finite array, strictly
  increasing, with size matching `X.shape[1]`.
- `latent_ids`: tuple of non-empty strings of length `n_rows`.
- `view_ids`: tuple of non-empty strings of length `n_rows`, all unique.
- `instrument_key`, `measurement_mode`: non-empty strings.
- `view_config`, `preprocessing_state`, `noise_state`, `metadata`:
  non-empty `dict` instances.
- `view_config`, `preprocessing_state`, `noise_state`, and `metadata`
  must not contain target or latent leakage keys at any nested path
  (`y`, `target(s)`, `target_clean`, `target_noisy`,
  `concentration(s)`, `latent_feature(s)`, including prefixed forms
  such as `y_min` or `concentration_row_sum_min`).
- `metadata.risk_gates` must be a dict and must equal at least
  `{"A3_failed_documented": True, "B2_realism_failed": True}`.

Violations raise `SpectralViewBatchError` carrying the structured
failure list (`reason`, `field`, `message`).

`from_synthetic_dataset_run`:

- Reuses `run.X` and `run.wavelengths` as-is (cast to `float64`
  contiguous arrays). No preprocessing is applied.
- Requires `run.X.shape[0] == len(latent_batch.latent_ids)` and checks
  the C1 provenance/metadata against the A2 run (`builder_config_name`,
  `random_state`, domain, instrument, measurement mode, and component
  keys) before copying the canonical latent ids in order.
- Reads `instrument_key` and `measurement_mode` from
  `run.builder_config["features"]`.
- Builds a stable wavelength signature (`n_wavelengths`, `first_nm`,
  `last_nm`, `step_nm` when uniform) and a stable `view_config`
  signature via canonical JSON, then derives each `view_id` from the
  `(prefix, latent_id, instrument_key, measurement_mode, wavelength
  signature, view_config signature, row index)` payload through
  `hashlib.sha256`.
- Defaults `view_config` to a single-render description carrying the
  C2 phase tag, instrument/mode, the wavelength summary, and the
  source contract `SyntheticDatasetRun`. A custom non-empty
  `view_config` may be supplied to differentiate variants.
- Sets `preprocessing_state` to declare that no preprocessing was
  applied (`preprocessing_applied=False`, empty `operations`).
- Sets `noise_state` to record the source nuisance noise level (when
  present) and to declare that the C2 view does not add or remove
  noise; A2 already includes generation noise and nuisance effects.
- Records C2 metadata: phase tag, source contract
  `CanonicalLatentBatch`, source latent count, builder config name,
  a filtered A2 spectral validation summary that excludes target and
  latent/concentration fields, `claims.realism=False`,
  `claims.transfer=False`, and the inherited risk gates.

`subset`:

- Validates 1D, in-range integer indices and slices `X`, `latent_ids`,
  and `view_ids` deterministically. Wavelengths and metadata are
  preserved. `dataclasses.replace` re-runs validation on the result.

`assert_aligned_to`:

- Raises `SpectralViewBatchError` (`alignment_mismatch`) when
  `self.latent_ids` does not match the target
  `CanonicalLatentBatch.latent_ids` exactly (same order).

## C3 readiness

C2 leaves the C3 surface intentionally empty:

- The contract is single-view and preprocessing-free; richer view
  factories and preprocessing pipelines can be layered on top without
  changing the latent surface or the C2 contract.
- `view_ids` are deterministic and unique within a batch, so
  later phases can reference samples by `(latent_id, view_id)` pairs
  once multi-view rendering is introduced.
- `preprocessing_state` and `noise_state` are explicit non-empty dicts
  with `preprocessing_applied=False` / `noise_added_in_view=False`, so
  later phases can override them without weakening the contract.

No part of the C3 stack is implemented here, by design.

## Limitations

- Only one rendered view per latent id is carried; multi-view rendering
  is deferred.
- Spectra are taken as-is from `SyntheticDatasetRun.X`; no
  preprocessing or augmentation pipeline is applied.
- Validation is structural only. C2 does not assert anything about
  realism, transferability, or model behaviour.
- Determinism is verified at the contract level: `view_ids` depend on
  a documented payload `(prefix, latent_id, instrument, mode,
  wavelength signature, view_config signature, row index)`. Underlying
  numerical reproducibility of `SyntheticDatasetRun` is the
  responsibility of A2.

## Risk gates

- `A3_failed_documented` — fitted-only real-fit adapter (Phase A3) is
  still failing and is documented as such. C2 does not depend on A3 and
  does not lift this gate.
- `B2_realism_failed` — synthetic vs real realism scorecards (Phase B2)
  remain blocked. C2 makes no realism claim.

Both flags are recorded inside every batch under `metadata.risk_gates`.

## Decision

C2 contract implemented; no realism or transfer claim is derived from
this batch.
