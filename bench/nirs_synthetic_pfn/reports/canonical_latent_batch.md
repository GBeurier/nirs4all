# Canonical Latent Batch (Phase C1)

## Objective

Define the bench-side `CanonicalLatentBatch` contract that turns one A2
`SyntheticDatasetRun` into a frozen, validated container of:

- deterministic latent ids,
- mixture concentrations and component keys,
- a minimal numeric latent feature matrix,
- clean / noisy targets,
- batch / group / split labels,
- structured metadata namespaces (domain, component, instrument, target,
  split, view, optical, baseline, scatter, environment,
  sample-presentation, provenance).

C1 only declares this contract surface. It does not render spectra
(C2 / `SpectralViewBatch`), does not construct a multi-view factory, and
does not implement any encoder or downstream PFN ingestion path (C3).

## Scope

In scope:

- `CanonicalLatentBatch` frozen dataclass with `__post_init__` validation.
- `CanonicalLatentBatchError(ValueError)` carrying a structured failure list.
- `CanonicalLatentBatch.from_synthetic_dataset_run` constructor that consumes
  the existing A2 `SyntheticDatasetRun` (`X`, `y`, `wavelengths`, `metadata`,
  `latent_metadata`, `prior_config`, `builder_config`, `validation_summary`).
- `subset(indices, split_label=None)` returning a new validated batch.
- `to_dict()` and `to_light_dict()` serialisation helpers.
- Deterministic ids derived from `seed`, run `name`, `domain_key`,
  `instrument_key`, `measurement_mode`, and sample index via `hashlib.sha256`.
- New `tests/test_canonical_latents.py` covering conversion, id stability,
  subset semantics, validation failures, non-numeric array rejection, and
  light/full serialisation.

Out of scope (deferred):

- `SpectralViewBatch` (C2) — only an `intended_contract` placeholder is
  carried in `view_metadata`, with `rendered_view_count = 0`.
- Multi-view factory and per-view spectrum rendering.
- Latent encoders or PFN training adapters (C3).
- Any new realism or transfer evaluation.

## Files changed

- `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/latents.py` (new)
- `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/__init__.py`
  (re-export `CanonicalLatentBatch`, `CanonicalLatentBatchError`)
- `bench/nirs_synthetic_pfn/tests/test_canonical_latents.py` (new)
- `bench/nirs_synthetic_pfn/reports/canonical_latent_batch.md` (new, this
  report)

No file inside `nirs4all/` is modified. C1 is implemented entirely under
`bench/nirs_synthetic_pfn/`.

## Commands

Targeted suite:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src \
    pytest bench/nirs_synthetic_pfn/tests/test_canonical_latents.py -q
```

Full bench suite:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src \
    pytest bench/nirs_synthetic_pfn/tests -q
```

Static checks on the new files:

```bash
ruff check \
    bench/nirs_synthetic_pfn

PYTHONPATH=bench/nirs_synthetic_pfn/src \
    mypy \
    bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/latents.py \
    bench/nirs_synthetic_pfn/tests/test_canonical_latents.py
```

## Results

- `test_canonical_latents.py`: 22 passed.
- Full bench suite: 76 passed, 4 sklearn `y residual` warnings inherited
  from existing transfer tests (unchanged by C1).
- `ruff check bench/nirs_synthetic_pfn`: all checks passed.
- `mypy` on `latents.py` and `test_canonical_latents.py`: success, no issues
  found.

## Contract checks

`CanonicalLatentBatch.__post_init__` enforces:

- `latent_ids`: non-empty, non-empty strings, unique, length `n`.
- `concentrations`: 2D `(n, c)`, numeric and finite, values `>= 0`,
  row sums `~= 1.0` (atol/rtol `1e-6`), `c == len(component_keys)`.
- `component_keys`: non-empty.
- `latent_features`: 2D `(n, f)`, numeric and finite,
  `f == len(latent_feature_names)`.
- `target_clean` / `target_noisy`: numeric and finite, identical shape, first
  dimension equal to `n`.
- `batch_ids`, `group_ids`: length `n`.
- `split_labels`: `None` or length `n`.
- `domain_metadata`, `component_metadata`, `instrument_metadata`,
  `target_metadata`, `split_metadata`, `view_metadata`, `optical_metadata`,
  `baseline_metadata`, `scatter_metadata`, `environment_metadata`,
  `sample_presentation_metadata`, `provenance`: non-empty `dict`.

Violations raise `CanonicalLatentBatchError` carrying the structured
failure list (`reason`, `field`, `message`).

`from_synthetic_dataset_run`:

- Reuses `run.latent_metadata["concentrations"]`, already validated by
  `build_synthetic_dataset_run` and consistent with B1 expectations
  (finite, 2D, in `[0, 1]`, row-normalized).
- Derives latent ids from `(seed, run name, domain, instrument, mode,
  index)` via `sha256`. Two A2 runs with identical inputs produce
  identical id tuples (covered by
  `test_latent_ids_are_deterministic_across_identical_runs`).
- Builds `latent_features` from existing nuisance fields:
  `temperature_c`, `particle_size_um`, `noise_level`, plus
  `batch_id_numeric` when `latent_metadata["batch_ids"]` is convertible
  to floats.
- Carries explicit component spectrum reference keys from
  `builder_config.features.components` and explicit optical metadata noting
  that A2 exposes instrument/mode geometry but no standalone path length.
- Sets `target_clean = target_noisy = run.y`. The A2 contract does not
  expose a separate noiseless target; this is recorded in
  `target_metadata` (`target_clean_equals_target_noisy=True`,
  `note=...`) and in `provenance` (`target_clean_source`,
  `target_noisy_source`).
- Captures A1 / A2 provenance through `provenance.a1_provenance` and
  `provenance.a2_validation_summary`.

`subset`:

- Validates 1D, in-range integer indices.
- Slices arrays and id tuples deterministically.
- When `split_label` is given, fills `split_labels` with the label for
  every kept row; otherwise it slices the existing `split_labels`.
- Returns a new validated batch via `dataclasses.replace`, so all
  contract checks re-run.

## C2 / C3 readiness

C1 leaves the C2 / C3 surface intentionally empty:

- `view_metadata` carries `intended_contract = "SpectralViewBatch"` and
  `rendered_view_count = 0`. C2 will populate spectral views without
  changing the latent surface.
- `latent_features` is intentionally minimal (3-4 numeric columns) so
  C3 encoders can extend the latent representation without breaking
  existing consumers.
- Ids and metadata are deterministic, so C2 and downstream phases can
  cross-reference samples by id alone.

No part of the C2/C3 stack is implemented here, by design.

## Limitations

- `target_noisy` is identical to `target_clean` because A2 only exposes
  one target array. A future phase that produces a separately-noised
  target should overwrite `target_noisy` and update `target_metadata`.
- `latent_features` is restricted to fields already present in
  `SyntheticDatasetRun`; no new physics or chemistry is introduced.
- Validation is structural only. C1 does not assert anything about
  realism, transferability, or downstream model behaviour.
- Determinism is verified at the contract level: ids depend on a
  documented payload `(prefix, seed, domain, instrument, mode, index)`.
  Underlying numerical reproducibility of `SyntheticDatasetRun` is the
  responsibility of A2.

## Gate flags

- `A3_failed_documented` — fitted-only real-fit adapter (Phase A3) is
  still failing and is documented as such. C1 does not depend on A3 and
  does not lift this gate.
- `B2_realism_failed` — synthetic vs real realism scorecards (Phase B2)
  remain blocked. C1 makes no realism claim.

Both flags are recorded inside every batch under
`provenance.risk_gates`.

## Decision

C1 contract implemented; no downstream realism/transfer claim.
