# Phase C3 — Multi-view factory aligned to a single CanonicalLatentBatch

## Scope

Phase C3 introduces a **minimal** factory that produces several
`SpectralViewBatch` objects all aligned to the **same**
`CanonicalLatentBatch`. The factory only applies row-wise structural
preprocessing (`identity` / `center` / `snv`) and optional deterministic
additive Gaussian noise to the spectra already produced by A2. It does
**not**:

- rerender spectra from concentrations,
- swap the optical configuration (instrument / measurement mode),
- train an encoder, define a contrastive loss, or build a dataloader,
- introduce any new realism or transfer claim.

C3 is bench-side only and changes nothing under `nirs4all/`.

## Risk gate reminder

The two prior risk gates remain **negative** and are explicitly carried
through in every C3 batch's `metadata.risk_gates` and `metadata.claims`:

- `A3_failed_documented = True` (fitted-only real-fit adapter remains
  failed),
- `B2_realism_failed = True` (synthetic vs real realism scorecards remain
  failed),
- `claims.realism = False`,
- `claims.transfer = False`.

C3 does not weaken or revisit either gate.

## Files modified

- `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/multiview.py`
  (new) — `SpectralViewVariantConfig` and
  `build_same_latent_spectral_views`.
- `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/__init__.py` —
  re-export the new symbols.
- `bench/nirs_synthetic_pfn/tests/test_multiview_factory.py` (new) —
  C3 contract tests.
- `bench/nirs_synthetic_pfn/reports/multiview_same_latent.md` (this
  report).

No file under `nirs4all/` was modified.

## Public surface

```python
@dataclass(frozen=True)
class SpectralViewVariantConfig:
    view_key: str
    preprocessing: Literal["identity", "center", "snv"] = "identity"
    noise_std: float = 0.0
    instrument_key: str | None = None
    measurement_mode: str | None = None
    random_seed_offset: int = 0


def build_same_latent_spectral_views(
    run: SyntheticDatasetRun,
    latent_batch: CanonicalLatentBatch,
    variants: Sequence[SpectralViewVariantConfig] | None = None,
    *,
    random_seed: int | None = None,
) -> tuple[SpectralViewBatch, ...]
```

Default variants when `variants is None`:

| view_key      | preprocessing | noise_std | random_seed_offset |
| ------------- | ------------- | --------- | ------------------ |
| `identity`    | `identity`    | `0.0`     | `0`                |
| `snv_noisy`   | `snv`         | `1e-3`    | `1`                |

## Construction details

Before building variants, the factory independently rebuilds the
canonical latent batch from the same run using the default canonical
`latent_id_prefix` and rejects any mismatch in
canonical `latent_ids`, concentrations, component keys, latent feature
names/values, targets, batch ids, or group ids. This prevents a view from
silently labeling `run.X` with mutated or reordered latent arrays.
This is a strict C3 limitation: a `CanonicalLatentBatch` built with a
custom `latent_id_prefix` is intentionally rejected because its
`latent_ids` no longer match the canonical ids derived from the run.

For each variant the factory then:

1. Calls
   `SpectralViewBatch.from_synthetic_dataset_run(run, latent_batch,
   view_config=..., view_id_prefix=...)`. This delegates the
   cross-batch alignment check (component keys, instrument, measurement
   mode, seed, builder name) to the C2 contract.
2. If the variant is `identity` with `noise_std == 0.0`, returns the base
   batch unchanged.
3. Otherwise, applies the preprocessing row-wise on `run.X`:
   - `center`: subtract the per-row mean.
   - `snv`: subtract the per-row mean and divide by the per-row std
     (zero-std rows fall back to a unit divisor).
4. Adds deterministic Gaussian noise when `noise_std > 0` using a seed
   sequence derived from
   `(random_seed or 0, random_seed_offset, sha256(view_key))`.
5. Replaces `X`, `preprocessing_state` and `noise_state` on the base
   batch via `dataclasses.replace`, so the `SpectralViewBatch.__post_init__`
   contract (finite `X`, monotonic wavelengths, unique `view_ids`,
   namespace non-emptiness, recursive leakage check, risk gates) is
   re-evaluated for the perturbed batch.

`view_config` carries the variant identity (`view_key`, `preprocessing`,
`noise_std`, `random_seed`, `random_seed_offset`) plus the *requested*
instrument / measurement mode separately from the *actual* top-level
ones. Because rerender is unsupported, the top-level `instrument_key` and
`measurement_mode` always follow the run; if a variant requests a
different value, the factory raises a `SpectralViewBatchError` with
reason `rerender_unsupported`.

`view_ids` are deterministic and globally unique across variants of a
single call: the SHA256 input includes the per-variant prefix and the
full `view_config` (which contains `view_key`), so different variants
produce disjoint `view_id` sets.

## Leakage safety

`SpectralViewBatch.__post_init__` already rejects, recursively, any of
the following keys (and their `<key>_*` prefixes) inside `view_config`,
`preprocessing_state`, `noise_state` and `metadata`:

- `y`, `target`, `targets`, `concentration`, `concentrations`,
  `target_clean`, `target_noisy`, `latent_feature`, `latent_features`.

The C3 namespaces use only structural keys (`phase`, `view_kind`,
`view_key`, `preprocessing`, `method`, `operations`, `noise_std`,
`view_noise_std`, `noise_added_in_view`, `random_seed`,
`random_seed_offset`, `noise_distribution`, `instrument_key`,
`measurement_mode`, `requested_instrument_key`,
`requested_measurement_mode`, `source`, `note`). None of those are leaky.
The contract tests assert this both via the validating constructor and
via an independent recursive walker.

## Commands

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src \
    pytest bench/nirs_synthetic_pfn/tests/test_multiview_factory.py -q

PYTHONPATH=bench/nirs_synthetic_pfn/src \
    pytest \
        bench/nirs_synthetic_pfn/tests/test_canonical_latents.py \
        bench/nirs_synthetic_pfn/tests/test_spectral_views.py \
        bench/nirs_synthetic_pfn/tests/test_multiview_factory.py -q

PYTHONPATH=bench/nirs_synthetic_pfn/src \
    pytest bench/nirs_synthetic_pfn/tests -q

ruff check bench/nirs_synthetic_pfn

PYTHONPATH=bench/nirs_synthetic_pfn/src \
    mypy bench/nirs_synthetic_pfn/src/nirsyntheticpfn/data/multiview.py \
         bench/nirs_synthetic_pfn/tests/test_multiview_factory.py
```

## Results

- `pytest test_multiview_factory.py`: 11 passed.
- `pytest test_canonical_latents.py + test_spectral_views.py +
  test_multiview_factory.py`: 67 passed (no C1/C2 regression).
- `pytest bench/nirs_synthetic_pfn/tests`: 121 passed, 4 sklearn PLS
  constant-residual warnings.
- `ruff check bench/nirs_synthetic_pfn`: All checks passed.
- `mypy multiview.py + test_multiview_factory.py`: Success, no issues.

## Test coverage (C3)

- `test_default_factory_produces_at_least_two_aligned_views`: ≥ 2 valid
  `SpectralViewBatch`, exact `latent_ids` alignment, finite `X`,
  identical `wavelengths`, unchanged `instrument_key` / `measurement_mode`,
  required risk gates and false realism / transfer claims.
- `test_default_factory_is_deterministic_under_same_seed`: same
  `random_seed` and same default variants ⇒ identical `X` arrays and
  identical `view_ids` across two independent constructions.
- `test_view_ids_differ_between_variants_within_same_call`: `view_ids`
  are unique within each variant **and** disjoint across variants in a
  single call.
- `test_non_identity_variant_changes_X_and_records_state`: the
  non-identity variant has `X != run.X`, exposes a non-identity
  `preprocessing_state.method`, and faithfully reports
  `noise_added_in_view` / `view_noise_std` / `noise_distribution`.
- `test_factory_constructions_have_no_leakage_in_namespaces`: explicit
  recursive walker confirms zero leaky keys in `view_config`,
  `preprocessing_state`, `noise_state`, `metadata`.
- `test_factory_rejects_duplicate_view_keys`: duplicate `view_key`
  raises `duplicate_view_key`.
- `test_factory_rejects_negative_noise_std`: negative `noise_std`
  raises `invalid_noise_std`.
- `test_factory_rejects_unsupported_instrument_override`: a variant
  requesting a different instrument raises `rerender_unsupported`.
- `test_factory_rejects_empty_variants`: empty variants tuple raises
  `empty_variants`.
- `test_factory_rejects_latent_arrays_that_do_not_match_run`: a mutated
  latent array with otherwise valid provenance raises
  `same_latent_mismatch`.
- `test_custom_variants_produce_expected_count_and_alignment`: custom
  3-variant call returns exactly 3 batches; `center` rows have zero
  mean; `snv` rows have zero mean and unit std (within tolerance); all
  three batches share the same `latent_ids`.

## Limitations

- Spectra are **not** rerendered from concentrations. C3 only applies
  bench-side row-wise transforms and additive noise on the existing
  `run.X`. Hence `instrument_key` / `measurement_mode` cannot be
  changed; any such request is rejected.
- No encoder, no contrastive loss, no dataloader, no training is
  introduced.
- `target_clean == target_noisy` is inherited from C1 because A2 does
  not expose a separate noiseless target. This is the same caveat
  documented for C1/C2.
- The factory does not improve realism or transfer. **A3 and B2 remain
  failed.** C3 only delivers a structural alignment of multiple views to
  the same canonical latent batch.

## Next

Out of scope for C3 and not implemented here: encoder, contrastive
training, real rerender from concentrations through the A2 generator,
PFN ingestion, or any realism / transfer claim.
