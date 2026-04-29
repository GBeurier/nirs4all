"""Phase C3 minimal multi-view factory for :class:`SpectralViewBatch`.

C3 is intentionally narrow. This module produces several
:class:`SpectralViewBatch` objects aligned to the same
:class:`CanonicalLatentBatch` by applying deterministic structural
preprocessing (``identity`` / ``center`` / ``snv``) and optional additive
Gaussian noise to the spectra already produced by A2. It does **not**:

- rerender spectra from concentrations,
- swap the optical configuration (instrument / measurement mode),
- train an encoder, define a contrastive loss, or build a dataloader,
- introduce any new realism or transfer claim.

Risk gates inherited from earlier phases remain unchanged:
- ``A3_failed_documented`` (fitted-only real-fit adapter remains failed),
- ``B2_realism_failed`` (synthetic vs real realism scorecards remain failed).

The factory delegates the cross-batch alignment check to
:meth:`SpectralViewBatch.from_synthetic_dataset_run`, then overrides the
spectra, the ``preprocessing_state`` and the ``noise_state`` namespaces to
faithfully describe the bench-side perturbation that has been applied.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from nirsyntheticpfn.adapters.builder_adapter import SyntheticDatasetRun
from nirsyntheticpfn.data.latents import CanonicalLatentBatch
from nirsyntheticpfn.data.views import SpectralViewBatch, SpectralViewBatchError

__all__ = [
    "SpectralViewVariantConfig",
    "build_same_latent_spectral_views",
]

PreprocessingMethod = Literal["identity", "center", "snv"]
_VALID_PREPROCESSING: tuple[str, ...] = ("identity", "center", "snv")


@dataclass(frozen=True)
class SpectralViewVariantConfig:
    """Immutable description of one bench-side spectral view variant.

    Attributes:
        view_key: Non-empty identifier unique within a single
            :func:`build_same_latent_spectral_views` call. It is used both
            as part of the ``view_id`` prefix and as a salt for the noise
            seed sequence so different variants get different deterministic
            noise realisations.
        preprocessing: Row-wise preprocessing applied to ``run.X``. Only
            ``"identity"``, ``"center"`` and ``"snv"`` are accepted.
        noise_std: Standard deviation of the additive Gaussian noise
            applied after preprocessing. Must be a finite, non-negative
            float. Defaults to ``0.0`` (no extra noise).
        instrument_key: Optional instrument override. If not ``None``, it
            must equal the run's instrument; the C3 minimal factory does
            not rerender spectra and therefore refuses to silently accept a
            different instrument.
        measurement_mode: Optional measurement mode override; same
            semantics as ``instrument_key``.
        random_seed_offset: Integer offset combined with the function-level
            ``random_seed`` and a hash of ``view_key`` to seed the noise
            RNG deterministically.
    """

    view_key: str
    preprocessing: PreprocessingMethod = "identity"
    noise_std: float = 0.0
    instrument_key: str | None = None
    measurement_mode: str | None = None
    random_seed_offset: int = 0


_DEFAULT_VARIANTS: tuple[SpectralViewVariantConfig, ...] = (
    SpectralViewVariantConfig(view_key="identity"),
    SpectralViewVariantConfig(
        view_key="snv_noisy",
        preprocessing="snv",
        noise_std=1e-3,
        random_seed_offset=1,
    ),
)


def build_same_latent_spectral_views(
    run: SyntheticDatasetRun,
    latent_batch: CanonicalLatentBatch,
    variants: Sequence[SpectralViewVariantConfig] | None = None,
    *,
    random_seed: int | None = None,
) -> tuple[SpectralViewBatch, ...]:
    """Build several :class:`SpectralViewBatch` aligned to ``latent_batch``.

    For each variant, ``run.X`` and ``run.wavelengths`` are reused (no
    rerender). A row-wise preprocessing is applied, then deterministic
    additive Gaussian noise is added when ``variant.noise_std > 0``. All
    returned batches share the same ``latent_ids`` (matching
    ``latent_batch.latent_ids`` exactly), the same ``wavelengths`` and the
    same top-level ``instrument_key`` / ``measurement_mode`` as the run.

    Args:
        run: A2 ``SyntheticDatasetRun`` that produced the spectra.
        latent_batch: Canonical latent batch derived from the same run.
        variants: Sequence of variant configurations. Defaults to a small
            deterministic pair: one ``"identity"`` view and one
            ``"snv_noisy"`` view with a small additive noise.
        random_seed: Optional integer seed mixed into the noise RNG. When
            ``None``, the noise still depends on ``view_key`` and
            ``random_seed_offset`` only.

    Returns:
        A tuple of :class:`SpectralViewBatch`, one per variant, in the
        order of ``variants``.

    Raises:
        SpectralViewBatchError: If variants are invalid (duplicate
            ``view_key``, negative or non-finite ``noise_std``, unknown
            preprocessing, or unsupported instrument/mode override) or if
            the latent batch does not align with the run.
    """
    chosen: tuple[SpectralViewVariantConfig, ...] = (
        tuple(variants) if variants is not None else _DEFAULT_VARIANTS
    )
    if not chosen:
        raise SpectralViewBatchError([{
            "reason": "empty_variants",
            "field": "variants",
            "message": "variants must contain at least one SpectralViewVariantConfig",
        }])

    run_instrument_key = str(run.builder_config["features"]["instrument"])
    run_measurement_mode = str(run.builder_config["features"]["measurement_mode"])
    builder_name = str(run.builder_config["name"])

    _validate_variants(
        chosen,
        run_instrument_key=run_instrument_key,
        run_measurement_mode=run_measurement_mode,
    )
    _validate_latent_batch_matches_run(run, latent_batch)

    batches: list[SpectralViewBatch] = []
    for variant in chosen:
        view_config = _build_view_config(
            variant=variant,
            run_instrument_key=run_instrument_key,
            run_measurement_mode=run_measurement_mode,
            random_seed=random_seed,
        )
        view_id_prefix = f"{builder_name}__view__{variant.view_key}"

        base = SpectralViewBatch.from_synthetic_dataset_run(
            run,
            latent_batch,
            view_config=view_config,
            view_id_prefix=view_id_prefix,
        )

        if variant.preprocessing == "identity" and variant.noise_std == 0.0:
            batches.append(base)
            continue

        X_pre = _apply_preprocessing(base.X, variant.preprocessing)
        X_final = _apply_noise(
            X_pre,
            noise_std=float(variant.noise_std),
            random_seed=random_seed,
            random_seed_offset=int(variant.random_seed_offset),
            view_key=variant.view_key,
        )

        new_preprocessing_state: dict[str, Any] = {
            "phase": "C3",
            "preprocessing_applied": variant.preprocessing != "identity",
            "method": variant.preprocessing,
            "operations": (
                [] if variant.preprocessing == "identity" else [variant.preprocessing]
            ),
            "note": (
                "Bench-side row-wise preprocessing applied to run.X. "
                "C3 does not rerender spectra from concentrations."
            ),
        }
        new_noise_state: dict[str, Any] = {
            "phase": "C3",
            "noise_added_in_view": variant.noise_std > 0.0,
            "view_noise_std": float(variant.noise_std),
            "source_noise_level": base.noise_state["source_noise_level"],
            "random_seed": int(random_seed) if random_seed is not None else None,
            "random_seed_offset": int(variant.random_seed_offset),
            "noise_distribution": (
                "gaussian" if variant.noise_std > 0.0 else None
            ),
            "note": (
                "Bench-side additive Gaussian noise applied after preprocessing. "
                "Source A2 noise is preserved; this is an additional structural "
                "perturbation."
            ),
        }
        batches.append(replace(
            base,
            X=np.ascontiguousarray(X_final, dtype=float),
            preprocessing_state=new_preprocessing_state,
            noise_state=new_noise_state,
        ))

    return tuple(batches)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_latent_batch_matches_run(
    run: SyntheticDatasetRun,
    latent_batch: CanonicalLatentBatch,
) -> None:
    expected = CanonicalLatentBatch.from_synthetic_dataset_run(run)
    failures: list[dict[str, str]] = []

    if tuple(latent_batch.latent_ids) != tuple(expected.latent_ids):
        failures.append(_same_latent_failure(
            field="latent_ids",
            message=(
                "latent_batch.latent_ids must be the canonical ids derived "
                "from this SyntheticDatasetRun, in row order."
            ),
        ))
    if tuple(latent_batch.component_keys) != tuple(expected.component_keys):
        failures.append(_same_latent_failure(
            field="component_keys",
            message="latent_batch.component_keys do not match this run.",
        ))
    if tuple(latent_batch.latent_feature_names) != tuple(expected.latent_feature_names):
        failures.append(_same_latent_failure(
            field="latent_feature_names",
            message="latent_batch.latent_feature_names do not match this run.",
        ))
    if tuple(latent_batch.batch_ids) != tuple(expected.batch_ids):
        failures.append(_same_latent_failure(
            field="batch_ids",
            message="latent_batch.batch_ids do not match this run.",
        ))
    if tuple(latent_batch.group_ids) != tuple(expected.group_ids):
        failures.append(_same_latent_failure(
            field="group_ids",
            message="latent_batch.group_ids do not match this run.",
        ))

    _append_array_mismatch(
        failures,
        observed=latent_batch.concentrations,
        expected=expected.concentrations,
        field="concentrations",
    )
    _append_array_mismatch(
        failures,
        observed=latent_batch.latent_features,
        expected=expected.latent_features,
        field="latent_features",
    )
    _append_array_mismatch(
        failures,
        observed=latent_batch.target_clean,
        expected=expected.target_clean,
        field="target_clean",
    )
    _append_array_mismatch(
        failures,
        observed=latent_batch.target_noisy,
        expected=expected.target_noisy,
        field="target_noisy",
    )

    if failures:
        raise SpectralViewBatchError(failures)


def _append_array_mismatch(
    failures: list[dict[str, str]],
    *,
    observed: np.ndarray,
    expected: np.ndarray,
    field: str,
) -> None:
    observed_arr = np.asarray(observed)
    expected_arr = np.asarray(expected)
    if observed_arr.shape != expected_arr.shape or not np.array_equal(
        observed_arr,
        expected_arr,
    ):
        failures.append(_same_latent_failure(
            field=field,
            message=f"latent_batch.{field} does not match this run.",
        ))


def _same_latent_failure(*, field: str, message: str) -> dict[str, str]:
    return {
        "reason": "same_latent_mismatch",
        "field": field,
        "message": message,
    }


def _validate_variants(
    variants: tuple[SpectralViewVariantConfig, ...],
    *,
    run_instrument_key: str,
    run_measurement_mode: str,
) -> None:
    failures: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for variant in variants:
        if not isinstance(variant.view_key, str) or not variant.view_key:
            failures.append({
                "reason": "invalid_view_key",
                "field": "variants",
                "message": "variant.view_key must be a non-empty string",
            })
            continue
        if variant.view_key in seen_keys:
            failures.append({
                "reason": "duplicate_view_key",
                "field": "variants",
                "message": f"duplicate variant.view_key={variant.view_key!r}",
            })
        seen_keys.add(variant.view_key)
        if variant.preprocessing not in _VALID_PREPROCESSING:
            failures.append({
                "reason": "invalid_preprocessing",
                "field": "variants",
                "message": (
                    f"variant {variant.view_key!r} preprocessing="
                    f"{variant.preprocessing!r} not in {_VALID_PREPROCESSING}"
                ),
            })
        if (
            not isinstance(variant.noise_std, (int, float))
            or not np.isfinite(float(variant.noise_std))
            or float(variant.noise_std) < 0.0
        ):
            failures.append({
                "reason": "invalid_noise_std",
                "field": "variants",
                "message": (
                    f"variant {variant.view_key!r} noise_std="
                    f"{variant.noise_std!r} must be a finite non-negative float"
                ),
            })
        if (
            variant.instrument_key is not None
            and variant.instrument_key != run_instrument_key
        ):
            failures.append({
                "reason": "rerender_unsupported",
                "field": "variants",
                "message": (
                    f"variant {variant.view_key!r} requested instrument_key="
                    f"{variant.instrument_key!r} but the C3 minimal factory does "
                    "not rerender spectra; only the run's instrument is supported."
                ),
            })
        if (
            variant.measurement_mode is not None
            and variant.measurement_mode != run_measurement_mode
        ):
            failures.append({
                "reason": "rerender_unsupported",
                "field": "variants",
                "message": (
                    f"variant {variant.view_key!r} requested measurement_mode="
                    f"{variant.measurement_mode!r} but the C3 minimal factory does "
                    "not rerender spectra; only the run's measurement_mode is "
                    "supported."
                ),
            })
    if failures:
        raise SpectralViewBatchError(failures)


def _build_view_config(
    *,
    variant: SpectralViewVariantConfig,
    run_instrument_key: str,
    run_measurement_mode: str,
    random_seed: int | None,
) -> dict[str, Any]:
    return {
        "phase": "C3",
        "view_kind": "preprocessing_noise_view",
        "view_key": variant.view_key,
        "preprocessing": variant.preprocessing,
        "noise_std": float(variant.noise_std),
        "random_seed": int(random_seed) if random_seed is not None else None,
        "random_seed_offset": int(variant.random_seed_offset),
        "instrument_key": run_instrument_key,
        "measurement_mode": run_measurement_mode,
        "requested_instrument_key": variant.instrument_key,
        "requested_measurement_mode": variant.measurement_mode,
        "source": "build_same_latent_spectral_views",
        "note": (
            "Structural view aligned to the same CanonicalLatentBatch; "
            "applies preprocessing/noise on run.X without rerendering. "
            "Top-level instrument_key / measurement_mode follow the run."
        ),
    }


def _apply_preprocessing(X: np.ndarray, method: str) -> np.ndarray:
    if method == "identity":
        return np.asarray(X, dtype=float)
    if method == "center":
        arr = np.asarray(X, dtype=float)
        mean = arr.mean(axis=1, keepdims=True)
        return np.asarray(arr - mean, dtype=float)
    if method == "snv":
        arr = np.asarray(X, dtype=float)
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True, ddof=0)
        safe_std = np.where(std == 0.0, 1.0, std)
        return np.asarray((arr - mean) / safe_std, dtype=float)
    raise SpectralViewBatchError([{
        "reason": "invalid_preprocessing",
        "field": "variants",
        "message": f"unknown preprocessing method={method!r}",
    }])


def _apply_noise(
    X: np.ndarray,
    *,
    noise_std: float,
    random_seed: int | None,
    random_seed_offset: int,
    view_key: str,
) -> np.ndarray:
    if noise_std <= 0.0:
        return X
    view_key_entropy = int(
        hashlib.sha256(view_key.encode("utf-8")).hexdigest()[:16],
        16,
    )
    seed_seq = np.random.SeedSequence([
        int(random_seed) if random_seed is not None else 0,
        int(random_seed_offset),
        view_key_entropy,
    ])
    rng = np.random.default_rng(seed_seq)
    noise = rng.normal(0.0, float(noise_std), size=X.shape)
    return X + noise
