"""Phase C2 spectral view batch contract.

This module defines :class:`SpectralViewBatch`, the bench-side spectral
container that aligns rendered spectra to a
:class:`nirsyntheticpfn.data.latents.CanonicalLatentBatch`. C2 is intentionally
narrow: it carries one set of spectra (already produced by A2) per latent id,
with deterministic view ids and structured ``view_config`` /
``preprocessing_state`` / ``noise_state`` / ``metadata`` namespaces. It does
**not** implement a multi-view factory, an encoder, or any realism / transfer
claim. C3 (multi-view rendering, encoders, PFN ingestion) is out of scope.

Risk gates inherited from earlier phases:
- ``A3_failed_documented`` (fitted-only real-fit adapter remains failed).
- ``B2_realism_failed`` (synthetic vs real realism scorecards remain failed).

The contract is independent from those gates: this module only aligns
existing A2 outputs to a canonical latent batch and does not introduce any
new realism or transfer claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, cast

import numpy as np

from nirsyntheticpfn.adapters.builder_adapter import SyntheticDatasetRun
from nirsyntheticpfn.data.latents import CanonicalLatentBatch

__all__ = [
    "SpectralViewBatch",
    "SpectralViewBatchError",
]

_LEAKY_METADATA_KEYS = (
    "y",
    "target",
    "targets",
    "concentration",
    "target_clean",
    "target_noisy",
    "concentrations",
    "latent_feature",
    "latent_features",
)
_REQUIRED_RISK_GATES: dict[str, bool] = {
    "A3_failed_documented": True,
    "B2_realism_failed": True,
}
_REQUIRED_NON_EMPTY_DICTS = (
    "view_config",
    "preprocessing_state",
    "noise_state",
    "metadata",
)


class SpectralViewBatchError(ValueError):
    """Raised when a :class:`SpectralViewBatch` fails its contract checks."""

    def __init__(self, failures: list[dict[str, str]]) -> None:
        self.failures = failures
        summary = "; ".join(
            f"{failure.get('reason', 'unknown')}:{failure.get('field', '?')}"
            for failure in failures
        )
        super().__init__(summary or "invalid SpectralViewBatch")


@dataclass(frozen=True)
class SpectralViewBatch:
    """Spectral view batch aligned to a :class:`CanonicalLatentBatch`.

    The batch is a frozen contract object. It carries:

    - ``X``: 2D float spectra ``(n_rows, n_wavelengths)``.
    - ``wavelengths``: 1D strictly increasing wavelength axis (nm).
    - ``latent_ids``: tuple of canonical latent ids (one per row).
    - ``view_ids``: deterministic per-row view ids unique within the batch.
    - ``view_config``: structured description of the rendered view.
    - ``instrument_key`` / ``measurement_mode``: optical configuration.
    - ``preprocessing_state``: declares whether any preprocessing was applied
      to ``X`` (C2 default: no preprocessing).
    - ``noise_state``: declares the source noise level and whether the C2
      view itself adds or removes noise (C2 default: no view-level noise).
    - ``metadata``: bench-level metadata. The contract forbids leakage keys
      (``y``, ``target``, ``target_clean``, ``target_noisy``,
      ``concentrations``, ``latent_features``) and requires both A3 / B2
      risk gates to be carried.

    The link to C1 is by ``latent_ids`` only; targets, concentrations and
    latent feature matrices are not stored on this batch.
    """

    X: np.ndarray
    wavelengths: np.ndarray
    latent_ids: tuple[str, ...]
    view_ids: tuple[str, ...]
    view_config: dict[str, Any]
    instrument_key: str
    measurement_mode: str
    preprocessing_state: dict[str, Any]
    noise_state: dict[str, Any]
    metadata: dict[str, Any]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        failures: list[dict[str, str]] = []

        X = np.asarray(self.X)
        if X.ndim != 2:
            failures.append({
                "reason": "shape_mismatch",
                "field": "X",
                "message": f"expected 2D, got shape={X.shape}",
            })
        x_finite_failure = _finite_numeric_failure(
            X, field="X", message="X must be numeric and finite"
        )
        if X.size and x_finite_failure is not None:
            failures.append(x_finite_failure)

        wavelengths = np.asarray(self.wavelengths)
        if wavelengths.ndim != 1:
            failures.append({
                "reason": "shape_mismatch",
                "field": "wavelengths",
                "message": f"expected 1D, got shape={wavelengths.shape}",
            })
        wavelengths_finite_failure = _finite_numeric_failure(
            wavelengths,
            field="wavelengths",
            message="wavelengths must be numeric and finite",
        )
        if wavelengths.size and wavelengths_finite_failure is not None:
            failures.append(wavelengths_finite_failure)
        if (
            wavelengths.ndim == 1
            and wavelengths.size > 1
            and wavelengths_finite_failure is None
            and not bool(np.all(np.diff(wavelengths) > 0))
        ):
            failures.append({
                "reason": "non_monotonic_wavelengths",
                "field": "wavelengths",
                "message": "wavelengths must be strictly increasing",
            })
        if (
            X.ndim == 2
            and wavelengths.ndim == 1
            and X.shape[1] != wavelengths.size
        ):
            failures.append({
                "reason": "shape_mismatch",
                "field": "wavelengths",
                "message": (
                    f"X.shape[1]={X.shape[1]} != wavelengths.size={wavelengths.size}"
                ),
            })

        n_rows = X.shape[0] if X.ndim == 2 else 0
        if X.ndim == 2 and n_rows == 0:
            failures.append({
                "reason": "empty_X",
                "field": "X",
                "message": "X must contain at least one row",
            })
        if wavelengths.ndim == 1 and wavelengths.size == 0:
            failures.append({
                "reason": "empty_wavelengths",
                "field": "wavelengths",
                "message": "wavelengths must contain at least one value",
            })

        if len(self.latent_ids) != n_rows:
            failures.append({
                "reason": "shape_mismatch",
                "field": "latent_ids",
                "message": (
                    f"len(latent_ids)={len(self.latent_ids)} != n_rows={n_rows}"
                ),
            })
        if any(not isinstance(lid, str) or not lid for lid in self.latent_ids):
            failures.append({
                "reason": "invalid_latent_ids",
                "field": "latent_ids",
                "message": "latent_ids must be non-empty strings",
            })

        if len(self.view_ids) != n_rows:
            failures.append({
                "reason": "shape_mismatch",
                "field": "view_ids",
                "message": (
                    f"len(view_ids)={len(self.view_ids)} != n_rows={n_rows}"
                ),
            })
        if any(not isinstance(vid, str) or not vid for vid in self.view_ids):
            failures.append({
                "reason": "invalid_view_ids",
                "field": "view_ids",
                "message": "view_ids must be non-empty strings",
            })
        if len(set(self.view_ids)) != len(self.view_ids):
            failures.append({
                "reason": "duplicate_view_ids",
                "field": "view_ids",
                "message": "view_ids must be unique",
            })

        if not isinstance(self.instrument_key, str) or not self.instrument_key:
            failures.append({
                "reason": "invalid_instrument_key",
                "field": "instrument_key",
                "message": "instrument_key must be a non-empty string",
            })
        if not isinstance(self.measurement_mode, str) or not self.measurement_mode:
            failures.append({
                "reason": "invalid_measurement_mode",
                "field": "measurement_mode",
                "message": "measurement_mode must be a non-empty string",
            })

        for name in _REQUIRED_NON_EMPTY_DICTS:
            value = getattr(self, name)
            if not isinstance(value, dict) or not value:
                failures.append({
                    "reason": "empty_metadata",
                    "field": name,
                    "message": f"{name} must be a non-empty dict",
                })
            elif leakage_paths := _find_leakage_paths(value):
                for path in leakage_paths:
                    failures.append({
                        "reason": "metadata_leakage",
                        "field": name,
                        "message": (
                            f"{name} must not carry target or latent leakage "
                            f"at {path}"
                        ),
                    })

        if isinstance(self.metadata, dict):
            risk_gates = self.metadata.get("risk_gates")
            if not isinstance(risk_gates, dict):
                failures.append({
                    "reason": "missing_risk_gates",
                    "field": "metadata",
                    "message": "metadata.risk_gates must be a dict",
                })
            else:
                for gate, expected in _REQUIRED_RISK_GATES.items():
                    if risk_gates.get(gate) is not expected:
                        failures.append({
                            "reason": "missing_risk_gates",
                            "field": "metadata",
                            "message": (
                                f"metadata.risk_gates[{gate!r}] must equal {expected!r}"
                            ),
                        })

        if failures:
            raise SpectralViewBatchError(failures)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_synthetic_dataset_run(
        cls,
        run: SyntheticDatasetRun,
        latent_batch: CanonicalLatentBatch,
        *,
        view_config: dict[str, Any] | None = None,
        view_id_prefix: str | None = None,
    ) -> SpectralViewBatch:
        """Build a spectral view batch aligned to ``latent_batch``.

        ``run.X`` and ``run.wavelengths`` are reused as-is; no preprocessing
        is applied and no additional noise is introduced. The number of rows
        in ``run.X`` must equal ``len(latent_batch.latent_ids)`` and the
        canonical latent ids are copied into ``latent_ids`` in order.

        ``view_ids`` are derived deterministically from
        ``(view_id_prefix, latent_id, instrument, mode, wavelength summary,
        view_config, row index)`` using ``hashlib.sha256``. The default
        ``view_id_prefix`` is ``f"{builder_config['name']}__view"``.

        ``view_config`` defaults to a single-render description for C2;
        callers may pass a custom non-empty dict to differentiate variants.
        """
        X = np.ascontiguousarray(np.asarray(run.X, dtype=float))
        wavelengths = np.ascontiguousarray(np.asarray(run.wavelengths, dtype=float))
        builder_config = run.builder_config
        features = builder_config["features"]
        instrument_key = str(features["instrument"])
        measurement_mode = str(features["measurement_mode"])

        n_rows = X.shape[0] if X.ndim == 2 else 0
        alignment_failures: list[dict[str, str]] = []
        if n_rows != len(latent_batch.latent_ids):
            alignment_failures.append({
                "reason": "shape_mismatch",
                "field": "latent_ids",
                "message": (
                    f"run.X rows ({n_rows}) do not match "
                    f"latent_batch.latent_ids ({len(latent_batch.latent_ids)})"
                ),
            })
        alignment_failures.extend(
            _latent_batch_run_alignment_failures(latent_batch, builder_config)
        )
        if alignment_failures:
            raise SpectralViewBatchError(alignment_failures)

        wavelength_summary = _wavelength_summary(wavelengths)
        prefix = (
            view_id_prefix
            if view_id_prefix is not None
            else f"{builder_config['name']}__view"
        )

        if view_config is None:
            resolved_view_config: dict[str, Any] = {
                "phase": "C2",
                "view_kind": "single_render",
                "instrument_key": instrument_key,
                "measurement_mode": measurement_mode,
                "wavelength_summary": wavelength_summary,
                "source": "SyntheticDatasetRun",
                "note": (
                    "C2 default view: single rendered spectrum per latent id; "
                    "no multi-view factory or augmentation is applied."
                ),
            }
        else:
            resolved_view_config = dict(view_config)

        wavelength_signature = _stable_signature(wavelength_summary)
        view_config_signature = _stable_signature(resolved_view_config)
        view_ids = tuple(
            _deterministic_view_id(
                prefix=prefix,
                latent_id=latent_batch.latent_ids[i],
                instrument_key=instrument_key,
                measurement_mode=measurement_mode,
                wavelength_signature=wavelength_signature,
                view_config_signature=view_config_signature,
                index=i,
            )
            for i in range(n_rows)
        )

        nuisance = builder_config.get("nuisance", {})
        raw_noise_level = nuisance.get("noise_level") if isinstance(nuisance, dict) else None
        source_noise_level = (
            float(raw_noise_level) if raw_noise_level is not None else None
        )
        noise_state: dict[str, Any] = {
            "phase": "C2",
            "noise_added_in_view": False,
            "source_noise_level": source_noise_level,
            "note": (
                "A2 SyntheticDatasetRun.X already includes generation noise "
                "and nuisance effects (instrument, environment, scattering, "
                "edge artifacts, batch effects). C2 does not add or remove "
                "noise."
            ),
        }

        preprocessing_state: dict[str, Any] = {
            "phase": "C2",
            "preprocessing_applied": False,
            "operations": [],
            "note": (
                "C2 carries spectra as produced by A2; no preprocessing "
                "(SNV, MSC, derivatives, scaling, etc.) is applied here."
            ),
        }

        metadata: dict[str, Any] = {
            "phase": "C2",
            "source_contract": "CanonicalLatentBatch",
            "source_latent_count": len(latent_batch.latent_ids),
            "builder_config_name": str(builder_config["name"]),
            "a2_spectral_validation_summary": _spectral_validation_summary(
                run.validation_summary
            ),
            "claims": {
                "realism": False,
                "transfer": False,
                "note": (
                    "C2 declares structural alignment of spectra to canonical "
                    "latents only. No realism or transfer claim is derived."
                ),
            },
            "risk_gates": {
                "A3_failed_documented": True,
                "B2_realism_failed": True,
            },
        }

        return cls(
            X=X,
            wavelengths=wavelengths,
            latent_ids=tuple(latent_batch.latent_ids),
            view_ids=view_ids,
            view_config=resolved_view_config,
            instrument_key=instrument_key,
            measurement_mode=measurement_mode,
            preprocessing_state=preprocessing_state,
            noise_state=noise_state,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Subset / alignment / serialisation
    # ------------------------------------------------------------------
    def subset(self, indices: Sequence[int] | np.ndarray) -> SpectralViewBatch:
        """Return a new batch restricted to the given row indices."""
        raw = np.asarray(indices)
        if raw.ndim != 1:
            raise SpectralViewBatchError([{
                "reason": "invalid_indices",
                "field": "indices",
                "message": f"indices must be 1D, got shape={raw.shape}",
            }])
        if not np.issubdtype(raw.dtype, np.integer):
            raise SpectralViewBatchError([{
                "reason": "invalid_indices",
                "field": "indices",
                "message": f"indices must be integers, got dtype={raw.dtype}",
            }])
        idx = raw.astype(np.intp, copy=False)
        n = len(self.latent_ids)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n):
            raise SpectralViewBatchError([{
                "reason": "invalid_indices",
                "field": "indices",
                "message": f"indices out of range [0, {n})",
            }])
        idx_list = idx.tolist()
        return replace(
            self,
            X=self.X[idx],
            latent_ids=tuple(self.latent_ids[i] for i in idx_list),
            view_ids=tuple(self.view_ids[i] for i in idx_list),
        )

    def assert_aligned_to(self, latent_batch: CanonicalLatentBatch) -> None:
        """Raise :class:`SpectralViewBatchError` if not aligned to ``latent_batch``."""
        if tuple(self.latent_ids) != tuple(latent_batch.latent_ids):
            raise SpectralViewBatchError([{
                "reason": "alignment_mismatch",
                "field": "latent_ids",
                "message": (
                    "SpectralViewBatch.latent_ids must match "
                    "CanonicalLatentBatch.latent_ids exactly (same order)."
                ),
            }])

    def to_dict(self) -> dict[str, Any]:
        """Return a fully-serialisable view including ``X``."""
        out: dict[str, Any] = {}
        for f in fields(self):
            out[f.name] = _to_builtin(getattr(self, f.name))
        return out

    def to_light_dict(self) -> dict[str, Any]:
        """Return a metadata-only view without the heavy ``X`` array."""
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "X":
                continue
            out[f.name] = _to_builtin(getattr(self, f.name))
        out["n_samples"] = len(self.latent_ids)
        out["n_wavelengths"] = int(np.asarray(self.wavelengths).size)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wavelength_summary(wavelengths: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(wavelengths, dtype=float)
    if arr.size == 0:
        return {
            "n_wavelengths": 0,
            "first_nm": None,
            "last_nm": None,
            "step_nm": None,
        }
    diffs = np.diff(arr)
    if diffs.size and bool(np.allclose(diffs, diffs[0], rtol=1e-9, atol=1e-9)):
        step: float | None = float(diffs[0])
    else:
        step = None
    return {
        "n_wavelengths": int(arr.size),
        "first_nm": float(arr[0]),
        "last_nm": float(arr[-1]),
        "step_nm": step,
    }


def _spectral_validation_summary(validation_summary: dict[str, Any]) -> dict[str, Any]:
    """Keep only A2 validation fields that describe spectra, not targets/latents."""
    failures = [
        dict(failure)
        for failure in validation_summary.get("failures", [])
        if isinstance(failure, dict) and failure.get("field") in {"X", "wavelengths"}
    ]
    checks = validation_summary.get("checks", {})
    summary = validation_summary.get("summary", {})

    out: dict[str, Any] = {
        "status": validation_summary.get("status"),
        "spectral_failures": failures,
        "checks": {},
        "summary": {},
    }
    if isinstance(checks, dict) and "wavelengths_monotonic" in checks:
        out["checks"]["wavelengths_monotonic"] = bool(checks["wavelengths_monotonic"])
    if isinstance(summary, dict):
        for key in ("X_shape", "wavelength_range_nm", "X_min", "X_max"):
            if key in summary:
                out["summary"][key] = summary[key]
    return out


def _latent_batch_run_alignment_failures(
    latent_batch: CanonicalLatentBatch,
    builder_config: dict[str, Any],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    provenance = latent_batch.provenance
    domain_metadata = latent_batch.domain_metadata
    instrument_metadata = latent_batch.instrument_metadata
    features = builder_config["features"]
    domain = builder_config["domain"]

    expected_name = str(builder_config["name"])
    if provenance.get("builder_config_name") != expected_name:
        failures.append(_alignment_failure(
            "latent batch provenance builder_config_name does not match run"
        ))

    raw_seed = provenance.get("random_state")
    try:
        observed_seed = int(raw_seed) if raw_seed is not None else None
    except (TypeError, ValueError):
        observed_seed = None
    if observed_seed != int(builder_config["random_state"]):
        failures.append(_alignment_failure(
            "latent batch provenance random_state does not match run"
        ))

    if str(domain_metadata.get("domain_key")) != str(domain["key"]):
        failures.append(_alignment_failure(
            "latent batch domain_key does not match run"
        ))
    if str(instrument_metadata.get("instrument_key")) != str(features["instrument"]):
        failures.append(_alignment_failure(
            "latent batch instrument_key does not match run"
        ))
    if str(instrument_metadata.get("measurement_mode")) != str(features["measurement_mode"]):
        failures.append(_alignment_failure(
            "latent batch measurement_mode does not match run"
        ))

    run_component_keys = tuple(str(k) for k in features.get("components", ()))
    if run_component_keys and tuple(latent_batch.component_keys) != run_component_keys:
        failures.append(_alignment_failure(
            "latent batch component_keys do not match run"
        ))
    return failures


def _alignment_failure(message: str) -> dict[str, str]:
    return {
        "reason": "alignment_mismatch",
        "field": "latent_ids",
        "message": message,
    }


def _stable_signature(value: Any) -> str:
    payload = json.dumps(_to_builtin(value), sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deterministic_view_id(
    *,
    prefix: str,
    latent_id: str,
    instrument_key: str,
    measurement_mode: str,
    wavelength_signature: str,
    view_config_signature: str,
    index: int,
) -> str:
    payload = "|".join([
        str(prefix),
        str(latent_id),
        str(instrument_key),
        str(measurement_mode),
        str(wavelength_signature),
        str(view_config_signature),
        str(int(index)),
    ])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}__{index:06d}__{digest}"


def _finite_numeric_failure(
    array: np.ndarray,
    *,
    field: str,
    message: str,
) -> dict[str, str] | None:
    try:
        finite = np.isfinite(array)
    except TypeError:
        return {
            "reason": "non_numeric",
            "field": field,
            "message": message,
        }
    if not bool(np.all(finite)):
        return {
            "reason": "non_finite",
            "field": field,
            "message": message,
        }
    return None


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    return cast(Any, value)


def _find_leakage_paths(value: Any, *, path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if _is_leaky_key(key_text):
                paths.append(child_path)
            paths.extend(_find_leakage_paths(child, path=child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_find_leakage_paths(child, path=f"{path}[{index}]"))
    return paths


def _is_leaky_key(key: str) -> bool:
    normalized = key.lower()
    if normalized in _LEAKY_METADATA_KEYS:
        return True
    return any(
        normalized.startswith(f"{prefix}_")
        for prefix in _LEAKY_METADATA_KEYS
    )
