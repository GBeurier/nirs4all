"""Phase C1 canonical latent batch contract.

This module defines :class:`CanonicalLatentBatch`, the bench-level canonical
container for one A2 ``SyntheticDatasetRun`` rendered into a (latent, target)
representation. C2 (``SpectralViewBatch``) and C3 (encoders, multi-view
rendering) are intentionally **out of scope**. The batch only carries the
contract surface required by downstream phases; no realism or transfer claims
are made here.

Risk gates inherited from earlier phases:
- ``A3_failed_documented`` (fitted-only real-fit adapter remains failed).
- ``B2_realism_failed`` (synthetic vs real realism scorecards remain failed).

The contract is independent from those gates: this module only canonicalizes
existing A2 outputs and does not introduce any new realism or transfer claim.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any, cast

import numpy as np

from nirsyntheticpfn.adapters.builder_adapter import SyntheticDatasetRun

__all__ = [
    "CanonicalLatentBatch",
    "CanonicalLatentBatchError",
]

_REQUIRED_NON_EMPTY_METADATA = (
    "domain_metadata",
    "component_metadata",
    "instrument_metadata",
    "target_metadata",
    "split_metadata",
    "view_metadata",
    "optical_metadata",
    "baseline_metadata",
    "scatter_metadata",
    "environment_metadata",
    "sample_presentation_metadata",
    "provenance",
)

_INTENDED_VIEW_CONTRACT = "SpectralViewBatch"


class CanonicalLatentBatchError(ValueError):
    """Raised when a :class:`CanonicalLatentBatch` fails its contract checks."""

    def __init__(self, failures: list[dict[str, str]]) -> None:
        self.failures = failures
        summary = "; ".join(
            f"{failure.get('reason', 'unknown')}:{failure.get('field', '?')}"
            for failure in failures
        )
        super().__init__(summary or "invalid CanonicalLatentBatch")


@dataclass(frozen=True)
class CanonicalLatentBatch:
    """Canonical latent + target view of one A2 synthetic dataset run.

    The batch is a frozen contract object: it stores deterministic ids, the
    component concentrations used to build the run, a minimal set of numeric
    latent features, the (clean, noisy) targets, and structured metadata
    namespaces. Downstream phases consume this object; they do not patch it.

    The class only captures information already present in
    :class:`SyntheticDatasetRun` plus deterministic ids and a few numeric
    latent features derived from existing nuisance fields. It does **not**
    render any spectrum nor evaluate realism.
    """

    latent_ids: tuple[str, ...]
    concentrations: np.ndarray
    component_keys: tuple[str, ...]
    latent_features: np.ndarray
    latent_feature_names: tuple[str, ...]
    target_clean: np.ndarray
    target_noisy: np.ndarray
    batch_ids: tuple[Any, ...]
    group_ids: tuple[Any, ...]
    split_labels: tuple[str, ...] | None
    domain_metadata: dict[str, Any]
    component_metadata: dict[str, Any]
    instrument_metadata: dict[str, Any]
    target_metadata: dict[str, Any]
    split_metadata: dict[str, Any] = field(default_factory=dict)
    view_metadata: dict[str, Any] = field(default_factory=dict)
    optical_metadata: dict[str, Any] = field(default_factory=dict)
    baseline_metadata: dict[str, Any] = field(default_factory=dict)
    scatter_metadata: dict[str, Any] = field(default_factory=dict)
    environment_metadata: dict[str, Any] = field(default_factory=dict)
    sample_presentation_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        failures: list[dict[str, str]] = []

        n_ids = len(self.latent_ids)
        if n_ids == 0:
            failures.append({
                "reason": "empty_latent_ids",
                "field": "latent_ids",
                "message": "latent_ids must not be empty",
            })
        if any(not isinstance(lid, str) or not lid for lid in self.latent_ids):
            failures.append({
                "reason": "invalid_latent_ids",
                "field": "latent_ids",
                "message": "latent_ids must be non-empty strings",
            })
        if len(set(self.latent_ids)) != n_ids:
            failures.append({
                "reason": "duplicate_latent_ids",
                "field": "latent_ids",
                "message": "latent_ids must be unique",
            })

        # concentrations
        concentrations = np.asarray(self.concentrations)
        if concentrations.ndim != 2:
            failures.append({
                "reason": "shape_mismatch",
                "field": "concentrations",
                "message": f"expected 2D, got shape={concentrations.shape}",
            })
        else:
            if concentrations.shape[0] != n_ids:
                failures.append({
                    "reason": "shape_mismatch",
                    "field": "concentrations",
                    "message": (
                        f"first dim {concentrations.shape[0]} != n_ids {n_ids}"
                    ),
                })
            if concentrations.shape[1] != len(self.component_keys):
                failures.append({
                    "reason": "shape_mismatch",
                    "field": "concentrations",
                    "message": (
                        f"second dim {concentrations.shape[1]} != "
                        f"len(component_keys) {len(self.component_keys)}"
                    ),
                })
            concentrations_failure = _finite_numeric_failure(
                concentrations,
                field="concentrations",
                message="concentrations must be numeric and finite",
            )
            if concentrations_failure is not None:
                failures.append(concentrations_failure)
            elif concentrations.size:
                if float(np.min(concentrations)) < -1e-9:
                    failures.append({
                        "reason": "negative_concentrations",
                        "field": "concentrations",
                        "message": "concentrations must be >= 0",
                    })
                row_sums = concentrations.sum(axis=1)
                if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-6):
                    failures.append({
                        "reason": "row_sum_mismatch",
                        "field": "concentrations",
                        "message": (
                            "concentration rows must sum to ~1.0; observed "
                            f"range=({float(np.min(row_sums))}, {float(np.max(row_sums))})"
                        ),
                    })

        if not self.component_keys:
            failures.append({
                "reason": "empty_component_keys",
                "field": "component_keys",
                "message": "component_keys must not be empty",
            })
        if any(not isinstance(key, str) or not key for key in self.component_keys):
            failures.append({
                "reason": "invalid_component_keys",
                "field": "component_keys",
                "message": "component_keys must be non-empty strings",
            })
        if len(set(self.component_keys)) != len(self.component_keys):
            failures.append({
                "reason": "duplicate_component_keys",
                "field": "component_keys",
                "message": "component_keys must be unique",
            })

        # latent_features
        latent_features = np.asarray(self.latent_features)
        if latent_features.ndim != 2:
            failures.append({
                "reason": "shape_mismatch",
                "field": "latent_features",
                "message": f"expected 2D, got shape={latent_features.shape}",
            })
        else:
            if latent_features.shape[0] != n_ids:
                failures.append({
                    "reason": "shape_mismatch",
                    "field": "latent_features",
                    "message": (
                        f"first dim {latent_features.shape[0]} != n_ids {n_ids}"
                    ),
                })
            if latent_features.shape[1] != len(self.latent_feature_names):
                failures.append({
                    "reason": "shape_mismatch",
                    "field": "latent_features",
                    "message": (
                        f"second dim {latent_features.shape[1]} != "
                        f"len(latent_feature_names) {len(self.latent_feature_names)}"
                    ),
                })
            latent_features_failure = _finite_numeric_failure(
                latent_features,
                field="latent_features",
                message="latent_features must be numeric and finite",
            )
            if latent_features.size and latent_features_failure is not None:
                failures.append(latent_features_failure)

        if any(
            not isinstance(name, str) or not name
            for name in self.latent_feature_names
        ):
            failures.append({
                "reason": "invalid_latent_feature_names",
                "field": "latent_feature_names",
                "message": "latent_feature_names must be non-empty strings",
            })
        if len(set(self.latent_feature_names)) != len(self.latent_feature_names):
            failures.append({
                "reason": "duplicate_latent_feature_names",
                "field": "latent_feature_names",
                "message": "latent_feature_names must be unique",
            })

        # targets
        target_clean = np.asarray(self.target_clean)
        target_noisy = np.asarray(self.target_noisy)
        if target_clean.shape != target_noisy.shape:
            failures.append({
                "reason": "shape_mismatch",
                "field": "target_noisy",
                "message": (
                    f"target_clean shape {target_clean.shape} != "
                    f"target_noisy shape {target_noisy.shape}"
                ),
            })
        if target_clean.shape[:1] != (n_ids,):
            failures.append({
                "reason": "shape_mismatch",
                "field": "target_clean",
                "message": (
                    f"first dim {target_clean.shape[:1]} != (n_ids,) ({n_ids},)"
                ),
            })
        target_clean_failure = _finite_numeric_failure(
            target_clean,
            field="target_clean",
            message="target_clean must be numeric and finite",
        )
        if target_clean.size and target_clean_failure is not None:
            failures.append(target_clean_failure)
        target_noisy_failure = _finite_numeric_failure(
            target_noisy,
            field="target_noisy",
            message="target_noisy must be numeric and finite",
        )
        if target_noisy.size and target_noisy_failure is not None:
            failures.append(target_noisy_failure)

        # batch / group / split
        if len(self.batch_ids) != n_ids:
            failures.append({
                "reason": "shape_mismatch",
                "field": "batch_ids",
                "message": f"len(batch_ids) {len(self.batch_ids)} != n_ids {n_ids}",
            })
        if len(self.group_ids) != n_ids:
            failures.append({
                "reason": "shape_mismatch",
                "field": "group_ids",
                "message": f"len(group_ids) {len(self.group_ids)} != n_ids {n_ids}",
            })
        if self.split_labels is not None and len(self.split_labels) != n_ids:
            failures.append({
                "reason": "shape_mismatch",
                "field": "split_labels",
                "message": (
                    f"len(split_labels) {len(self.split_labels)} != n_ids {n_ids}"
                ),
            })

        # required metadata namespaces non-empty
        for name in _REQUIRED_NON_EMPTY_METADATA:
            value = getattr(self, name)
            if not isinstance(value, dict) or not value:
                failures.append({
                    "reason": "empty_metadata",
                    "field": name,
                    "message": f"{name} must be a non-empty dict",
                })

        if failures:
            raise CanonicalLatentBatchError(failures)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_synthetic_dataset_run(
        cls,
        run: SyntheticDatasetRun,
        *,
        latent_id_prefix: str | None = None,
    ) -> CanonicalLatentBatch:
        """Build a canonical latent batch from one A2 ``SyntheticDatasetRun``.

        The conversion is purely structural: arrays already validated by
        ``build_synthetic_dataset_run`` are reused, deterministic ids are
        derived from ``builder_config`` (seed, name, domain, instrument, mode),
        and a minimal set of numeric latent features is assembled from
        existing nuisance fields (temperature, particle size, noise level,
        batch id when available).

        ``target_clean`` and ``target_noisy`` are populated from ``run.y``
        because A2 does not expose a separate noiseless target. This is
        documented under ``target_metadata`` and ``provenance``.
        """
        builder_config = run.builder_config
        latent_metadata = run.latent_metadata
        nuisance = builder_config["nuisance"]
        domain = builder_config["domain"]
        features = builder_config["features"]
        target_cfg = builder_config["target"]

        seed = int(builder_config["random_state"])
        run_name = str(builder_config["name"])
        domain_key = str(domain["key"])
        instrument_key = str(features["instrument"])
        measurement_mode = str(features["measurement_mode"])
        prefix = latent_id_prefix if latent_id_prefix is not None else run_name

        try:
            concentrations = np.asarray(latent_metadata["concentrations"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise CanonicalLatentBatchError([{
                "reason": "non_numeric",
                "field": "concentrations",
                "message": "run.latent_metadata['concentrations'] must be numeric",
            }]) from exc
        n_samples = concentrations.shape[0]
        component_keys = tuple(str(k) for k in latent_metadata["component_keys"])

        latent_ids = tuple(
            _deterministic_latent_id(
                prefix=prefix,
                seed=seed,
                domain_key=domain_key,
                instrument_key=instrument_key,
                measurement_mode=measurement_mode,
                index=i,
            )
            for i in range(n_samples)
        )

        temperatures = np.asarray(
            latent_metadata.get("temperature_c", [nuisance["temperature_c"]] * n_samples),
            dtype=float,
        ).reshape(-1)
        if temperatures.size != n_samples:
            temperatures = np.full(n_samples, float(nuisance["temperature_c"]))

        particle_size = float(nuisance["particle_size_um"])
        noise_level = float(nuisance["noise_level"])

        raw_batch_ids = latent_metadata.get("batch_ids")
        batch_ids_tuple, batch_ids_numeric = _normalize_batch_ids(raw_batch_ids, n_samples)

        latent_feature_columns: list[np.ndarray] = [
            temperatures.astype(float),
            np.full(n_samples, particle_size, dtype=float),
            np.full(n_samples, noise_level, dtype=float),
        ]
        latent_feature_names: list[str] = [
            "temperature_c",
            "particle_size_um",
            "noise_level",
        ]
        if batch_ids_numeric is not None:
            latent_feature_columns.append(batch_ids_numeric.astype(float))
            latent_feature_names.append("batch_id_numeric")

        latent_features = np.column_stack(latent_feature_columns) if latent_feature_columns else np.empty((n_samples, 0))

        y = np.asarray(run.y)
        target_clean = y.copy()
        target_noisy = y.copy()

        target_metadata: dict[str, Any] = {
            "type": target_cfg["type"],
            "mapping": target_cfg.get("mapping"),
            "target_clean_equals_target_noisy": True,
            "source": "SyntheticDatasetRun.y",
            "note": (
                "A2 SyntheticDatasetRun exposes a single target array. "
                "target_clean and target_noisy are populated from run.y; "
                "no separate noiseless target is available."
            ),
        }
        if target_cfg["type"] == "regression":
            target_metadata["range"] = target_cfg.get("range")
            target_metadata["nonlinearity"] = target_cfg.get("nonlinearity")
            target_metadata["component_keys"] = target_cfg.get("component_keys")
        else:
            target_metadata["n_classes"] = target_cfg.get("n_classes")
            target_metadata["separation_key"] = target_cfg.get("separation_key")
            target_metadata["separation_method"] = target_cfg.get("separation_method")

        domain_metadata: dict[str, Any] = {
            "domain_key": domain_key,
            "category": domain.get("category"),
            "product_key": domain.get("product_key"),
            "aggregate_key": domain.get("aggregate_key"),
            "complexity": domain.get("complexity"),
        }

        component_metadata: dict[str, Any] = {
            "component_keys": list(component_keys),
            "spectra_reference_keys": list(features.get("components", component_keys)),
            "spectra_reference_source": "builder_config.features.components",
            "concentration_transform": latent_metadata.get("concentration_transform"),
        }

        instrument_metadata: dict[str, Any] = {
            "instrument_key": instrument_key,
            "measurement_mode": measurement_mode,
            "wavelength_range_nm": list(features["wavelength_range"]),
            "wavelength_step_nm": float(features["wavelength_step"]),
            "n_wavelengths": int(np.asarray(run.wavelengths).size),
        }

        environment_metadata: dict[str, Any] = {
            "temperature_c_min": float(np.min(temperatures)) if temperatures.size else None,
            "temperature_c_max": float(np.max(temperatures)) if temperatures.size else None,
            "temperature_enabled": nuisance.get("environment", {}).get("temperature_enabled"),
            "moisture_enabled": nuisance.get("environment", {}).get("moisture_enabled"),
        }

        scatter_metadata: dict[str, Any] = {
            "particle_size_um": particle_size,
            "particle_size_enabled": nuisance.get("scatter", {}).get("particle_size_enabled"),
            "emsc_enabled": nuisance.get("scatter", {}).get("emsc_enabled"),
        }

        baseline_metadata: dict[str, Any] = {
            "baseline_amplitude": nuisance.get("custom_params", {}).get("baseline_amplitude"),
        }

        optical_metadata: dict[str, Any] = {
            "matrix_type": nuisance.get("matrix_type"),
            "instrument_key": instrument_key,
            "measurement_mode": measurement_mode,
            "path_length_mm": nuisance.get("path_length_mm"),
            "optical_geometry_source": (
                "A2 SyntheticNIRSGenerator instrument and measurement_mode; "
                "no explicit path length is exposed by SyntheticDatasetRun."
            ),
            "noise_level": noise_level,
        }

        sample_presentation_metadata: dict[str, Any] = {
            "edge_artifacts": nuisance.get("edge_artifacts"),
            "batch_effects": nuisance.get("batch_effects"),
        }

        view_metadata: dict[str, Any] = {
            "intended_contract": _INTENDED_VIEW_CONTRACT,
            "rendered_view_count": 0,
            "note": "C1 only declares intent; spectral views are produced by C2.",
        }

        split_metadata: dict[str, Any] = {
            "partition": builder_config.get("partition"),
        }

        provenance: dict[str, Any] = {
            "phase": "C1",
            "source": "SyntheticDatasetRun",
            "builder_config_name": run_name,
            "random_state": seed,
            "seed": seed,
            "a1_provenance": run.metadata.get("provenance_a1"),
            "a2_validation_summary": run.validation_summary,
            "target_clean_source": "run.y",
            "target_noisy_source": "run.y",
            "risk_gates": {
                "A3_failed_documented": True,
                "B2_realism_failed": True,
            },
            "claims": {
                "realism": False,
                "transfer": False,
                "note": (
                    "C1 declares structural canonicalization only. No realism "
                    "or transfer claim is derived from this batch."
                ),
            },
        }

        return cls(
            latent_ids=latent_ids,
            concentrations=np.ascontiguousarray(concentrations, dtype=float),
            component_keys=component_keys,
            latent_features=np.ascontiguousarray(latent_features, dtype=float),
            latent_feature_names=tuple(latent_feature_names),
            target_clean=np.ascontiguousarray(target_clean),
            target_noisy=np.ascontiguousarray(target_noisy),
            batch_ids=batch_ids_tuple,
            group_ids=batch_ids_tuple,
            split_labels=None,
            domain_metadata=domain_metadata,
            component_metadata=component_metadata,
            instrument_metadata=instrument_metadata,
            target_metadata=target_metadata,
            split_metadata=split_metadata,
            view_metadata=view_metadata,
            optical_metadata=optical_metadata,
            baseline_metadata=baseline_metadata,
            scatter_metadata=scatter_metadata,
            environment_metadata=environment_metadata,
            sample_presentation_metadata=sample_presentation_metadata,
            provenance=provenance,
        )

    # ------------------------------------------------------------------
    # Subset / serialisation
    # ------------------------------------------------------------------
    def subset(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        split_label: str | None = None,
    ) -> CanonicalLatentBatch:
        """Return a new batch restricted to the given row indices.

        ``split_label`` populates ``split_labels`` for the resulting batch
        (one label per kept row) when provided; otherwise the existing
        ``split_labels`` are sliced.
        """
        raw_idx = np.asarray(indices)
        if raw_idx.ndim != 1:
            raise CanonicalLatentBatchError([{
                "reason": "invalid_indices",
                "field": "indices",
                "message": f"indices must be 1D, got shape={raw_idx.shape}",
            }])
        if not np.issubdtype(raw_idx.dtype, np.integer):
            raise CanonicalLatentBatchError([{
                "reason": "invalid_indices",
                "field": "indices",
                "message": f"indices must be integers, got dtype={raw_idx.dtype}",
            }])
        idx = raw_idx.astype(np.intp, copy=False)
        n = len(self.latent_ids)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n):
            raise CanonicalLatentBatchError([{
                "reason": "invalid_indices",
                "field": "indices",
                "message": f"indices out of range [0, {n})",
            }])

        new_ids = tuple(self.latent_ids[i] for i in idx.tolist())
        new_batch_ids = tuple(self.batch_ids[i] for i in idx.tolist())
        new_group_ids = tuple(self.group_ids[i] for i in idx.tolist())

        if split_label is not None:
            new_split_labels: tuple[str, ...] | None = tuple(
                str(split_label) for _ in range(idx.size)
            )
        elif self.split_labels is not None:
            new_split_labels = tuple(self.split_labels[i] for i in idx.tolist())
        else:
            new_split_labels = None

        return replace(
            self,
            latent_ids=new_ids,
            concentrations=self.concentrations[idx],
            latent_features=self.latent_features[idx],
            target_clean=self.target_clean[idx],
            target_noisy=self.target_noisy[idx],
            batch_ids=new_batch_ids,
            group_ids=new_group_ids,
            split_labels=new_split_labels,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a fully-serialisable view including arrays."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = _to_builtin(value)
        return out

    def to_light_dict(self) -> dict[str, Any]:
        """Return a metadata-only view without the heavy numerical arrays."""
        heavy = {
            "concentrations",
            "latent_features",
            "target_clean",
            "target_noisy",
        }
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name in heavy:
                continue
            out[f.name] = _to_builtin(getattr(self, f.name))
        out["n_samples"] = len(self.latent_ids)
        out["n_components"] = len(self.component_keys)
        out["n_latent_features"] = len(self.latent_feature_names)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deterministic_latent_id(
    *,
    prefix: str,
    seed: int,
    domain_key: str,
    instrument_key: str,
    measurement_mode: str,
    index: int,
) -> str:
    payload = "|".join(
        [
            str(prefix),
            str(seed),
            str(domain_key),
            str(instrument_key),
            str(measurement_mode),
            str(int(index)),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}__{index:06d}__{digest}"


def _normalize_batch_ids(
    raw: Any,
    n_samples: int,
) -> tuple[tuple[Any, ...], np.ndarray | None]:
    if raw is None:
        return tuple(0 for _ in range(n_samples)), None

    if isinstance(raw, np.ndarray):
        sequence: Iterable[Any] = raw.tolist()
    elif isinstance(raw, (list, tuple)):
        sequence = list(raw)
    else:
        sequence = [raw] * n_samples

    sequence_list = list(sequence)

    numeric_array: np.ndarray | None
    if len(sequence_list) != n_samples:
        numeric_array = None
    else:
        try:
            numeric_array = np.asarray([float(x) for x in sequence_list], dtype=float)
            if not np.isfinite(numeric_array).all():
                numeric_array = None
        except (TypeError, ValueError):
            numeric_array = None

    return tuple(sequence_list), numeric_array


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
