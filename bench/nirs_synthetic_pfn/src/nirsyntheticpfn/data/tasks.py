"""Phase D3 PFN-style prior task contract.

This module defines :class:`NIRSPriorTask`, the bench-side container that
turns one :class:`~nirsyntheticpfn.data.latents.CanonicalLatentBatch` and
one aligned :class:`~nirsyntheticpfn.data.views.SpectralViewBatch` into a
PFN-style ``(context, query)`` task with explicit, per-split spectra,
labels, latent ids, view ids, metadata, and provenance.

D3 widens the D1 / D2 contract to multi-output tasks while keeping every
other invariant unchanged:

- Single-output regression and classification keep their D1 behaviour:
  ``y_context`` / ``y_query`` are 1D ``float`` arrays, ``n_outputs=1``.
- Multi-output regression and multi-output classification are now
  accepted: ``y_context`` / ``y_query`` are 2D ``(n_rows, n_outputs)``
  arrays. Single-column 2D inputs are normalised back to 1D so the
  single-output convention stays unique.
- Classification (single-output and multi-output) labels must be
  integer-like and finite for every output column.
- A3 / B2 risk gates remain explicitly negative on every task; no
  realism or transfer claim is introduced by D3.

The contract still guarantees a clean separation between spectra /
features (``X_*``) and labels / targets (``y_*``): per-split metadata
namespaces (``metadata_context`` / ``metadata_query``) must not carry
any target or latent leakage at any nested path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any, cast

import numpy as np

from nirsyntheticpfn.data.latents import CanonicalLatentBatch
from nirsyntheticpfn.data.views import SpectralViewBatch

__all__ = [
    "NIRSPriorTask",
    "NIRSPriorTaskError",
]

_PER_SPLIT_LEAKY_KEYS: tuple[str, ...] = (
    "y",
    "target",
    "targets",
    "concentration",
    "concentrations",
    "target_clean",
    "target_noisy",
    "latent_feature",
    "latent_features",
)
"""Forbidden keys (and ``<key>_*`` prefixed forms) inside per-split metadata.

Per-split namespaces (``metadata_context`` / ``metadata_query``) must not
expose any target/latent values of the rows they describe. Configuration
namespaces such as ``prior_config`` legitimately use prefixed forms like
``target_prior`` / ``concentration_prior`` and are checked separately
against :data:`_PER_SAMPLE_LEAKY_KEYS` only.
"""

_PER_SAMPLE_LEAKY_KEYS: tuple[str, ...] = (
    "y",
    "target_clean",
    "target_noisy",
    "concentrations",
    "latent_features",
)
"""Per-sample leaky keys (exact match, any depth).

Used for ``prior_config`` and ``provenance`` where structural prior
descriptions (``target_prior``, ``concentration_prior``) are legitimate
but the actual per-sample arrays must not be embedded.
"""

_REQUIRED_RISK_GATES: dict[str, bool] = {
    "A3_failed_documented": True,
    "B2_realism_failed": True,
}

_REQUIRED_NON_EMPTY_DICTS: tuple[str, ...] = (
    "metadata_context",
    "metadata_query",
    "target_semantics",
    "latent_params",
    "prior_config",
    "split_policy",
    "provenance",
)


class NIRSPriorTaskError(ValueError):
    """Raised when a :class:`NIRSPriorTask` fails its contract checks."""

    def __init__(self, failures: list[dict[str, str]]) -> None:
        self.failures = failures
        summary = "; ".join(
            f"{failure.get('reason', 'unknown')}:{failure.get('field', '?')}"
            for failure in failures
        )
        super().__init__(summary or "invalid NIRSPriorTask")


@dataclass(frozen=True)
class NIRSPriorTask:
    """PFN-style prior task built from a canonical latent / spectral view pair.

    The task is a frozen contract object. It carries one ``(context,
    query)`` split derived from a single ``CanonicalLatentBatch`` /
    ``SpectralViewBatch`` pair: spectra, single-output labels, deterministic
    ids, optical configuration, structured metadata namespaces, and
    provenance with the inherited A3 / B2 risk gates.
    """

    task_id: str
    X_context: np.ndarray
    y_context: np.ndarray
    X_query: np.ndarray
    y_query: np.ndarray
    wavelengths_context: np.ndarray
    wavelengths_query: np.ndarray
    context_latent_ids: tuple[str, ...]
    query_latent_ids: tuple[str, ...]
    context_view_ids: tuple[str, ...]
    query_view_ids: tuple[str, ...]
    metadata_context: dict[str, Any]
    metadata_query: dict[str, Any]
    domain_key: str
    instrument_context: str
    instrument_query: str
    measurement_mode: str
    target_name: str
    target_type: str
    target_semantics: dict[str, Any]
    latent_params: dict[str, Any]
    prior_config: dict[str, Any]
    split_policy: dict[str, Any]
    task_seed: int
    provenance: dict[str, Any]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        failures: list[dict[str, str]] = []

        for field_name in ("y_context", "y_query"):
            value = np.asarray(getattr(self, field_name))
            if value.ndim == 2 and value.shape[1] == 1:
                object.__setattr__(
                    self,
                    field_name,
                    np.ascontiguousarray(value.reshape(-1), dtype=value.dtype),
                )

        if not isinstance(self.task_id, str) or not self.task_id:
            failures.append(
                _failure("invalid_task_id", "task_id", "task_id must be a non-empty string")
            )

        n_ctx = self._validate_split(
            failures,
            X=self.X_context,
            y=self.y_context,
            wavelengths=self.wavelengths_context,
            latent_ids=self.context_latent_ids,
            view_ids=self.context_view_ids,
            split="context",
        )
        n_query = self._validate_split(
            failures,
            X=self.X_query,
            y=self.y_query,
            wavelengths=self.wavelengths_query,
            latent_ids=self.query_latent_ids,
            view_ids=self.query_view_ids,
            split="query",
        )

        if n_ctx == 0:
            failures.append(
                _failure("empty_split", "X_context", "context split must not be empty")
            )
        if n_query == 0:
            failures.append(
                _failure("empty_split", "X_query", "query split must not be empty")
            )

        # Cross-split y consistency: same dimensionality, and matching
        # number of outputs when 2D.
        y_ctx_arr = np.asarray(self.y_context)
        y_q_arr = np.asarray(self.y_query)
        if y_ctx_arr.ndim != y_q_arr.ndim:
            failures.append(
                _failure(
                    "shape_mismatch",
                    "y_query",
                    f"y_context.ndim={y_ctx_arr.ndim} != y_query.ndim={y_q_arr.ndim}",
                )
            )
        elif y_ctx_arr.ndim == 2 and y_ctx_arr.shape[1] != y_q_arr.shape[1]:
            failures.append(
                _failure(
                    "shape_mismatch",
                    "y_query",
                    (
                        f"y_context.shape[1]={y_ctx_arr.shape[1]} != "
                        f"y_query.shape[1]={y_q_arr.shape[1]}"
                    ),
                )
            )

        # Latent / view id disjointness (no leakage of the same physical
        # row between context and query).
        if self.context_latent_ids and self.query_latent_ids:
            overlap_latent = set(self.context_latent_ids) & set(self.query_latent_ids)
            if overlap_latent:
                failures.append(
                    _failure(
                        "overlapping_split_ids",
                        "context_latent_ids",
                        f"context and query share latent_ids: {sorted(overlap_latent)}",
                    )
                )
        if self.context_view_ids and self.query_view_ids:
            overlap_view = set(self.context_view_ids) & set(self.query_view_ids)
            if overlap_view:
                failures.append(
                    _failure(
                        "overlapping_split_ids",
                        "context_view_ids",
                        f"context and query share view_ids: {sorted(overlap_view)}",
                    )
                )
        if self.context_latent_ids and len(set(self.context_latent_ids)) != len(
            self.context_latent_ids
        ):
            failures.append(
                _failure(
                    "duplicate_split_ids",
                    "context_latent_ids",
                    "context_latent_ids must be unique within the split",
                )
            )
        if self.query_latent_ids and len(set(self.query_latent_ids)) != len(
            self.query_latent_ids
        ):
            failures.append(
                _failure(
                    "duplicate_split_ids",
                    "query_latent_ids",
                    "query_latent_ids must be unique within the split",
                )
            )
        if self.context_view_ids and len(set(self.context_view_ids)) != len(
            self.context_view_ids
        ):
            failures.append(
                _failure(
                    "duplicate_split_ids",
                    "context_view_ids",
                    "context_view_ids must be unique within the split",
                )
            )
        if self.query_view_ids and len(set(self.query_view_ids)) != len(
            self.query_view_ids
        ):
            failures.append(
                _failure(
                    "duplicate_split_ids",
                    "query_view_ids",
                    "query_view_ids must be unique within the split",
                )
            )

        if not isinstance(self.domain_key, str) or not self.domain_key:
            failures.append(
                _failure("invalid_domain_key", "domain_key", "domain_key must be a non-empty string")
            )
        if not isinstance(self.instrument_context, str) or not self.instrument_context:
            failures.append(
                _failure(
                    "invalid_instrument_key",
                    "instrument_context",
                    "instrument_context must be a non-empty string",
                )
            )
        if not isinstance(self.instrument_query, str) or not self.instrument_query:
            failures.append(
                _failure(
                    "invalid_instrument_key",
                    "instrument_query",
                    "instrument_query must be a non-empty string",
                )
            )
        if not isinstance(self.measurement_mode, str) or not self.measurement_mode:
            failures.append(
                _failure(
                    "invalid_measurement_mode",
                    "measurement_mode",
                    "measurement_mode must be a non-empty string",
                )
            )
        if not isinstance(self.target_name, str) or not self.target_name:
            failures.append(
                _failure("invalid_target_name", "target_name", "target_name must be a non-empty string")
            )
        if self.target_type not in {"regression", "classification"}:
            failures.append(
                _failure(
                    "invalid_target_type",
                    "target_type",
                    f"target_type must be 'regression' or 'classification', got {self.target_type!r}",
                )
            )
        elif self.target_type == "classification":
            for field, labels in (
                ("y_context", np.asarray(self.y_context)),
                ("y_query", np.asarray(self.y_query)),
            ):
                if labels.size and not _is_integer_like(labels):
                    failures.append(
                        _failure(
                            "invalid_classification_labels",
                            field,
                            (
                                f"{field} must contain integer-like class labels "
                                "for every output column"
                            ),
                        )
                    )

        if not isinstance(self.task_seed, int) or isinstance(self.task_seed, bool):
            failures.append(
                _failure("invalid_task_seed", "task_seed", "task_seed must be an int")
            )

        # Required non-empty dicts.
        for name in _REQUIRED_NON_EMPTY_DICTS:
            value = getattr(self, name)
            if not isinstance(value, dict) or not value:
                failures.append(
                    _failure("empty_metadata", name, f"{name} must be a non-empty dict")
                )

        # Per-split metadata: strict leakage check (no targets/latents).
        for name in ("metadata_context", "metadata_query"):
            value = getattr(self, name)
            if isinstance(value, dict):
                for path in _find_leakage_paths(value, leaky_keys=_PER_SPLIT_LEAKY_KEYS):
                    failures.append(
                        _failure(
                            "metadata_leakage",
                            name,
                            f"{name} must not carry target or latent leakage at {path}",
                        )
                    )

        # prior_config / provenance: forbid only per-sample exact keys.
        for name in ("prior_config", "provenance"):
            value = getattr(self, name)
            if isinstance(value, dict):
                for path in _find_leakage_paths(
                    value,
                    leaky_keys=_PER_SAMPLE_LEAKY_KEYS,
                    prefix_match=False,
                ):
                    failures.append(
                        _failure(
                            "metadata_leakage",
                            name,
                            f"{name} must not embed per-sample target/latent values at {path}",
                        )
                    )

        # Required risk gates inside provenance.
        if isinstance(self.provenance, dict):
            risk_gates = self.provenance.get("risk_gates")
            if not isinstance(risk_gates, dict):
                failures.append(
                    _failure(
                        "missing_risk_gates",
                        "provenance",
                        "provenance.risk_gates must be a dict",
                    )
                )
            else:
                for gate, expected in _REQUIRED_RISK_GATES.items():
                    if risk_gates.get(gate) is not expected:
                        failures.append(
                            _failure(
                                "missing_risk_gates",
                                "provenance",
                                f"provenance.risk_gates[{gate!r}] must equal {expected!r}",
                            )
                        )

        # target_semantics required keys.
        if isinstance(self.target_semantics, dict) and self.target_semantics:
            for required_key in (
                "target_source",
                "target_name",
                "target_type",
                "target_clean_equals_target_noisy",
                "n_outputs",
                "output_names",
                "multi_output_supported",
            ):
                if required_key not in self.target_semantics:
                    failures.append(
                        _failure(
                            "incomplete_target_semantics",
                            "target_semantics",
                            f"target_semantics must include {required_key!r}",
                        )
                    )
            source = self.target_semantics.get("target_source")
            if source not in {"target_clean", "target_noisy"}:
                failures.append(
                    _failure(
                        "invalid_target_source",
                        "target_semantics",
                        f"target_semantics.target_source must be 'target_clean' or 'target_noisy', got {source!r}",
                    )
                )
            if self.target_semantics.get("target_name") != self.target_name:
                failures.append(
                    _failure(
                        "inconsistent_target_semantics",
                        "target_semantics",
                        "target_semantics.target_name must match target_name",
                    )
                )
            if self.target_semantics.get("target_type") != self.target_type:
                failures.append(
                    _failure(
                        "inconsistent_target_semantics",
                        "target_semantics",
                        "target_semantics.target_type must match target_type",
                    )
                )
            clean_equals_noisy = self.target_semantics.get(
                "target_clean_equals_target_noisy"
            )
            if not isinstance(clean_equals_noisy, bool):
                failures.append(
                    _failure(
                        "incomplete_target_semantics",
                        "target_semantics",
                        "target_semantics.target_clean_equals_target_noisy must be a bool",
                    )
                )
            n_outputs_value = self.target_semantics.get("n_outputs")
            output_names_value = self.target_semantics.get("output_names")
            multi_supported = self.target_semantics.get("multi_output_supported")
            if not isinstance(n_outputs_value, int) or isinstance(n_outputs_value, bool) or n_outputs_value < 1:
                failures.append(
                    _failure(
                        "incomplete_target_semantics",
                        "target_semantics",
                        "target_semantics.n_outputs must be a positive int",
                    )
                )
            if not isinstance(output_names_value, list) or not all(
                isinstance(name, str) and name for name in output_names_value
            ):
                failures.append(
                    _failure(
                        "incomplete_target_semantics",
                        "target_semantics",
                        "target_semantics.output_names must be a non-empty list of non-empty strings",
                    )
                )
            elif (
                isinstance(n_outputs_value, int)
                and not isinstance(n_outputs_value, bool)
                and len(output_names_value) != n_outputs_value
            ):
                failures.append(
                    _failure(
                        "incomplete_target_semantics",
                        "target_semantics",
                        "target_semantics.output_names length must match n_outputs",
                    )
                )
            if multi_supported is not True:
                failures.append(
                    _failure(
                        "incomplete_target_semantics",
                        "target_semantics",
                        "target_semantics.multi_output_supported must be True (D3 contract)",
                    )
                )
            # Cross-check n_outputs against actual y arrays.
            actual_outputs_ctx = (
                int(y_ctx_arr.shape[1]) if y_ctx_arr.ndim == 2 else 1 if y_ctx_arr.ndim == 1 else 0
            )
            actual_outputs_q = (
                int(y_q_arr.shape[1]) if y_q_arr.ndim == 2 else 1 if y_q_arr.ndim == 1 else 0
            )
            if (
                isinstance(n_outputs_value, int)
                and not isinstance(n_outputs_value, bool)
                and actual_outputs_ctx
                and actual_outputs_ctx != n_outputs_value
            ):
                failures.append(
                    _failure(
                        "inconsistent_target_semantics",
                        "target_semantics",
                        (
                            f"target_semantics.n_outputs={n_outputs_value} does not match "
                            f"y_context output count={actual_outputs_ctx}"
                        ),
                    )
                )
            if (
                isinstance(n_outputs_value, int)
                and not isinstance(n_outputs_value, bool)
                and actual_outputs_q
                and actual_outputs_q != n_outputs_value
            ):
                failures.append(
                    _failure(
                        "inconsistent_target_semantics",
                        "target_semantics",
                        (
                            f"target_semantics.n_outputs={n_outputs_value} does not match "
                            f"y_query output count={actual_outputs_q}"
                        ),
                    )
                )

        if failures:
            raise NIRSPriorTaskError(failures)

    @staticmethod
    def _validate_split(
        failures: list[dict[str, str]],
        *,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: np.ndarray,
        latent_ids: tuple[str, ...],
        view_ids: tuple[str, ...],
        split: str,
    ) -> int:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        wl_arr = np.asarray(wavelengths)

        x_field = f"X_{split}"
        y_field = f"y_{split}"
        wl_field = f"wavelengths_{split}"
        lid_field = f"{split}_latent_ids"
        vid_field = f"{split}_view_ids"

        if X_arr.ndim != 2:
            failures.append(
                _failure(
                    "shape_mismatch",
                    x_field,
                    f"expected 2D, got shape={X_arr.shape}",
                )
            )
        x_finite = _finite_numeric_failure(
            X_arr, field=x_field, message=f"{x_field} must be numeric and finite"
        )
        if X_arr.size and x_finite is not None:
            failures.append(x_finite)

        if y_arr.ndim not in (1, 2):
            failures.append(
                _failure(
                    "shape_mismatch",
                    y_field,
                    (
                        f"expected 1D (single-output) or 2D (multi-output) array, "
                        f"got shape={y_arr.shape}"
                    ),
                )
            )
        elif y_arr.ndim == 2 and y_arr.shape[1] < 1:
            failures.append(
                _failure(
                    "shape_mismatch",
                    y_field,
                    f"{y_field} 2D array must have at least one output column, got shape={y_arr.shape}",
                )
            )
        y_finite = _finite_numeric_failure(
            y_arr, field=y_field, message=f"{y_field} must be numeric and finite"
        )
        if y_arr.size and y_finite is not None:
            failures.append(y_finite)

        if wl_arr.ndim != 1:
            failures.append(
                _failure(
                    "shape_mismatch",
                    wl_field,
                    f"expected 1D, got shape={wl_arr.shape}",
                )
            )
        wl_finite = _finite_numeric_failure(
            wl_arr,
            field=wl_field,
            message=f"{wl_field} must be numeric and finite",
        )
        if wl_arr.size and wl_finite is not None:
            failures.append(wl_finite)
        if (
            wl_arr.ndim == 1
            and wl_arr.size > 1
            and wl_finite is None
            and not bool(np.all(np.diff(wl_arr) > 0))
        ):
            failures.append(
                _failure(
                    "non_monotonic_wavelengths",
                    wl_field,
                    f"{wl_field} must be strictly increasing",
                )
            )

        n_rows = X_arr.shape[0] if X_arr.ndim == 2 else 0
        if X_arr.ndim == 2 and wl_arr.ndim == 1 and X_arr.shape[1] != wl_arr.size:
            failures.append(
                _failure(
                    "shape_mismatch",
                    wl_field,
                    f"{x_field}.shape[1]={X_arr.shape[1]} != {wl_field}.size={wl_arr.size}",
                )
            )
        if y_arr.ndim in (1, 2) and y_arr.shape[0] != n_rows:
            failures.append(
                _failure(
                    "shape_mismatch",
                    y_field,
                    f"{y_field}.shape[0]={y_arr.shape[0]} != n_rows={n_rows}",
                )
            )
        if len(latent_ids) != n_rows:
            failures.append(
                _failure(
                    "shape_mismatch",
                    lid_field,
                    f"len({lid_field})={len(latent_ids)} != n_rows={n_rows}",
                )
            )
        if any(not isinstance(lid, str) or not lid for lid in latent_ids):
            failures.append(
                _failure(
                    "invalid_latent_ids",
                    lid_field,
                    f"{lid_field} must be non-empty strings",
                )
            )
        if len(view_ids) != n_rows:
            failures.append(
                _failure(
                    "shape_mismatch",
                    vid_field,
                    f"len({vid_field})={len(view_ids)} != n_rows={n_rows}",
                )
            )
        if any(not isinstance(vid, str) or not vid for vid in view_ids):
            failures.append(
                _failure(
                    "invalid_view_ids",
                    vid_field,
                    f"{vid_field} must be non-empty strings",
                )
            )

        return n_rows

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_batches(
        cls,
        latent_batch: CanonicalLatentBatch,
        spectral_view: SpectralViewBatch,
        context_indices: Sequence[int] | np.ndarray,
        query_indices: Sequence[int] | np.ndarray,
        *,
        target_source: str = "target_noisy",
        target_name: str | None = None,
        split_policy: dict[str, Any] | None = None,
        task_seed: int | None = None,
    ) -> NIRSPriorTask:
        """Build a :class:`NIRSPriorTask` from one ``(latents, view)`` pair.

        ``spectral_view.assert_aligned_to(latent_batch)`` is invoked before
        any work to guarantee row alignment between the two batches.
        ``context_indices`` and ``query_indices`` are integer indices into
        the shared row order; they must be 1D, non-empty, contain only
        unique in-range integers, and be disjoint from each other.

        ``target_source`` selects whether labels come from
        ``latent_batch.target_clean`` or ``latent_batch.target_noisy``.
        Single-output targets are stored as 1D arrays; multi-output
        targets (``y.shape[1] > 1``) are stored as 2D arrays. A 2D
        single-column target is normalised back to 1D so the
        single-output convention stays unique.
        """
        if not isinstance(spectral_view, SpectralViewBatch):
            raise NIRSPriorTaskError(
                [_failure("invalid_input", "spectral_view", "spectral_view must be a SpectralViewBatch")]
            )
        if not isinstance(latent_batch, CanonicalLatentBatch):
            raise NIRSPriorTaskError(
                [_failure("invalid_input", "latent_batch", "latent_batch must be a CanonicalLatentBatch")]
            )
        if target_source not in {"target_clean", "target_noisy"}:
            raise NIRSPriorTaskError(
                [
                    _failure(
                        "invalid_target_source",
                        "target_source",
                        f"target_source must be 'target_clean' or 'target_noisy', got {target_source!r}",
                    )
                ]
            )

        # Cross-batch alignment (raises SpectralViewBatchError on mismatch).
        spectral_view.assert_aligned_to(latent_batch)

        n_total = len(latent_batch.latent_ids)

        ctx_idx = _coerce_indices(context_indices, n_total=n_total, field="context_indices")
        query_idx = _coerce_indices(query_indices, n_total=n_total, field="query_indices")

        overlap = set(ctx_idx.tolist()) & set(query_idx.tolist())
        if overlap:
            raise NIRSPriorTaskError(
                [
                    _failure(
                        "overlapping_indices",
                        "context_indices",
                        f"context_indices and query_indices must be disjoint; overlap={sorted(overlap)}",
                    )
                ]
            )

        # Target source selection. D3 accepts single-output (1D) and
        # multi-output (2D) targets. A 2D single-column array is
        # normalised back to 1D so the single-output convention stays
        # unique downstream.
        target_array = (
            latent_batch.target_clean
            if target_source == "target_clean"
            else latent_batch.target_noisy
        )
        target_array = np.asarray(target_array, dtype=float)
        if target_array.ndim not in (1, 2):
            raise NIRSPriorTaskError(
                [
                    _failure(
                        "shape_mismatch",
                        "target",
                        (
                            f"target array must be 1D or 2D, got shape={target_array.shape}"
                        ),
                    )
                ]
            )
        if target_array.ndim == 2 and target_array.shape[1] == 1:
            target_array = target_array.reshape(-1)
        if target_array.shape[0] != n_total:
            raise NIRSPriorTaskError(
                [
                    _failure(
                        "shape_mismatch",
                        "target",
                        f"target.shape[0]={target_array.shape[0]} != n_total {n_total}",
                    )
                ]
            )

        n_outputs = 1 if target_array.ndim == 1 else int(target_array.shape[1])

        # Subset arrays.
        X_ctx = np.ascontiguousarray(spectral_view.X[ctx_idx], dtype=float)
        X_q = np.ascontiguousarray(spectral_view.X[query_idx], dtype=float)
        y_ctx = np.ascontiguousarray(target_array[ctx_idx], dtype=float)
        y_q = np.ascontiguousarray(target_array[query_idx], dtype=float)
        wavelengths = np.ascontiguousarray(spectral_view.wavelengths, dtype=float)

        ctx_latent_ids = tuple(latent_batch.latent_ids[i] for i in ctx_idx.tolist())
        query_latent_ids = tuple(latent_batch.latent_ids[i] for i in query_idx.tolist())
        ctx_view_ids = tuple(spectral_view.view_ids[i] for i in ctx_idx.tolist())
        query_view_ids = tuple(spectral_view.view_ids[i] for i in query_idx.tolist())

        # Resolve target metadata.
        c1_target_metadata = dict(latent_batch.target_metadata)
        target_type = str(c1_target_metadata.get("type", "regression"))
        if target_type == "classification" and not _is_integer_like(target_array):
            raise NIRSPriorTaskError(
                [
                    _failure(
                        "invalid_classification_labels",
                        "target",
                        (
                            "classification targets must contain integer-like "
                            "class labels for every output column"
                        ),
                    )
                ]
            )
        target_clean_equals_target_noisy = bool(
            c1_target_metadata.get("target_clean_equals_target_noisy", False)
        )

        target_component_keys = list(c1_target_metadata.get("component_keys") or [])
        metadata_output_names = c1_target_metadata.get("output_names")
        if target_name is None:
            if n_outputs == 1:
                if target_type == "regression" and target_component_keys:
                    resolved_target_name = f"target__{target_component_keys[0]}"
                elif target_type == "classification":
                    resolved_target_name = "class_label"
                else:
                    resolved_target_name = "target"
            else:
                resolved_target_name = (
                    "multi_label" if target_type == "classification" else "multi_target"
                )
        else:
            resolved_target_name = str(target_name)

        # Resolve output_names: prefer explicit metadata, then component_keys
        # for multi-output regression when sizes match, else fallback to a
        # canonical numbered scheme.
        if (
            isinstance(metadata_output_names, (list, tuple))
            and len(metadata_output_names) == n_outputs
            and all(isinstance(name, str) and name for name in metadata_output_names)
        ):
            output_names = [str(name) for name in metadata_output_names]
        elif n_outputs == 1:
            output_names = [resolved_target_name]
        elif (
            target_type == "regression"
            and target_component_keys
            and len(target_component_keys) == n_outputs
        ):
            output_names = [f"target__{key}" for key in target_component_keys]
        else:
            output_names = [f"{resolved_target_name}_{i}" for i in range(n_outputs)]

        if n_outputs == 1:
            target_semantics_note = (
                "D3 single-output task. target_clean and target_noisy share "
                "the same array because A2 SyntheticDatasetRun does not expose "
                "a separate noiseless target. The contract supports multi-output "
                "but this task is single-output."
            )
        else:
            target_semantics_note = (
                f"D3 multi-output task with n_outputs={n_outputs}. target_clean "
                "and target_noisy share the same array because A2 "
                "SyntheticDatasetRun does not expose a separate noiseless "
                "target. y_context and y_query are stored as 2D "
                "(n_rows, n_outputs) arrays."
            )

        target_semantics: dict[str, Any] = {
            "target_source": target_source,
            "target_name": resolved_target_name,
            "target_type": target_type,
            "target_clean_equals_target_noisy": target_clean_equals_target_noisy,
            "n_outputs": int(n_outputs),
            "output_names": list(output_names),
            "multi_output_supported": True,
            "note": target_semantics_note,
        }
        if target_type == "regression":
            target_semantics["target_range"] = c1_target_metadata.get("range")
            target_semantics["target_nonlinearity"] = c1_target_metadata.get("nonlinearity")
            target_semantics["component_keys"] = list(target_component_keys)
        else:
            target_semantics["n_classes"] = c1_target_metadata.get("n_classes")
            target_semantics["separation_key"] = c1_target_metadata.get("separation_key")
            target_semantics["separation_method"] = c1_target_metadata.get("separation_method")

        # Optical configuration.
        domain_key = str(latent_batch.domain_metadata.get("domain_key", ""))
        instrument_key = str(latent_batch.instrument_metadata.get("instrument_key", ""))
        measurement_mode = str(latent_batch.instrument_metadata.get("measurement_mode", ""))

        # Prior config + latent params (no per-sample arrays).
        a1_provenance = latent_batch.provenance.get("a1_provenance")
        if isinstance(a1_provenance, dict):
            prior_config = dict(a1_provenance.get("source_prior_config") or {})
        else:
            prior_config = {}
        if not prior_config:
            prior_config = {
                "phase": "D1",
                "source": "CanonicalLatentBatch.provenance.a1_provenance",
                "note": (
                    "No raw A1 prior config exposed by the latent batch; D1 "
                    "records a stub describing the absent source. Structural "
                    "domain/instrument/measurement metadata remains available "
                    "via latent_params and provenance."
                ),
            }

        latent_params: dict[str, Any] = {
            "phase": "D1",
            "component_keys": list(latent_batch.component_keys),
            "latent_feature_names": list(latent_batch.latent_feature_names),
            "n_components": len(latent_batch.component_keys),
            "n_latent_features": len(latent_batch.latent_feature_names),
            "concentration_transform": latent_batch.component_metadata.get(
                "concentration_transform"
            ),
            "domain_complexity": latent_batch.domain_metadata.get("complexity"),
            "note": (
                "Structural description of the latent space carried by the "
                "source CanonicalLatentBatch. No per-sample target or latent "
                "feature value is embedded."
            ),
        }

        # Split policy.
        if split_policy is None:
            resolved_split_policy: dict[str, Any] = {
                "phase": "D1",
                "kind": "explicit_indices",
                "n_context": int(ctx_idx.size),
                "n_query": int(query_idx.size),
                "n_total": int(n_total),
                "indices_disjoint": True,
                "source": "from_batches.context_indices/query_indices",
                "note": (
                    "Caller-provided integer index splits. D2 context/query "
                    "samplers are out of scope."
                ),
            }
        else:
            resolved_split_policy = dict(split_policy)
            resolved_split_policy.setdefault("phase", "D1")
            resolved_split_policy.setdefault("n_context", int(ctx_idx.size))
            resolved_split_policy.setdefault("n_query", int(query_idx.size))
            resolved_split_policy.setdefault("n_total", int(n_total))
            resolved_split_policy.setdefault("indices_disjoint", True)

        # Resolve task seed deterministically.
        if task_seed is None:
            inherited = latent_batch.provenance.get("seed")
            try:
                resolved_task_seed = int(inherited) if inherited is not None else 0
            except (TypeError, ValueError):
                resolved_task_seed = 0
        else:
            if isinstance(task_seed, bool) or not isinstance(task_seed, (int, np.integer)):
                raise NIRSPriorTaskError(
                    [
                        _failure(
                            "invalid_task_seed",
                            "task_seed",
                            "task_seed must be an int, not bool or float",
                        )
                    ]
                )
            resolved_task_seed = int(task_seed)

        # Per-split metadata (no targets / latents).
        wavelength_summary = _wavelength_summary(wavelengths)
        common_split_metadata: dict[str, Any] = {
            "phase": "D1",
            "wavelength_summary": wavelength_summary,
            "instrument_key": instrument_key,
            "measurement_mode": measurement_mode,
            "view_id_count": int(spectral_view.X.shape[0]),
            "source_view_phase": spectral_view.metadata.get("phase"),
            "source_view_kind": spectral_view.view_config.get("view_kind"),
            "preprocessing_applied": bool(
                spectral_view.preprocessing_state.get("preprocessing_applied", False)
            ),
            "noise_added_in_view": bool(
                spectral_view.noise_state.get("noise_added_in_view", False)
            ),
        }
        metadata_context: dict[str, Any] = {
            **common_split_metadata,
            "split": "context",
            "n_rows": int(ctx_idx.size),
        }
        metadata_query: dict[str, Any] = {
            **common_split_metadata,
            "split": "query",
            "n_rows": int(query_idx.size),
        }

        sampler_implemented = bool(resolved_split_policy.get("phase") == "D2")
        if sampler_implemented:
            limitations_note = (
                "D2 declarative context/query sampler produced this split. "
                "D3 multi-output regression and classification are supported."
            )
        else:
            limitations_note = (
                "Task built from explicit integer index splits (D1 path). "
                "D3 multi-output regression and classification are supported."
            )

        # Provenance with risk gates and explicit no-realism / no-transfer.
        provenance: dict[str, Any] = {
            "phase": "D3" if n_outputs > 1 else "D1",
            "source_contracts": [
                "CanonicalLatentBatch",
                "SpectralViewBatch",
            ],
            "builder_config_name": latent_batch.provenance.get("builder_config_name"),
            "random_state": latent_batch.provenance.get("random_state"),
            "task_seed": int(resolved_task_seed),
            "target_source": target_source,
            "target_clean_equals_target_noisy": target_clean_equals_target_noisy,
            "instrument_key": instrument_key,
            "measurement_mode": measurement_mode,
            "domain_key": domain_key,
            "n_outputs": int(n_outputs),
            "claims": {
                "realism": False,
                "transfer": False,
                "note": (
                    "Structural canonicalization of (context, query) tasks "
                    "only. No realism or transfer claim is derived."
                ),
            },
            "risk_gates": dict(_REQUIRED_RISK_GATES),
            "limitations": {
                "context_query_sampler_implemented": sampler_implemented,
                "multi_output_supported": True,
                "note": limitations_note,
            },
        }

        # Build a deterministic task id from a stable signature of all
        # ingredients that can vary between tasks.
        task_id = _deterministic_task_id(
            builder_config_name=str(latent_batch.provenance.get("builder_config_name") or ""),
            random_state=latent_batch.provenance.get("random_state"),
            task_seed=resolved_task_seed,
            target_source=target_source,
            target_name=resolved_target_name,
            target_type=target_type,
            n_outputs=n_outputs,
            output_names=tuple(output_names),
            context_latent_ids=ctx_latent_ids,
            query_latent_ids=query_latent_ids,
            context_view_ids=ctx_view_ids,
            query_view_ids=query_view_ids,
            split_policy=resolved_split_policy,
        )

        return cls(
            task_id=task_id,
            X_context=X_ctx,
            y_context=y_ctx,
            X_query=X_q,
            y_query=y_q,
            wavelengths_context=wavelengths,
            wavelengths_query=wavelengths,
            context_latent_ids=ctx_latent_ids,
            query_latent_ids=query_latent_ids,
            context_view_ids=ctx_view_ids,
            query_view_ids=query_view_ids,
            metadata_context=metadata_context,
            metadata_query=metadata_query,
            domain_key=domain_key,
            instrument_context=instrument_key,
            instrument_query=instrument_key,
            measurement_mode=measurement_mode,
            target_name=resolved_target_name,
            target_type=target_type,
            target_semantics=target_semantics,
            latent_params=latent_params,
            prior_config=prior_config,
            split_policy=resolved_split_policy,
            task_seed=int(resolved_task_seed),
            provenance=provenance,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a fully-serialisable view including arrays and labels."""
        out: dict[str, Any] = {}
        for f in fields(self):
            out[f.name] = _to_builtin(getattr(self, f.name))
        return out

    def to_light_dict(self) -> dict[str, Any]:
        """Return a metadata-only view without the heavy arrays or labels."""
        heavy = {"X_context", "X_query", "y_context", "y_query"}
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name in heavy:
                continue
            out[f.name] = _to_builtin(getattr(self, f.name))
        out["n_context"] = int(np.asarray(self.X_context).shape[0])
        out["n_query"] = int(np.asarray(self.X_query).shape[0])
        out["n_wavelengths_context"] = int(np.asarray(self.wavelengths_context).size)
        out["n_wavelengths_query"] = int(np.asarray(self.wavelengths_query).size)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _failure(reason: str, field: str, message: str) -> dict[str, str]:
    return {"reason": reason, "field": field, "message": message}


def _coerce_indices(
    indices: Sequence[int] | np.ndarray,
    *,
    n_total: int,
    field: str,
) -> np.ndarray:
    raw = np.asarray(indices)
    if raw.ndim != 1:
        raise NIRSPriorTaskError(
            [_failure("invalid_indices", field, f"{field} must be 1D, got shape={raw.shape}")]
        )
    if raw.size == 0:
        raise NIRSPriorTaskError(
            [_failure("empty_split", field, f"{field} must not be empty")]
        )
    if not np.issubdtype(raw.dtype, np.integer):
        raise NIRSPriorTaskError(
            [
                _failure(
                    "invalid_indices",
                    field,
                    f"{field} must be integers, got dtype={raw.dtype}",
                )
            ]
        )
    idx = raw.astype(np.intp, copy=False)
    if int(idx.min()) < 0 or int(idx.max()) >= n_total:
        raise NIRSPriorTaskError(
            [_failure("invalid_indices", field, f"{field} out of range [0, {n_total})")]
        )
    if len(set(idx.tolist())) != idx.size:
        raise NIRSPriorTaskError(
            [_failure("duplicate_indices", field, f"{field} must contain unique values")]
        )
    return idx


def _wavelength_summary(wavelengths: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(wavelengths, dtype=float)
    if arr.size == 0:
        return {"n_wavelengths": 0, "first_nm": None, "last_nm": None, "step_nm": None}
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


def _stable_signature(value: Any) -> str:
    payload = json.dumps(_to_builtin(value), sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deterministic_task_id(
    *,
    builder_config_name: str,
    random_state: Any,
    task_seed: int,
    target_source: str,
    target_name: str,
    target_type: str,
    n_outputs: int,
    output_names: tuple[str, ...],
    context_latent_ids: tuple[str, ...],
    query_latent_ids: tuple[str, ...],
    context_view_ids: tuple[str, ...],
    query_view_ids: tuple[str, ...],
    split_policy: dict[str, Any],
) -> str:
    payload = {
        "builder_config_name": builder_config_name,
        "random_state": _to_builtin(random_state),
        "task_seed": int(task_seed),
        "target_source": target_source,
        "target_name": target_name,
        "target_type": target_type,
        "n_outputs": int(n_outputs),
        "output_names": list(output_names),
        "context_latent_ids": list(context_latent_ids),
        "query_latent_ids": list(query_latent_ids),
        "context_view_ids": list(context_view_ids),
        "query_view_ids": list(query_view_ids),
        "split_policy": _to_builtin(split_policy),
    }
    digest = _stable_signature(payload)[:16]
    prefix = builder_config_name or "task"
    return f"{prefix}__d1__{digest}"


def _finite_numeric_failure(
    array: np.ndarray,
    *,
    field: str,
    message: str,
) -> dict[str, str] | None:
    try:
        finite = np.isfinite(array)
    except TypeError:
        return _failure("non_numeric", field, message)
    if not bool(np.all(finite)):
        return _failure("non_finite", field, message)
    return None


def _is_integer_like(array: np.ndarray) -> bool:
    arr = np.asarray(array)
    if arr.size == 0:
        return True
    try:
        as_float = np.asarray(arr, dtype=float)
    except (TypeError, ValueError):
        return False
    if not bool(np.all(np.isfinite(as_float))):
        return False
    return bool(np.all(np.equal(as_float, np.round(as_float))))


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


def _find_leakage_paths(
    value: Any,
    *,
    leaky_keys: tuple[str, ...],
    prefix_match: bool = True,
    path: str = "$",
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if _is_leaky_key(key_text, leaky_keys=leaky_keys, prefix_match=prefix_match):
                paths.append(child_path)
            paths.extend(
                _find_leakage_paths(
                    child,
                    leaky_keys=leaky_keys,
                    prefix_match=prefix_match,
                    path=child_path,
                )
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(
                _find_leakage_paths(
                    child,
                    leaky_keys=leaky_keys,
                    prefix_match=prefix_match,
                    path=f"{path}[{index}]",
                )
            )
    return paths


def _is_leaky_key(
    key: str,
    *,
    leaky_keys: tuple[str, ...],
    prefix_match: bool,
) -> bool:
    normalized = key.lower()
    if normalized in leaky_keys:
        return True
    if prefix_match:
        tokens = tuple(token for token in normalized.split("_") if token)
        return any(
            normalized.startswith(f"{leaky_key}_")
            or normalized.endswith(f"_{leaky_key}")
            or f"_{leaky_key}_" in normalized
            or tuple(leaky_key.split("_")) in _token_windows(tokens, leaky_key)
            for leaky_key in leaky_keys
        )
    return False


def _token_windows(tokens: tuple[str, ...], leaky_key: str) -> tuple[tuple[str, ...], ...]:
    leaky_tokens = tuple(leaky_key.split("_"))
    width = len(leaky_tokens)
    if width == 0 or len(tokens) < width:
        return ()
    return tuple(tokens[index : index + width] for index in range(len(tokens) - width + 1))
