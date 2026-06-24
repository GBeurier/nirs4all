"""Data-plan envelope + fold-set builders for the nirs4all → dag-ml(-data) bridge.

dag-ml-data owns the contract *types* and the fingerprint algorithm; this module only
assembles the JSON inputs from a ``SpectroDataset`` + its :class:`IdentityMap` and hands
them to the ``dag_ml_data`` wheel, which computes every fingerprint internally and derives
``coordinator_relations`` from the ``SampleRelationTable``. We never compute a fingerprint
or hand-build a coordinator relation — that keeps the materialize-time fingerprint gate
passing by construction.

Declares **identity only**, never X/y values:

* the ``DatasetSchema`` (one source, NIR spectra ``signal_1d`` → ``tabular_numeric``),
* the ``DataPlan`` (materialize → adapt ``spectra.flatten`` → join to ``port:X``),
* the ``SampleRelationTable`` (one row per observation; input field names
  ``origin_id`` / ``repetition_id`` / ``augmented`` / ``excluded``).

Folds are a **separate** first-class contract (``FoldSet``), validated against the
CV-universe relations by ``validate_fold_set_against_sample_relations`` — never carried in
the envelope. That validator enforces a clean OOF **partition** (each sample validated
exactly once): ``KFold`` satisfies it; ``ShuffleSplit`` does not (a known OOF-semantics gap
flagged for the execution/mechanism phase).

``dag_ml_data`` is an optional dependency (``nirs4all[dagml]``); imports are guarded.
Scope: single-source / no-repetition baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset

    from .identity import IdentityMap

_DEFAULT_SOURCE_ID = "src0"


def _import_dag_ml_data() -> Any:
    try:
        import dag_ml_data
    except ImportError as exc:  # pragma: no cover - exercised only without the wheel
        raise ImportError("dag-ml-data is not installed; install with `pip install nirs4all[dagml]`") from exc
    return dag_ml_data


def _num_wavelengths(dataset: SpectroDataset) -> int:
    num_features = dataset.num_features
    return int(num_features if isinstance(num_features, int) else num_features[0])


def _dataset_schema(dataset: SpectroDataset, source_id: str, sample_id_strings: list[str]) -> dict[str, Any]:
    n_samples = len(sample_id_strings)
    return {
        "dataset_id": f"nirs4all.{dataset.name}",
        "sample_ids": sample_id_strings,
        "sources": [
            {
                "id": source_id,
                "name": dataset.name,
                "type_id": "dense_signal",
                "modality": "spectroscopy",
                "native_representation": {
                    "id": "signal_1d",
                    "type_id": "dense_signal",
                    "rank": 2,
                    "axes": [
                        {"name": "sample", "kind": "sample", "unit": None, "size": n_samples, "variable": False},
                        {"name": "wavelength", "kind": "wavelength", "unit": None, "size": _num_wavelengths(dataset), "variable": False},
                    ],
                    "container": "ndarray",
                    "dtype": "float64",
                    "sparse": False,
                    "ragged": False,
                },
                "sample_key": "sample_id",
                "granularity": "per_sample",
                "schema": {},
                "tags": {},
            }
        ],
        "targets": {
            "y": {
                "id": "tabular_numeric",
                "type_id": "table",
                "rank": 2,
                "axes": [
                    {"name": "sample", "kind": "sample", "unit": None, "size": n_samples, "variable": False},
                    {"name": "target", "kind": "target", "unit": None, "size": 1, "variable": False},
                ],
                "container": "dataframe",
                "dtype": "float64",
                "sparse": False,
                "ragged": False,
            }
        },
        "metadata": {},
    }


def _data_plan(dataset: SpectroDataset, source_id: str) -> dict[str, Any]:
    return {
        "id": f"plan.{dataset.name}",
        "steps": [
            {"kind": "materialize", "source_id": source_id, "adapter_id": None, "input_representation": None, "output_representation": "signal_1d", "fit_scope": "stateless", "requires_user_choice": False, "metadata": {"output": f"src:{source_id}"}},
            {"kind": "adapt", "source_id": source_id, "adapter_id": "spectra.flatten", "input_representation": "signal_1d", "output_representation": "tabular_numeric", "fit_scope": "stateless", "requires_user_choice": False, "metadata": {"input": f"src:{source_id}", "output": "step:adapt:0"}},
            {"kind": "join", "source_id": None, "adapter_id": None, "input_representation": "tabular_numeric", "output_representation": "tabular_numeric", "fit_scope": "stateless", "requires_user_choice": False, "metadata": {"inputs": ["step:adapt:0"], "output": "port:X"}},
        ],
        "output_representation": "tabular_numeric",
        "issues": [],
    }


def sample_relations(identity: IdentityMap, *, source_id: str = _DEFAULT_SOURCE_ID, sample_ints: list[int] | None = None) -> dict[str, Any]:
    """The ``SampleRelationTable`` rows, optionally scoped to a sample-int subset.

    Pass ``sample_ints`` (e.g. the CV training pool) to scope the relations for fold
    validation; omit it for the full-dataset relations the envelope declares.
    """
    chosen = identity.identities if sample_ints is None else [identity.identities[i] for i in _positions(identity, sample_ints)]
    return {
        "rows": [
            {
                "observation_id": sample.observation_id,
                "sample_id": sample.sample_id,
                "source_id": source_id,
                "target_id": "y",
                "group_id": None,
                "origin_id": None,
                "repetition_id": None,
                "augmented": sample.augmented,
                "excluded": False,
                "metadata": {},
            }
            for sample in chosen
        ]
    }


def _positions(identity: IdentityMap, sample_ints: list[int]) -> list[int]:
    index = {sample.sample_int: position for position, sample in enumerate(identity.identities)}
    return [index[sample_int] for sample_int in sample_ints]


def build_envelope(dataset: SpectroDataset, identity: IdentityMap, *, source_id: str = _DEFAULT_SOURCE_ID, sample_ints: list[int] | None = None) -> dict[str, Any]:
    """Build the validated ``CoordinatorDataPlanEnvelope``.

    Pass ``sample_ints`` to scope the envelope to a sample universe — e.g. the CV training
    pool, so the schema + ``coordinator_relations`` match the embedded ``FoldSet`` and pass
    ``validate_data_envelope_relations`` (every relation must live inside the fold set).
    Omit it for a whole-dataset envelope.

    The wheel computes all fingerprints and derives ``coordinator_relations``; a successful
    return means the envelope is contract-valid (the materialize-time fingerprint gate
    will accept it).
    """
    dag_ml_data = _import_dag_ml_data()
    chosen = identity.identities if sample_ints is None else [identity.identities[i] for i in _positions(identity, sample_ints)]
    envelope = dag_ml_data.build_coordinator_data_plan_envelope(
        _dataset_schema(dataset, source_id, [sample.sample_id for sample in chosen]),
        _data_plan(dataset, source_id),
        sample_relations(identity, source_id=source_id, sample_ints=sample_ints),
    )
    return dict(envelope.to_dict())


def build_fold_set(identity: IdentityMap, folds: list[tuple[list[int], list[int]]], *, set_id: str = "nirs4all.folds") -> dict[str, Any]:
    """Translate ``(train_ints, validation_ints)`` folds into a dag-ml-data ``FoldSet``.

    Pure identity translation — sample ints become stable wire ids. The validation distribution is
    auto-detected: a clean OOF partition (each sample validated exactly once, KFold-style) stays the
    default ``Partition`` mode (``partition_mode`` omitted → byte-identical fold set), while
    resampling CV (ShuffleSplit / repeated KFold, where a sample is validated 0 or 2+ times) is
    tagged ``"resampled"`` so dag-ml relaxes OOF completeness (the per-fold leakage guard holds).
    """
    pool: list[int] = []
    seen: set[int] = set()
    validation_counts: dict[int, int] = {}
    for train_ints, validation_ints in folds:
        for sample_int in (*train_ints, *validation_ints):
            if sample_int not in seen:
                seen.add(sample_int)
                pool.append(sample_int)
        for sample_int in validation_ints:
            validation_counts[sample_int] = validation_counts.get(sample_int, 0) + 1
    is_oof_partition = len(validation_counts) == len(pool) and all(count == 1 for count in validation_counts.values())
    fold_set: dict[str, Any] = {
        "id": set_id,
        "sample_ids": [identity.to_wire(sample_int) for sample_int in pool],
        "folds": [
            {
                "fold_id": f"fold{index}",
                "train_sample_ids": [identity.to_wire(sample_int) for sample_int in train_ints],
                "validation_sample_ids": [identity.to_wire(sample_int) for sample_int in validation_ints],
            }
            for index, (train_ints, validation_ints) in enumerate(folds)
        ],
    }
    if not is_oof_partition:
        fold_set["partition_mode"] = "resampled"
    return fold_set
