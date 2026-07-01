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

``dag_ml_data`` is a CORE dependency since the ADR-17 cutover; imports are still guarded so a
broken wheel missing the native backend surfaces a clear error rather than a raw ImportError.
Scope: single-source / no-repetition baseline.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset

    from .identity import IdentityMap

_DEFAULT_SOURCE_ID = "src0"

# The dag-ml-data representation id for an early-fusion multi-source model input: the engine
# (host-side, via the resolver) joins the N per-source blocks by sample_id into one fused matrix.
_FEATURE_BLOCK_SET = "feature_block_set"


def source_ids(dataset: SpectroDataset) -> list[str]:
    """The per-source ids the schema/plan/binding agree on.

    Single-source keeps the legacy ``["src0"]`` (BYTE-IDENTICAL); a multi-source dataset
    (``features_sources() > 1``) emits ``["src0", "src1", …]`` — one per ``FeatureSource``.
    """
    n_sources = dataset.features_sources()
    if n_sources <= 1:
        return [_DEFAULT_SOURCE_ID]
    return [f"src{k}" for k in range(n_sources)]


def source_order(dataset: SpectroDataset) -> list[str]:
    """Legacy by_source branch keys in exact source-index order.

    Mirrors ``BranchController._get_source_names`` without importing the controller:
    use ``dataset.source_name(i)`` when a dataset exposes it, else fall back to
    ``source_i``. These are the user-facing keys legacy by_source dict bodies
    resolve against, distinct from the native data-plan ids (``src0``/``src1``).
    """
    n_sources = dataset.features_sources()
    names: list[str] = []
    try:
        for index in range(n_sources):
            name = dataset.source_name(index) if hasattr(dataset, "source_name") else None
            names.append(str(name) if name else f"source_{index}")
    except Exception:  # noqa: BLE001 - match legacy's defensive fallback
        return [f"source_{index}" for index in range(n_sources)]
    return names


def _params_fingerprint(transform_id: str) -> str:
    """Deterministic 64-hex digest for an augmentation transform (sorted-params discipline).

    Only the transform id is available at the relation grain, so the digest is over it; it is
    stable across runs and satisfies the dag-ml-data contract (``params_fingerprint`` must be a
    64-character hex digest).
    """
    return hashlib.sha256(transform_id.encode("utf-8")).hexdigest()


def _import_dag_ml_data() -> Any:
    try:
        import dag_ml_data
    except ImportError as exc:  # pragma: no cover - exercised only without the wheel
        raise ImportError("dag-ml-data is not installed; it is a core dependency — reinstall with `pip install nirs4all`") from exc
    return dag_ml_data


def _build_coordinator_envelope(dag_ml_data: Any, schema: dict[str, Any], plan: dict[str, Any], relations: dict[str, Any]) -> dict[str, Any]:
    """Build a coordinator envelope through either dag-ml-data Python API generation.

    Newer dag-ml-data wheels expose a typed convenience wrapper whose return value has ``to_dict()``;
    older/minimal wheels expose only the raw JSON helper. The host contract stays a plain dict either way.
    """
    builder = getattr(dag_ml_data, "build_coordinator_data_plan_envelope", None)
    if builder is not None:
        envelope = builder(schema, plan, relations)
        return dict(envelope.to_dict())

    json_builder = dag_ml_data.build_coordinator_data_plan_envelope_json
    return dict(json.loads(json_builder(json.dumps(schema), json.dumps(plan), json.dumps(relations))))


def _num_wavelengths(dataset: SpectroDataset, source_idx: int = 0) -> int:
    """Feature count of one source. Single-source ``num_features`` is an int; multi-source a list."""
    num_features = dataset.num_features
    if isinstance(num_features, int):
        return int(num_features)
    return int(num_features[source_idx])


def num_targets(dataset: SpectroDataset) -> int:
    """The number of target columns (``y`` width). 1 for the single-target baseline."""
    return int(dataset._target_accessor._block.num_targets)


def target_names(dataset: SpectroDataset) -> list[str]:
    """Per-target names ``["y0", "y1", …]`` — nirs4all stores no target headers, so they are
    synthesized positionally. Used as the per-target metric suffix (``rmse:y0``/``rmse:y1``)."""
    return [f"y{i}" for i in range(num_targets(dataset))]


def _target_representation(dataset: SpectroDataset, n_samples: int) -> dict[str, Any]:
    """The ``targets.y`` representation spec.

    Single-target keeps the legacy ``tabular_numeric`` table (BYTE-IDENTICAL); a multi-target
    dataset (``num_targets>1``) emits ``target_numeric_matrix`` / ``target_categorical_matrix``
    (per ``is_classification``) with a ``target`` axis of width ``num_targets`` — mirroring
    nirs4all-io-dagml ``target_matrix_representation`` (lib.rs:256-269). The per-target columns ride
    inside this one sample-keyed block, so the fold/OOF partition (over samples) stays leakage-safe.
    """
    k = num_targets(dataset)
    if k <= 1:
        return {
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
    categorical = dataset.is_classification
    return {
        "id": "target_categorical_matrix" if categorical else "target_numeric_matrix",
        "type_id": "target",
        "rank": 2,
        "axes": [
            {"name": "sample", "kind": "sample", "unit": None, "size": n_samples, "variable": False},
            {"name": "target", "kind": "target", "unit": None, "size": k, "variable": False},
        ],
        "container": "array",
        "dtype": "string" if categorical else "float64",
        "sparse": False,
        "ragged": False,
    }


def _signal_source(dataset: SpectroDataset, source_id: str, source_idx: int, n_samples: int) -> dict[str, Any]:
    """One ``signal_1d`` source descriptor for feature source ``source_idx`` (per-source feature count)."""
    return {
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
                {"name": "wavelength", "kind": "wavelength", "unit": None, "size": _num_wavelengths(dataset, source_idx), "variable": False},
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


def _dataset_schema(dataset: SpectroDataset, sources: list[str], sample_id_strings: list[str]) -> dict[str, Any]:
    """The ``DatasetSchema``: one ``signal_1d`` source per feature source (``sources``).

    Single-source emits exactly one ``src0`` (BYTE-IDENTICAL); multi-source emits N per-source
    ``signal_1d`` sources whose blocks the data plan joins into a ``feature_block_set`` for early
    fusion — mirroring nirs4all-io-dagml ``build_dag_ml_data_parts`` (lib.rs:486-528).
    """
    n_samples = len(sample_id_strings)
    return {
        "dataset_id": f"nirs4all.{dataset.name}",
        "sample_ids": sample_id_strings,
        "sources": [_signal_source(dataset, source_id, source_idx, n_samples) for source_idx, source_id in enumerate(sources)],
        "targets": {"y": _target_representation(dataset, n_samples)},
        "metadata": {},
    }


def _data_plan(dataset: SpectroDataset, sources: list[str]) -> dict[str, Any]:
    """The ``DataPlan``: materialize each source, then join.

    SINGLE-SOURCE (BYTE-IDENTICAL): materialize ``signal_1d`` → adapt ``spectra.flatten`` →
    join to ``tabular_numeric`` (output_representation ``tabular_numeric``).

    MULTI-SOURCE (early fusion): materialize each per-source ``signal_1d``, then a single Join to
    ``feature_block_set`` — the N per-source blocks are fused by sample_id host-side (the resolver's
    ``x_rows(concat_source=True)``). Mirrors nirs4all-io-dagml (lib.rs:551-585).
    """
    if len(sources) == 1:
        source_id = sources[0]
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
    steps: list[dict[str, Any]] = [
        {"kind": "materialize", "source_id": source_id, "adapter_id": None, "input_representation": None, "output_representation": "signal_1d", "fit_scope": "stateless", "requires_user_choice": False, "metadata": {"output": f"src:{source_id}"}}
        for source_id in sources
    ]
    steps.append(
        {"kind": "join", "source_id": None, "adapter_id": None, "input_representation": "signal_1d", "output_representation": _FEATURE_BLOCK_SET, "fit_scope": "stateless", "requires_user_choice": False, "metadata": {"inputs": [f"src:{source_id}" for source_id in sources], "output": "port:X"}}
    )
    return {
        "id": f"plan.{dataset.name}",
        "steps": steps,
        "output_representation": _FEATURE_BLOCK_SET,
        "issues": [],
    }


def _source_layout(dataset: SpectroDataset, sources: list[str]) -> dict[str, Any]:
    """Explicit by-source layout contract for multi-source feature concat.

    ``source_order`` is the legacy branch-key order (``source_0`` etc., or
    dataset-provided source names); ``source_ids`` is the native data-plan order
    (``src0`` etc.). Consumers must map dict bodies by these keys, not by guessing
    from insertion order.
    """
    names = source_order(dataset)
    column_start = 0
    blocks: list[dict[str, Any]] = []
    per_source_outputs: dict[str, dict[str, Any]] = {}
    for index, (source_name, source_id) in enumerate(zip(names, sources, strict=True)):
        column_count = _num_wavelengths(dataset, index)
        output = {
            "feature_set_id": f"x:{source_id}",
            "representation_id": "tabular_numeric",
            "adapter_id": f"preprocess:{source_id}",
            "fit_scope": "fold_train",
        }
        block = {
            "source_name": source_name,
            "source_id": source_id,
            "source_index": index,
            "preprocessing_output": output,
            "column_start": column_start,
            "column_count": column_count,
            "feature_names": [str(name) for name in (dataset.headers(index) or [])],
        }
        blocks.append(block)
        per_source_outputs[source_name] = {"source_id": source_id, "source_index": index, **output}
        column_start += column_count
    return {
        "kind": "by_source_concat",
        "source_order": names,
        "source_ids": sources,
        "blocks": blocks,
        "per_source_preprocessing_outputs": per_source_outputs,
        "concat_layout": {
            "strategy": "concat",
            "axis": "feature",
            "source_order": names,
            "source_ids": sources,
            "total_column_count": column_start,
            "output_source_index": 0,
            "preserves_storage_roundtrip": True,
        },
        "concat": {
            "feature_set_id": "x",
            "representation_id": "tabular_numeric",
            "axis": "feature",
            "total_column_count": column_start,
            "preserve_source_order": True,
            "namespace_columns": True,
        },
    }


def sample_relations(
    identity: IdentityMap,
    *,
    source_id: str | None = _DEFAULT_SOURCE_ID,
    sample_ints: list[int] | None = None,
    excluded_sample_ints: set[int] | None = None,
    metadata_by_sample: dict[str, dict[int, Any]] | None = None,
    tags_by_sample: dict[int, list[str]] | None = None,
    augmentation_by_sample: dict[int, str] | None = None,
    group_by_sample: dict[int, str] | None = None,
) -> dict[str, Any]:
    """The ``SampleRelationTable`` rows, optionally scoped to a sample-int subset.

    Pass ``sample_ints`` (e.g. the CV training pool) to scope the relations for fold
    validation; omit it for the full-dataset relations the envelope declares.

    Pass ``excluded_sample_ints`` to emit ``excluded: true`` for those rows — the opt-in
    leakage-pure exclude mode (``keep_in_oof=True``): Phase 1's native ``excluded`` bit then
    drops them from each fold's TRAIN view while keeping them in validation/OOF and predict.
    Omit it (the default) and every row is ``excluded: false`` — the legacy mode removes
    excluded samples from the CV universe entirely (they are simply absent from the pool).

    Pass ``metadata_by_sample`` (``{column: {sample_int: value}}``) to emit per-sample
    metadata onto each relation (Slice 1 ``metadata`` field). The native
    ``fan_out_data_aware_branches`` discovers a separation branch's partition values from
    these relation metadata values, so a ``by_metadata`` branch criterion column must be
    present here. Omit it and every row carries empty ``metadata``.

    Pass ``tags_by_sample`` (``{sample_int: [tag, ...]}``) to emit per-sample tag labels onto each
    relation. Rows without tags omit the ``tags`` field so untagged fingerprints stay on the native
    contract's skip-if-empty path.

    Pass ``group_by_sample`` (``{sample_int: group_value}``) to emit ``group_id`` for a
    **repetition** dataset — several stored rows that share one physical sample carry the same
    group value (their repetition column). dag-ml-data's
    ``validate_fold_set_against_sample_relations`` then refuses any fold that splits a group
    across train/validation (native group-leakage validation), so all repetitions of a sample
    must stay on the same fold side. Omit it and every row keeps ``group_id = None``.

    Augmented rows (``identity`` minted on a dataset that already holds the synthetic rows)
    emit ``origin_id`` = the origin row's ``observation_id`` (``to_wire(origin_int)``, distinct
    from the child's own ``observation_id``); their ``sample_id`` stays the origin's grouping
    key. Pass ``augmentation_by_sample`` (``{sample_int: transform_id}``) to attach the
    structured ``augmentation`` metadata (``{transform_id, params_fingerprint}``) for those
    rows. Base rows keep ``origin_id = None`` and no augmentation metadata (the dag-ml-data
    contract rejects an origin or augmentation on a non-augmented row).
    """
    excluded = excluded_sample_ints or set()
    metadata_columns = metadata_by_sample or {}
    tag_labels = tags_by_sample or {}
    augmentation_ids = augmentation_by_sample or {}
    group_ids = group_by_sample or {}
    chosen = identity.identities if sample_ints is None else [identity.identities[i] for i in _positions(identity, sample_ints)]
    rows: list[dict[str, Any]] = []
    for sample in chosen:
        row: dict[str, Any] = {
            "observation_id": sample.observation_id,
            "sample_id": sample.sample_id,
            "source_id": source_id,
            "target_id": "y",
            "group_id": group_ids.get(sample.sample_int),
            "origin_id": (identity.to_wire(sample.origin_int) if sample.augmented else None),
            "repetition_id": None,
            "augmented": sample.augmented,
            "excluded": sample.sample_int in excluded,
            "metadata": {column: values[sample.sample_int] for column, values in metadata_columns.items() if sample.sample_int in values},
        }
        tags = tag_labels.get(sample.sample_int)
        if tags:
            row["tags"] = list(tags)
        if sample.augmented and sample.sample_int in augmentation_ids:
            transform_id = augmentation_ids[sample.sample_int]
            row["augmentation"] = {"transform_id": transform_id, "params_fingerprint": _params_fingerprint(transform_id)}
        rows.append(row)
    return {"rows": rows}


def _positions(identity: IdentityMap, sample_ints: list[int]) -> list[int]:
    index = {sample.sample_int: position for position, sample in enumerate(identity.identities)}
    return [index[sample_int] for sample_int in sample_ints]


def build_envelope(
    dataset: SpectroDataset,
    identity: IdentityMap,
    *,
    source_id: str = _DEFAULT_SOURCE_ID,
    sample_ints: list[int] | None = None,
    excluded_sample_ints: set[int] | None = None,
    metadata_by_sample: dict[str, dict[int, Any]] | None = None,
    tags_by_sample: dict[int, list[str]] | None = None,
    augmentation_by_sample: dict[int, str] | None = None,
    group_by_sample: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Build the validated ``CoordinatorDataPlanEnvelope``.

    Pass ``sample_ints`` to scope the envelope to a sample universe — e.g. the CV training
    pool, so the schema + ``coordinator_relations`` match the embedded ``FoldSet`` and pass
    ``validate_data_envelope_relations`` (every relation must live inside the fold set).
    Omit it for a whole-dataset envelope.

    Pass ``excluded_sample_ints`` (a subset of ``sample_ints``) to mark those rows
    ``excluded: true`` — the opt-in ``keep_in_oof=True`` exclude mode where Phase 1's native
    bit drops them from fold TRAIN but keeps them in validation/OOF. Default (``None``) marks
    every row not-excluded.

    Pass ``metadata_by_sample`` (``{column: {sample_int: value}}``) to emit per-sample
    metadata onto each relation — the native ``fan_out_data_aware_branches`` reads a
    ``by_metadata`` separation branch's partition values from these relation metadata values.

    Pass ``tags_by_sample`` (``{sample_int: [tag, ...]}``) to emit per-sample tag labels onto the
    native relations. Untagged rows omit the field.

    Pass ``augmentation_by_sample`` (``{sample_int: transform_id}``) to attach the structured
    ``augmentation`` metadata to augmented rows (their ``origin_id`` is always emitted from the
    identity grain).

    Pass ``group_by_sample`` (``{sample_int: group_value}``) to emit ``group_id`` on each
    relation for a **repetition** dataset — repetitions of one physical sample share a group
    value, and dag-ml-data refuses a fold that splits a group across train/validation. For a
    repetition dataset every relation is its own ``sample_id`` (each stored row is scored
    individually at the repetition grain), so the schema's sample axis is one entry per row.

    The wheel computes all fingerprints and derives ``coordinator_relations``; a successful
    return means the envelope is contract-valid (the materialize-time fingerprint gate
    will accept it).
    """
    dag_ml_data = _import_dag_ml_data()
    chosen = identity.identities if sample_ints is None else [identity.identities[i] for i in _positions(identity, sample_ints)]
    # SINGLE-SOURCE: the caller's ``source_id`` (default ``src0``) is the one source, and each
    # relation carries it (BYTE-IDENTICAL). MULTI-SOURCE (early fusion): one ``signal_1d`` source
    # per ``FeatureSource`` (``src0..srcK``), the plan joins them to ``feature_block_set``, and each
    # relation's ``source_id`` is None (a sample's blocks span every source — relations are
    # sample-grain), mirroring nirs4all-io-dagml.
    sources = source_ids(dataset)
    multi_source = len(sources) > 1
    relation_source_id = None if multi_source else source_id
    # The schema's sample axis is the SAMPLE grain (one entry per distinct sample, which must be unique);
    # augmented children share their origin's sample_id, so dedup order-preservingly. The observation grain
    # (one row per stored row) lives in the relations, not the schema.
    schema = _dataset_schema(dataset, sources if multi_source else [source_id], list(dict.fromkeys(sample.sample_id for sample in chosen)))
    plan = _data_plan(dataset, sources if multi_source else [source_id])
    relations = sample_relations(
        identity,
        source_id=relation_source_id,
        sample_ints=sample_ints,
        excluded_sample_ints=excluded_sample_ints,
        metadata_by_sample=metadata_by_sample,
        tags_by_sample=tags_by_sample,
        augmentation_by_sample=augmentation_by_sample,
        group_by_sample=group_by_sample,
    )
    out = _build_coordinator_envelope(dag_ml_data, schema, plan, relations)
    if multi_source:
        out["plan"]["source_layout"] = _source_layout(dataset, sources)
    return out


def build_fold_set(identity: IdentityMap, folds: list[tuple[list[int], list[int]]], *, set_id: str = "nirs4all.folds") -> dict[str, Any]:
    """Translate ``(train_ints, validation_ints)`` folds into a dag-ml-data ``FoldSet``.

    Pure identity translation — sample ints become stable wire ids. The validation distribution is
    auto-detected: a clean OOF partition (each sample validated exactly once, KFold-style) stays the
    default ``Partition`` mode (``partition_mode`` omitted → byte-identical fold set), while
    resampling CV (ShuffleSplit / repeated KFold, where a sample is validated 0 or 2+ times) is
    tagged ``"resampled"`` so dag-ml relaxes OOF completeness (the per-fold leakage guard holds).
    """
    seen: set[int] = set()
    validation_counts: dict[int, int] = {}
    for train_ints, validation_ints in folds:
        for sample_int in (*train_ints, *validation_ints):
            seen.add(sample_int)
        for sample_int in validation_ints:
            validation_counts[sample_int] = validation_counts.get(sample_int, 0) + 1
    # The CV pool drives the REFIT (FullTrain) materialization ORDER: dag-ml preserves the host order of
    # fold_set.sample_ids when it materializes the full-train universe (merge.rs). Order the pool by
    # STORAGE order (ascending sample int == the train partition's storage order), NOT fold-first-seen —
    # legacy refit trains on `dataset.x(train)` in storage order, so a fold-first-seen pool (sample 0
    # lands late) would feed a fixed-seed RF/GBR a different bootstrap draw and diverge the refit model.
    # Same SET as fold-first-seen, just the storage ORDER. Per-fold training uses each fold's own
    # train_sample_ids (unchanged); OOF coverage reads sample_ids as a set (order-insensitive).
    pool: list[int] = sorted(seen)
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
