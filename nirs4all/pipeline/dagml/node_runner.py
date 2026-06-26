"""Node runner — execute one dag-ml ``NodeTask`` with real nirs4all operators + data.

This is the heart of the host-controller side of the migration: it replaces the shipped
conformance adapter's hash-synthesized features with the **real** ``SpectroDataset`` rows
(via :class:`MaterializationResolver`) and the **real** operator (via ``route_graph_node``),
mirroring the dag-ml ``NodeTask``/``NodeResult`` JSON contract exactly (verified against
``dag-ml/examples/adapters/sklearn_process_controller.py``).

Frame contract (per phase): ``data_views`` is keyed by ``data:<input>`` with a
``partition`` — FIT_CV gets ``fold_train`` + a sibling ``…:validation`` (``fold_validation``)
and predicts the validation samples (``partition: validation``); REFIT gets ``full_train``,
fits, persists a model artifact, predicts the train samples (``partition: final``); PREDICT
reloads the artifact and predicts the ``predict`` view. The ``NodeTask`` carries ``node_plan``
(params, kind) but **not** the operator class, so the runner takes a ``node_lookup`` that maps
``node_id`` → the compiled graph node (which carries the operator) — both produced nirs4all
side by ``build_dagml_plan``.

**Scope (honest):** only ``model``/``tuner`` nodes produce predictions (matching dag-ml);
``transform``/``y_transform`` nodes are passthrough output-handles here. Real cross-node
feature chaining (e.g. SNV→PLS, where the model must see *transformed* features) is the
unresolved A3 gap — the process-adapter delivers only ``sample_ids`` per node, so a model
after a transform would re-fetch raw features. Until that is designed, this runner is
numerically correct for **model-on-raw-features** graphs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sklearn.pipeline import make_pipeline

from nirs4all.pipeline.dagml_bridge import _META_MODEL_CONTROLLER_ID

from .operator_routing import route_graph_node

if TYPE_CHECKING:
    from .resolver import MaterializationResolver

_PREDICTION_PARTITION = {"FIT_CV": "validation", "REFIT": "final", "PREDICT": "final", "EXPLAIN": "final"}


def _is_multi_block_model(model: Any) -> bool:
    """True when ``model`` natively consumes a LIST of per-source blocks (MB-PLS intermediate fusion).

    The single canonical multi-block consumer is :class:`~nirs4all.operators.models.sklearn.MBPLS`,
    whose ``fit``/``predict`` branch on ``isinstance(X, list)`` to fuse the blocks internally (per-block
    standardize + super-score). Detected by class identity (not duck-typing) so an ordinary estimator
    is never accidentally handed a list. The import is lazy — MBPLS lives behind an optional model
    subpackage and most pipelines never reference it.
    """
    try:
        from nirs4all.operators.models.sklearn.mbpls import MBPLS
    except ImportError:  # pragma: no cover - MBPLS is in the core install, but stay defensive
        return False
    return isinstance(model, MBPLS)


class _MultiBlockEstimator:
    """Fit a multi-block model on a LIST of per-source blocks, applying the X-chain PER BLOCK (S5).

    INTERMEDIATE fusion = each source's per-block preprocessing runs first, then the multi-block model
    fuses the transformed blocks. The upstream X-transform chain (e.g. SNV) is a stateful sklearn
    transformer; for early fusion (concat) one chain fits the whole matrix, but for intermediate fusion
    each block gets its OWN cloned chain fit on THAT block's fold-train rows — a block's train statistics
    never touch another block (and never its own validation rows, since the chain is fit inside the
    fold's train materialization, exactly like the single-block estimator). The fitted per-block chains
    + the fitted model travel together through the REFIT→PREDICT artifact handle.

    LEAKAGE: identical fold/holdout discipline as the single-block path — every per-block transform is
    a per-fold fit on fold-train only; block alignment across sources stays identity-keyed (the resolver
    delivers sample-aligned blocks). The model's own block weights are part of the fitted artifact.
    """

    def __init__(self, model: Any, chain_template: list[Any]) -> None:
        self._model = model
        # The routed upstream X-transform operators in furthest-upstream-first order. One independent
        # clone of every step is fit per block at fit time (the block count is only known then). Applied
        # as a bare transform sequence (NOT a sklearn Pipeline): a transforms-only Pipeline has no final
        # estimator, so its ``check_is_fitted`` would reject ``transform`` for a stateless transformer
        # like SNV — applying the steps directly matches the operator's own fit/transform contract.
        self._chain_template = chain_template
        self._block_chains: list[list[Any]] = []

    def _fit_transform_block(self, steps: list[Any], block: np.ndarray) -> np.ndarray:
        out = block
        for step in steps:
            out = np.asarray(step.fit_transform(out))
        return out

    @staticmethod
    def _transform_block(steps: list[Any], block: np.ndarray) -> np.ndarray:
        out = block
        for step in steps:
            out = np.asarray(step.transform(out))
        return out

    def fit(self, blocks: list[np.ndarray], y: Any) -> _MultiBlockEstimator:
        from sklearn.base import clone

        self._block_chains = [[clone(step) for step in self._chain_template] for _ in blocks]
        transformed = [self._fit_transform_block(steps, block) for steps, block in zip(self._block_chains, blocks, strict=True)]
        self._model.fit(transformed, y)
        return self

    def predict(self, blocks: list[np.ndarray]) -> np.ndarray:
        transformed = [self._transform_block(steps, block) for steps, block in zip(self._block_chains, blocks, strict=True)]
        return np.asarray(self._model.predict(transformed))


def _source_index(node: dict[str, Any]) -> int | None:
    """The ``source_index`` a by_source branch model is bound to (S4), or ``None`` for any other node.

    A by_source separation branch fans out one model node PER SOURCE, each carrying
    ``metadata.source_index`` so the node materializes only that ONE source's feature block (a
    feature-axis selection, all samples) instead of the early-fusion concat — late fusion by source.
    Read from the compiled graph node (available in every phase, so PREDICT selects the same source
    its REFIT model was fit on). Absent for single-source / duplication / separation-by-metadata nodes.
    """
    value = (node.get("metadata") or {}).get("source_index")
    return int(value) if value is not None else None


def _view_by_partition(task: dict[str, Any], partition: str) -> dict[str, Any] | None:
    for view in task.get("data_views", {}).values():
        if view.get("partition") == partition:
            return cast("dict[str, Any]", view)
    return None


def _sample_ids(view: dict[str, Any] | None, *, allow_empty: bool = False) -> list[str]:
    if view is None:
        raise ValueError("data view is missing")
    ids = view.get("sample_ids")
    if not ids and not allow_empty:
        raise ValueError("data view is missing sample_ids")
    return list(ids or [])


def _branch_view_keep(sample_id: str, selector: dict[str, Any], sample_metadata: dict[str, dict[str, Any]]) -> bool:
    """Whether ``sample_id`` matches a branch view selector (by_metadata equality / by_tag membership)."""
    meta = sample_metadata.get(sample_id, {})
    for key, value in (selector.get("metadata") or {}).items():
        if meta.get(key) != value:
            return False
    tags = selector.get("tags") or []
    return not (tags and not (set(tags) & set(meta.get("__tags__", []))))


def _filter_by_branch_view(view: dict[str, Any] | None, sample_metadata: dict[str, dict[str, Any]] | None) -> None:
    """Restrict a data view's ``sample_ids`` to those matching its ``branch_view`` selector, in place.

    A separation-branch fan-out gives each per-partition model node a ``branch_view`` selector
    (``{"metadata": {key: value}}`` / ``{"tags": [...]}``), but the runtime delivers the FULL
    fold's sample_ids — the host data provider is expected to apply the selector. The process
    adapter applies it here, using a ``sample_id → metadata`` map, so each branch model fits and
    predicts ONLY its partition (else every branch covers the full fold → the native concat-merge
    handler rejects the overlap). A no-op when the view carries no ``branch_view``.
    """
    if view is None or sample_metadata is None:
        return
    branch_view = view.get("branch_view")
    if not branch_view:
        return
    selector = branch_view.get("selector") or {}
    view["sample_ids"] = [sample_id for sample_id in (view.get("sample_ids") or []) if _branch_view_keep(sample_id, selector, sample_metadata)]


def _branch_selector(task: dict[str, Any]) -> dict[str, Any] | None:
    """The branch_view selector shared by this task's data views (None for a non-branch node)."""
    for view in task.get("data_views", {}).values():
        branch_view = view.get("branch_view")
        if branch_view:
            return cast("dict[str, Any]", branch_view.get("selector") or {})
    return None


def _variant_overrides(task: dict[str, Any], node_id: str) -> dict[str, Any]:
    """Collect per-node generated param overrides from the task's variant, if any."""
    overrides: dict[str, Any] = {}
    variant = task.get("variant")
    if variant is None:
        return overrides
    for choice in variant.get("choices", {}).values():
        for override in choice.get("param_overrides", []):
            if override.get("node_id") == node_id:
                overrides.update(override.get("params", {}))
    return overrides


def _train_predict_ids(task: dict[str, Any], sample_metadata: dict[str, dict[str, Any]] | None = None) -> tuple[list[str], list[str]]:
    """(train_sample_ids, predict_sample_ids) for the task's phase, by partition.

    Each view's sample_ids are first restricted to its ``branch_view`` selector (a no-op for a
    non-branch node), so a separation-branch model fits/predicts only its partition.
    """
    phase = task["phase"]
    # A branch (fanned) node carries a branch_view: after filtering, a partition can be empty for a
    # given fold (e.g. a small group has no validation sample in one fold), which is legitimate — emit
    # empty predictions for that scope. A non-branch node with an empty fold is a real error.
    is_branch = _branch_selector(task) is not None
    if phase == "FIT_CV":
        train_view, val_view = _view_by_partition(task, "fold_train"), _view_by_partition(task, "fold_validation")
        _filter_by_branch_view(train_view, sample_metadata)
        _filter_by_branch_view(val_view, sample_metadata)
        return _sample_ids(train_view, allow_empty=is_branch), _sample_ids(val_view, allow_empty=is_branch)
    if phase == "REFIT":
        train_view = _view_by_partition(task, "full_train")
        _filter_by_branch_view(train_view, sample_metadata)
        train = _sample_ids(train_view, allow_empty=is_branch)
        return train, train
    if phase == "PREDICT":
        predict_view = _view_by_partition(task, "predict")
        _filter_by_branch_view(predict_view, sample_metadata)
        return [], _sample_ids(predict_view)
    raise ValueError(f"unsupported phase {phase!r}")


def _artifact_id(node_id: str, variant_label: str) -> str:
    return f"artifact:{node_id}:nirs4all:refit:{variant_label}"


def _upstream_x_chain(node_id: str, edges: list[dict[str, Any]] | None) -> list[str]:
    """Ordered upstream node ids feeding ``node_id``'s ``x`` input (furthest-upstream first).

    Walks the linear data-edge chain backward so the model node can apply the real
    preprocessing (e.g. SNV) before fitting — the process-adapter delivers only sample_ids
    per node, so cross-node feature flow is reconstructed here. Linear chains only.
    """
    incoming: dict[str, str] = {}
    for edge in edges or []:
        if edge["target"]["port_name"] == "x" and edge["contract"]["kind"] == "data":
            incoming[edge["target"]["node_id"]] = edge["source"]["node_id"]
    chain: list[str] = []
    current = incoming.get(node_id)
    while current is not None:
        chain.append(current)
        current = incoming.get(current)
    chain.reverse()
    return chain


def _stable_handle(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF


def _output_handles(task: dict[str, Any], handle: int) -> dict[str, Any]:
    """Output port handles by node kind (model→out+oof, join→out+prediction, else out+x_out)."""
    controller_id = task["node_plan"]["controller_id"]
    kind = task["node_plan"].get("kind")
    outputs = {"out": {"handle": handle, "kind": "data", "owner_controller": controller_id}}
    if kind in ("model", "tuner"):
        outputs["oof"] = {"handle": handle, "kind": "prediction", "owner_controller": controller_id}
    elif kind == "prediction_join":
        outputs["prediction"] = {"handle": handle, "kind": "prediction", "owner_controller": controller_id}
    else:
        outputs["x_out"] = {"handle": handle, "kind": "data", "owner_controller": controller_id}
    return outputs


def _build_result(task: dict[str, Any], predictions: list[dict[str, Any]], artifacts: list[dict[str, Any]], artifact_handles: dict[str, Any], regression_targets: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Assemble the schema-complete ``NodeResult`` the runtime validates (outputs + full lineage).

    ``regression_targets`` (the real ``y_true`` for the predicted samples) lets dag-ml score the
    predictions natively into the bundle's ScoreSet — emitted only when we predict labelled samples.
    """
    node_plan = task["node_plan"]
    node_id = node_plan["node_id"]
    phase = task["phase"]
    variant_label = task.get("variant_id") or "base"
    fold_label = task.get("fold_id") or "nofold"
    metrics = {"nirs4all_adapter": 1.0}
    if predictions and predictions[0]["values"]:
        flat = [row[0] for row in predictions[0]["values"]]
        metrics["prediction_mean"] = float(sum(flat) / len(flat))
    return {
        "node_id": node_id,
        "outputs": _output_handles(task, _stable_handle(f"{node_id}:{phase}:{variant_label}:{fold_label}")),
        "predictions": predictions,
        "shape_deltas": [],
        "artifacts": artifacts,
        "artifact_handles": artifact_handles,
        "regression_targets": regression_targets or [],
        "lineage": {
            "record_id": f"lineage:{node_id}:{phase}:{variant_label}:{fold_label}",
            "run_id": task["run_id"],
            "node_id": node_id,
            "phase": phase,
            "controller_id": node_plan["controller_id"],
            "controller_version": node_plan["controller_version"],
            "variant_id": task.get("variant_id"),
            "fold_id": task.get("fold_id"),
            "branch_path": task.get("branch_path", []),
            "input_lineage": [],
            "artifact_refs": artifacts,
            "params_fingerprint": node_plan["params_fingerprint"],
            "data_model_shape_fingerprint": None,
            "aggregation_policy_fingerprint": None,
            "seed": task.get("seed"),
            "unsafe_flags": [],
            "metrics": metrics,
        },
    }


def run_model_node(
    task: dict[str, Any],
    resolver: MaterializationResolver,
    node_lookup: Callable[[str], dict[str, Any]],
    model_store: MutableMapping[int, Any],
    edges: list[dict[str, Any]] | None = None,
    y_transform_node: dict[str, Any] | None = None,
    sample_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute a model-kind ``NodeTask`` with the real operator + real data; return a ``NodeResult``.

    When ``edges`` are supplied, the model node reconstructs its upstream X-transform chain
    (e.g. SNV→PLS) into a leakage-safe sklearn ``Pipeline`` fit on fold-train only — the
    process-adapter delivers only sample_ids per node, so the chain is applied here. A
    ``y_transform_node`` (a nirs4all ``y_processing`` step — a floating graph node with no edge
    to the model) is fit on fold-train ``y``, applied before fitting, and **inverse-transformed**
    over the predictions, exactly reproducing nirs4all target scaling.

    ``sample_metadata`` (``{wire_id: {col: value}}``) lets a separation-branch model honor its
    ``branch_view`` selector: train/predict ids are restricted to the partition, and the held-out
    TEST prediction (the refit best_rmse path) is likewise restricted to the branch's partition.
    """
    node_plan = task["node_plan"]
    node_id = node_plan["node_id"]
    controller_id = node_plan["controller_id"]
    phase = task["phase"]
    variant_label = task.get("variant_id") or "base"
    fold_label = task.get("fold_id") or "nofold"

    train_ids, predict_ids = _train_predict_ids(task, sample_metadata)
    artifact_id = _artifact_id(node_id, variant_label)
    artifact_handle = _stable_handle(artifact_id)

    # An empty fold-train can only happen for a separation-branch partition with no sample in this
    # fold (the branch_view filtered it out). The model cannot fit, so emit an empty NodeResult — the
    # other partitions cover the fold, and the native concat-merge reassembles full coverage.
    if phase != "PREDICT" and not train_ids:
        return _build_result(task, [], [], {}, [])

    estimator: Any
    y_transform: Any
    # BY_SOURCE (S4): a late-fusion-by-source branch model is bound to ONE source via
    # ``metadata.source_index`` — it sees only that source's feature block (all samples, that source's
    # columns), NOT the early-fusion concat or the MB-PLS block list. Read from the graph node so every
    # phase (incl. PREDICT, which reloads the estimator) selects the same source. ``None`` for any other
    # node (single-source / duplication / separation-by-metadata) → the unchanged concat/multi-block path.
    source_index = _source_index(node_lookup(node_id))
    # INTERMEDIATE FUSION (S5): a multi-block model (MB-PLS) consumes a LIST of per-source blocks, NOT
    # the early-fusion concat. ``multi_block`` is true ONLY when BOTH the model is a multi-block consumer
    # AND the dataset actually has >1 source — a single-source MB-PLS stays the early-fusion concat path
    # (the degenerate one-block list would be identical, so we keep the simpler concat for it). At PREDICT
    # the estimator is reloaded; its wrapper type tells us the persisted model was multi-block.
    if phase == "PREDICT":
        bundle = model_store[artifact_handle]
        estimator, y_transform = bundle["estimator"], bundle["y_transform"]
        multi_block = isinstance(estimator, _MultiBlockEstimator)
    else:
        model = route_graph_node(node_lookup(node_id), variant_overrides=_variant_overrides(task, node_id))
        upstream = [route_graph_node(node_lookup(upstream_id)) for upstream_id in _upstream_x_chain(node_id, edges)]
        multi_block = _is_multi_block_model(model) and resolver.is_multi_source()
        if multi_block:
            estimator = _MultiBlockEstimator(model, upstream)
        else:
            estimator = make_pipeline(*upstream, model) if upstream else model
        y_transform = route_graph_node(y_transform_node) if y_transform_node is not None else None
        # Fit views materialize TRAINING rows (FIT_CV fold_train, REFIT full_train): the view carries
        # BASE ids (dag-ml keeps the FoldSet a clean base-grain OOF partition) + include_augmented_train,
        # so the host expands each base id to base + its augmented children — those synthetic rows train.
        # A no-op when no augmentation ran. A child's target is its origin's y (resolve_targets keys by
        # the origin's sample_id). include_augmented=True so the leakage guard permits the children here.
        # FOLD-LOCAL augmentation: a fold's own children only join THAT fold's fit-train — FIT_CV uses
        # the task's fold_id, REFIT the "refit" key (the full-train pass). A child fit inside fold K's
        # train is therefore never expanded into fold L's fit, so a stateful augmenter cannot leak.
        fold_label = task.get("fold_id") if phase == "FIT_CV" else "refit"
        fit_ids = resolver.expand_with_augmented_children(train_ids, fold_label)
        # MULTI-BLOCK (S5): materialize the per-source blocks as a LIST (concat_source=False); the
        # wrapper applies the X-chain per block and fits ``model.fit([X1,X2,…], y)``. BY_SOURCE (S4):
        # materialize ONLY the bound source's block (one 2D matrix — late fusion by source). Otherwise the
        # single concat matrix (single-source OR early-fusion multi-source) is fit as before.
        x_train: Any
        if multi_block:
            x_train = [np.asarray(block, dtype=float) for block in resolver.resolve_feature_blocks(fit_ids, include_augmented=True)["blocks"]]
        elif source_index is not None:
            x_train = np.asarray(resolver.resolve_source_block(fit_ids, source_index, include_augmented=True)["values"], dtype=float)
        else:
            x_train = np.asarray(resolver.resolve_features(fit_ids, include_augmented=True)["values"], dtype=float)
        y_train = np.asarray(resolver.resolve_targets(resolver.target_sample_ids(fit_ids))["values"], dtype=float)
        # MULTI-TARGET (S0): resolve_targets returns a 2D (n, n_targets) block, so y_train is already 2D
        # — pass it through unraveled (PLSRegression(n_targets>1)/MultiOutputRegressor consume 2D y) and
        # scale per-column. SINGLE-TARGET stays 1D (byte-identical legacy reshape(-1,1).ravel()).
        if y_train.ndim > 1 and y_train.shape[1] > 1:
            y_fit = y_transform.fit_transform(y_train) if y_transform is not None else y_train
        else:
            y_fit = y_transform.fit_transform(y_train.reshape(-1, 1)).ravel() if y_transform is not None else y_train
        estimator.fit(x_train, y_fit)

    def _predict(ids: list[str], include_augmented: bool) -> list[list[float]]:
        x: Any
        if multi_block:
            x = [np.asarray(block, dtype=float) for block in resolver.resolve_feature_blocks(ids, include_augmented=include_augmented)["blocks"]]
        elif source_index is not None:
            x = np.asarray(resolver.resolve_source_block(ids, source_index, include_augmented=include_augmented)["values"], dtype=float)
        else:
            x = np.asarray(resolver.resolve_features(ids, include_augmented=include_augmented)["values"], dtype=float)
        pred = np.asarray(estimator.predict(x), dtype=float).reshape(len(ids), -1)
        scaled = np.asarray(y_transform.inverse_transform(pred), dtype=float).reshape(len(ids), -1) if y_transform is not None else pred
        return [[float(value) for value in row] for row in scaled]

    # What to predict: the phase's own partition; and — at REFIT — also the held-out TEST partition
    # so dag-ml scores the final model's test RMSE (nirs4all's best_rmse). The CV fold set does not
    # cover test, but dag-ml only scope-checks `validation` blocks (runtime validate_prediction_scope),
    # so a `test` block for the refit model is accepted and scored natively.
    #
    # The TEST block is emitted with `fold_id=None` (the OFF-FOLD convention dag-ml keys on): a REFIT
    # runs `fold_id=None`, and dag-ml's `reassemble_branch_merge_off_fold` /
    # `collect_off_fold_prediction_input` read each base producer's `partition=Test`, `fold_id=None`
    # block (`fold_id.is_none()`). Emitting `fold_id=None` therefore makes a base/branch model's TEST
    # prediction reach the concat/fusion merge producer (a scored Test block → best_rmse) and the
    # stacking meta-node (the `...oof:refit` off-fold input → it predicts the holdout). A single
    # (non-branch) model reaches no merge node — its `(test, None)` block is read straight back by
    # `_scores_to_run_result` for best_rmse, exactly like the prior `(test, "final")` block.
    #
    # include_augmented per spec mirrors the leakage guard: it is True ONLY when the predicted ids are
    # TRAINING rows — REFIT predicting its own full_train ("final"-train score). The predict ids are the
    # view's BASE ids, so resolve_features fetches base rows only (the "final"-train score is over base
    # train); the synthetic children influenced the FIT, never a scored holdout. FIT_CV's validation/OOF
    # view, the REFIT held-out TEST, and PREDICT are non-fit holdout views, so include_augmented=False
    # makes resolve_features REFUSE any augmented child there (the origin-boundary leakage guard).
    predict_is_train = phase == "REFIT"  # REFIT predict_ids == full_train (training rows); FIT_CV/PREDICT are holdout
    specs: list[tuple[list[str], str, str | None, bool]] = [
        (predict_ids, _PREDICTION_PARTITION[phase], task.get("fold_id") if phase == "FIT_CV" else None, predict_is_train)
    ]
    if phase == "REFIT":
        test_ids = resolver.partition_wire_ids("test")
        # A separation-branch refit model only ever trained on its partition, so its TEST prediction
        # must be restricted to that partition too (the test ids are fetched directly, not via a view).
        selector = _branch_selector(task)
        if selector is not None and sample_metadata is not None:
            test_ids = [sample_id for sample_id in test_ids if _branch_view_keep(sample_id, selector, sample_metadata)]
        if test_ids:
            specs.append((test_ids, "test", None, False))

    # One prediction block per spec, each paired 1:1 with an exactly-matching y_true block (dag-ml
    # scoring requires target units == prediction units). dag-ml matches block↔target by unit set.
    # Skip an empty spec (a branch partition with no validation sample in this fold).
    predictions: list[dict[str, Any]] = []
    regression_targets: list[dict[str, Any]] = []
    for spec_ids, partition, spec_fold, spec_include_augmented in specs:
        if not spec_ids:
            continue
        # MULTI-TARGET (S0): resolve_targets returns list-of-rows (n, n_targets); _predict already builds
        # 2D rows, so both blocks widen to k columns and carry per-target names (rmse:y0/rmse:y1 keys +
        # macro-mean). SINGLE-TARGET stays a flat list → [[v]] rows + ["y"] (BYTE-IDENTICAL legacy emit).
        true_y = resolver.resolve_targets(spec_ids)["values"]
        multi_target = bool(true_y) and isinstance(true_y[0], list)
        names = [f"y{i}" for i in range(len(true_y[0]))] if multi_target else ["y"]
        true_values = [[float(value) for value in row] for row in true_y] if multi_target else [[float(value)] for value in true_y]
        predictions.append(
            {
                "prediction_id": f"pred:{node_id}:{phase}:{variant_label}:{fold_label}:{partition}",
                "producer_node": node_id,
                "partition": partition,
                "fold_id": spec_fold,
                "sample_ids": spec_ids,
                "values": _predict(spec_ids, spec_include_augmented),
                "target_names": names,
            }
        )
        regression_targets.append(
            {
                "level": "sample",
                "unit_ids": [{"level": "sample", "id": sample_id} for sample_id in spec_ids],
                "values": true_values,
                "target_names": names,
            }
        )

    artifacts: list[dict[str, Any]] = []
    artifact_handles: dict[str, Any] = {}
    if phase == "REFIT":
        model_store[artifact_handle] = {"estimator": estimator, "y_transform": y_transform}
        artifacts.append({"id": artifact_id, "kind": "sklearn_estimator", "controller_id": controller_id, "backend": "joblib"})
        artifact_handles[artifact_id] = {"handle": artifact_handle, "kind": "model", "owner_controller": controller_id}

    return _build_result(task, predictions, artifacts, artifact_handles, regression_targets)


def _meta_feature_matrix(specs: list[dict[str, Any]], node_id: str) -> tuple[list[str], np.ndarray]:
    """Build ``(sample_ids, X_meta)`` from base prediction-input specs, concatenated per producer.

    One column block per base producer in the order ``specs`` is given (the caller passes them in the
    deterministic sorted-producer order), aligned by ``sample_id`` — the first spec's sample order is
    canonical and every other spec must cover the same samples (a missing one is a hard error, never a
    silent zero). Mirrors dag-ml's own ``join_oof_features`` column ordering.
    """
    sample_ids = list(specs[0]["sample_ids"])
    rows_by_sample: dict[str, list[float]] = {sample_id: [] for sample_id in sample_ids}
    for spec in specs:
        spec_rows = {sample_id: [float(value) for value in row] for sample_id, row in zip(spec["sample_ids"], spec["values"], strict=True)}
        for sample_id in sample_ids:
            row = spec_rows.get(sample_id)
            if row is None:
                raise ValueError(f"meta-model node {node_id!r}: base producer {spec.get('producer_node')!r} is missing prediction for sample {sample_id!r}")
            rows_by_sample[sample_id].extend(row)
    return sample_ids, np.asarray([rows_by_sample[sample_id] for sample_id in sample_ids], dtype=float)


def _ordered_oof_specs(prediction_inputs: dict[str, Any], *, suffix: str | None) -> list[dict[str, Any]]:
    """The meta-node's base specs in canonical producer order, selecting one delivery kind.

    dag-ml keys each base producer's Validation OOF under ``"{producer}.{port}"`` and its off-fold
    (REFIT-test / PREDICT) block under ``"{producer}.{port}:refit"`` / ``":predict"`` (see runtime
    ``collect_off_fold_prediction_input``). ``suffix=None`` selects the Validation OOF inputs (FIT_CV
    training + the REFIT fit pool); ``suffix="refit"`` / ``"predict"`` selects the off-fold inputs.

    Ordering is by the **base OOF key** (the suffix stripped) so the meta-feature columns line up
    one-for-one with the FIT_CV matrix regardless of which delivery kind is selected — the column for
    ``branch:0.node:0`` is always first whether read from ``…oof`` or ``…oof:refit``.
    """
    tag = f":{suffix}" if suffix is not None else None
    selected: dict[str, dict[str, Any]] = {}
    for key, spec in prediction_inputs.items():
        if tag is None:
            if not (key.endswith(":refit") or key.endswith(":predict")):
                selected[key] = spec
        elif key.endswith(tag):
            selected[key[: -len(tag)]] = spec
    return [selected[base_key] for base_key in sorted(selected)]


def run_meta_model_node(
    task: dict[str, Any],
    resolver: MaterializationResolver,
    node_lookup: Callable[[str], dict[str, Any]],
    model_store: MutableMapping[int, Any],
) -> dict[str, Any]:
    """Execute a STACKING meta-model node from its base branches' OOF + off-fold predictions (#10).

    The meta-node is a ``model``-kind node whose ``x`` is not data but the base branches' predictions,
    delivered by dag-ml as ``prediction_inputs`` (Option A: each spec carries ``values`` aligned 1:1
    with ``sample_ids``). dag-ml delivers two delivery kinds, keyed by suffix
    (:func:`_ordered_oof_specs`):

    * ``"{producer}.{port}"`` — the base producer's **Validation OOF** (the ``requires_oof`` edge
      refuses any train block, so it is leakage-safe). This is what the meta-learner trains on.
    * ``"{producer}.{port}:refit"`` / ``":predict"`` — the base producer's **off-fold** block (REFIT
      held-out Test / PREDICT new-data Final, ``fold_id=None``), delivered ONLY in REFIT/PREDICT and
      used ONLY to PREDICT the holdout, never to train.

    The meta-feature matrix has one column block per base producer in **deterministic producer order**
    (sorted base key), aligned by ``sample_id``, mirroring dag-ml's ``join_oof_features``.

    * FIT_CV (fold K): build ``X_meta`` from the fold-K Validation OOF, fit the MetaModel on
      ``(X_meta, y_train)``, predict those rows, emit the meta-model's own OOF
      (``partition=validation``, ``fold_id=foldK``) + targets. The cross-fold OOF average of the meta
      producer is ``cv_best_score``.
    * REFIT: fit + persist the meta-model on the FULL Validation OOF (all folds), then build the TEST
      meta-feature matrix from the base ``:refit`` off-fold inputs (the base models' held-out Test
      predictions), predict the holdout, and emit a scored ``(test, fold_id=None)`` block + targets so
      dag-ml scores ``best_rmse``. LEAKAGE INVARIANT: the meta-model is fit on Validation OOF ONLY (the
      ``:refit`` Test predictions never enter the fit); the Test meta-features come from the base
      models' Test predictions, never their OOF/train.
    """
    node_plan = task["node_plan"]
    node_id = node_plan["node_id"]
    controller_id = node_plan["controller_id"]
    phase = task["phase"]
    variant_label = task.get("variant_id") or "base"
    fold_label = task.get("fold_id") or "nofold"

    prediction_inputs = task.get("prediction_inputs") or {}
    if not prediction_inputs:
        raise ValueError(f"meta-model node {node_id!r} received no prediction_inputs (no base branch OOF)")

    if phase == "PREDICT":
        # PREDICT replays the persisted meta-learner over the base producers' PREDICT-set predictions
        # (the `:predict` off-fold inputs); there are no Validation OOF inputs to fit on. dag-ml only
        # delivers `:predict` inputs here when each base producer emitted a PREDICT block, so a missing
        # one is a real wiring error.
        artifact_handle = _stable_handle(_artifact_id(node_id, variant_label))
        estimator = model_store[artifact_handle]["estimator"]
        predict_specs = _ordered_oof_specs(prediction_inputs, suffix="predict")
        if not predict_specs:
            raise ValueError(f"meta-model node {node_id!r} REFIT/PREDICT received no `:predict` off-fold inputs (no base predict-set predictions)")
        sample_ids, x_meta = _meta_feature_matrix(predict_specs, node_id)
        pred = np.asarray(estimator.predict(x_meta), dtype=float).reshape(len(sample_ids), -1)
        predictions = [_meta_prediction_block(node_id, phase, variant_label, fold_label, "final", None, sample_ids, pred)]
        true_y = resolver.resolve_targets(sample_ids)["values"]
        regression_targets = [_meta_target_block(sample_ids, [float(value) for value in true_y])]
        return _build_result(task, predictions, [], {}, regression_targets)

    # FIT_CV + REFIT both fit on Validation OOF. The OOF specs (no `:refit`/`:predict` suffix) are the
    # meta-learner's training features; in FIT_CV they are this fold's OOF, in REFIT the full OOF.
    oof_specs = _ordered_oof_specs(prediction_inputs, suffix=None)
    if not oof_specs:
        raise ValueError(f"meta-model node {node_id!r} received no Validation OOF inputs to fit on")
    sample_ids, x_meta = _meta_feature_matrix(oof_specs, node_id)
    y_meta = np.asarray(resolver.resolve_targets(sample_ids)["values"], dtype=float)

    artifact_id = _artifact_id(node_id, variant_label)
    artifact_handle = _stable_handle(artifact_id)
    fit_estimator: Any = route_graph_node(node_lookup(node_id), variant_overrides=_variant_overrides(task, node_id))
    fit_estimator.fit(x_meta, y_meta)

    fold_predictions: list[dict[str, Any]] = []
    fold_targets: list[dict[str, Any]] = []
    if phase == "FIT_CV":
        # The meta-model's own OOF for this fold's samples (scored → cv_best_score).
        pred = np.asarray(fit_estimator.predict(x_meta), dtype=float).reshape(len(sample_ids), -1)
        fold_predictions.append(_meta_prediction_block(node_id, phase, variant_label, fold_label, "validation", task.get("fold_id"), sample_ids, pred))
        fold_targets.append(_meta_target_block(sample_ids, [float(value) for value in y_meta]))

    artifacts: list[dict[str, Any]] = []
    artifact_handles: dict[str, Any] = {}
    if phase == "REFIT":
        model_store[artifact_handle] = {"estimator": fit_estimator, "y_transform": None}
        artifacts.append({"id": artifact_id, "kind": "sklearn_estimator", "controller_id": controller_id, "backend": "joblib"})
        artifact_handles[artifact_id] = {"handle": artifact_handle, "kind": "model", "owner_controller": controller_id}

        # Predict the held-out TEST set from the base producers' `:refit` off-fold predictions (their
        # held-out Test predictions). The meta-model was fit on Validation OOF ONLY (above), so this is
        # leakage-safe: the Test meta-features come from base Test predictions, never OOF/train. Emit a
        # `(test, fold_id=None)` block so dag-ml scores best_rmse (off-fold convention, like a base model).
        test_specs = _ordered_oof_specs(prediction_inputs, suffix="refit")
        if test_specs:
            test_ids, x_test = _meta_feature_matrix(test_specs, node_id)
            test_pred = np.asarray(fit_estimator.predict(x_test), dtype=float).reshape(len(test_ids), -1)
            fold_predictions.append(_meta_prediction_block(node_id, phase, variant_label, fold_label, "test", None, test_ids, test_pred))
            test_true = resolver.resolve_targets(test_ids)["values"]
            fold_targets.append(_meta_target_block(test_ids, [float(value) for value in test_true]))

    return _build_result(task, fold_predictions, artifacts, artifact_handles, fold_targets)


def _meta_prediction_block(node_id: str, phase: str, variant_label: str, fold_label: str, partition: str, fold_id: str | None, sample_ids: list[str], values: np.ndarray) -> dict[str, Any]:
    """A meta-node prediction block for one partition (validation OOF / test / final)."""
    return {
        "prediction_id": f"pred:{node_id}:{phase}:{variant_label}:{fold_label}:{partition}",
        "producer_node": node_id,
        "partition": partition,
        "fold_id": fold_id,
        "sample_ids": sample_ids,
        "values": [[float(value) for value in row] for row in values],
        "target_names": ["y"],
    }


def _meta_target_block(sample_ids: list[str], true_y: list[float]) -> dict[str, Any]:
    """The y_true block paired 1:1 with a meta-node prediction block (dag-ml scores against it)."""
    return {
        "level": "sample",
        "unit_ids": [{"level": "sample", "id": sample_id} for sample_id in sample_ids],
        "values": [[value] for value in true_y],
        "target_names": ["y"],
    }


def run_node(
    task: dict[str, Any],
    resolver: MaterializationResolver,
    node_lookup: Callable[[str], dict[str, Any]],
    model_store: MutableMapping[int, Any],
    edges: list[dict[str, Any]] | None = None,
    y_transform_node: dict[str, Any] | None = None,
    sample_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dispatch a ``NodeTask`` by node kind.

    A STACKING meta-model node (a ``model``-kind node bound to ``controller:nirs4all.meta_model``,
    recognised by its ``prediction_inputs`` of base-branch OOF) runs the meta-model over those OOF
    meta-features. Other ``model``/``tuner`` nodes execute and (with ``edges``/``y_transform_node``)
    apply their upstream X-transform chain + target scaling; ``transform``/``y_transform`` are
    passthrough output-handles (the model node reconstructs the chain), so each preprocessing step is
    applied exactly once, at the model node. A ``prediction_join`` (concat-merge) node is also a
    passthrough here — the dag-ml runtime reassembles it natively before any controller runs.

    ``sample_metadata`` (``{wire_id: {col: value}}``) honors separation-branch ``branch_view``
    selectors so each fanned model node sees only its partition.
    """
    node_plan = task["node_plan"]
    kind = node_plan["kind"]
    if kind in ("model", "tuner"):
        if node_plan["controller_id"] == _META_MODEL_CONTROLLER_ID:
            return run_meta_model_node(task, resolver, node_lookup, model_store)
        return run_model_node(task, resolver, node_lookup, model_store, edges, y_transform_node, sample_metadata)
    return _build_result(task, [], [], {})
