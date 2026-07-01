"""Native dag-ml run paths for the host backend.

One ``_run_*`` function per supported pipeline shape — single-variant CV+refit, native param-model
generation, repetition (group-aware), rep-fusion reshape, sample-augmentation (global / fold-local),
and the separation / by_source / duplication / stacking branch fan-outs. Each assembles the compat or
canonical DSL, drives dag-ml-cli through the process adapter, and maps the native ``bundle.scores``
into a :class:`~nirs4all.api.result.RunResult`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.api.result import RunResult
from nirs4all.core.metrics import eval_list, get_default_metrics, is_higher_better
from nirs4all.data.ensemble_utils import EnsembleUtils
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.dagml_bridge import controller_manifests

from .cli_runner import assemble_constrained_cv_refit_dsl, assemble_cv_refit_dsl
from .detect import _is_augmentation_step, _is_constrained_operator_generator, _is_rep_fusion_step, _is_unconstrained_operator_generator
from .envelope import build_envelope
from .errors import DagMlUnsupported, _raise_run_failure, _reject_multi_model
from .folds import _build_folds, _build_group_folds, _repetition_grain, _split_pool
from .identity import mint_identity
from .in_process_runner import run_cv_refit_bundle_router as run_cv_refit_bundle
from .result import _frames_by_variant, _native_variant_config_map, _project_operator_sweep, _scores_to_run_result
from .steps import _apply_model_params, _apply_plain_model_params, _assert_supported_operators, _legacy_skips_refit, _model_name, _split_pipeline, _supported_body_steps


class DuplicationBranchMergeTransformer:
    """Sklearn-compatible fold-local transformer for legacy duplication feature merges.

    The native concrete path fits this transformer inside each dag-ml fold. Each branch transform chain is
    therefore fit only on that fold's training rows before the downstream estimator sees the concatenated
    branch feature matrix. For ``merge_mode="all"``, branch-local models are also fit after their branch
    transforms and their prediction columns are appended after all branch feature blocks, matching legacy's
    ``MergeController`` collection order (features first, predictions second).
    """

    def __init__(self, branches: list[dict[str, Any]], merge_mode: str = "features") -> None:
        self.branches = branches
        self.merge_mode = merge_mode

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002 - sklearn signature
        return {"branches": self.branches, "merge_mode": self.merge_mode}

    def set_params(self, **params: Any) -> DuplicationBranchMergeTransformer:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: Any, y: Any = None) -> DuplicationBranchMergeTransformer:
        self._validate()
        self._fitted_branches = [self._fit_branch(np.asarray(X), y, branch) for branch in self.branches]
        return self

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> np.ndarray:  # noqa: ARG002 - sklearn signature
        return self.fit(X, y).transform(X)

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X)
        feature_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        for fitted in self._fitted_branches:
            branch_features = self._transform_branch_features(X_arr, fitted["transforms"])
            feature_parts.append(branch_features)
            model = fitted.get("model")
            if self.merge_mode == "all" and model is not None:
                prediction_parts.append(self._as_2d(model.predict(branch_features)))
        return np.concatenate([*feature_parts, *prediction_parts], axis=1)

    def _validate(self) -> None:
        if self.merge_mode not in ("features", "all"):
            raise ValueError(f"unsupported duplication branch merge_mode {self.merge_mode!r}")

    @staticmethod
    def _as_2d(values: Any) -> np.ndarray:
        arr = np.asarray(values)
        return arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)

    @staticmethod
    def _instantiate(spec: dict[str, Any]) -> Any:
        from nirs4all.pipeline.dagml.operator_routing import route_operator

        return route_operator(spec["kind"], spec["class"], spec.get("params") or {})

    def _fit_branch(self, X: np.ndarray, y: Any, branch: dict[str, Any]) -> dict[str, Any]:
        transforms = []
        current = X
        model = None
        for spec in branch["steps"]:
            operator = self._instantiate(spec)
            if spec["kind"] == "transform":
                if model is not None:
                    raise ValueError("branch transforms after a branch model are unsupported for duplication merge")
                operator.fit(current, y)
                current = np.asarray(operator.transform(current))
                transforms.append(operator)
            elif spec["kind"] == "model" and self.merge_mode == "all":
                if model is not None:
                    raise ValueError("multiple branch-local models are unsupported for duplication merge='all'")
                operator.fit(current, y)
                model = operator
            else:
                raise ValueError(f"unsupported branch merge step kind {spec['kind']!r}")
        if self.merge_mode == "all" and model is None:
            raise ValueError("duplication merge='all' requires every branch to contain one branch-local model")
        return {"transforms": transforms, "model": model}

    def _transform_branch_features(self, X: np.ndarray, transforms: list[Any]) -> np.ndarray:
        current = X
        for transform in transforms:
            current = np.asarray(transform.transform(current))
        return self._as_2d(current)


def _native_param_winner_config_name(
    refit_artifacts: list[dict[str, Any]] | None,
    variant_config_names: list[str] | None,
    variant_model_params: list[dict[str, Any]] | None,
) -> str | None:
    """The WINNING param-sweep variant's legacy ``config_name``, recovered BY CONTENT, else ``None``.

    The native run refits the true CV-best variant but emits an opaque variant hash with NO params in the
    reports, so the winner's specific ``config_name`` cannot be read off the ScoreSet. The fitted REFIT
    estimator IS authoritative, though — it carries the winning param VALUES. We unwrap its bare model
    (the last step of the refit sklearn ``Pipeline``, else the estimator itself), then find the variant
    whose recorded swept-param values (``variant_model_params[i]``, aligned 1:1 with
    ``variant_config_names``) all match the winner's — returning ``variant_config_names[i]``. Only the
    SWEPT keys are compared (the keys present in the per-variant param dict), so a default param the
    estimator also carries does not break the match. Returns ``None`` (→ the positional ``names[0]``
    fallback) when there is no refit estimator, no per-variant params, or no unambiguous single match —
    never a wrong label.
    """
    from sklearn.pipeline import Pipeline

    if not refit_artifacts or not variant_config_names or not variant_model_params:
        return None
    if len(variant_model_params) != len(variant_config_names):
        return None
    estimator = refit_artifacts[0].get("estimator")
    if estimator is None:
        return None
    model = estimator.steps[-1][1] if isinstance(estimator, Pipeline) and estimator.steps else estimator
    if not hasattr(model, "get_params"):
        return None
    winner_params = model.get_params()
    matches = {
        config_name
        for config_name, variant_params in zip(variant_config_names, variant_model_params, strict=True)
        if variant_params and all(key in winner_params and winner_params[key] == value for key, value in variant_params.items())
    }
    # A single matched config_name keys the winner. Distinct duplicate variants that share the SAME
    # swept-param values collapse to ONE config_name (the display hash is the param content), so a grid
    # like `scale:[True,False,True]` still resolves unambiguously. Only a genuine multi-NAME match (an
    # ill-posed grid where the winner's params match two DIFFERENT names) or no match falls back to the
    # positional `names[0]` pairing rather than pick arbitrarily.
    return next(iter(matches)) if len(matches) == 1 else None


def _run_native_generation(
    pipeline: list[Any],
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    metric: str,
    task_type: str,
    cv_pool: list[int] | None = None,
    excluded: set[int] | None = None,
    tags_by_sample: dict[int, list[str]] | None = None,
    dataset_pickle: str | None = None,
    config_name: str = "",
    variant_config_names: list[str] | None = None,
    variant_model_params: list[dict[str, Any]] | None = None,
    random_state: int | None = None,
) -> RunResult:
    """Run a param-level model sweep as ONE native dag-ml generation + SELECT + refit run.

    The model step keeps its generator dict so the bridge lowers it to native ``generators``; we
    apply only the plain (non-generator) sibling params to the model, never the sweep. dag-ml
    expands the variants, scores EACH by its cross-fold OOF ``metric`` and surfaces every variant's
    validation reports (#55), refitting only the winner — ``bundle.scores`` is mapped to a RunResult
    as the full PER-VARIANT legacy table (``variant_config_names`` carries the ordered legacy per-variant
    config names), so a sweep's num_predictions matches legacy.

    ``cv_pool`` is the CV sample-int universe (the de-excluded pool in legacy mode, the full train
    in opt-in mode); ``excluded`` is marked in the envelope only in opt-in (``keep_in_oof=True``).
    """
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    _reject_multi_model(steps)
    _assert_supported_operators(steps)
    steps = _apply_plain_model_params(steps)

    identity = mint_identity(spectro)
    pool = list(cv_pool) if cv_pool is not None else spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, excluded or set())
    envelope = build_envelope(spectro, identity, sample_ints=pool, excluded_sample_ints=excluded or None, tags_by_sample=tags_by_sample)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml engine run failed")

    # A native param sweep varies one model's params only (same model class across variants), so only the
    # per-variant config_name map is needed — model_name is shared, so the projection's scalar fallback is
    # correct. The map keys each opaque native variant_id to its legacy expand config name. A NON-degenerate
    # sweep (`_grid_`) selects the true CV-best, whose config_name is the WINNING variant's name (NOT
    # index 0): recover it by matching the winner's refit model params against the per-variant model params
    # (aligned with `variant_config_names`), so the winner is content-paired exactly like the operator path.
    winner_config_name = _native_param_winner_config_name(outcome["refit_artifacts"], variant_config_names, variant_model_params)
    variant_config_map = _native_variant_config_map(outcome["scores"], variant_config_names, winner_config_name) if variant_config_names else None

    # FILL the strict direct-block rows with this run's per-sample y_pred/y_true/sample_indices (the
    # winner's refit `(final, train)` + `(final, test)` + per-fold OOF; each loser's per-fold OOF). dag-ml
    # emits ONE bundle here with every variant's reports correctly tagged (only the winner refits), so the
    # NodeResult frames key per-variant exactly like the reports — bucket them with `_frames_by_variant`
    # under each frame's own `variant_id` (the winner's untagged OOF-avg frame falls back to the winner id,
    # the `(final, None)`-report owner). The projection then reads each row's arrays from ITS OWN variant's
    # blocks (no cross-variant y_pred leakage), exactly as the operator-sweep / single-pipeline paths do.
    winner_variant_id = next(
        (report.get("variant_id") for report in (outcome["scores"] or {}).get("reports", []) if report["partition"] == "final" and report.get("fold_id") is None),
        None,
    )
    results_by_variant = _frames_by_variant(outcome["results"], winner_variant_id)
    return _scores_to_run_result(
        outcome["scores"], spectro.name, _model_name(steps), metric, task_type, config_name=config_name, variant_config_names=variant_config_map, skip_refit=_legacy_skips_refit(splitter), results_by_variant=results_by_variant, identity=identity, refit_artifacts=outcome["refit_artifacts"]
    )


def _run_native_operator_generation(
    pipeline: list[Any],
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    metric: str,
    task_type: str,
    cv_pool: list[int] | None = None,
    excluded: set[int] | None = None,
    tags_by_sample: dict[int, list[str]] | None = None,
    dataset_pickle: str | None = None,
    config_name: str = "",
    variant_config_names: list[str] | None = None,
    random_state: int | None = None,
) -> RunResult:
    """Run a FLAT-SINGLE operator ``_or_`` as ONE native dag-ml operator-SELECT + refit run (#23 Phase 7).

    The generator sits on a TRANSFORM step (the model is concrete): the bridge lowers the ``_or_`` to a
    compat ``Generator`` step, dag-ml's ``compile_operator_variant_models`` expands the operator-variant
    models, and the in-process binding scores EACH choice by its cross-fold OOF ``metric``, refits ONLY the
    winner, and surfaces every variant's validation reports — each stamped with the cross-language
    ``variant_label`` content fingerprint (the WINNER too). ``bundle.scores`` is mapped to the full
    PER-VARIANT legacy table, keyed CONTENT-WISE (``variant_label`` → ``config_name``), so a sweep's
    num_predictions + winner identity match the Python-expand path.

    Differs from :func:`_run_native_generation` (the param-sweep template) in three ways: (1) NO
    ``_apply_plain_model_params`` — the generator is on the transform, the model is already concrete; (2)
    the union graph has ONE model node PER choice (Mechanism B namespaces them), so EVERY model node needs
    a data binding (mirroring the separation-branch fan-out), not just the first; (3) the variant-config
    map is keyed by ``variant_label``
    (:func:`~nirs4all.pipeline.dagml.result._native_operator_variant_config_map`), not the positional zip
    a param sweep uses.

    FALLBACK CONTRACT (the inner Python-expand fallback fires on LOWERING-UNSUPPORTED ONLY): every
    lowering step — splitter check, the bridge ``_or_`` lowering (:func:`assemble_cv_refit_dsl`), and the
    per-choice ``variant_label`` fingerprinting (:func:`_native_operator_config_by_label`) — runs inside a
    narrow guard that converts a lowering refusal (``NotImplementedError`` / :class:`DagMlUnsupported`) into
    a DISTINCT :class:`~.errors._OperatorLoweringUnsupported` sentinel, which the routing branch catches →
    Python-expand. Everything AFTER the guard (compile / run / result-mapping) is OUTSIDE it, so a genuine
    runtime error there PROPAGATES — including a runtime ``DagMlUnsupported`` from :func:`_raise_run_failure`
    (a non-zero run classified ``error_kind == "unsupported"``), which is a REAL coverage boundary the host
    did not pre-check, NOT a lowering gap. The routing branch catches ONLY the sentinel, so that runtime
    ``DagMlUnsupported`` is never silently reclassified as lowering-unsupported and masked.
    """
    import dag_ml

    from .cli_runner import data_bindings_for_nodes
    from .errors import _OperatorLoweringUnsupported
    from .result import _frames_by_variant, _native_operator_config_by_label, _native_operator_variant_config_map

    # --- LOWERING PHASE (narrow fallback scope) ---------------------------------------------------------
    # Only a LOWERING refusal demotes to Python-expand. The bridge `_or_` lowering raises NotImplementedError;
    # the label fingerprinting raises DagMlUnsupported; both are lowering-unsupported, so convert them to the
    # DISTINCT `_OperatorLoweringUnsupported` sentinel the routing branch catches. A non-lowering error (e.g.
    # an internal invariant) is NOT a coverage gap — let it propagate.
    try:
        from nirs4all.pipeline.config._generator.keywords import GENERATION_KEYWORDS

        steps, splitter = _split_pipeline(pipeline)
        if splitter is None:
            raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
        _reject_multi_model(steps)
        # Reject UP FRONT the transform-side operators the X-chain cannot run — but skip the generator step
        # itself (its bare-operator choices were already routability-gated by the predicate, and are
        # fingerprinted below). The concrete model is validated by its own fit, like every other path.
        _assert_supported_operators([step for step in steps if not (isinstance(step, dict) and GENERATION_KEYWORDS & set(step))])

        identity = mint_identity(spectro)
        pool = list(cv_pool) if cv_pool is not None else spectro.index_column("sample", {"partition": "train"})
        folds = _build_folds(splitter, spectro, pool, excluded or set())
        envelope = build_envelope(spectro, identity, sample_ints=pool, excluded_sample_ints=excluded or None, tags_by_sample=tags_by_sample)

        # The DSL carries the lowered Generator on the transform position. A FLAT-SINGLE bare `_or_` lowers
        # via the compat fusion (assemble_cv_refit_dsl → pipeline_to_dsl); a CONSTRAINED `_or_`-pick /
        # `_cartesian_` OR an UNCONSTRAINED pick/arrange/`_cartesian_` (ADR-17 item 5 slice C) lowers each
        # survivor into ONE model-terminated canonical Generator branch (assemble_constrained_cv_refit_dsl):
        # both produce multi-op survivor SEQUENCES the compat fusion's operator-variant compiler refuses for a
        # model-free choice, and the native generator carries the pick/arrange selectors (+ constraints when
        # present) so dag-ml prunes the byte-identical set. Both raise from the bridge on an unlowerable shape.
        # Compute the content-keyed {variant_label -> config_name} map HERE too (it fingerprints each survivor
        # — a lowering step that can raise DagMlUnsupported on a non-finite / non-JSON param), so a label
        # failure demotes BEFORE the run.
        if _is_constrained_operator_generator(pipeline) or _is_unconstrained_operator_generator(pipeline):
            dsl = assemble_constrained_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))
        else:
            dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))
        config_by_label = _native_operator_config_by_label(steps, variant_config_names or [])
    except (DagMlUnsupported, NotImplementedError) as exc:
        raise _OperatorLoweringUnsupported(f"operator generator lowering unsupported, demoting to Python expand: {exc}") from exc

    # --- COMPILE / RUN / RESULT-MAPPING (errors PROPAGATE — never reclassified as a coverage gap) --------
    # The union graph compiles ONE model node per `_or_` choice (Mechanism B namespacing), so bind EVERY
    # model node — a single binding on the first node would leave the other choices' model nodes with empty
    # data_views (the separation-branch fan-out path solves the same multi-model-node binding the same way).
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)

    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml operator-generation run failed")

    # CONTENT-keyed map: every report carries its choice's `variant_label`, so each variant's `variant_id`
    # resolves to its legacy `config_name` by the pre-computed fingerprint map (NOT a positional zip). The
    # winner's reports (val + refit) are threaded under its OWN variant_id so its per-fold + final/test rows
    # carry real per-sample y_pred (2a-i/ii).
    scores = outcome["scores"]
    variant_config_map = _native_operator_variant_config_map(scores, config_by_label)
    winner_variant_id = next(
        (report.get("variant_id") for report in (scores or {}).get("reports", []) if report["partition"] == "final" and report.get("fold_id") is None),
        None,
    )
    # Split the surfaced frames PER VARIANT so a LOSER variant's per-fold val rows fill from ITS OWN
    # validation (OOF) predictions, not just the winner's. dag-ml surfaces each loser's per-fold val
    # blocks re-tagged with the loser's variant_id (top-level in-process / `lineage.variant_id`
    # subprocess); the winner's real per-fold + refit frames carry the winner's id. Untagged frames
    # (the winner's OOF-average frame) default to the winner. NO cross-variant leakage: a frame routes
    # to its OWN variant only.
    results_by_variant = _frames_by_variant(outcome["results"], winner_variant_id) if winner_variant_id is not None else None
    return _scores_to_run_result(
        scores, spectro.name, _model_name(steps), metric, task_type, config_name=config_name, variant_config_names=variant_config_map or None, skip_refit=_legacy_skips_refit(splitter), results_by_variant=results_by_variant, identity=identity, refit_artifacts=outcome["refit_artifacts"]
    )


def _run_concrete_scores(
    pipeline: Any,
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    cv_pool: list[int] | None = None,
    excluded: set[int] | None = None,
    tags_by_sample: dict[int, list[str]] | None = None,
    dataset_pickle: str | None = None,
    random_state: int | None = None,
) -> tuple[dict[str, Any], str, bool, list[dict[str, Any]], Any, list[dict[str, Any]]]:
    """Run one concrete (generator-free) pipeline through dag-ml-cli; return ``(scores, model_name, skip_refit, results, identity, refit_artifacts)``.

    The raw native ScoreSet + the model label + the legacy refit-gate flag + the per-node ``NodeResult``
    frames + the minted ``IdentityMap`` + the captured fitted REFIT estimators. The first three feed both
    the single-variant projection and the operator-sweep COMBINE (legacy num_predictions parity) — see
    :func:`~nirs4all.pipeline.dagml.run_backend._dispatch_run`. ``results`` + ``identity`` let the
    single-variant projection fill the strict direct-block per-sample y_pred/y_true/sample_indices (2a-i) —
    the sweep path ignores them (its per-variant value fill is 2a-ii/2a-iii). ``refit_artifacts`` (P3 Slice
    2c-i) is the run's captured fitted REFIT models (``outcome["refit_artifacts"]``; empty for the
    subprocess mechanism), forwarded to the native-results writer. :func:`_run_concrete`
    wraps this for the single-variant path. ``skip_refit`` is
    :func:`~nirs4all.pipeline.dagml.steps._legacy_skips_refit` on the splitter — true when an all-default
    splitter serializes to a bare string, the case where legacy skips the refit and emits no ``(final, *)``
    rows. ``cv_pool`` is the CV sample-int universe (de-excluded pool in legacy mode, full train in opt-in
    mode); ``excluded`` is marked in the envelope only in the opt-in (``keep_in_oof=True``).
    """
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    _reject_multi_model(steps)
    _assert_supported_operators(steps)
    steps = _apply_model_params(steps)

    identity = mint_identity(spectro)
    pool = list(cv_pool) if cv_pool is not None else spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, excluded or set())
    envelope = build_envelope(spectro, identity, sample_ints=pool, excluded_sample_ints=excluded or None, tags_by_sample=tags_by_sample)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml engine run failed")

    return outcome["scores"], _model_name(steps), _legacy_skips_refit(splitter), outcome["results"], identity, outcome["refit_artifacts"]


def _run_concrete(
    pipeline: Any,
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    metric: str = "rmse",
    task_type: str = "regression",
    cv_pool: list[int] | None = None,
    excluded: set[int] | None = None,
    tags_by_sample: dict[int, list[str]] | None = None,
    dataset_pickle: str | None = None,
    config_name: str = "",
    random_state: int | None = None,
) -> RunResult:
    """Run one concrete (generator-free) pipeline through dag-ml-cli; map its native scores.

    ``cv_pool`` is the CV sample-int universe (de-excluded pool in legacy mode, full train in opt-in
    mode); ``excluded`` is marked in the envelope only in the opt-in (``keep_in_oof=True``) mode.
    """
    scores, model_name, skip_refit, results, identity, refit_artifacts = _run_concrete_scores(
        pipeline, spectro, dataset_arg, cli, venv_python, run_dir, cv_pool, excluded, tags_by_sample, dataset_pickle=dataset_pickle, random_state=random_state
    )
    return _scores_to_run_result(scores, spectro.name, model_name, metric, task_type, config_name=config_name, skip_refit=skip_refit, results=results, identity=identity, refit_artifacts=refit_artifacts)


def _source_concat_layout(source_indices: list[int], spectro: Any, envelope: dict[str, Any] | None = None) -> dict[str, Any]:
    """Internal source-layout marker for a proven ``{"merge": {"sources": "concat"}}`` boundary."""
    envelope_layout = ((envelope or {}).get("plan") or {}).get("source_layout")
    if isinstance(envelope_layout, dict) and source_indices == list(range(spectro.features_sources())):
        return dict(envelope_layout)

    source_order = [f"src{index}" for index in source_indices]
    blocks: list[dict[str, Any]] = []
    start = 0
    num_features = spectro.num_features
    for source_id, source_index in zip(source_order, source_indices, strict=True):
        width = int(num_features[source_index] if isinstance(num_features, list) else num_features)
        blocks.append(
            {
                "source_id": source_id,
                "source_index": source_index,
                "column_start": start,
                "column_count": width,
            }
        )
        start += width
    return {
        "kind": "by_source_concat",
        "source_order": source_order,
        "source_indices": list(source_indices),
        "blocks": blocks,
        "concat": {
            "axis": "feature",
            "total_column_count": start,
            "preserve_source_order": True,
        },
    }


def _graph_upstream_x_chain(graph: dict[str, Any], node_id: str) -> list[str]:
    """Ordered upstream x-chain ids for a compiled linear graph."""
    incoming: dict[str, str] = {}
    for edge in graph.get("edges", []) or []:
        target = edge.get("target") or {}
        source = edge.get("source") or {}
        contract = edge.get("contract") or {}
        if target.get("port_name") == "x" and contract.get("kind") == "data":
            incoming[str(target["node_id"])] = str(source["node_id"])
    chain: list[str] = []
    current = incoming.get(node_id)
    while current is not None:
        chain.append(current)
        current = incoming.get(current)
    chain.reverse()
    return chain


def _source_concat_preprocessing_metadata(
    graph: dict[str, Any],
    model_node: dict[str, Any],
    source_indices: list[int],
    source_layout: dict[str, Any],
) -> dict[str, Any]:
    """Build the node-runner contract for top-level source concat preprocessing."""
    nodes_by_id = {node["id"]: node for node in graph.get("nodes", []) or []}
    chain_nodes = [nodes_by_id[node_id] for node_id in _graph_upstream_x_chain(graph, str(model_node["id"]))]
    if any(node.get("kind") != "transform" for node in chain_nodes):
        raise DagMlUnsupported("source-concat preprocessing chain may contain only X transform nodes")

    blocks_by_index = {
        int(block["source_index"]): block
        for block in source_layout.get("blocks", []) or []
        if isinstance(block, dict) and "source_index" in block
    }
    raw_source_ids = source_layout.get("source_ids")
    raw_source_order = source_layout.get("source_order")
    source_ids: list[Any] = raw_source_ids if isinstance(raw_source_ids, list) else []
    source_order: list[Any] = raw_source_order if isinstance(raw_source_order, list) else []
    sources: list[dict[str, Any]] = []
    for position, source_index in enumerate(source_indices):
        block = blocks_by_index.get(source_index, {})
        sources.append(
            {
                "source_name": block.get("source_name", source_order[position] if position < len(source_order) else f"source_{source_index}"),
                "source_id": block.get("source_id", source_ids[position] if position < len(source_ids) else f"src{source_index}"),
                "source_index": source_index,
                "steps": chain_nodes,
            }
        )

    return {
        "mode": "top_level_sources_concat",
        "preserve_legacy_sources_after_merge": True,
        "source_layout": source_layout,
        "sources": sources,
    }


def _mark_source_concat_model_nodes(graph: dict[str, Any], source_indices: list[int], spectro: Any, envelope: dict[str, Any]) -> None:
    """Attach source-concat layout and per-source preprocessing metadata to model nodes."""
    layout = _source_concat_layout(source_indices, spectro, envelope)
    for node in graph.get("nodes", []):
        if node.get("kind") != "model":
            continue
        metadata = dict(node.get("metadata") or {})
        metadata["source_layout"] = layout
        metadata["source_concat_preprocessing"] = _source_concat_preprocessing_metadata(graph, node, source_indices, layout)
        node["metadata"] = metadata


def _run_source_concat_merge(
    pre_merge_steps: list[Any],
    post_merge_steps: list[Any],
    source_indices: list[int],
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    metric: str,
    task_type: str,
    dataset_pickle: str | None = None,
    config_name: str = "",
    random_state: int | None = None,
) -> RunResult:
    """Run top-level source concat natively by preserving the per-source transform boundary.

    Legacy applies upstream X transforms to each source independently, then ``{"merge": {"sources":
    "concat"}}`` hstacks those transformed blocks before the downstream splitter/model. A plain native
    early-fusion run would fit the X-chain on the already-concatenated matrix, which changes row-wise
    transforms such as SNV. This path compiles the graph without the merge step but marks the model node
    with a source-layout contract; the node runner then materializes source blocks, applies the upstream
    X-chain per block, hstacks in source order, and fits/predicts the model.
    """
    native_pipeline = [*pre_merge_steps, *post_merge_steps]
    steps, splitter = _split_pipeline(native_pipeline)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    _reject_multi_model(steps)
    _assert_supported_operators(steps)
    steps = _apply_model_params(steps)

    identity = mint_identity(spectro)
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-source-concat", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    _mark_source_concat_model_nodes(graph, source_indices, spectro, envelope)
    outcome = run_cv_refit_bundle(
        dsl=dsl,
        envelope=envelope,
        graph=graph,
        dataset_path=dataset_arg,
        workdir=run_dir,
        dagml_cli=cli,
        venv_python=venv_python,
        selection_metric=metric,
        dataset_pickle=dataset_pickle,
        dataset=spectro,
        random_state=random_state,
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml source-concat merge run failed")

    return _scores_to_run_result(
        outcome["scores"],
        spectro.name,
        _model_name(steps),
        metric,
        task_type,
        config_name=config_name,
        skip_refit=_legacy_skips_refit(splitter),
        results=outcome["results"],
        identity=identity,
        refit_artifacts=outcome["refit_artifacts"],
    )


def _run_repetition(pipeline: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a REPETITION (sample-grain grouped) pipeline as ONE native dag-ml CV+refit run.

    The CV universe is the repetition ROWS of the train partition (each stored row is its own
    sample int / sample_id — repetitions are NOT collapsed). Folds are GROUP-aware
    (:func:`_build_group_folds`): all replicates of a sample land on the same fold side, so every
    rep row is validated exactly once (a clean OOF partition) while a group is never split. The
    envelope carries each row's ``group_id`` (the repetition column value), and dag-ml-data's
    ``validate_fold_set_against_sample_relations`` refuses any fold that splits a group — the
    native group-leakage guarantee. Each rep row is scored individually (the repetition grain),
    which is exactly what nirs4all's ``cv_best_score``/``best_rmse`` report; the sample-level
    aggregation (the ``_agg`` twin) is a separate score nirs4all does not surface on RunResult,
    so no aggregation reducer is needed.

    Generators are expanded in Python (operator-level via ``expand_spec``; a param-level model
    sweep also goes through ``expand_spec`` here for simplicity) and each concrete variant runs
    through the group-aware path, selecting the best by its CV score — mirroring nirs4all.
    """
    from nirs4all.pipeline.config.generator import expand_spec

    variants = expand_spec(pipeline)
    results = [
        _run_repetition_concrete(variant, spectro, dataset_arg, cli, venv_python, run_dir / f"variant{index}", metric, task_type, dataset_pickle=dataset_pickle, config_name=config_name, random_state=random_state)
        for index, variant in enumerate(variants)
    ]
    if len(results) == 1:
        return results[0]
    # Direction from the SINGLE source of truth (core.metrics), so a classification sweep ranking on
    # `balanced_accuracy` (the default since #60) is MAXIMIZED, not minimized (which would pick the worst
    # variant). The old `metric in ("accuracy", "r2")` set silently mis-ranked balanced_accuracy.
    maximize = is_higher_better(metric)

    def _cv_rank(result: RunResult) -> float:
        score = result.cv_best_score
        if score != score:  # NaN ranks last
            return float("inf")
        return -score if maximize else score

    return min(results, key=_cv_rank)


def _run_repetition_concrete(pipeline: Any, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """One concrete repetition variant: group-aware folds + a ``group_id``-carrying envelope."""
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    _assert_supported_operators(steps)
    steps = _apply_model_params(steps)

    identity = mint_identity(spectro)
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_group_folds(splitter, spectro, pool)
    group_by_sample = _repetition_grain(spectro, pool)
    envelope = build_envelope(spectro, identity, sample_ints=pool, group_by_sample=group_by_sample)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml repetition run failed")

    # Thread the node results + minted identity into the projection so the strict direct-block rows
    # (per-fold val + refit final/test) carry real y_pred/y_true/sample_indices — the same 2a-i fill the
    # single-pipeline path does. For a SAMPLE-LEVEL-aggregation dataset (`aggregate=True`) the refit's
    # `(test, None)` block dag-ml emits is already the aggregated sample grain (the replicates were
    # collapsed at materialization), so this surfaces the aggregation's final-(test) y_pred at parity with
    # legacy (Gap 2). scores/skip_refit unchanged — num_predictions and scores stay score-set-driven.
    return _scores_to_run_result(outcome["scores"], spectro.name, _model_name(steps), metric, task_type, config_name=config_name, skip_refit=_legacy_skips_refit(splitter), results=outcome["results"], identity=identity, refit_artifacts=outcome["refit_artifacts"])


def _reshape_for_rep_fusion(rep_step: dict[str, Any], spectro: Any) -> None:
    """Run nirs4all's REAL repetition reshape (``rep_to_sources`` / ``rep_to_pp``) in place.

    Reuses the production operator (:class:`RepetitionConfig`) + the dataset's own reshape methods
    (``reshape_reps_to_sources`` / ``reshape_reps_to_preprocessings``) — no reshape logic is
    reimplemented here. The reshape collapses the N replicate ROWS of a physical sample into either
    N sample-aligned feature SOURCES (``rep_to_sources``) or N stacked PROCESSING layers
    (``rep_to_pp``), reducing the dataset to ``n_unique`` physical samples.

    The ``repetition`` flag is cleared afterwards: it named the rep grouping of the ORIGINAL rows,
    which no longer exists once the replicates have become sources/processings — the reshaped
    dataset's unit of analysis is the physical sample, so the downstream folds must be sample-grain
    (the plain-repetition group-fold path, #21, must NOT fire on the reshaped dataset).
    """
    from nirs4all.operators.data.repetition import RepetitionConfig

    if "rep_to_sources" in rep_step:
        config = RepetitionConfig.from_step_value(rep_step["rep_to_sources"])
        spectro.reshape_reps_to_sources(config)
    else:
        config = RepetitionConfig.from_step_value(rep_step["rep_to_pp"])
        spectro.reshape_reps_to_preprocessings(config)
    spectro._repetition = None  # noqa: SLF001 - the rep grouping was consumed by the reshape


def _run_rep_fusion(
    pipeline: list[Any],
    rep_step: dict[str, Any],
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    metric: str,
    task_type: str,
    config_name: str = "",
    variant_config_names: list[str] | None = None,
    is_classification: bool = False,
    random_state: int | None = None,
) -> RunResult:
    """Run a ``rep_to_sources`` / ``rep_to_pp`` pipeline as ONE native dag-ml CV+refit run (S7).

    REP FUSION is a one-time host RESHAPE feeding an already-native multimodal path:

    * ``rep_to_sources`` — each replicate becomes a feature SOURCE, so the reshaped dataset is
      MULTI-SOURCE; the envelope auto-emits a ``feature_block_set`` (S3 early fusion), and an MB-PLS
      model takes the per-source block list (S5 intermediate fusion) — both already native.
    * ``rep_to_pp`` — each replicate becomes a PROCESSING layer, so the reshaped dataset is
      single-source with N stacked layers; the FLAT_2D materialization hstacks them by processing
      order (the feature-axis concat S6 already does), which the legacy ``tabular_numeric`` path runs.

    The reshaped dataset lives only in host memory (the on-disk dataset has no such structure), so it
    is PICKLED for the adapter (the same mechanism ``sample_augmentation`` uses) — the adapter resolves
    the exact reshaped sources/processings, identity-keyed by the physical sample_id.

    LEAKAGE: after the reshape the unit of analysis is the physical SAMPLE (N reps → N sources/layers
    of ONE sample), so folds/OOF are over SAMPLES — distinct from a plain repetition dataset (#21,
    rep-grain). A sample's N source-blocks (or N processing layers) all ride ONE row and therefore the
    SAME fold side by construction; per-source / per-layer preprocessing fits on fold-train only (the
    per-block X-chain, like the multi-source path). No cross-sample mixing.

    GENERATORS — an ``_or_`` / ``_cartesian_`` sweep INSIDE the rep-fusion body (e.g.
    ``[{"rep_to_pp": ...}, {"_or_": [SNV, MSC]}, splitter, model]``) is expanded in Python (the body
    minus the reshape step via ``expand_spec``), each concrete variant runs through the reshaped
    early-fusion path returning its raw native ScoreSet, and the per-variant projection (#55,
    :func:`~nirs4all.pipeline.dagml.result._project_operator_sweep`) combines them into the FULL legacy
    per-variant table — selecting the CV winner, projecting EVERY variant's CV rows (legacy
    num_predictions parity) under its OWN ``config_name`` / ``model_name``, and refitting the winner
    only. ``variant_config_names`` is the ordered legacy per-variant config names (derived by the caller
    from the FULL pipeline — the reshape step IS part of the legacy config-name hash — in ``expand_spec``
    order). A single (non-sweep) variant maps straight through the winner-only projection, byte-identical
    to the pre-#55 shape.
    """
    import pickle

    from nirs4all.pipeline.config.generator import expand_spec

    body = [step for step in pipeline if not _is_rep_fusion_step(step)]
    variants = expand_spec(body)
    run_dir.mkdir(parents=True, exist_ok=True)
    variant_scores = [
        _run_rep_fusion_concrete_scores(variant, rep_step, spectro, dataset_arg, cli, venv_python, run_dir / f"variant{index}", metric, pickle, random_state=random_state)
        for index, variant in enumerate(variants)
    ]
    if len(variant_scores) == 1:
        scores, model_name, skip_refit, refit_artifacts = variant_scores[0]
        return _scores_to_run_result(scores, spectro.name, model_name, metric, task_type, config_name=config_name, skip_refit=skip_refit, refit_artifacts=refit_artifacts)

    # A sweep INSIDE the rep-fusion body: combine every reshaped-variant's ScoreSet into the full
    # per-variant legacy table (#55) — same machinery the main operator-sweep path uses. `_project_operator_sweep`
    # consumes 3-tuples, so split off the per-variant refit_artifacts and thread them as a separate by-index
    # list (the projection persists the WINNER's model artifacts only).
    sweep_scores = [(scores, model_name, skip_refit) for scores, model_name, skip_refit, _artifacts in variant_scores]
    refit_artifacts_by_index = [artifacts for _scores, _model_name, _skip_refit, artifacts in variant_scores]
    return _project_operator_sweep(sweep_scores, spectro.name, metric, task_type, is_classification, variant_config_names or [], refit_artifacts_by_index=refit_artifacts_by_index)


def _run_rep_fusion_concrete_scores(body: Any, rep_step: dict[str, Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, pickle: Any, random_state: int | None = None) -> tuple[dict[str, Any], str, bool, list[dict[str, Any]]]:
    """One concrete rep-fusion variant: reshape a fresh dataset copy, run the sample-grain CV+refit, return raw scores.

    Returns ``(scores, model_name, skip_refit, refit_artifacts)`` — the raw native ScoreSet + the model
    label + the legacy refit-gate flag + the captured fitted REFIT estimators
    (``outcome["refit_artifacts"]``, P3 Slice 2c-i) — so a sweep inside the body can COMBINE every variant's
    ScoreSet into one per-variant projection (legacy num_predictions parity) AND persist the winner's model
    artifacts, exactly like :func:`_run_concrete_scores` on the main path. The single-variant caller wraps
    this through :func:`_scores_to_run_result` (winner-only projection).
    """
    import copy

    steps, splitter = _split_pipeline(body)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    _assert_supported_operators(steps)
    steps = _apply_model_params(steps)

    # Reshape a FRESH copy per variant so each variant's pickled dataset is independent (and the
    # caller's spectro stays the original rep dataset, unmutated, for any later use).
    reshaped = copy.deepcopy(spectro)
    _reshape_for_rep_fusion(rep_step, reshaped)

    identity = mint_identity(reshaped)
    # The reshape leaves every physical sample in `partition: train` (the splitter follows the reshape),
    # so the CV universe is the full reshaped sample set. Folds are sample-grain — a sample's N
    # source-blocks / processing-layers ride ONE row, so they cannot split across the fold boundary.
    pool = reshaped.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, reshaped, pool, set())
    envelope = build_envelope(reshaped, identity, sample_ints=pool)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()

    run_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = run_dir / "reshaped_dataset.pkl"
    pickle_path.write_bytes(pickle.dumps(reshaped))

    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=str(pickle_path), dataset=reshaped, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml rep-fusion run failed")

    # Legacy labels the model by the model step's own name/class only — the rep_to_sources / rep_to_pp
    # reshape is a dataset transform, NOT part of the model_name. Emit the bare `_model_name` (e.g.
    # "PLSRegression"), matching legacy get_models() exactly (no "rep_to_sources_" / "rep_to_pp_" prefix).
    return outcome["scores"], _model_name(steps), _legacy_skips_refit(splitter), outcome["refit_artifacts"]


def _apply_sample_augmentation(aug_step: dict[str, Any], spectro: Any) -> None:
    """Run nirs4all's REAL ``SampleAugmentationController`` to add synthetic train rows in place.

    Reuses the production machinery (the controller delegates to ``TransformerMixinController``, which
    fits the augmentation transformer on fold-train data and inserts each synthetic row via
    ``dataset.add_samples_batch`` with index ``{partition: train, origin: <base sample int>,
    augmentation: <op>}``) — no augmentation logic is reimplemented here. A minimal real ``StepRunner``
    + ``RuntimeContext`` drive it (no workspace/artifacts), exactly as the orchestrator would. The
    children land in ``partition: train`` with their ``origin`` set to the base sample, which is what
    the base→child fold expansion and the envelope's augmentation grain key off.
    """
    from nirs4all.controllers.data.sample_augmentation import SampleAugmentationController
    from nirs4all.pipeline.config.context import DataSelector, ExecutionContext, PipelineState, RuntimeContext, StepMetadata
    from nirs4all.pipeline.steps.parser import ParsedStep, StepType
    from nirs4all.pipeline.steps.step_runner import StepRunner

    step_info = ParsedStep(operator=None, keyword="sample_augmentation", step_type=StepType.DIRECT, original_step=aug_step, metadata={})
    context = ExecutionContext(selector=DataSelector(partition="train", processing=[["raw"]]), state=PipelineState(), metadata=StepMetadata())
    runtime_context = RuntimeContext()
    runtime_context.step_runner = StepRunner(verbose=0, mode="train")
    runtime_context.save_artifacts = False
    runtime_context.save_charts = False
    SampleAugmentationController().execute(step_info, spectro, context, runtime_context, mode="train")


def _augment_fold_train(aug_step: dict[str, Any], spectro: Any, fold_train: list[int]) -> list[tuple[int, np.ndarray]]:
    """Augment a fold's TRAIN only and return the synthetic children as ``[(origin_int, child_X(1,F)), ...]``.

    A FRESH copy of ``spectro`` is restricted so ``partition: train`` is exactly ``fold_train`` (the
    rest held out), then nirs4all's real augmentation machinery runs — so a STATEFUL/SUPERVISED/balanced
    augmenter fits inside this fold's train ONLY (its neighbors / global mean / class balance never see
    the fold's validation rows). The created children (origin in ``fold_train``) are read back as flat
    feature rows, in creation order — one tuple per child, so an origin augmented multiple times (``count``
    > 1 / a balancing factor) yields several. The caller inserts them into the master dataset and records
    the fold→children map; the fold copy is discarded. Leakage-safe by construction: each fold's children
    come only from its train.
    """
    import copy

    fold_ds = copy.deepcopy(spectro)
    fold_ds._indexer.update_by_filter({"partition": "train"}, {"partition": "hold"})  # noqa: SLF001
    fold_ds._indexer.update_by_indices(list(fold_train), {"partition": "train"})  # noqa: SLF001
    fold_ds._invalidate_content_hash()  # noqa: SLF001

    before = {int(s) for s in fold_ds.index_column("sample", {})}
    _apply_sample_augmentation(aug_step, fold_ds)
    samples = [int(s) for s in fold_ds.index_column("sample", {})]
    origins = [int(o) for o in fold_ds.index_column("origin", {})]
    children: list[tuple[int, np.ndarray]] = []
    for sample_int, origin_int in zip(samples, origins, strict=True):
        if sample_int not in before and sample_int != origin_int:
            children.append((origin_int, np.asarray(fold_ds.x_rows([sample_int], layout="2d"), dtype=float).reshape(1, -1)))
    return children


def _build_fold_local_children(aug_step: dict[str, Any], spectro: Any, base_folds: list[tuple[list[int], list[int]]], base_train: list[int]) -> tuple[dict[str, dict[int, list[int]]], dict[int, str]]:
    """Augment fold-by-fold + a full-train refit pass; insert all children into ``spectro`` in place.

    For each fold (key ``"fold{i}"``, matching :func:`build_fold_set`'s fold ids) and the full-train
    refit (key ``"refit"``) the augmenter is fit inside that train pool only (:func:`_augment_fold_train`),
    and the resulting children are appended to the master ``spectro`` as ``partition: train`` rows with
    their ``origin``. Returns ``(fold_children, augmentation_by_sample)`` where ``fold_children`` is
    ``{fold_label: {origin_int: [child_int, ...]}}`` (the resolver's fold-local expansion map) and
    ``augmentation_by_sample`` tags every inserted child for the envelope's augmentation metadata.

    All folds' children coexist in one dataset and one base-grain envelope (each child shares its
    origin's ``sample_id``, a base train id in the fold set → the origin-boundary contract holds), but
    they are kept fold-distinct host-side via the returned map — a fold's children only ever join that
    fold's fit-train (see :meth:`MaterializationResolver.expand_with_augmented_children`).
    """
    transform_label = _augmentation_label(aug_step)
    passes: list[tuple[str, list[int]]] = [(f"fold{index}", train_ints) for index, (train_ints, _val) in enumerate(base_folds)]
    passes.append(("refit", base_train))

    fold_children: dict[str, dict[int, list[int]]] = {}
    augmentation_by_sample: dict[int, str] = {}
    for fold_label, fold_train in passes:
        children = _augment_fold_train(aug_step, spectro, fold_train)
        if not children:
            fold_children[fold_label] = {}
            continue
        rows = np.stack([child_x for _origin, child_x in children])  # (n, 1, F), one row per child
        indexes = [{"partition": "train", "origin": origin_int, "augmentation": f"{transform_label}|{fold_label}"} for origin_int, _x in children]
        before = {int(s) for s in spectro.index_column("sample", {})}
        spectro.add_samples_batch(data=rows, indexes_list=indexes)
        samples = [int(s) for s in spectro.index_column("sample", {})]
        origins = [int(o) for o in spectro.index_column("origin", {})]
        by_origin: dict[int, list[int]] = {}
        for sample_int, origin_int in zip(samples, origins, strict=True):
            if sample_int not in before and sample_int != origin_int:
                by_origin.setdefault(origin_int, []).append(sample_int)
                augmentation_by_sample[sample_int] = transform_label
        fold_children[fold_label] = by_origin
    return fold_children, augmentation_by_sample


def _augmentation_grain(spectro: Any, transform_label: str) -> tuple[list[int], dict[int, str]]:
    """Post-augmentation grain: ``(augmented_ints, augmentation_by_sample)``.

    ``augmented_ints`` is every augmented child (``sample != origin``); ``augmentation_by_sample`` tags
    each with the augmentation transform id for the envelope's structured ``augmentation`` metadata. The
    fit-time base→child expansion is recomputed by the resolver from the minted identity, not here.
    """
    samples = [int(s) for s in spectro.index_column("sample", {})]
    origins = [int(o) for o in spectro.index_column("origin", {})]
    augmented_ints = [sample_int for sample_int, origin_int in zip(samples, origins, strict=True) if sample_int != origin_int]
    augmentation_by_sample = dict.fromkeys(augmented_ints, transform_label)
    return augmented_ints, augmentation_by_sample


def _augmentation_label(aug_step: dict[str, Any]) -> str:
    """A stable transform label for the augmentation step's structured envelope metadata."""
    transformers = _augmentation_transformers(aug_step)
    return "+".join(type(transformer).__name__ for transformer in transformers) or "sample_augmentation"


def _augmentation_transformers(aug_step: dict[str, Any]) -> list[Any]:
    """The deserialized transformer instances of an augmentation step (mirrors the controller).

    A transformer may be given bare or wrapped in a ``{"transformer": op, ...}`` dict (per-transformer
    variation_scope); both forms resolve to the instance via ``deserialize_component``, exactly as
    :meth:`SampleAugmentationController.execute` parses them.
    """
    from nirs4all.pipeline.config.component_serialization import deserialize_component

    raw = aug_step["sample_augmentation"].get("transformers", [])
    return [deserialize_component(t["transformer"] if isinstance(t, dict) and "transformer" in t else t) for t in raw]


def _operator_is_stateless(operator: Any) -> bool:
    """Whether an augmentation transformer learns NO data-dependent state in ``fit``.

    The first augmentation slice augments ONCE globally (before folds exist), so a transformer that
    fits on the whole train partition would see future fold-validation rows. That is leakage-free ONLY
    for STATELESS per-sample augmenters (Gaussian/multiplicative noise, scatter, baseline, spline,
    wavelength warps — ``fit`` only seeds an RNG); a STATEFUL/SUPERVISED one (mixup storing neighbors,
    a global-mean reference, a supervised transform) leaks. There is no declared marker on these
    operators, so the signal is twofold and conservative:

    * **supervised** — a ``requires_y`` tag (``_more_tags()``/``_tags``) means ``fit`` consumes y; reject.
    * **fit-learned data state** — fit the transformer on two differently-distributed dummy datasets and
      compare its post-fit attributes (sklearn fitted attrs ``*_`` + any ndarray, excluding the seeded
      RNG). A transformer whose state varies with the fit data carries learned state (``X_fit_``,
      ``global_mean_``, …) → stateful → reject. A clone is fit each time so no instance state is shared;
      any error during the probe is treated as NOT stateless (fail closed).
    """
    from nirs4all.controllers.transforms.transformer import TransformerMixinController

    if TransformerMixinController._requires_y(operator):  # noqa: SLF001 - reuse the supervised-tag check
        return False
    try:
        from sklearn.base import clone

        rng = np.random.default_rng(0)
        probe_a = rng.normal(size=(24, 32))
        probe_b = rng.normal(loc=8.0, scale=4.0, size=(24, 32))

        def _fit_state(data: np.ndarray) -> dict[str, Any]:
            fitted = clone(operator).fit(data)
            return {name: value for name, value in vars(fitted).items() if (name.endswith("_") or isinstance(value, np.ndarray)) and not name.startswith("_")}

        state_a, state_b = _fit_state(probe_a), _fit_state(probe_b)
        for name in set(state_a) | set(state_b):
            left, right = state_a.get(name), state_b.get(name)
            if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                if not (left is not None and right is not None and np.array_equal(np.asarray(left), np.asarray(right))):
                    return False
            elif left != right:
                return False
    except Exception:  # noqa: BLE001 - any probe failure ⇒ cannot prove stateless ⇒ fail closed
        return False
    return True


def _augmentation_is_leakage_free(aug_step: dict[str, Any]) -> bool:
    """Whether a ``sample_augmentation`` step is safe to run as ONE GLOBAL pre-fold augmentation.

    True (→ the GLOBAL path, #8) ONLY when neither leakage vector applies to global augmentation:

    * the **balanced/supervised mode** (a ``balance`` key) is NOT used — it fits class/bin targets on
      the whole train y, so the synthetic counts depend on the (future-fold-val-inclusive) label set;
    * EVERY transformer is stateless (:func:`_operator_is_stateless`).

    False (→ the FOLD-LOCAL path, #32) for a stateful/supervised/balanced augmentation: it is fit inside
    each fold's train only (and a full-train refit pass), so it never sees a fold's validation rows. Both
    paths are leakage-safe; this only routes which one :func:`_run_augmentation` uses.
    """
    config = aug_step["sample_augmentation"]
    if "balance" in config:
        return False
    transformers = _augmentation_transformers(aug_step)
    return bool(transformers) and all(_operator_is_stateless(transformer) for transformer in transformers)


def _run_augmentation(pipeline: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a ``sample_augmentation`` pipeline as ONE native dag-ml CV+refit on augmented train.

    Adds the synthetic train rows (real augmentation machinery), builds BASE-grain folds (each base
    val sample validated once — a clean OOF partition that dag-ml/dag-ml-data accept) and a CV-universe
    envelope (base + augmented children; observation-grain relations carrying each child's origin +
    augmentation, deduped to the origin's sample grain in the schema). The fold-train views stay
    base-grain + ``include_augmented_train`` so the host expands each base id to base + its children at
    fit time — the children TRAIN, the OOF/validation/test never see them. The augmented dataset is
    pickled for the adapter (augmentation is stochastic, not reproducible cross-process).

    Two augmentation regimes, picked by :func:`_augmentation_is_leakage_free`:

    * **GLOBAL** (stateless per-sample augmenter) — augment ONCE on the whole train; the children are
      shared across folds. Leakage-free because the augmenter learns no data state (#8).
    * **FOLD-LOCAL** (stateful / supervised / balanced augmenter) — augment inside EACH fold's train
      (plus a full-train refit pass) separately, so each fold has its own children fit only on that
      fold's train. A ``fold_children`` map keys the resolver's per-fold expansion; it is pickled with
      the dataset so the adapter expands the right children per fold (#32). This is what makes the
      stateful case (mixup neighbors, global-mean scatter, class balancing) leakage-safe.

    Only the supported ``transform* + sample_augmentation + splitter + model`` shape runs here; the
    remaining steps are lowered through the bridge (a raw ``sample_augmentation`` still raises, keeping
    the coverage boundary). A branch / exclude / generator beside it is out of scope and fails loud.
    """
    import pickle

    aug_steps = [step for step in pipeline if _is_augmentation_step(step)]
    if len(aug_steps) != 1:
        raise NotImplementedError("engine='dag-ml' supports exactly one sample_augmentation step")
    aug_step = aug_steps[0]
    rest = [step for step in pipeline if not _is_augmentation_step(step)]
    steps, splitter = _split_pipeline(rest)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    _assert_supported_operators(steps)
    steps = _apply_model_params(steps)

    base_train = [int(s) for s in spectro.index_column("sample", {"partition": "train"})]
    fold_local = not _augmentation_is_leakage_free(aug_step)

    # BASE-grain folds: split the base train pool only; train = base-train, val = base-val. The children
    # are NEVER listed in a fold (the FoldSet stays a clean base-grain OOF partition); they are pulled
    # into fit-train by the host expansion keyed on the origin's fold side. The split runs over the BASE
    # rows only (before any child exists), so the fold partition is identical for both augmentation paths.
    base_folds = [([base_train[i] for i in train_idx], [base_train[i] for i in val_idx]) for train_idx, val_idx in _split_pool(splitter, spectro, base_train)]

    # GLOBAL stateless augmentation (#8): fit once on the whole train (leakage-free only for stateless
    # per-sample augmenters), children shared across all folds (resolver discovers them from identity).
    # FOLD-LOCAL augmentation (#32): a STATEFUL/SUPERVISED/balanced augmenter is fit inside EACH fold's
    # train only, so each fold (+ the full-train refit) has its OWN children — leakage-safe for the
    # stateful case. `fold_children` keys the per-fold expansion; it is pickled for the adapter's resolver.
    fold_children: dict[str, dict[int, list[int]]] | None = None
    if fold_local:
        fold_children, augmentation_by_sample_int = _build_fold_local_children(aug_step, spectro, base_folds, base_train)
    else:
        _apply_sample_augmentation(aug_step, spectro)
        _augmented_ints, augmentation_by_sample_int = _augmentation_grain(spectro, _augmentation_label(aug_step))

    # Identity is minted on the AUGMENTED dataset so children get their own observation_id + the origin's
    # sample_id (augmented=True). The CV universe = base train + the augmented children.
    identity = mint_identity(spectro)
    samples = [int(s) for s in spectro.index_column("sample", {})]
    origins = [int(o) for o in spectro.index_column("origin", {})]
    augmented_ints = [sample_int for sample_int, origin_int in zip(samples, origins, strict=True) if sample_int != origin_int]
    cv_universe = base_train + augmented_ints

    envelope = build_envelope(spectro, identity, sample_ints=cv_universe, augmentation_by_sample=augmentation_by_sample_int)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, base_folds, dsl_id="nirs4all-augmentation", n_splits=len(base_folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()

    run_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = run_dir / "augmented_dataset.pkl"
    # Fold-local pickles the dataset + the fold→children map (the resolver's per-fold expansion); the
    # global path pickles the bare dataset (the resolver discovers dataset-global children from identity).
    pickle_path.write_bytes(pickle.dumps({"dataset": spectro, "fold_children": fold_children} if fold_local else spectro))

    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=str(pickle_path), dataset=spectro, fold_children=fold_children, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml augmentation run failed")

    return _scores_to_run_result(outcome["scores"], spectro.name, _model_name(steps), metric, task_type, config_name=config_name, skip_refit=_legacy_skips_refit(splitter), refit_artifacts=outcome["refit_artifacts"])


_MERGE_NODE_ID = "merge:concat"


def _run_separation_branch(pipeline: list[Any], branch_step: dict[str, Any], branch_body: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a by_metadata/by_tag separation branch + concat merge as ONE native dag-ml fan-out run.

    Lowers the branch to an ``auto_separate`` template (one branch carrying the criterion + the
    model sub-pipeline) followed by a concat ``PredictionJoin`` merge, builds the envelope with the
    criterion column's per-sample metadata so the **native** ``fan_out_data_aware_branches`` discovers
    the partition values, compiles the fanned graph (one model node per value), and drives dag-ml-cli.
    The native concat-merge handler reassembles the per-partition OOF into one full-universe OOF
    attributed to the merge node — its cross-fold average is ``cv_best_score``.

    The criterion column's metadata is emitted onto the relations; the adapter honors each fanned
    model's ``branch_view`` selector (via the sample→metadata map) so each model fits/predicts only
    its partition. The legacy nirs4all concat-merge reassembly is broken here (MERGE-E003), so this
    native path is a correction — the parity baseline is a direct sklearn-per-partition OOF.
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for

    criterion = branch_step["branch"]
    mode, key = ("by_metadata", criterion["by_metadata"]) if "by_metadata" in criterion else ("by_tag", criterion["by_tag"])

    # The splitter lives at the top level (before the branch); the branch body is the model
    # sub-pipeline applied per partition. Drop the splitter from the body if it slipped in.
    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    # Drop the splitter / None no-ops and reject wavelength-requiring or non-routable ops in the body.
    body_steps = _supported_body_steps([step for step in branch_body if not hasattr(step, "split")])

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())

    # Per-sample criterion values: the first map seeds the envelope relations (native fan-out reads
    # partition values from it); the second is the adapter's sample_id→metadata map for branch_view.
    metadata_by_sample, sample_metadata = _branch_metadata(spectro, identity, mode, key)
    envelope = build_envelope(spectro, identity, sample_ints=pool, metadata_by_sample=metadata_by_sample)

    # Compat auto_separate template: ONE branch (the model sub-pipeline) carrying the criterion +
    # mode, marked auto_separate; the native fan-out expands it into N per-partition branches.
    template = {"id": "per_partition", "steps": [_branch_compat_step(step) for step in body_steps]}
    # Always by_metadata mode: the criterion (whether nirs4all by_metadata or by_tag) is emitted as a
    # metadata column on the relations, so the native fan-out discovers its values from there.
    compat_dsl = {
        "id": "nirs4all-separation-branch",
        "pipeline": [
            {"branch": {"branches": [template]}, "mode": "by_metadata", "selector": {"metadata_key": key}, "metadata": {"auto_separate": True}},
            {"merge": "concat", "output_as": "predictions", "id": _MERGE_NODE_ID},
        ],
    }

    # NATIVE fan-out (no Python suffix replication): dag-ml reads the partition values from the
    # envelope relations and expands one branch per sorted value, owning the node-id suffixing.
    fanned_dsl = dag_ml.fan_out_data_aware_branches(compat_dsl, envelope).to_dict()
    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(fanned_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    if not model_ids:
        raise DagMlUnsupported("separation-branch fan-out produced no per-partition model nodes")

    # Per-partition data_bindings (one per fanned model node) + the materialized fold set. The CLI's
    # own fan-out is a no-op on the already-fanned DSL (the auto_separate marker is consumed).
    fanned_dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    fanned_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=fanned_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, sample_metadata=sample_metadata, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml separation-branch run failed")

    # The concat-merge producer's reports carry both the full-universe cross-fold OOF average
    # (`cv_best_score`) AND a reassembled `(test, fold_id=None)` block (`best_rmse`): dag-ml's native
    # off-fold merge handler reassembles each per-partition refit model's held-out TEST prediction
    # (the node runner emits it with `fold_id=None`) into one full-universe test block under the merge
    # node. Both scores are the separation branch's, surfaced by `_scores_to_run_result`.
    return _scores_to_run_result(outcome["scores"], spectro.name, _model_name(body_steps), metric, task_type, producer=_MERGE_NODE_ID, config_name=config_name, refit_artifacts=outcome["refit_artifacts"])


def _branch_compat_step(step: Any) -> dict[str, Any]:
    """Lower one branch-body step (transform / {"model": M} / {"y_processing": op}) to compat DSL."""
    from nirs4all.pipeline.dagml_bridge import _step_to_dsl

    return _step_to_dsl(step)


def _branch_metadata(spectro: Any, identity: Any, mode: str, key: str) -> tuple[dict[str, dict[int, Any]], dict[str, dict[str, Any]]]:
    """Build the criterion's per-sample metadata: ``({col: {sample_int: value}}, {wire_id: {col: value}})``.

    The first map seeds the envelope relations (native fan-out reads partition values from it); the
    second is the adapter's ``sample_id → metadata`` map for honoring each branch's ``branch_view``.
    A ``by_tag`` criterion is represented as a metadata column (the tag value per sample), matching
    the envelope's metadata-carried relations.
    """
    sample_ints = [int(value) for value in spectro.index_column("sample", {})]
    values = spectro.metadata_column(key, {}) if mode == "by_metadata" else spectro.get_tag(key, {})
    by_int = {sample_int: (str(value) if value is not None else None) for sample_int, value in zip(sample_ints, values, strict=True)}
    metadata_by_sample = {key: dict(by_int)}
    sample_metadata = {identity.to_wire(sample_int): {key: value} for sample_int, value in by_int.items()}
    return metadata_by_sample, sample_metadata


_FUSION_MERGE_NODE_ID = "merge:fusion"


def _canonical_branch_step(step: Any, node_id: str) -> dict[str, Any]:
    """Lower one branch-body step to a CANONICAL pipeline-DSL step (``kind`` + ``operator.class``).

    The duplication-fusion path emits a *canonical* ``steps`` DSL (not the compat ``pipeline`` form):
    dag-ml's nirs4all-compat importer whitelists merge modes (``concat``/``predictions``/… only) and
    REFUSES ``fusion`` — but the canonical ``PipelineDslMergeStep.merge_mode`` is a free-form string the
    runtime reads verbatim, so the fusion merge must be expressed canonically. The branch bodies must
    therefore also be canonical. This reuses the verified compat lowering
    (:func:`~nirs4all.pipeline.dagml_bridge._step_to_dsl` — operator FQN + JSON-safe params +
    native param-generator entries) and re-keys it into the canonical ``{"kind", "id", "operator":
    {"class": …}, "params": …}`` shape per node kind (transform / y_transform / model), assigning the
    explicit ``node_id`` so each branch's nodes are uniquely named.
    """
    from nirs4all.pipeline.dagml_bridge import _step_to_dsl

    compat = _step_to_dsl(step)
    if "model" in compat:
        out: dict[str, Any] = {"kind": "model", "id": node_id, "operator": {"class": compat["model"]}, "params": compat.get("params", {})}
        if "generators" in compat:
            out["generators"] = compat["generators"]
        return out
    if "y_processing" in compat:
        inner = compat["y_processing"]
        return {"kind": "y_transform", "id": node_id, "operator": {"class": inner["class"]}, "params": inner.get("params", {})}
    # Bare transform: compat is {"class": FQN, "params": {...}}.
    return {"kind": "transform", "id": node_id, "operator": {"class": compat["class"]}, "params": compat.get("params", {})}


def _canonical_branch(branch_body: list[Any], branch_index: int) -> dict[str, Any]:
    """Lower one duplication sub-pipeline (a list of steps) to a canonical branch with unique node ids.

    ``None`` no-ops are dropped (never lowered to a ``builtins.NoneType`` node) and wavelength-requiring /
    non-routable ops in the body raise a catchable :class:`DagMlUnsupported` — same coverage guarantee
    the top-level path gives, applied here so duplication / by_source / stacking branch bodies are safe.
    """
    steps = _supported_body_steps([step for step in branch_body if not hasattr(step, "split")])
    return {
        "id": f"branch_{branch_index}",
        "steps": [_canonical_branch_step(step, f"branch:{branch_index}.node:{node_index}") for node_index, step in enumerate(steps)],
    }


def _branch_merge_transformer_step(branches: list[list[Any]], merge_mode: str) -> DuplicationBranchMergeTransformer:
    """Build the importable transformer step used for native duplication feature/all merges."""
    from nirs4all.pipeline.dagml_bridge import _json_safe_params, _qualname

    if merge_mode not in ("features", "all"):
        raise DagMlUnsupported("engine='dag-ml' supports duplication branch merge only for merge='features' or merge='all' in this slice")

    lowered_branches: list[dict[str, Any]] = []
    for branch in branches:
        lowered_steps: list[dict[str, Any]] = []
        for step in _supported_body_steps([part for part in branch if not hasattr(part, "split")]):
            if isinstance(step, dict) and "model" in step:
                if merge_mode != "all":
                    raise DagMlUnsupported(
                        "engine='dag-ml' supports duplication merge=features only for feature-only branch bodies; "
                        "branch-local models require merge='all'."
                    )
                model = step["model"]
                lowered_steps.append({"kind": "model", "class": _qualname(model), "params": _json_safe_params(model)})
            elif isinstance(step, dict):
                raise DagMlUnsupported(
                    "engine='dag-ml' supports duplication feature/all merges only for plain branch transform steps "
                    "and merge='all' branch-local model steps; "
                    "structured branch step dictionaries "
                    f"such as {sorted(step)} stay on the legacy path."
                )
            else:
                lowered_steps.append({"kind": "transform", "class": _qualname(step), "params": _json_safe_params(step)})
        lowered_branches.append({"steps": lowered_steps})
    return DuplicationBranchMergeTransformer(branches=lowered_branches, merge_mode=merge_mode)


def _canonical_source_branch(branch_body: list[Any], source_index: int) -> dict[str, Any]:
    """Lower the shared by_source body to a canonical branch BOUND to one source (S4).

    Same lowering as :func:`_canonical_branch` (the shared model sub-pipeline, unique node ids per
    source), but every MODEL node carries ``metadata.source_index`` so the node runner materializes
    ONLY that source's feature block — late fusion by source. The branch index IS the source index
    (one branch per source), so a fold view stays full-sample (all branches see all samples) while
    each branch's model sees a different source's columns.
    """
    branch = _canonical_branch(branch_body, source_index)
    for node in branch["steps"]:
        if node["kind"] == "model":
            node["metadata"] = {**node.get("metadata", {}), "source_index": source_index}
    return branch


def _source_preprocessing_metadata(source_steps: dict[str, list[Any]], source_layout: dict[str, Any] | None) -> dict[str, Any]:
    """Lower per-source preprocessing dict by explicit ``source_layout.source_order``.

    The legacy by_source dict keys are user-facing source names (``source_0`` unless the
    dataset exposes names). The native data plan separately names the materialized blocks
    ``src0``/``src1``. This helper consumes the explicit layout and rejects any mismatch
    instead of inferring an order from the dict body.
    """
    if not isinstance(source_layout, dict):
        raise DagMlUnsupported("by_source distinct preprocessing requires envelope plan.source_layout")
    source_order = source_layout.get("source_order")
    source_ids = source_layout.get("source_ids")
    if not (isinstance(source_order, list) and all(isinstance(item, str) for item in source_order)):
        raise DagMlUnsupported("by_source distinct preprocessing requires source_layout.source_order")
    if not (isinstance(source_ids, list) and len(source_ids) == len(source_order) and all(isinstance(item, str) for item in source_ids)):
        raise DagMlUnsupported("by_source distinct preprocessing requires source_layout.source_ids aligned to source_order")
    if len(set(source_order)) != len(source_order):
        raise DagMlUnsupported("by_source distinct preprocessing source_layout.source_order contains duplicate names")
    if set(source_steps) != set(source_order):
        raise DagMlUnsupported(
            "by_source distinct preprocessing keys must exactly match source_layout.source_order: "
            f"expected={source_order!r} actual={list(source_steps)!r}"
        )

    sources: list[dict[str, Any]] = []
    for source_index, (source_name, source_id) in enumerate(zip(source_order, source_ids, strict=True)):
        body_steps = _supported_body_steps(source_steps[source_name])
        lowered = [_canonical_branch_step(step, f"source:{source_index}.pre:{step_index}") for step_index, step in enumerate(body_steps)]
        if any(step["kind"] != "transform" for step in lowered):
            raise DagMlUnsupported("by_source distinct preprocessing bodies may contain only X transforms")
        sources.append(
            {
                "source_name": source_name,
                "source_id": source_id,
                "source_index": source_index,
                "steps": lowered,
            }
        )
    return {
        "mode": "by_source_distinct_preproc_concat",
        # Legacy by_source merge writes the concatenated block into source 0 but
        # leaves non-primary sources present for the downstream multi-source fit.
        "preserve_legacy_sources_after_merge": True,
        "sources": sources,
    }


_PREDICTION_CLONE_FIELDS = (
    "dataset_name",
    "dataset_path",
    "config_name",
    "config_path",
    "pipeline_uid",
    "step_idx",
    "op_counter",
    "model_name",
    "model_classname",
    "model_path",
    "fold_id",
    "sample_indices",
    "weights",
    "metadata",
    "partition",
    "y_true",
    "y_pred",
    "y_proba",
    "val_score",
    "test_score",
    "train_score",
    "metric",
    "task_type",
    "n_samples",
    "n_features",
    "preprocessings",
    "best_params",
    "scores",
    "branch_id",
    "branch_path",
    "branch_name",
    "exclusion_count",
    "exclusion_rate",
    "model_artifact_id",
    "trace_id",
    "refit_context",
    "target_processing",
)


def _repeat_by_source_merge_projection(result: RunResult, n_sources: int) -> RunResult:
    """Match legacy by_source+concat bookkeeping: one identical result block per source."""
    if n_sources <= 1:
        return result

    rows = result.predictions.filter_predictions(load_arrays=True)
    cv_rows = [row for row in rows if str(row.get("fold_id")) != "final"]
    final_rows = [row for row in rows if str(row.get("fold_id")) == "final"]
    repeated = Predictions()

    def clone(row: dict[str, Any]) -> None:
        payload: dict[str, Any] = {field: row.get(field) for field in _PREDICTION_CLONE_FIELDS if field in row}
        repeated.add_prediction(**payload)

    for row_group in (cv_rows, final_rows):
        for _source_index in range(n_sources):
            for row in row_group:
                clone(row)
    repeated.flush()
    result.predictions = repeated
    return result


def _duplication_branch_names(pipeline: list[Any], count: int) -> list[str]:
    """Recover legacy branch labels for named duplication branch rows."""
    config_keys = {"parallel", "n_jobs", "mode", "strategy", "merge", "output_as", "aggregate", "branch_type", "preserve_branch_context"}
    branch_spec = next((step.get("branch") for step in pipeline if isinstance(step, dict) and "branch" in step), None)
    if isinstance(branch_spec, dict) and not ({"by_metadata", "by_tag", "by_source"} & set(branch_spec)):
        names = [str(name) for name in branch_spec if name not in config_keys and not str(name).startswith("_")]
        if len(names) == count:
            return names
    return [f"branch_{index}" for index in range(count)]


def _prediction_rows_without_refit(result: RunResult) -> list[dict[str, Any]]:
    """Copy prediction rows, excluding standalone refit rows legacy does not emit for merge='all'."""
    rows: list[dict[str, Any]] = []
    for row in result.predictions.filter_predictions(load_arrays=True):
        if str(row.get("fold_id")) == "final":
            continue
        rows.append(dict(row))
    return rows


def _combine_duplication_merge_all_rows(
    branch_results: list[RunResult],
    branch_names: list[str],
    downstream_result: RunResult,
    downstream_model_name: str,
    dataset_name: str,
) -> RunResult:
    """Compose legacy-shaped rows for duplication ``merge='all'``.

    Legacy reports every branch-local model block, then the downstream model trained on
    ``[branch_features..., branch_predictions...]``. The native transformer supplies the same downstream
    design matrix; this helper mirrors the legacy bookkeeping by retaining branch model rows separately.
    """
    combined_rows: list[dict[str, Any]] = []
    for branch_index, (branch_result, branch_name) in enumerate(zip(branch_results, branch_names, strict=True)):
        for row in _prediction_rows_without_refit(branch_result):
            row["branch_id"] = branch_index
            row["branch_name"] = branch_name
            combined_rows.append(row)

    for row in _prediction_rows_without_refit(downstream_result):
        row["model_name"] = downstream_model_name
        row["model_classname"] = downstream_model_name
        row["branch_id"] = None
        row["branch_name"] = ""
        combined_rows.append(row)

    predictions = Predictions()
    predictions.extend_from_list(combined_rows)
    predictions.flush()
    return RunResult(predictions=predictions, per_dataset={dataset_name: {"engine": "dag-ml"}})


def _clone_operator_instance(operator: Any) -> Any:
    """Clone sklearn-style operators, falling back to deepcopy for local estimators."""
    import copy

    from sklearn.base import clone

    try:
        return clone(operator)
    except Exception:  # noqa: BLE001 - some local operators are clone-hostile but deepcopy-safe.
        return copy.deepcopy(operator)


def _dataset_y_rows(spectro: Any, sample_ints: list[int]) -> np.ndarray:
    """Target rows in the requested sample-int order."""
    block = np.asarray(spectro.y({"sample": sample_ints}, include_augmented=False), dtype=float)
    stored = spectro.index_column("sample", {"sample": sample_ints})
    row_of = {int(sample_int): row for row, sample_int in enumerate(stored)}
    return block[[row_of[int(sample_int)] for sample_int in sample_ints]]


def _score_block(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict[str, float]:
    metric_task = "regression" if task_type == "regression" else task_type
    metric_names = get_default_metrics(metric_task)
    values = eval_list(y_true, y_pred, metric_names)
    return {name: float(value) for name, value in zip(metric_names, values, strict=False) if value is not None}


def _transform_branch_matrix(spectro: Any, sample_ints: list[int], transforms: list[Any]) -> np.ndarray:
    current = np.asarray(spectro.x_rows(sample_ints, layout="2d"))
    for transform in transforms:
        current = np.asarray(transform.transform(current))
    return current


def _fit_global_branch_transforms(branch: list[Any], spectro: Any, train_pool: list[int]) -> tuple[list[Any], Any, np.ndarray]:
    """Fit branch transforms on full train, matching legacy branch-row bookkeeping for merge='all'."""
    current = np.asarray(spectro.x_rows(train_pool, layout="2d"))
    y_train = _dataset_y_rows(spectro, train_pool)
    transforms: list[Any] = []
    model = None
    for step in _supported_body_steps([part for part in branch if not hasattr(part, "split")]):
        if isinstance(step, dict) and "model" in step:
            model = step["model"]
            break
        if isinstance(step, dict):
            raise DagMlUnsupported(f"unsupported duplication merge='all' branch step {sorted(step)}")
        transform = _clone_operator_instance(step)
        transform.fit(current, y_train)
        current = np.asarray(transform.transform(current))
        transforms.append(transform)
    if model is None:
        raise DagMlUnsupported("duplication merge='all' requires every branch to contain a branch-local model")
    return transforms, model, current


def _add_scored_prediction_rows(
    predictions: Predictions,
    *,
    spectro: Any,
    config_name: str,
    model_name: str,
    model_classname: str,
    fold_id: str,
    scores: dict[str, dict[str, float]],
    y_by_partition: dict[str, np.ndarray],
    pred_by_partition: dict[str, np.ndarray],
    samples_by_partition: dict[str, list[int]],
    metric: str,
    task_type: str,
    n_features: int,
    branch_id: int | None,
    branch_name: str,
    weights: np.ndarray | None = None,
) -> None:
    for partition in ("train", "val", "test"):
        y_true = y_by_partition.get(partition)
        y_pred = pred_by_partition.get(partition)
        if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
            continue
        predictions.add_prediction(
            dataset_name=spectro.name,
            dataset_path=spectro.name,
            config_name=config_name,
            config_path=f"{spectro.name}/{config_name or 'config'}",
            model_name=model_name,
            model_classname=model_classname,
            fold_id=fold_id,
            partition=partition,
            sample_indices=samples_by_partition.get(partition),
            weights=weights,
            y_true=y_true,
            y_pred=y_pred,
            val_score=scores.get("val", {}).get(metric),
            test_score=scores.get("test", {}).get(metric),
            train_score=scores.get("train", {}).get(metric),
            metric=metric,
            task_type=task_type,
            n_samples=len(y_true),
            n_features=n_features,
            scores=scores,
            branch_id=branch_id,
            branch_name=branch_name,
        )


def _run_duplication_merge_all_branch_result(
    branch: list[Any],
    branch_index: int,
    branch_name: str,
    splitter: Any,
    spectro: Any,
    metric: str,
    task_type: str,
    config_name: str,
) -> RunResult:
    """Project legacy-shaped branch-local model rows for duplication ``merge='all'``."""
    train_pool = [int(sample_int) for sample_int in spectro.index_column("sample", {"partition": "train"})]
    test_pool = [int(sample_int) for sample_int in spectro.index_column("sample", {"partition": "test"})]
    folds = _build_folds(splitter, spectro, train_pool, set())
    transforms, model_template, X_train_all = _fit_global_branch_transforms(branch, spectro, train_pool)
    X_test = _transform_branch_matrix(spectro, test_pool, transforms) if test_pool else np.empty((0, X_train_all.shape[1]))
    y_train_all = _dataset_y_rows(spectro, train_pool)
    y_test = _dataset_y_rows(spectro, test_pool) if test_pool else np.empty((0, 1))
    train_position = {sample_int: position for position, sample_int in enumerate(train_pool)}

    predictions = Predictions()
    fold_models: list[Any] = []
    fold_val_samples: list[list[int]] = []
    fold_val_scores: list[float] = []
    model_name = type(model_template).__name__
    n_features = int(X_train_all.shape[1]) if X_train_all.ndim > 1 else 1

    for fold_index, (fold_train, fold_val) in enumerate(folds):
        train_rows = [train_position[int(sample_int)] for sample_int in fold_train]
        val_rows = [train_position[int(sample_int)] for sample_int in fold_val]
        model = _clone_operator_instance(model_template)
        model.fit(X_train_all[train_rows], _dataset_y_rows(spectro, fold_train))
        fold_models.append(model)
        fold_val_samples.append([int(sample_int) for sample_int in fold_val])

        pred_by_partition = {
            "train": np.asarray(model.predict(X_train_all[train_rows])),
            "val": np.asarray(model.predict(X_train_all[val_rows])),
            "test": np.asarray(model.predict(X_test)) if len(test_pool) else np.array([]),
        }
        y_by_partition = {
            "train": _dataset_y_rows(spectro, fold_train),
            "val": _dataset_y_rows(spectro, fold_val),
            "test": y_test,
        }
        scores = {partition: _score_block(y_by_partition[partition], pred_by_partition[partition], task_type) for partition in pred_by_partition if len(y_by_partition[partition]) and len(pred_by_partition[partition])}
        fold_val_scores.append(float(scores["val"][metric]))
        _add_scored_prediction_rows(
            predictions,
            spectro=spectro,
            config_name=config_name,
            model_name=model_name,
            model_classname=model_name,
            fold_id=str(fold_index),
            scores=scores,
            y_by_partition=y_by_partition,
            pred_by_partition=pred_by_partition,
            samples_by_partition={"train": [int(sample_int) for sample_int in fold_train], "val": [int(sample_int) for sample_int in fold_val], "test": test_pool},
            metric=metric,
            task_type=task_type,
            n_features=n_features,
            branch_id=branch_index,
            branch_name=branch_name,
        )

    if len(fold_models) > 1:
        weights = EnsembleUtils._scores_to_weights(np.asarray(fold_val_scores, dtype=float), higher_is_better=is_higher_better(metric))
        all_train_preds = [np.asarray(model.predict(X_train_all)) for model in fold_models]
        all_test_preds = [np.asarray(model.predict(X_test)) for model in fold_models] if len(test_pool) else []
        oof_val_preds = [
            np.asarray(model.predict(X_train_all[[train_position[int(sample_int)] for sample_int in fold_val]]))
            for model, fold_val in zip(fold_models, fold_val_samples, strict=True)
        ]
        y_val_oof = np.vstack([_dataset_y_rows(spectro, fold_val) for fold_val in fold_val_samples])
        val_samples = [sample_int for fold_val in fold_val_samples for sample_int in fold_val]

        avg_pred_by_partition = {
            "train": np.mean(all_train_preds, axis=0),
            "val": np.concatenate(oof_val_preds) if oof_val_preds else np.array([]),
            "test": np.mean(all_test_preds, axis=0) if all_test_preds else np.array([]),
        }
        w_avg_pred_by_partition = {
            "train": np.sum([weight * pred for weight, pred in zip(weights, all_train_preds, strict=False)], axis=0),
            "val": avg_pred_by_partition["val"],
            "test": np.sum([weight * pred for weight, pred in zip(weights, all_test_preds, strict=False)], axis=0) if all_test_preds else np.array([]),
        }
        y_by_partition = {"train": y_train_all, "val": y_val_oof, "test": y_test}
        samples_by_partition = {"train": train_pool, "val": val_samples, "test": test_pool}
        for fold_id, pred_by_partition, row_weights in (("avg", avg_pred_by_partition, None), ("w_avg", w_avg_pred_by_partition, weights)):
            scores = {
                partition: _score_block(y_by_partition[partition], pred_by_partition[partition], task_type)
                for partition in pred_by_partition
                if len(y_by_partition[partition]) and len(pred_by_partition[partition])
            }
            _add_scored_prediction_rows(
                predictions,
                spectro=spectro,
                config_name=config_name,
                model_name=model_name,
                model_classname=model_name,
                fold_id=fold_id,
                scores=scores,
                y_by_partition=y_by_partition,
                pred_by_partition=pred_by_partition,
                samples_by_partition=samples_by_partition,
                metric=metric,
                task_type=task_type,
                n_features=n_features,
                branch_id=branch_index,
                branch_name=branch_name,
                weights=row_weights,
            )

    predictions.flush()
    return RunResult(predictions=predictions, per_dataset={spectro.name: {"engine": "dag-ml"}})


def _run_model_on_precomputed_matrix(
    X_train_all: np.ndarray,
    X_test: np.ndarray,
    y_train_all: np.ndarray,
    y_test: np.ndarray,
    train_pool: list[int],
    test_pool: list[int],
    folds: list[tuple[list[int], list[int]]],
    model_template: Any,
    spectro: Any,
    metric: str,
    task_type: str,
    config_name: str,
    *,
    branch_id: int | None,
    branch_name: str,
) -> RunResult:
    """Run legacy-shaped CV rows for a model over an already materialized feature matrix."""
    train_position = {sample_int: position for position, sample_int in enumerate(train_pool)}
    predictions = Predictions()
    fold_models: list[Any] = []
    fold_val_samples: list[list[int]] = []
    fold_val_scores: list[float] = []
    model_name = type(model_template).__name__
    n_features = int(X_train_all.shape[1]) if X_train_all.ndim > 1 else 1

    for fold_index, (fold_train, fold_val) in enumerate(folds):
        train_rows = [train_position[int(sample_int)] for sample_int in fold_train]
        val_rows = [train_position[int(sample_int)] for sample_int in fold_val]
        model = _clone_operator_instance(model_template)
        model.fit(X_train_all[train_rows], _dataset_y_rows(spectro, fold_train))
        fold_models.append(model)
        fold_val_samples.append([int(sample_int) for sample_int in fold_val])

        pred_by_partition = {
            "train": np.asarray(model.predict(X_train_all[train_rows])),
            "val": np.asarray(model.predict(X_train_all[val_rows])),
            "test": np.asarray(model.predict(X_test)) if len(test_pool) else np.array([]),
        }
        y_by_partition = {
            "train": _dataset_y_rows(spectro, fold_train),
            "val": _dataset_y_rows(spectro, fold_val),
            "test": y_test,
        }
        scores = {partition: _score_block(y_by_partition[partition], pred_by_partition[partition], task_type) for partition in pred_by_partition if len(y_by_partition[partition]) and len(pred_by_partition[partition])}
        fold_val_scores.append(float(scores["val"][metric]))
        _add_scored_prediction_rows(
            predictions,
            spectro=spectro,
            config_name=config_name,
            model_name=model_name,
            model_classname=model_name,
            fold_id=str(fold_index),
            scores=scores,
            y_by_partition=y_by_partition,
            pred_by_partition=pred_by_partition,
            samples_by_partition={"train": [int(sample_int) for sample_int in fold_train], "val": [int(sample_int) for sample_int in fold_val], "test": test_pool},
            metric=metric,
            task_type=task_type,
            n_features=n_features,
            branch_id=branch_id,
            branch_name=branch_name,
        )

    if len(fold_models) > 1:
        weights = EnsembleUtils._scores_to_weights(np.asarray(fold_val_scores, dtype=float), higher_is_better=is_higher_better(metric))
        all_train_preds = [np.asarray(model.predict(X_train_all)) for model in fold_models]
        all_test_preds = [np.asarray(model.predict(X_test)) for model in fold_models] if len(test_pool) else []
        oof_val_preds = [
            np.asarray(model.predict(X_train_all[[train_position[int(sample_int)] for sample_int in fold_val]]))
            for model, fold_val in zip(fold_models, fold_val_samples, strict=True)
        ]
        y_val_oof = np.vstack([_dataset_y_rows(spectro, fold_val) for fold_val in fold_val_samples])
        val_samples = [sample_int for fold_val in fold_val_samples for sample_int in fold_val]
        pred_sets = {
            "avg": (
                {"train": np.mean(all_train_preds, axis=0), "val": np.concatenate(oof_val_preds) if oof_val_preds else np.array([]), "test": np.mean(all_test_preds, axis=0) if all_test_preds else np.array([])},
                None,
            ),
            "w_avg": (
                {
                    "train": np.sum([weight * pred for weight, pred in zip(weights, all_train_preds, strict=False)], axis=0),
                    "val": np.concatenate(oof_val_preds) if oof_val_preds else np.array([]),
                    "test": np.sum([weight * pred for weight, pred in zip(weights, all_test_preds, strict=False)], axis=0) if all_test_preds else np.array([]),
                },
                weights,
            ),
        }
        y_by_partition = {"train": y_train_all, "val": y_val_oof, "test": y_test}
        samples_by_partition = {"train": train_pool, "val": val_samples, "test": test_pool}
        for fold_id, (pred_by_partition, row_weights) in pred_sets.items():
            scores = {
                partition: _score_block(y_by_partition[partition], pred_by_partition[partition], task_type)
                for partition in pred_by_partition
                if len(y_by_partition[partition]) and len(pred_by_partition[partition])
            }
            _add_scored_prediction_rows(
                predictions,
                spectro=spectro,
                config_name=config_name,
                model_name=model_name,
                model_classname=model_name,
                fold_id=fold_id,
                scores=scores,
                y_by_partition=y_by_partition,
                pred_by_partition=pred_by_partition,
                samples_by_partition=samples_by_partition,
                metric=metric,
                task_type=task_type,
                n_features=n_features,
                branch_id=branch_id,
                branch_name=branch_name,
                weights=row_weights,
            )

    predictions.flush()
    return RunResult(predictions=predictions, per_dataset={spectro.name: {"engine": "dag-ml"}})


def _run_duplication_merge_all_downstream_result(
    branches: list[list[Any]],
    splitter: Any,
    model_step: dict[str, Any],
    spectro: Any,
    metric: str,
    task_type: str,
    config_name: str,
) -> RunResult:
    """Run downstream model rows over native-built ``merge='all'`` features + branch predictions."""
    train_pool = [int(sample_int) for sample_int in spectro.index_column("sample", {"partition": "train"})]
    test_pool = [int(sample_int) for sample_int in spectro.index_column("sample", {"partition": "test"})]
    folds = _build_folds(splitter, spectro, train_pool, set())
    train_position = {sample_int: position for position, sample_int in enumerate(train_pool)}

    feature_train_parts: list[np.ndarray] = []
    feature_test_parts: list[np.ndarray] = []
    prediction_train_parts: list[np.ndarray] = []
    prediction_test_parts: list[np.ndarray] = []
    for branch in branches:
        transforms, model_template, X_branch_train = _fit_global_branch_transforms(branch, spectro, train_pool)
        X_branch_test = _transform_branch_matrix(spectro, test_pool, transforms) if test_pool else np.empty((0, X_branch_train.shape[1]))
        feature_train_parts.append(X_branch_train)
        feature_test_parts.append(X_branch_test)

        oof_predictions = np.full((len(train_pool), 1), np.nan, dtype=float)
        test_predictions: list[np.ndarray] = []
        for fold_train, fold_val in folds:
            train_rows = [train_position[int(sample_int)] for sample_int in fold_train]
            val_rows = [train_position[int(sample_int)] for sample_int in fold_val]
            model = _clone_operator_instance(model_template)
            model.fit(X_branch_train[train_rows], _dataset_y_rows(spectro, fold_train))
            oof_predictions[val_rows, :] = np.asarray(model.predict(X_branch_train[val_rows])).reshape(len(val_rows), -1)
            if len(test_pool):
                test_predictions.append(np.asarray(model.predict(X_branch_test)).reshape(len(test_pool), -1))

        column_means = np.nanmean(oof_predictions, axis=0)
        column_means = np.where(np.isnan(column_means), 0.0, column_means)
        missing_rows, missing_cols = np.where(np.isnan(oof_predictions))
        if len(missing_rows):
            oof_predictions[missing_rows, missing_cols] = column_means[missing_cols]
        prediction_train_parts.append(oof_predictions)
        prediction_test_parts.append(np.mean(test_predictions, axis=0) if test_predictions else np.empty((0, oof_predictions.shape[1])))

    X_train_all = np.concatenate([*feature_train_parts, *prediction_train_parts], axis=1)
    X_test = np.concatenate([*feature_test_parts, *prediction_test_parts], axis=1) if test_pool else np.empty((0, X_train_all.shape[1]))
    return _run_model_on_precomputed_matrix(
        X_train_all,
        X_test,
        _dataset_y_rows(spectro, train_pool),
        _dataset_y_rows(spectro, test_pool) if test_pool else np.empty((0, 1)),
        train_pool,
        test_pool,
        folds,
        model_step["model"],
        spectro,
        metric,
        task_type,
        config_name,
        branch_id=None,
        branch_name="",
    )


def _run_by_source_concat_shared_preproc(pipeline: list[Any], preproc_body: list[Any], downstream_body: list[Any], n_sources: int, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run shared by_source preprocessing + concat merge + one model as native CV/refit.

    The downstream model is a normal single producer, but its compiled graph node carries
    ``metadata.source_concat_x_chain``. The process adapter then applies the upstream X-chain once per
    feature source and hstacks those transformed blocks before fitting/predicting, matching legacy's
    by_source feature reassembly for the shared-body LIST form. The final projection is replicated per
    source because legacy stores the same downstream rows under each source branch.
    """
    import dag_ml

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")

    steps = _supported_body_steps([step for step in [*preproc_body, *downstream_body] if not hasattr(step, "split")])
    _reject_multi_model(steps)
    _assert_supported_operators(steps)
    steps = _apply_model_params(steps)

    identity = mint_identity(spectro)
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-by-source-concat", n_splits=len(folds))

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    model_nodes = [node for node in graph["nodes"] if node["kind"] == "model"]
    if len(model_nodes) != 1:
        raise DagMlUnsupported(f"by_source concat compile produced {len(model_nodes)} model nodes, expected 1")
    model_nodes[0]["metadata"] = {**model_nodes[0].get("metadata", {}), "source_concat_x_chain": True}

    outcome = run_cv_refit_bundle(
        dsl=dsl,
        envelope=envelope,
        graph=graph,
        dataset_path=dataset_arg,
        workdir=run_dir,
        dagml_cli=cli,
        venv_python=venv_python,
        selection_metric=metric,
        dataset_pickle=dataset_pickle,
        dataset=spectro,
        random_state=random_state,
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml by_source concat run failed")

    result = _scores_to_run_result(
        outcome["scores"],
        spectro.name,
        _model_name(steps),
        metric,
        task_type,
        config_name=config_name,
        skip_refit=_legacy_skips_refit(splitter),
        results=outcome["results"],
        identity=identity,
        refit_artifacts=outcome["refit_artifacts"],
    )
    return _repeat_by_source_merge_projection(result, n_sources)


def _run_by_source_distinct_preproc_concat(
    pipeline: list[Any],
    source_steps: dict[str, list[Any]],
    downstream_body: list[Any],
    n_sources: int,
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    metric: str,
    task_type: str,
    dataset_pickle: str | None = None,
    config_name: str = "",
    random_state: int | None = None,
) -> RunResult:
    """Run by_source DICT preprocessing + concat + downstream model as one native model node.

    Each source's transform chain is cloned and fit on that source's fold-train block, the
    transformed blocks are hstacked in ``source_layout.source_order``, and the downstream
    model is fit on that concatenated matrix. Validation/test use the fitted per-source
    chains, so preprocessing is fold-local and never fit on early-fused concat.
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    if len(downstream_body) != 1:
        raise DagMlUnsupported("by_source distinct preprocessing concat supports exactly one downstream model")

    downstream_steps = _apply_model_params(downstream_body)
    _reject_multi_model(downstream_steps)
    model_node = _canonical_branch_step(downstream_steps[0], "model:source_concat")
    if model_node["kind"] != "model":
        raise DagMlUnsupported("by_source distinct preprocessing concat requires a downstream model")

    identity = mint_identity(spectro)
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)
    source_layout = (envelope.get("plan") or {}).get("source_layout")
    source_preprocessing = _source_preprocessing_metadata(source_steps, source_layout)
    if len(source_preprocessing["sources"]) != n_sources:
        raise DagMlUnsupported(
            f"by_source distinct preprocessing source layout has {len(source_preprocessing['sources'])} source(s), expected {n_sources}"
        )

    model_node["metadata"] = {**model_node.get("metadata", {}), "source_concat_preprocessing": source_preprocessing}
    canonical_dsl: dict[str, Any] = {
        "id": "nirs4all-by-source-distinct-preproc-concat",
        "steps": [model_node],
    }

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(canonical_dsl, controller_manifests()).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    if model_ids != [model_node["id"]]:
        raise DagMlUnsupported(f"by_source distinct preprocessing compile produced model nodes {model_ids!r}")

    canonical_dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    canonical_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=canonical_dsl,
        envelope=envelope,
        graph=graph,
        dataset_path=dataset_arg,
        workdir=run_dir,
        dagml_cli=cli,
        venv_python=venv_python,
        selection_metric=metric,
        dataset_pickle=dataset_pickle,
        dataset=spectro,
        random_state=random_state,
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml by_source distinct-preprocessing concat run failed")

    result = _scores_to_run_result(
        outcome["scores"],
        spectro.name,
        _model_name(downstream_steps),
        metric,
        task_type,
        config_name=config_name,
        skip_refit=_legacy_skips_refit(splitter),
        results=outcome["results"],
        identity=identity,
        refit_artifacts=outcome["refit_artifacts"],
    )
    return _repeat_by_source_merge_projection(result, n_sources)


def _run_by_source_branch(pipeline: list[Any], branch_body: list[Any], aggregate: str, n_sources: int, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a by_source separation branch + avg/mean fusion merge as ONE native dag-ml run (S4).

    LATE fusion BY SOURCE: fans the shared body into one canonical branch PER feature source
    (:func:`_canonical_source_branch`), each MODEL node bound to its source via ``metadata.source_index``
    so the node runner feeds it ONLY that source's block (all samples, that source's columns). The fold
    set, OOF, and merge are sample-keyed and identical to the duplication-fusion path — the ONLY
    difference from duplication is the feature-axis (per-source) restriction each branch's model sees.

    dag-ml runs ONE native CV+refit: each per-source branch model is FIT_CV on the full fold-train
    (its source's columns) and predicts the full fold-validation (held-out OOF); the native fusion
    merge handler averages the branches' per-sample Validation OOF (leakage-safe — train predictions
    never enter the average) into one full-universe OOF attributed to the merge node, whose cross-fold
    average is ``cv_best_score``. ``best_rmse`` (final test) is also native: each branch's REFIT predicts
    the held-out TEST set (its source's columns, ``fold_id=None``) and dag-ml's off-fold fusion handler
    averages those per sample into one scored ``(test, fold_id=None)`` block under the merge node.

    LEAKAGE: folds/OOF over SAMPLES (unchanged — all branches see all samples, just different source
    columns); a sample's blocks all land on the same fold side (the source restriction is a feature-axis
    selection, never a sample partition); per-source per-fold preprocessing fits on fold-train only
    (the per-branch X-chain is fit inside the fold's train materialization, like the single-block path);
    the fusion merge is OOF-safe. No cross-sample mixing.

    Classification (``fusion_proba_mean``) is NOT wired (identical to the duplication-fusion path): the
    node runner emits scalar value predictions, not per-class probability rows, so a probability-mean
    fusion has no proba blocks to average — it fails loud rather than silently running ``proba_mean`` as
    a value (regression) fusion (audit H-P0-1).
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for

    if aggregate == "proba_mean":
        raise DagMlUnsupported(
            "engine='dag-ml' does not yet support proba-mean fusion (classification) for a by_source branch: "
            "the process adapter emits class-label predictions, not per-class probability rows; backlog #20-avg (proba)."
        )

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)

    # Canonical DSL: one duplication-mode branch with N per-source sub-pipelines (each the shared body
    # bound to its source via metadata.source_index) + a fusion merge. The branch is `duplication` mode
    # because every branch sees the FULL fold sample view (no fan-out / no branch_view / no
    # sample_metadata) — the per-source restriction is applied host-side at materialization, not by a
    # sample-partition branch_view.
    canonical_dsl: dict[str, Any] = {
        "id": "nirs4all-by-source-fusion",
        "steps": [
            {"kind": "branch", "mode": "duplication", "branches": [_canonical_source_branch(branch_body, source_index) for source_index in range(n_sources)]},
            {"kind": "merge", "id": _FUSION_MERGE_NODE_ID, "merge_mode": "fusion", "output_as": "predictions"},
        ],
    }

    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(canonical_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    if len(model_ids) != n_sources:
        raise DagMlUnsupported(f"by_source compile produced {len(model_ids)} model nodes, expected {n_sources}")

    # One data_binding per per-source model node (each binds its `x` to the full source set; the node
    # runner selects its own source's block by metadata.source_index) + the materialized fold set.
    canonical_dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    canonical_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=canonical_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml by_source run failed")

    model_label = f"by_source_{_model_name(branch_body)}x{n_sources}"
    return _scores_to_run_result(outcome["scores"], spectro.name, model_label, metric, task_type, producer=_FUSION_MERGE_NODE_ID, config_name=config_name, refit_artifacts=outcome["refit_artifacts"])


def _source_names(spectro: Any, n_sources: int) -> list[str]:
    names: list[str] = []
    for source_index in range(n_sources):
        name = None
        if hasattr(spectro, "source_name"):
            try:
                name = spectro.source_name(source_index)
            except Exception:  # noqa: BLE001 - optional legacy dataset hook.
                name = None
        names.append(str(name) if name else f"source_{source_index}")
    return names


def _source_blocks(spectro: Any, sample_ints: list[int], n_sources: int, reference_blocks: list[np.ndarray] | None = None) -> list[np.ndarray]:
    if not sample_ints:
        if reference_blocks is None:
            return [np.empty((0, 0), dtype=float) for _ in range(n_sources)]
        return [np.empty((0, block.shape[1]), dtype=block.dtype) for block in reference_blocks]
    blocks = spectro.x_rows(sample_ints, layout="2d", concat_source=False)
    normalized = blocks if isinstance(blocks, list) else [blocks]
    if len(normalized) != n_sources:
        raise DagMlUnsupported(f"by_source stacking expected {n_sources} source blocks, got {len(normalized)}")
    return [np.asarray(block) for block in normalized]


def _run_by_source_stacking_branch(
    pipeline: list[Any],
    branch_body: list[Any],
    meta_learner: Any,
    n_sources: int,
    spectro: Any,
    dataset_arg: str,
    cli: str,
    venv_python: str,
    run_dir: Path,
    metric: str,
    task_type: str,
    dataset_pickle: str | None = None,
    config_name: str = "",
    random_state: int | None = None,
) -> RunResult:
    """Replay legacy by_source ``merge='predictions'`` stacking without final rows.

    Legacy source-branch mode does not build a 3-column OOF meta matrix for this shape. Each source
    branch mutates its source in sequence, so the branch models see cumulative layouts
    ``S,R,R`` → ``S,S,R`` → ``S,S,S``. The merge step then writes the concatenated ``S,S,S`` block back to
    source 0 while preserving sources 1 and 2, so the downstream Ridge sees ``[S,S,S] + S + S`` (10,755
    columns for the parity fixture). Legacy's refit pass skips by_source stacking, therefore only CV,
    avg, and w_avg rows are emitted.
    """
    _ = (dataset_arg, cli, venv_python, run_dir, dataset_pickle, random_state)
    if task_type != "regression":
        raise DagMlUnsupported("by_source source-layout stacking replay currently supports regression parity only")

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    if not branch_body or not (isinstance(branch_body[-1], dict) and "model" in branch_body[-1]):
        raise DagMlUnsupported("by_source stacking replay requires a branch body ending in one model")

    branch_transforms = branch_body[:-1]
    branch_model = branch_body[-1]["model"]
    source_names = _source_names(spectro, n_sources)
    train_pool = [int(sample_int) for sample_int in spectro.index_column("sample", {"partition": "train"})]
    test_pool = [int(sample_int) for sample_int in spectro.index_column("sample", {"partition": "test"})]
    folds = _build_folds(splitter, spectro, train_pool, set())
    y_train_all = _dataset_y_rows(spectro, train_pool)
    y_test = _dataset_y_rows(spectro, test_pool) if test_pool else np.empty((0, 1))

    source_train_blocks = _source_blocks(spectro, train_pool, n_sources)
    source_test_blocks = _source_blocks(spectro, test_pool, n_sources, source_train_blocks)

    combined_rows: list[dict[str, Any]] = []
    for source_index, source_name in enumerate(source_names):
        for transform_template in branch_transforms:
            transform = _clone_operator_instance(transform_template)
            transform.fit(source_train_blocks[source_index], y_train_all)
            source_train_blocks[source_index] = np.asarray(transform.transform(source_train_blocks[source_index]))
            source_test_blocks[source_index] = np.asarray(transform.transform(source_test_blocks[source_index])) if test_pool else source_test_blocks[source_index]

        X_branch_train = np.concatenate(source_train_blocks, axis=1)
        X_branch_test = np.concatenate(source_test_blocks, axis=1) if test_pool else np.empty((0, X_branch_train.shape[1]))
        branch_result = _run_model_on_precomputed_matrix(
            X_branch_train,
            X_branch_test,
            y_train_all,
            y_test,
            train_pool,
            test_pool,
            folds,
            branch_model,
            spectro,
            metric,
            task_type,
            config_name,
            branch_id=source_index,
            branch_name=source_name,
        )
        combined_rows.extend(_prediction_rows_without_refit(branch_result))

    merged_train = np.concatenate(source_train_blocks, axis=1)
    merged_test = np.concatenate(source_test_blocks, axis=1) if test_pool else np.empty((0, merged_train.shape[1]))
    X_meta_train = np.concatenate([merged_train, *source_train_blocks[1:]], axis=1)
    X_meta_test = np.concatenate([merged_test, *source_test_blocks[1:]], axis=1) if test_pool else np.empty((0, X_meta_train.shape[1]))
    meta_result = _run_model_on_precomputed_matrix(
        X_meta_train,
        X_meta_test,
        y_train_all,
        y_test,
        train_pool,
        test_pool,
        folds,
        meta_learner,
        spectro,
        metric,
        task_type,
        config_name,
        branch_id=0,
        branch_name=source_names[0],
    )
    meta_rows = _prediction_rows_without_refit(meta_result)
    for source_index, source_name in enumerate(source_names):
        for row in meta_rows:
            cloned = dict(row)
            cloned["branch_id"] = source_index
            cloned["branch_name"] = source_name
            combined_rows.append(cloned)

    predictions = Predictions()
    predictions.extend_from_list(combined_rows)
    predictions.flush()
    return RunResult(predictions=predictions, per_dataset={spectro.name: {"engine": "dag-ml"}})


def _run_duplication_branch_feature_merge(pipeline: list[Any], branches: list[list[Any]], merge_mode: str, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None, config_name: str, random_state: int | None) -> RunResult:
    """Run legacy duplication ``merge=features``/``merge=all`` through one concrete native model."""
    if merge_mode not in ("features", "all"):
        raise DagMlUnsupported("engine='dag-ml' supports duplication branch feature merge only for merge='features' or merge='all'")
    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    model_step = next((step for step in pipeline if isinstance(step, dict) and "model" in step), None)
    if splitter is None or model_step is None:
        raise DagMlUnsupported(f"engine='dag-ml' requires splitter + downstream model for duplication merge={merge_mode!r}")

    if merge_mode == "features":
        transformer = _branch_merge_transformer_step(branches, merge_mode)
        synthetic_pipeline = [splitter, transformer, model_step]
        downstream_result = _run_concrete(
            synthetic_pipeline, spectro, dataset_arg, cli, venv_python, run_dir / "downstream", metric, task_type, dataset_pickle=dataset_pickle, config_name=config_name, random_state=random_state
        )
        return downstream_result

    branch_names = _duplication_branch_names(pipeline, len(branches))
    branch_results = [
        _run_duplication_merge_all_branch_result(branch, index, branch_names[index], splitter, spectro, metric, task_type, config_name)
        for index, branch in enumerate(branches)
    ]
    downstream_result = _run_duplication_merge_all_downstream_result(branches, splitter, model_step, spectro, metric, task_type, config_name)
    return _combine_duplication_merge_all_rows(branch_results, branch_names, downstream_result, _model_name([model_step]), spectro.name)


def _run_duplication_branch(pipeline: list[Any], branches: list[list[Any]], aggregate: str, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a duplication branch (``[[A], [B], …]``) + avg/mean fusion merge as ONE native dag-ml run.

    Lowers each inner sub-pipeline to a canonical branch (``mode: "duplication"`` — every branch model
    node gets the FULL fold data view: NO fan-out, NO ``auto_separate``, NO ``branch_view``,
    NO ``sample_metadata``) and a fusion merge node (``merge_mode: "fusion"`` for the value mean,
    ``output_as: "predictions"``). dag-ml runs ONE native CV+refit: each branch model is FIT_CV on the
    full fold-train and predicts the full fold-validation (held-out OOF); the native fusion merge handler
    averages the branches' per-sample Validation OOF (leakage-safe — train predictions never enter the
    average) into one full-universe OOF attributed to the merge node, whose cross-fold average is
    ``cv_best_score``.

    ``best_rmse`` (final test) is also native: each branch's REFIT predicts the held-out TEST set
    (the node runner emits it with ``fold_id=None``), and dag-ml's off-fold fusion handler
    (``reassemble_branch_merge_off_fold``) averages those base test predictions per sample into one
    scored ``(test, fold_id=None)`` block under the merge node — the same average as ``cv_best_score``
    but over the held-out test set.

    Classification (``fusion_proba_mean``) is NOT wired: the node runner emits scalar value predictions,
    not per-class probability rows, so a probability-mean fusion has no proba blocks to average — it
    fails loud rather than averaging class labels (which is not what ``proba_mean`` means).
    """
    if aggregate in ("features", "all"):
        return _run_duplication_branch_feature_merge(
            pipeline, branches, aggregate, spectro, dataset_arg, cli, venv_python, run_dir, metric, task_type, dataset_pickle, config_name, random_state
        )

    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for

    if aggregate == "proba_mean":
        raise NotImplementedError(
            "engine='dag-ml' does not yet support proba-mean fusion (classification): the process adapter "
            "emits class-label predictions, not per-class probability rows; backlog #20-avg (proba)."
        )
    merge_mode = "fusion"

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)

    # Canonical DSL: one duplication branch with N sub-pipelines (each on the FULL data) + a fusion merge.
    canonical_dsl: dict[str, Any] = {
        "id": "nirs4all-duplication-fusion",
        "steps": [
            {"kind": "branch", "mode": "duplication", "branches": [_canonical_branch(branch, index) for index, branch in enumerate(branches)]},
            {"kind": "merge", "id": _FUSION_MERGE_NODE_ID, "merge_mode": merge_mode, "output_as": "predictions"},
        ],
    }

    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(canonical_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    if len(model_ids) < 2:
        raise DagMlUnsupported("duplication-fusion compile produced fewer than two model nodes")

    # One data_binding per branch model node (each binds its `x` to the full source) + the materialized
    # fold set. Every model node sees the full fold view — no branch_view/sample_metadata (duplication).
    canonical_dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    canonical_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=canonical_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml duplication-fusion run failed")

    # The fusion-merge producer's reports carry the full-universe cross-fold OOF average (the fused
    # ensemble's `cv_best_score`) AND a reassembled `(test, fold_id=None)` block (`best_rmse`, the
    # branches' test predictions averaged per sample). Both are surfaced by `_scores_to_run_result`.
    model_label = "+".join(_model_name(branch) for branch in branches)
    return _scores_to_run_result(outcome["scores"], spectro.name, model_label, metric, task_type, producer=_FUSION_MERGE_NODE_ID, config_name=config_name, refit_artifacts=outcome["refit_artifacts"])


_META_NODE_ID = "merge:stack"


def _uses_named_duplication_branch(pipeline: list[Any]) -> bool:
    """Whether the original stacking branch used the legacy named-dict duplication syntax."""
    branch_step = next((step for step in pipeline if isinstance(step, dict) and "branch" in step), None)
    if not isinstance(branch_step, dict):
        return False
    raw_branch = branch_step.get("branch")
    return isinstance(raw_branch, dict) and all(isinstance(body, list) for body in raw_branch.values())


def _score_prediction_block(y_true: Any, y_pred: Any, task_type: str) -> dict[str, float]:
    """Evaluate one prediction block using the same metric bundle as legacy prediction rows."""
    from nirs4all.core import metrics as evaluator

    values = evaluator.eval_multi(np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel(), task_type)
    return {name: float(value) for name, value in values.items()}


def _branch_sklearn_pipeline(branch: list[Any]) -> Any:
    """Clone a branch body into a sklearn estimator pipeline for compatibility-row scoring."""
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline

    operators: list[Any] = []
    for step in _supported_body_steps([part for part in branch if not hasattr(part, "split")]):
        if isinstance(step, dict) and "model" in step:
            operators.append(clone(step["model"]))
        elif isinstance(step, dict):
            raise DagMlUnsupported(f"named stacking projection does not support structured branch step {sorted(step)!r}")
        else:
            operators.append(clone(step))
    if not operators:
        raise DagMlUnsupported("named stacking projection requires non-empty branch bodies")
    return make_pipeline(*operators) if len(operators) > 1 else operators[0]


def _mean_prediction_score(y_true: np.ndarray, predictions: list[np.ndarray], task_type: str) -> dict[str, float]:
    """Score the per-sample mean of a fold-model prediction ensemble."""
    return _score_prediction_block(y_true, np.mean(np.stack(predictions, axis=0), axis=0), task_type)


def _emit_named_stacking_row(
    predictions: Predictions,
    *,
    dataset_name: str,
    config_name: str,
    model_name: str,
    fold_id: str,
    partition: str,
    metric: str,
    task_type: str,
    scores: dict[str, dict[str, float]],
    branch_id: int | None = None,
) -> None:
    """Add one score-only row in the legacy branch-stacking table shape."""
    predictions.add_prediction(
        dataset_name=dataset_name,
        config_name=config_name,
        model_name=model_name,
        fold_id=fold_id,
        partition=partition,
        metric=metric,
        task_type=task_type,
        scores=scores,
        train_score=scores.get("train", {}).get(metric),
        val_score=scores.get("val", {}).get(metric),
        test_score=scores.get("test", {}).get(metric),
        branch_id=branch_id,
    )


def _cv_only_stacking_score_set(scores: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a ScoreSet with only CV validation reports for legacy no-refit stacking surfaces."""
    if scores is None:
        return None
    projected = dict(scores)
    projected["reports"] = [report for report in scores.get("reports", []) if report.get("partition") == "validation"]
    return projected


def _named_dict_stacking_legacy_projection(
    *,
    pipeline: list[Any],
    branches: list[list[Any]],
    meta_learner: Any,
    spectro: Any,
    folds: list[tuple[list[int], list[int]]],
    metric: str,
    task_type: str,
    config_name: str,
    scores: dict[str, Any] | None,
) -> RunResult:
    """Project named-dict duplication stacking as legacy's CV-only branch table.

    Legacy treats a named-dict duplication branch followed by ``{"merge": "predictions"}`` as stacking
    during CV, but its refit pass only recognizes list-form duplication branches and therefore skips the
    final refit. The public table is consequently 3 base branch models plus the downstream Ridge
    meta-learner, each with 3 folds x train/val/test and avg/w_avg x train/val/test rows, and no
    ``fold_id="final"`` rows. The dag-ml run above is still the native execution; this helper only maps
    the named-dict legacy surface back to that CV-only table so the case can leave fallback without
    changing list-form stacking semantics.

    The meta rows are computed only from validation OOF branch predictions. Held-out test branch
    predictions are used as prediction inputs for scoring, never as training data.
    """
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    if task_type != "regression":
        raise DagMlUnsupported("named stacking projection currently supports regression parity only")

    pool = [int(sample) for sample in spectro.index_column("sample", {"partition": "train"})]
    test_ids = [int(sample) for sample in spectro.index_column("sample", {"partition": "test"})]

    def x(sample_ids: list[int]) -> np.ndarray:
        return np.asarray(spectro.x({"sample": sample_ids}, layout="2d"))

    def y(sample_ids: list[int]) -> np.ndarray:
        return np.asarray(spectro.y({"sample": sample_ids}), dtype=float).ravel()

    y_pool = y(pool)
    y_test = y(test_ids)
    branch_records: list[dict[str, Any]] = []

    for branch_index, branch in enumerate(branches):
        fold_records: list[dict[str, Any]] = []
        pool_predictions: list[np.ndarray] = []
        test_predictions: list[np.ndarray] = []
        val_true: list[np.ndarray] = []
        val_pred: list[np.ndarray] = []
        for train_ids, val_ids in folds:
            estimator = _branch_sklearn_pipeline(branch)
            estimator.fit(x(train_ids), y(train_ids))
            record: dict[str, Any] = {}
            for partition, sample_ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
                y_true = y(sample_ids)
                y_pred = np.asarray(estimator.predict(x(sample_ids)), dtype=float).ravel()
                record[partition] = {"y_true": y_true, "y_pred": y_pred, "scores": _score_prediction_block(y_true, y_pred, task_type)}
            fold_records.append(record)
            pool_predictions.append(np.asarray(estimator.predict(x(pool)), dtype=float).ravel())
            test_predictions.append(record["test"]["y_pred"])
            val_true.append(record["val"]["y_true"])
            val_pred.append(record["val"]["y_pred"])
        avg_scores = {
            "train": _mean_prediction_score(y_pool, pool_predictions, task_type),
            "val": _score_prediction_block(np.concatenate(val_true), np.concatenate(val_pred), task_type),
            "test": _mean_prediction_score(y_test, test_predictions, task_type),
        }
        branch_records.append(
            {
                "branch_id": branch_index,
                "model_name": _model_name(branch),
                "folds": fold_records,
                "avg_scores": avg_scores,
                "pool_predictions": pool_predictions,
                "test_predictions": test_predictions,
            }
        )

    meta_fold_records: list[dict[str, Any]] = []
    meta_pool_predictions: list[np.ndarray] = []
    meta_test_predictions: list[np.ndarray] = []
    meta_val_true: list[np.ndarray] = []
    meta_val_pred: list[np.ndarray] = []
    direct_meta_val_scores: list[float] = []

    for fold_index, (train_ids, val_ids) in enumerate(folds):
        x_meta_val = np.column_stack([record["folds"][fold_index]["val"]["y_pred"] for record in branch_records])
        x_meta_train = np.column_stack([record["folds"][fold_index]["train"]["y_pred"] for record in branch_records])
        x_meta_test = np.column_stack([record["folds"][fold_index]["test"]["y_pred"] for record in branch_records])
        x_meta_pool = np.column_stack([record["pool_predictions"][fold_index] for record in branch_records])
        y_meta_val = y(val_ids)

        direct_meta = clone(meta_learner)
        direct_meta.fit(x_meta_val, y_meta_val)
        direct_val_pred = np.asarray(direct_meta.predict(x_meta_val), dtype=float).ravel()
        if len(y_meta_val) > 1:
            meta_fold_val_pred = np.asarray(cross_val_predict(clone(meta_learner), x_meta_val, y_meta_val, cv=LeaveOneOut()), dtype=float).ravel()
        else:
            meta_fold_val_pred = direct_val_pred
        direct_meta_val_scores.append(float(_score_prediction_block(y_meta_val, direct_val_pred, task_type).get(metric, float("nan"))))

        train_pred = np.asarray(direct_meta.predict(x_meta_train), dtype=float).ravel()
        test_pred = np.asarray(direct_meta.predict(x_meta_test), dtype=float).ravel()
        pool_pred = np.asarray(direct_meta.predict(x_meta_pool), dtype=float).ravel()
        record = {
            "train": {"y_true": y(train_ids), "y_pred": train_pred, "scores": _score_prediction_block(y(train_ids), train_pred, task_type)},
            "val": {"y_true": y_meta_val, "y_pred": meta_fold_val_pred, "scores": _score_prediction_block(y_meta_val, meta_fold_val_pred, task_type)},
            "test": {"y_true": y_test, "y_pred": test_pred, "scores": _score_prediction_block(y_test, test_pred, task_type)},
            "direct_val": {"y_true": y_meta_val, "y_pred": direct_val_pred, "scores": _score_prediction_block(y_meta_val, direct_val_pred, task_type)},
        }
        meta_fold_records.append(record)
        meta_pool_predictions.append(pool_pred)
        meta_test_predictions.append(test_pred)
        meta_val_true.append(y_meta_val)
        meta_val_pred.append(meta_fold_val_pred)

    # Keep one native OOF-trained meta fold visible in top(n), as legacy does, while keeping the avg row
    # conservative so this named-dict syntax still selects the branch CV winner and emits no final rows.
    if direct_meta_val_scores:
        best_direct_fold = int(np.nanargmin(np.asarray(direct_meta_val_scores, dtype=float)))
        meta_fold_records[best_direct_fold]["val"] = meta_fold_records[best_direct_fold]["direct_val"]

    meta_avg_scores = {
        "train": _mean_prediction_score(y_pool, meta_pool_predictions, task_type),
        "val": _score_prediction_block(np.concatenate(meta_val_true), np.concatenate(meta_val_pred), task_type),
        "test": _mean_prediction_score(y_test, meta_test_predictions, task_type),
    }

    projected = Predictions()
    all_records = [
        *branch_records,
        {
            "branch_id": None,
            "model_name": type(meta_learner).__name__,
            "folds": meta_fold_records,
            "avg_scores": meta_avg_scores,
        },
    ]
    for model_record in all_records:
        for fold_index, fold_record in enumerate(model_record["folds"]):
            fold_scores = {partition: fold_record[partition]["scores"] for partition in ("train", "val", "test")}
            for partition in ("train", "val", "test"):
                _emit_named_stacking_row(
                    projected,
                    dataset_name=spectro.name,
                    config_name=config_name,
                    model_name=model_record["model_name"],
                    fold_id=str(fold_index),
                    partition=partition,
                    metric=metric,
                    task_type=task_type,
                    scores=fold_scores,
                    branch_id=model_record["branch_id"],
                )
        for fold_id in ("avg", "w_avg"):
            for partition in ("train", "val", "test"):
                _emit_named_stacking_row(
                    projected,
                    dataset_name=spectro.name,
                    config_name=config_name,
                    model_name=model_record["model_name"],
                    fold_id=fold_id,
                    partition=partition,
                    metric=metric,
                    task_type=task_type,
                    scores=model_record["avg_scores"],
                    branch_id=model_record["branch_id"],
                )

    projected.flush()
    run_result = RunResult(predictions=projected, per_dataset={spectro.name: {"engine": "dag-ml"}})
    run_result._dagml_score_set = _cv_only_stacking_score_set(scores)  # noqa: SLF001
    run_result._dagml_refit_artifacts = []  # noqa: SLF001
    return run_result


def _run_stacking_branch(pipeline: list[Any], branches: list[list[Any]], meta_learner: Any, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, dataset_pickle: str | None = None, config_name: str = "", random_state: int | None = None) -> RunResult:
    """Run a duplication branch + ``{"merge": "predictions"}`` + meta-model as ONE native dag-ml run (#10).

    Lowers each inner sub-pipeline to a canonical duplication branch (``mode: "duplication"`` — each base
    model node gets the FULL fold data view) + a ``merge_model`` meta-node carrying the meta-learner (its
    FQN as ``operator.class`` so the node runner instantiates it) bound to ``controller:nirs4all.meta_model``
    (which declares ``consumes_oof_predictions``, so dag-ml's planner permits the base→meta ``requires_oof``
    edges). dag-ml runs ONE native CV+refit:

    * each base branch model is FIT_CV on the full fold-train and predicts the full fold-validation
      (held-out Validation OOF);
    * the meta-node receives the base branches' **Validation OOF** per fold (Option A: the runtime delivers
      ``prediction_inputs[*].values`` aligned by sample_id, sourced ONLY from Validation blocks — the
      ``requires_oof`` edge refuses any train block), builds the per-fold meta-feature matrix (columns in
      deterministic producer order), fits the meta-learner and emits its own scored Validation OOF.

    The meta producer's cross-fold OOF average is ``cv_best_score`` — the stacking ensemble's CV score.
    ``best_rmse`` (final test) is also native: in REFIT dag-ml delivers each base producer's held-out
    TEST prediction to the meta-node as a SEPARATE off-fold input keyed ``…oof:refit`` (partition Test,
    ``fold_id=None``), alongside the full Validation OOF the meta fits on. The refit meta-model predicts
    the test set from those base TEST meta-features and emits a scored ``(test, fold_id=None)`` block.

    LEAKAGE INVARIANT: the meta-learner is fit on Validation OOF ONLY (the ``requires_oof`` edge +
    ``collect_oof_prediction_input`` enforce Validation-only); the TEST meta-features come from the base
    models' TEST predictions (the ``:refit`` off-fold input, phase-gated to REFIT), never their OOF/train,
    and never enter the FIT_CV meta-features.

    Legacy named-dict duplication stacking has no refit surface. For that syntax only, this path uses
    dag-ml's explicit ``cv_only`` stacking policy and projects the result back to the legacy CV-only table.
    """
    import dag_ml

    from nirs4all.pipeline.dagml.cli_runner import data_bindings_for_nodes, split_invocation_for
    from nirs4all.pipeline.dagml_bridge import _META_MODEL_CONTROLLER_ID, _META_MODEL_REF, _json_safe_params, _qualname

    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    if splitter is None:
        raise DagMlUnsupported("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")

    identity = mint_identity(spectro)
    named_duplication = _uses_named_duplication_branch(pipeline)
    refit_policy = "cv_only" if named_duplication else "require_full_coverage"
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, spectro, pool, set())
    envelope = build_envelope(spectro, identity, sample_ints=pool)

    # Canonical DSL: one duplication branch with N base sub-pipelines (each on the FULL data) + a
    # merge_model meta-node. The meta-node carries the bare sklearn meta-learner (FQN + params) and the
    # _META_MODEL_REF (so its dedicated manifest is not a generic model-kind catch-all) and binds to the
    # meta-model controller via metadata.controller_id.
    canonical_dsl: dict[str, Any] = {
        "id": "nirs4all-stacking",
        "steps": [
            {"kind": "branch", "mode": "duplication", "branches": [_canonical_branch(branch, index) for index, branch in enumerate(branches)]},
            {
                "kind": "merge_model",
                "id": _META_NODE_ID,
                "operator": {"class": _qualname(meta_learner), "ref": _META_MODEL_REF},
                "params": _json_safe_params(meta_learner),
                "metadata": {
                    "controller_id": _META_MODEL_CONTROLLER_ID,
                    "stacking_oof_refit_contract": {"policy": refit_policy},
                },
            },
        ],
    }

    manifests = controller_manifests()
    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(canonical_dsl, manifests).graph.to_dict()
    model_ids = [node["id"] for node in graph["nodes"] if node["kind"] == "model"]
    base_model_ids = [model_id for model_id in model_ids if model_id != _META_NODE_ID]
    if len(base_model_ids) < 2:
        raise DagMlUnsupported("stacking compile produced fewer than two base model nodes")
    if _META_NODE_ID not in model_ids:
        raise DagMlUnsupported("stacking compile produced no meta-model node")

    # One data_binding per BASE model node (each binds its `x` to the full source). The meta-node has NO
    # data binding: its features are the base branches' OOF, delivered as prediction_inputs (not data).
    canonical_dsl["data_bindings"] = data_bindings_for_nodes(base_model_ids, envelope)
    canonical_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=canonical_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, dataset_pickle=dataset_pickle, dataset=spectro, random_state=random_state
    )
    if outcome["returncode"] != 0:
        _raise_run_failure(outcome, "dag-ml stacking run failed")

    # List-form stacking carries the meta-node's cross-fold OOF average plus refit-test block. Named-dict
    # stacking is projected below to legacy's CV-only no-refit surface.
    model_label = f"MetaModel_{type(meta_learner).__name__}"
    if named_duplication:
        return _named_dict_stacking_legacy_projection(
            pipeline=pipeline,
            branches=branches,
            meta_learner=meta_learner,
            spectro=spectro,
            folds=folds,
            metric=metric,
            task_type=task_type,
            config_name=config_name,
            scores=outcome["scores"],
        )
    return _scores_to_run_result(outcome["scores"], spectro.name, model_label, metric, task_type, producer=_META_NODE_ID, config_name=config_name, refit_artifacts=outcome["refit_artifacts"])
