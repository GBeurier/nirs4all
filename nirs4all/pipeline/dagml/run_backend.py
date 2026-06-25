"""Run a nirs4all pipeline on the **dag-ml** engine and return a ``RunResult`` (ADR-17 backend).

This is the operational seam for ``engine="dag-ml"``: it assembles the executable compat DSL,
drives ``dag-ml-cli`` through the nirs4all process adapter, and maps dag-ml's **native**
``bundle.scores`` — per-fold validation RMSE/R², the cross-fold OOF average (``cv_best_score``)
and the final-test score (``best_rmse``), all computed in Rust — into an in-memory
:class:`~nirs4all.data.predictions.Predictions`, wrapped in a :class:`~nirs4all.api.result.RunResult`.

No workspace is created and no scoring happens Python-side: the numbers are dag-ml's. Supports the
vertical-slice shape (feature transforms + one model + an OOF/KFold-style splitter). Non-partition
CV (e.g. ``ShuffleSplit``) is not yet supported by the dag-ml ``FoldSet`` (see migration notes).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.api.result import RunResult
from nirs4all.data.config import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.dagml_bridge import controller_manifests

from .cli_runner import assemble_cv_refit_dsl, run_cv_refit_bundle
from .envelope import build_envelope
from .identity import mint_identity

_DEFAULT_CLI = Path(__file__).resolve().parents[4] / "dag-ml" / "target" / "release" / "dag-ml-cli"


def _split_pipeline(pipeline: list[Any]) -> tuple[list[Any], Any]:
    """Separate the cross-validator step (the object exposing ``.split``) from the operator steps."""
    splitter = next((step for step in pipeline if hasattr(step, "split")), None)
    steps = [step for step in pipeline if step is not splitter]
    return steps, splitter


def _is_exclude_step(step: Any) -> bool:
    return isinstance(step, dict) and "exclude" in step


def _excluded_from_pool(exclude_step: dict[str, Any], spectro: Any, pool_ints: list[int]) -> set[int]:
    """Excluded sample ints from ``pool_ints`` for one ``exclude_step``, mirroring ExcludeController.

    Fits each :class:`~nirs4all.operators.filters.base.SampleFilter` on the CURRENT kept pool's X/y
    (``include_augmented=False``) and combines the per-filter keep-masks by ``mode`` — exactly the
    legacy :class:`~nirs4all.controllers.data.exclude.ExcludeController` mask logic:

    * ``mode="any"`` → exclude if ANY filter flags = ``np.all`` of the keep-masks (exclude.py:193);
    * ``mode="all"`` → exclude only if ALL filters flag = ``np.any`` (exclude.py:196).

    Two legacy edge behaviors are replicated:

    * **Per-filter ``ValueError`` → neutral keep-all** (exclude.py:175-184): a filter that fails to
      fit/mask (e.g. insufficient data) contributes a keep-all mask rather than propagating.
    * **All-excluded guard** (exclude.py:213-222): if the COMBINED keep-mask would exclude every row,
      keep the first sample so exclusion never empties the pool.

    The engine consumes the result as identity (a sample-int set) instead of marking the indexer.
    """
    from nirs4all.controllers.data.exclude import ExcludeController

    controller = ExcludeController()
    filters, filter_mode, _cascade = controller._parse_config(exclude_step)  # noqa: SLF001 - reuse legacy parsing
    if not filters:
        raise ValueError("exclude keyword requires at least one filter")
    if not pool_ints:
        return set()

    x_pool = np.asarray(spectro.x({"sample": list(pool_ints)}, layout="2d", concat_source=True, include_augmented=False))
    y_pool = np.asarray(spectro.y({"sample": list(pool_ints)}, include_augmented=False))
    if y_pool.ndim > 1:
        y_pool = y_pool.flatten()
    if y_pool.size == 0:
        return set()
    # `spectro.x/y({"sample": ids})` returns ascending storage order, not request order; re-key so the
    # mask aligns to `pool_ints` exactly (the storage-vs-request trap the resolver also guards against).
    stored = spectro.index_column("sample", {"sample": list(pool_ints)})
    row_of = {int(sample_int): row for row, sample_int in enumerate(stored)}
    order = [row_of[int(sample_int)] for sample_int in pool_ints]
    x_pool, y_pool = x_pool[order], y_pool[order]

    masks: list[np.ndarray] = []
    for filter_obj in filters:
        try:
            filter_obj.fit(x_pool, y_pool)
            masks.append(filter_obj.get_mask(x_pool, y_pool))
        except ValueError:
            # exclude.py:175-184 — a filter that can't be applied contributes a neutral keep-all mask.
            masks.append(np.ones(len(pool_ints), dtype=bool))

    if len(masks) == 1:
        keep_mask = masks[0].copy()
    else:
        stacked = np.stack(masks, axis=0)
        keep_mask = np.all(stacked, axis=0) if filter_mode == "any" else np.any(stacked, axis=0)

    # exclude.py:213-222 — never empty the pool: if all rows would be excluded, keep the first.
    if not keep_mask.any():
        keep_mask[0] = True

    return {int(sample_int) for sample_int, keep in zip(pool_ints, keep_mask, strict=True) if not keep}


def _resolve_exclude(pipeline: list[Any], spectro: Any) -> tuple[list[Any], list[int], set[int]]:
    """Consume ALL ``exclude`` steps and return ``(pipeline_without_exclude, cv_pool, excluded)``.

    Mirrors the verified legacy + opt-in semantics:

    * **No exclude step** → ``(pipeline, full_train, set())``.
    * **``keep_in_oof=False`` (default = legacy parity)** → the CV pool is the train universe MINUS
      the excluded ints; excluded samples are absent from the folds AND the envelope (removed from
      the CV universe entirely, matching legacy: the splitter runs over ``include_excluded=False``).
      The native ``excluded`` bit is unused (``excluded`` set is empty for the envelope).
    * **``keep_in_oof=True`` (opt-in, leakage-pure)** → the CV pool is the FULL train universe; the
      excluded ints are marked in the envelope so Phase 1's native bit drops them from each fold's
      TRAIN while keeping them in validation/OOF.

    Multiple ``exclude`` steps are applied SEQUENTIALLY, exactly as legacy: each step's filter fits on
    the CURRENT kept train (``include_excluded=False``), i.e. the pool after the earlier steps'
    exclusions (exclude.py:135-137 reads ``include_excluded=False``), so the excluded set is built
    progressively. The ``keep_in_oof`` flag is honored from any exclude step (consistent across steps
    is the caller's contract). All ``exclude`` steps are removed from the remaining pipeline — none is
    lowered to a dag-ml node (the bridge still raises ``NotImplementedError`` for a raw ``exclude``).
    """
    train_ints = [int(sample_int) for sample_int in spectro.index_column("sample", {"partition": "train"})]
    exclude_steps = [step for step in pipeline if _is_exclude_step(step)]
    if not exclude_steps:
        return pipeline, train_ints, set()

    keep_in_oof = any(bool(step.get("keep_in_oof", False)) for step in exclude_steps)
    excluded: set[int] = set()
    for step in exclude_steps:
        current_pool = [sample_int for sample_int in train_ints if sample_int not in excluded]
        excluded |= _excluded_from_pool(step, spectro, current_pool)

    remaining = [step for step in pipeline if not _is_exclude_step(step)]
    if keep_in_oof:
        # Opt-in: keep excluded in the CV universe; mark them excluded in the envelope (native bit)
        # and (host-side) drop them from each fold's TRAIN below so the OOF is leakage-pure.
        return remaining, train_ints, excluded
    # Default (legacy): drop excluded from the CV universe entirely; envelope marks nothing excluded.
    pool = [sample_int for sample_int in train_ints if sample_int not in excluded]
    return remaining, pool, set()


def _build_folds(splitter: Any, pool: list[int], excluded: set[int]) -> list[tuple[list[int], list[int]]]:
    """Split ``pool`` and drop ``excluded`` from each fold's TRAIN, keeping them in VALIDATION.

    In legacy mode ``excluded`` is empty (excluded samples are already absent from ``pool``), so this
    is a plain split. In the opt-in (``keep_in_oof=True``) mode ``pool`` is the full train and
    ``excluded`` is non-empty: excluded samples stay in each fold's validation (predicted in OOF) but
    are removed from its train pool — the leakage-pure semantic, materialized in the host FoldSet (the
    adapter owns the split; dag-ml has no runtime splitter, so the FoldSet's ``train_sample_ids`` are
    authoritative for what the node trains on). The envelope still marks them ``excluded`` for lineage.
    """
    return [
        ([pool[i] for i in train_idx if pool[i] not in excluded], [pool[i] for i in val_idx])
        for train_idx, val_idx in splitter.split(pool)
    ]


# Keys on a step dict that are NOT model hyperparameters (mirrors StepParser.RESERVED_KEYWORDS).
_RESERVED_STEP_KEYS = frozenset({"model", "params", "metadata", "steps", "name", "finetune_params", "train_params", "refit_params", "fit_on_all", "force_layout", "na_policy", "fill_value", "y_processing"})


def _apply_model_params(steps: list[Any]) -> list[Any]:
    """Apply sibling hyperparameters to the model (e.g. `{"model": PLS(), "n_components": 9}`).

    Generators expand a param sweep into `{"model": M, "<param>": value}` steps; nirs4all applies the
    non-reserved siblings to the model via set_params. We do the same on a clone so concurrent
    variants do not share mutated state.
    """
    from sklearn.base import clone

    out: list[Any] = []
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            params = {key: value for key, value in step.items() if key not in _RESERVED_STEP_KEYS}
            if params:
                model = step["model"]
                model = clone(model) if hasattr(model, "set_params") else model
                model.set_params(**params)
                step = {key: value for key, value in step.items() if key in _RESERVED_STEP_KEYS}
                step["model"] = model
        out.append(step)
    return out


def _model_name(steps: list[Any]) -> str:
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            return type(step["model"]).__name__
    return "model"


# Step keywords whose presence forces the Python path even alongside a model param sweep: Optuna
# finetune / per-model train kwargs are not part of the native generation+SELECT contract, so a
# pipeline carrying them must NOT be mistaken for a clean param-sweep-only pipeline.
_FORCE_PYTHON_STEP_KEYS = frozenset({"finetune_params", "train_params"})


def _generation_kind(pipeline: list[Any]) -> str:
    """Classify a pipeline's generators: ``"none"``, ``"param_model"`` (native), or ``"operator"``.

    CONSERVATIVE by design — native (``"param_model"``) is returned ONLY when the pipeline is a clean
    model-param-sweep, i.e. ALL of:

    (a) at least one ``{"model": ...}`` step carries a natively-lowerable param sweep
        (:func:`~nirs4all.pipeline.dagml_bridge.is_param_generator_spec` — the exact ``_range_`` /
        ``_log_range_`` list forms), AND
    (b) NO other generator exists ANYWHERE — no generator keyword on a non-model step, no
        generator-valued model (multi-model ``{"model": {"_or_": ...}}``), no generator-shaped model
        sibling that is not natively lowerable (``_grid_``, dict-form, modifier-bearing), AND
    (c) NO step carries ``finetune_params`` or ``train_params``.

    Any other generator (or finetune/train_params) → ``"operator"`` (the correct Python ``expand_spec``
    path). ``"none"`` means no generators at all. When in doubt, this never returns ``"param_model"``.
    """
    from nirs4all.pipeline.config._generator.keywords import GENERATION_KEYWORDS, has_nested_generator_keywords
    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    has_param_model = False
    has_other = False
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if _FORCE_PYTHON_STEP_KEYS & set(step):
            has_other = True  # finetune/train_params are not in the native contract
        if "model" in step:
            # A generator-valued model (multi-model) is operator-level, not a clean param sweep.
            if has_nested_generator_keywords(step["model"]):
                has_other = True
            for key, value in step.items():
                if key in _RESERVED_STEP_KEYS:
                    continue
                if is_param_generator_spec(value):
                    has_param_model = True
                elif has_nested_generator_keywords(value):
                    # A generator-shaped sibling we cannot lower natively (e.g. `_grid_`, dict-form,
                    # or a modifier-bearing range) — Python expand owns it.
                    has_other = True
        elif GENERATION_KEYWORDS & set(step) or has_nested_generator_keywords(step):
            # Any generator on a non-model step (bare `_or_`/`_range_`/... or a nested one).
            has_other = True
    if has_other:
        return "operator"
    return "param_model" if has_param_model else "none"


def _is_separation_branch_step(step: Any) -> bool:
    """True for a separation branch by metadata/tag: ``{"branch": {"by_metadata"|"by_tag": ...}}``."""
    return isinstance(step, dict) and isinstance(step.get("branch"), dict) and bool({"by_metadata", "by_tag"} & set(step["branch"]))


def _is_concat_merge_step(step: Any) -> bool:
    return isinstance(step, dict) and step.get("merge") == "concat"


# Keys recognised inside a separation-branch criterion dict. `run_backend` honors ONLY the criterion
# (by_metadata/by_tag) + the shared `steps` body; `values` (explicit grouping), `min_samples`
# (cardinality drop) and per-branch selectors are NOT honored, so a branch carrying them must fall
# through to the loud bridge error rather than be silently run with default behavior.
_HANDLED_BRANCH_KEYS = frozenset({"by_metadata", "by_tag", "steps"})


def _detect_separation_branch(pipeline: list[Any]) -> tuple[dict[str, Any], list[Any]] | None:
    """Detect the EXACT handled shape, else return ``None`` (fail-loud via the bridge).

    Admits ONLY a pipeline that is exactly: the splitter + ONE by_metadata/by_tag separation branch
    (a single shared ``steps`` body containing the model) + ONE ``{"merge": "concat"}`` — nothing
    that ``_run_separation_branch`` does not actually honor. Returns ``(branch_step, branch_body)``
    when matched. ANY deviation returns ``None`` so the bridge's raw-branch ``NotImplementedError``
    fires (the coverage-boundary fail-loud guarantee), never a silent-wrong run. Specifically REJECTED:

    * a top-level operator/transform/``tag``/``y_processing`` step beside the branch (only the branch
      body is lowered, so a top-level step would be silently dropped) — out-of-scope follow-up;
    * an ``exclude`` step anywhere (the folds are built over the full pool with no excluded bit, so the
      exclusion would be silently lost) — exclude+branch is a follow-up slice;
    * an unhandled branch option (``values`` / ``min_samples`` / a per-branch ``selector`` / any key
      outside ``by_metadata``/``by_tag``/``steps``) — those grouping semantics are not honored;
    * a per-value dict ``steps`` (different sub-pipeline per partition), a missing model in the body,
      a model after the merge, a non-concat merge, or more than one branch/merge.
    """
    branch_steps = [step for step in pipeline if _is_separation_branch_step(step)]
    merge_steps = [step for step in pipeline if _is_concat_merge_step(step)]
    if len(branch_steps) != 1 or len(merge_steps) != 1:
        return None
    branch_step, merge_step = branch_steps[0], merge_steps[0]

    # The pipeline must be EXACTLY {splitter, branch, merge} — no other top-level steps. A top-level
    # transform / tag / y_processing / exclude / model would be silently ignored (only the branch body
    # is lowered), so its presence rejects the match → fail-loud.
    for step in pipeline:
        if step is branch_step or step is merge_step or hasattr(step, "split"):
            continue
        return None

    criterion = branch_step["branch"]
    # Only the criterion (by_metadata/by_tag) + the shared `steps` body are honored. Any other branch
    # option (values/min_samples/per-branch selector/...) is not → reject.
    if set(criterion) - _HANDLED_BRANCH_KEYS:
        return None

    body = criterion.get("steps")
    # Only the shared-body LIST form (one sub-pipeline applied per partition) with a model inside is
    # supported. The per-value dict form and a body without a model fall through to the bridge error.
    if not isinstance(body, list) or not any(isinstance(sub, dict) and "model" in sub for sub in body):
        return None
    return branch_step, body


def run_via_dagml(
    pipeline: Any,
    dataset: Any,
    *,
    dagml_cli: str | Path | None = None,
    venv_python: str | None = None,
    workdir: str | Path | None = None,
) -> RunResult:
    """Execute ``pipeline`` on ``dataset`` via dag-ml-cli; return a RunResult of dag-ml's native scores.

    Args:
        pipeline: nirs4all pipeline (feature transforms, one ``{"model": ...}`` step, and a splitter).
        dataset: Anything :class:`~nirs4all.data.config.DatasetConfigs` accepts (path/config).
        dagml_cli: Path to the ``dag-ml-cli`` binary (defaults to the sibling ``dag-ml`` build).
        venv_python: Python interpreter the process adapter re-execs under (defaults to the current).
        workdir: Scratch directory for the run inputs/outputs (defaults to a temp dir).

    Returns:
        A :class:`~nirs4all.api.result.RunResult` whose ``best_rmse`` is the native final-test score
        and ``cv_best_score`` is the native cross-fold OOF average.
    """
    cli = str(dagml_cli or _DEFAULT_CLI)
    if not Path(cli).exists():
        raise FileNotFoundError(f"dag-ml-cli binary not found at {cli}; build it (cargo build -p dag-ml-cli --release)")

    from nirs4all.core import detect_task_type
    from nirs4all.pipeline.config.generator import expand_spec

    spectro = DatasetConfigs(dataset).get_dataset_at(0)
    dataset_arg = str(spectro.dataset_path) if hasattr(spectro, "dataset_path") else str(dataset)
    base_dir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="n4a_dagml_"))

    is_classification = "classif" in str(detect_task_type(np.asarray(spectro.y({"partition": "train"}))))
    metric = "accuracy" if is_classification else "rmse"
    task_type = "classification" if is_classification else "regression"

    # Separation branch (by_metadata/by_tag) + concat merge → ONE native fan-out run: dag-ml fans the
    # branch into one model node per partition value (discovered from the envelope metadata/tags),
    # runs per-partition FIT_CV, and the native concat-merge handler reassembles a full-universe OOF.
    # Detected on the ORIGINAL pipeline (before exclude consumption) so an exclude step beside the
    # branch is still visible — exclude+branch is rejected (out of scope) rather than silently dropped.
    detected = _detect_separation_branch(list(pipeline))
    if detected is not None:
        branch_step, branch_body = detected
        return _run_separation_branch(list(pipeline), branch_step, branch_body, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "branch", metric, task_type)

    # Consume the `exclude` step (if any) BEFORE generator handling: run the SampleFilter operator(s)
    # in Python on the full CV train pool to get the excluded sample ints, then choose the CV universe
    # per the `keep_in_oof` mode. `cv_pool` is the sample-int universe the splitter runs over;
    # `excluded` is non-empty only in the opt-in (keep_in_oof=True) leakage-pure mode.
    pipeline, cv_pool, excluded = _resolve_exclude(list(pipeline), spectro)

    # Param-level model sweeps (`_range_`/`_log_range_`/`_grid_` on a model step) run as ONE native
    # dag-ml run: the bridge lowers them to native `generators`, the compiler expands variants, and
    # dag-ml generates + scores + SELECTs + refits the best (no Python expand). Operator-level
    # generators (`_or_`/`_cartesian_`, multi-model) stay on the Python `expand_spec` path below.
    if _generation_kind(list(pipeline)) == "param_model":
        return _run_native_generation(
            list(pipeline), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "native", metric, task_type, cv_pool, excluded
        )

    # Expand operator-level generators (_or_/_cartesian_/...) into concrete pipelines in Python
    # (nirs4all's own expansion), run each through the verified single-variant dag-ml path, then
    # select the best by its CV score — mirroring nirs4all selecting the best variant by its metric.
    variants = expand_spec(pipeline)
    results = [
        _run_concrete(variant, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / f"variant{index}", metric, task_type, cv_pool, excluded)
        for index, variant in enumerate(variants)
    ]
    if len(results) == 1:
        return results[0]

    def _cv_rank(result: RunResult) -> float:
        score = result.cv_best_score
        if score != score:  # NaN (no CV score) ranks last
            return float("inf")
        return -score if is_classification else score  # maximize accuracy, minimize RMSE

    return min(results, key=_cv_rank)


def _run_native_generation(pipeline: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str, cv_pool: list[int] | None = None, excluded: set[int] | None = None) -> RunResult:
    """Run a param-level model sweep as ONE native dag-ml generation + SELECT + refit run.

    The model step keeps its generator dict so the bridge lowers it to native ``generators``; we
    apply only the plain (non-generator) sibling params to the model, never the sweep. dag-ml
    expands the variants, scores each by its cross-fold OOF ``metric``, and refits the winner —
    ``bundle.scores`` is the selected variant's, mapped to a RunResult exactly like the single path.

    ``cv_pool`` is the CV sample-int universe (the de-excluded pool in legacy mode, the full train
    in opt-in mode); ``excluded`` is marked in the envelope only in opt-in (``keep_in_oof=True``).
    """
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    steps = _apply_plain_model_params(steps)

    identity = mint_identity(spectro)
    pool = list(cv_pool) if cv_pool is not None else spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, pool, excluded or set())
    envelope = build_envelope(spectro, identity, sample_ints=pool, excluded_sample_ints=excluded or None)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml engine run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    return _scores_to_run_result(bundle.get("scores"), spectro.name, _model_name(steps), metric, task_type)


def _apply_plain_model_params(steps: list[Any]) -> list[Any]:
    """Apply only the PLAIN (non-generator) sibling hyperparameters to the model, keeping sweeps.

    The native path lowers param-level sweeps (``_range_``/``_log_range_``/``_grid_``) to dag-ml
    ``generators``, so they must stay on the step dict; plain siblings (e.g. ``scale=False``) are
    set on a model clone, exactly like ``_apply_model_params`` but leaving the generator dicts in
    place for the bridge to lower.
    """
    from sklearn.base import clone

    from nirs4all.pipeline.dagml_bridge import is_param_generator_spec

    out: list[Any] = []
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            plain = {key: value for key, value in step.items() if key not in _RESERVED_STEP_KEYS and not is_param_generator_spec(value)}
            if plain:
                model = step["model"]
                model = clone(model) if hasattr(model, "set_params") else model
                model.set_params(**plain)
                kept = {key: value for key, value in step.items() if key in _RESERVED_STEP_KEYS or is_param_generator_spec(value)}
                kept["model"] = model
                step = kept
        out.append(step)
    return out


def _run_concrete(pipeline: Any, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str = "rmse", task_type: str = "regression", cv_pool: list[int] | None = None, excluded: set[int] | None = None) -> RunResult:
    """Run one concrete (generator-free) pipeline through dag-ml-cli; map its native scores.

    ``cv_pool`` is the CV sample-int universe (de-excluded pool in legacy mode, full train in opt-in
    mode); ``excluded`` is marked in the envelope only in the opt-in (``keep_in_oof=True``) mode.
    """
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    steps = _apply_model_params(steps)

    identity = mint_identity(spectro)
    pool = list(cv_pool) if cv_pool is not None else spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, pool, excluded or set())
    envelope = build_envelope(spectro, identity, sample_ints=pool, excluded_sample_ints=excluded or None)
    dsl = assemble_cv_refit_dsl(steps, identity, envelope, folds, dsl_id="nirs4all-pipeline", n_splits=len(folds))

    import dag_ml

    graph = dag_ml.compile_pipeline_dsl_artifact_with_controllers(dsl, controller_manifests()).graph.to_dict()
    outcome = run_cv_refit_bundle(
        dsl=dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml engine run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    return _scores_to_run_result(bundle.get("scores"), spectro.name, _model_name(steps), metric, task_type)


_MERGE_NODE_ID = "merge:concat"


def _run_separation_branch(pipeline: list[Any], branch_step: dict[str, Any], branch_body: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str) -> RunResult:
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
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    body_steps = [step for step in branch_body if not hasattr(step, "split")]

    identity = mint_identity(spectro)
    # The handled shape rejects any exclude step, so the CV universe is the full train pool.
    pool = spectro.index_column("sample", {"partition": "train"})
    folds = _build_folds(splitter, pool, set())

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
        raise RuntimeError("separation-branch fan-out produced no per-partition model nodes")

    # Per-partition data_bindings (one per fanned model node) + the materialized fold set. The CLI's
    # own fan-out is a no-op on the already-fanned DSL (the auto_separate marker is consumed).
    fanned_dsl["data_bindings"] = data_bindings_for_nodes(model_ids, envelope)
    fanned_dsl["split_invocation"] = split_invocation_for(identity, folds, n_splits=len(folds))

    outcome = run_cv_refit_bundle(
        dsl=fanned_dsl, envelope=envelope, graph=graph, dataset_path=dataset_arg, workdir=run_dir, dagml_cli=cli, venv_python=venv_python, selection_metric=metric, sample_metadata=sample_metadata
    )
    if outcome["returncode"] != 0:
        raise RuntimeError(f"dag-ml separation-branch run failed (rc={outcome['returncode']}):\n{outcome['stdout'][-2000:]}")

    bundle = json.loads((run_dir / "bundle.json").read_text())
    # The concat-merge producer's reports carry the full-universe cross-fold OOF average — that is the
    # separation branch's `cv_best_score`. `best_rmse` (final-test) stays NaN: the native concat-merge
    # handler reassembles the FIT_CV validation OOF, not the per-partition REFIT test predictions, so
    # no merged `(test, final)` report exists — which also matches legacy (top-level best_rmse is NaN
    # for a branch+merge pipeline). cv_best_score is the score a separation branch is meant to produce.
    return _scores_to_run_result(bundle.get("scores"), spectro.name, _model_name(body_steps), metric, task_type, producer=_MERGE_NODE_ID)


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


def _scores_to_run_result(scores: dict[str, Any] | None, dataset_name: str, model_name: str, metric: str = "rmse", task_type: str = "regression", producer: str | None = None) -> RunResult:
    """Map a dag-ml ScoreSet into a RunResult, mirroring nirs4all's entry shape.

    ``producer`` filters to one ``producer_node`` — e.g. a separation branch's concat-merge node,
    whose reports carry the full-universe OOF average (``cv_best_score``); ``None`` keeps all
    reports (the single-model path, where exactly one producer scores).

    dag-ml emits one report per (partition, fold). nirs4all's RunResult expects per-fold validation
    entries + a single combined **refit/final** entry that carries val (the CV score), test and train
    scores together — that combined entry is what `best`/`best_rmse`/`best_final` resolve. We build
    exactly that: per-fold `val` entries, an `avg` CV entry (`cv_best_score`), and one `final` entry
    (`fold_id="final"`, `partition="test"`) with val_score=OOF-average, test_score=final-test,
    train_score=final-train, and a combined `scores` dict.
    """
    reports = [report for report in (scores or {}).get("reports", []) if producer is None or report.get("producer_node") == producer]
    by_key = {(report["partition"], report.get("fold_id")): {name: float(value) for name, value in report["metrics"].items()} for report in reports}

    def add(fold_id: str | None, partition: str, scores_map: dict[str, dict[str, float]], *, val: float | None = None, test: float | None = None, train: float | None = None) -> None:
        predictions.add_prediction(dataset_name=dataset_name, model_name=model_name, fold_id=fold_id, partition=partition, metric=metric, task_type=task_type, scores=scores_map, val_score=val, test_score=test, train_score=train)

    # Two entries: the `avg` CV entry (cv_best_score) and the combined refit `final` entry that holds
    # val + test + train scores (best/best_rmse/best_accuracy resolve from it). Per-fold val entries
    # are omitted — they would let get_best(score_scope="all") rank a single fold first.
    #
    # _rank_candidates ranks an is_final entry by its OWN partition's metric, so the final entry uses
    # partition="val" (ranks on the CV metric, same axis as the avg) with the ranking value nudged a
    # negligible epsilon in the better direction (maximize accuracy, minimize rmse). That makes the
    # refit entry win the get_best tie over the avg for BOTH task directions; reported scores come
    # from the unmodified `scores` dict, so best_rmse/best_accuracy read the true final-test value.
    maximize = metric in ("accuracy", "r2")
    predictions = Predictions()
    avg = by_key.get(("validation", "avg"))
    test = by_key.get(("test", "final"))
    train = by_key.get(("final", None))
    if avg is not None:
        add("avg", "val", {"val": avg}, val=avg.get(metric))
    if test is not None or train is not None:
        rank_val = dict(avg) if avg is not None else {}
        if metric in rank_val:
            rank_val[metric] += 1e-9 if maximize else -1e-9
        combined: dict[str, dict[str, float]] = {"val": rank_val}
        if train is not None:
            combined["train"] = train
        if test is not None:
            combined["test"] = test
        add("final", "val", combined, val=(avg or {}).get(metric), test=(test or {}).get(metric), train=(train or {}).get(metric))

    predictions.flush()
    return RunResult(predictions=predictions, per_dataset={dataset_name: {"engine": "dag-ml"}})
