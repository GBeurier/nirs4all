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

    # Task type sets the selection metric: classification ranks by accuracy (higher is better),
    # regression by RMSE (lower is better). dag-ml emits both natively in every score report.
    is_classification = "classif" in str(detect_task_type(np.asarray(spectro.y({"partition": "train"}))))
    metric = "accuracy" if is_classification else "rmse"
    task_type = "classification" if is_classification else "regression"

    # Param-level model sweeps (`_range_`/`_log_range_`/`_grid_` on a model step) run as ONE native
    # dag-ml run: the bridge lowers them to native `generators`, the compiler expands variants, and
    # dag-ml generates + scores + SELECTs + refits the best (no Python expand). Operator-level
    # generators (`_or_`/`_cartesian_`, multi-model) stay on the Python `expand_spec` path below.
    if _generation_kind(list(pipeline)) == "param_model":
        return _run_native_generation(
            list(pipeline), spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / "native", metric, task_type
        )

    # Expand operator-level generators (_or_/_cartesian_/...) into concrete pipelines in Python
    # (nirs4all's own expansion), run each through the verified single-variant dag-ml path, then
    # select the best by its CV score — mirroring nirs4all selecting the best variant by its metric.
    variants = expand_spec(pipeline)
    results = [
        _run_concrete(variant, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / f"variant{index}", metric, task_type)
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


def _run_native_generation(pipeline: list[Any], spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str, task_type: str) -> RunResult:
    """Run a param-level model sweep as ONE native dag-ml generation + SELECT + refit run.

    The model step keeps its generator dict so the bridge lowers it to native ``generators``; we
    apply only the plain (non-generator) sibling params to the model, never the sweep. dag-ml
    expands the variants, scores each by its cross-fold OOF ``metric``, and refits the winner —
    ``bundle.scores`` is the selected variant's, mapped to a RunResult exactly like the single path.
    """
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    steps = _apply_plain_model_params(steps)

    identity = mint_identity(spectro)
    train = spectro.index_column("sample", {"partition": "train"})
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in splitter.split(train)]
    envelope = build_envelope(spectro, identity, sample_ints=train)
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


def _run_concrete(pipeline: Any, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path, metric: str = "rmse", task_type: str = "regression") -> RunResult:
    """Run one concrete (generator-free) pipeline through dag-ml-cli; map its native scores."""
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")
    steps = _apply_model_params(steps)

    identity = mint_identity(spectro)
    train = spectro.index_column("sample", {"partition": "train"})
    folds = [([train[i] for i in tr], [train[i] for i in va]) for tr, va in splitter.split(train)]
    envelope = build_envelope(spectro, identity, sample_ints=train)
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


def _scores_to_run_result(scores: dict[str, Any] | None, dataset_name: str, model_name: str, metric: str = "rmse", task_type: str = "regression") -> RunResult:
    """Map a dag-ml ScoreSet into a RunResult, mirroring nirs4all's entry shape.

    dag-ml emits one report per (partition, fold). nirs4all's RunResult expects per-fold validation
    entries + a single combined **refit/final** entry that carries val (the CV score), test and train
    scores together — that combined entry is what `best`/`best_rmse`/`best_final` resolve. We build
    exactly that: per-fold `val` entries, an `avg` CV entry (`cv_best_score`), and one `final` entry
    (`fold_id="final"`, `partition="test"`) with val_score=OOF-average, test_score=final-test,
    train_score=final-train, and a combined `scores` dict.
    """
    by_key = {(report["partition"], report.get("fold_id")): {name: float(value) for name, value in report["metrics"].items()} for report in (scores or {}).get("reports", [])}

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
