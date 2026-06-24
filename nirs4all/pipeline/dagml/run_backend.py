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


def _model_name(steps: list[Any]) -> str:
    for step in steps:
        if isinstance(step, dict) and "model" in step:
            return type(step["model"]).__name__
    return "model"


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

    from nirs4all.pipeline.config.generator import expand_spec

    spectro = DatasetConfigs(dataset).get_dataset_at(0)
    dataset_arg = str(spectro.dataset_path) if hasattr(spectro, "dataset_path") else str(dataset)
    base_dir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="n4a_dagml_"))

    # Expand generators (_or_/_range_/_grid_/...) into concrete pipelines in Python (nirs4all's own
    # expansion), run each through the verified single-variant dag-ml path, then select the best by
    # its cross-validation score — mirroring nirs4all selecting the best variant by its CV metric.
    variants = expand_spec(pipeline)
    results = [
        _run_concrete(variant, spectro, dataset_arg, cli, venv_python or sys.executable, base_dir / f"variant{index}")
        for index, variant in enumerate(variants)
    ]
    if len(results) == 1:
        return results[0]

    def _cv_rank(result: RunResult) -> float:
        score = result.cv_best_score
        return score if score == score else float("inf")  # NaN (no CV score) ranks last

    return min(results, key=_cv_rank)


def _run_concrete(pipeline: Any, spectro: Any, dataset_arg: str, cli: str, venv_python: str, run_dir: Path) -> RunResult:
    """Run one concrete (generator-free) pipeline through dag-ml-cli; map its native scores."""
    steps, splitter = _split_pipeline(pipeline)
    if splitter is None:
        raise ValueError("engine='dag-ml' requires a cross-validator step (e.g. KFold) in the pipeline")

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
    return _scores_to_run_result(bundle.get("scores"), spectro.name, _model_name(steps))


def _scores_to_run_result(scores: dict[str, Any] | None, dataset_name: str, model_name: str) -> RunResult:
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
        predictions.add_prediction(dataset_name=dataset_name, model_name=model_name, fold_id=fold_id, partition=partition, metric="rmse", task_type="regression", scores=scores_map, val_score=val, test_score=test, train_score=train)

    # Only the `avg` (cv_best) + combined `final` entries: per-fold val entries would make
    # get_best(score_scope="all") rank the lowest-val fold first (an entry with no test data),
    # breaking best_rmse when a fold's RMSE < the final-test RMSE.
    predictions = Predictions()
    avg = by_key.get(("validation", "avg"))
    test = by_key.get(("test", "final"))
    train = by_key.get(("final", None))
    if avg is not None:
        add("avg", "val", {"val": avg}, val=avg.get("rmse"))
    if test is not None or train is not None:
        combined: dict[str, dict[str, float]] = {}
        if avg is not None:
            combined["val"] = avg
        if train is not None:
            combined["train"] = train
        if test is not None:
            combined["test"] = test
        add("final", "test", combined, val=(avg or {}).get("rmse"), test=(test or {}).get("rmse"), train=(train or {}).get("rmse"))

    predictions.flush()
    return RunResult(predictions=predictions, per_dataset={dataset_name: {"engine": "dag-ml"}})
