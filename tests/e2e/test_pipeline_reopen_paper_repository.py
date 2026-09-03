from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

import nirs4all
from nirs4all.data.config import DatasetConfigs
from nirs4all.operators.transforms import StandardNormalVariate
from nirs4all.sklearn import NIRSPipeline
from tests.integration.parity._datasets import dataset_path

PIPELINE_DESCRIPTOR = {
    "schema_version": "n4a.e2e.saved_pipeline/v1",
    "name": "paper repository refit parity smoke",
    "pipeline": [
        {
            "role": "transform",
            "class": "nirs4all.operators.transforms.StandardNormalVariate",
            "params": {},
        },
        {
            "role": "split",
            "class": "sklearn.model_selection.ShuffleSplit",
            "params": {"n_splits": 2, "test_size": 0.25, "random_state": 42},
        },
        {
            "role": "model",
            "class": "sklearn.cross_decomposition.PLSRegression",
            "params": {"n_components": 5},
        },
    ],
}


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(child.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _load_saved_pipeline(path: Path) -> list[Any]:
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    if descriptor.get("schema_version") != PIPELINE_DESCRIPTOR["schema_version"]:
        raise AssertionError(f"unexpected saved pipeline schema: {descriptor.get('schema_version')!r}")
    reopened: list[Any] = []
    for step in descriptor["pipeline"]:
        class_name = step["class"]
        params = dict(step.get("params") or {})
        if class_name == "nirs4all.operators.transforms.StandardNormalVariate":
            reopened.append(StandardNormalVariate(**params))
        elif class_name == "sklearn.model_selection.ShuffleSplit":
            reopened.append(ShuffleSplit(**params))
        elif class_name == "sklearn.cross_decomposition.PLSRegression":
            reopened.append({"model": PLSRegression(**params), "name": "PLS_repository_refit"})
        else:
            raise AssertionError(f"unsupported saved pipeline class: {class_name}")
    return reopened


def _repository_refit_recipe() -> dict[str, Any]:
    return {
        "name": "paper repository refit parity smoke",
        "pipeline": [
            {
                "class": "nirs4all.operators.transforms.StandardNormalVariate",
                "params": {},
            },
            {
                "model": {
                    "class": "sklearn.cross_decomposition.PLSRegression",
                    "params": {"n_components": 5},
                },
                "name": "PLS_repository_refit",
            },
        ],
    }


def _write_web_upload_dataset(artifacts_dir: Path, dataset: Any) -> dict[str, Any]:
    x_train = np.asarray(dataset.x({"partition": "train"}, include_augmented=False, layout="2d"), dtype=float)
    y_train = _vector(dataset.y({"partition": "train"}, include_augmented=False))
    sample_indices = [int(sample) for sample in dataset.index_column("sample", {"partition": "train"})]
    assert x_train.shape[0] == y_train.shape[0] == len(sample_indices)
    axis = np.arange(x_train.shape[1], dtype=float)

    x_path = artifacts_dir / "repository_X_train.csv"
    y_path = artifacts_dir / "repository_y_train.csv"
    metadata_path = artifacts_dir / "repository_metadata.csv"
    np.savetxt(x_path, np.vstack([axis, x_train]), delimiter=",", fmt="%.17g")
    np.savetxt(y_path, y_train, delimiter=",", fmt="%.17g", header="target", comments="")
    metadata_path.write_text(
        "sample_id,dataset_id,source_sample_index\n"
        + "".join(f"python-regression-{index},python-regression-train,{index}\n" for index in sample_indices),
        encoding="utf-8",
    )

    files = {
        x_path.name: _sha256(x_path),
        y_path.name: _sha256(y_path),
        metadata_path.name: _sha256(metadata_path),
    }
    return {
        "schema_version": "n4a.e2e.web_upload_dataset.v1",
        "status": "passed",
        "path": ".",
        "files": files,
        "files_sha256": _stable_hash(files),
        "rows": int(x_train.shape[0]),
        "cols": int(x_train.shape[1]),
        "target_rows": int(y_train.shape[0]),
        "metadata_rows": len(sample_indices),
        "expected_badge": f"{x_train.shape[0]} samples × {x_train.shape[1]} wavelengths",
    }


def _assert_no_dagml_fallback(result: Any, warning_messages: list[str], native_root: Path) -> Path:
    fallback_warnings = [message for message in warning_messages if "falling back to the legacy engine" in message]
    assert not fallback_warnings, fallback_warnings
    assert result._is_dagml_engine(), "engine='dag-ml' must not report a legacy fallback result"  # noqa: SLF001
    native_results_dir = getattr(result, "_dagml_results_dir", None)
    assert native_results_dir is not None, "engine='dag-ml' with results_path must expose native results"
    native_results_path = Path(native_results_dir)
    assert native_results_path.is_relative_to(native_root)
    for filename in ("manifest.json", "score_set.json", "predictions.parquet"):
        assert (native_results_path / filename).is_file(), native_results_path / filename
    return native_results_path


def _run_reopened(engine: str, saved_pipeline: Path, artifacts_dir: Path) -> tuple[Any, list[str], Path | None]:
    kwargs: dict[str, Any] = {
        "engine": engine,
        "name": f"e2e_reopened_{engine.replace('-', '_')}",
        "verbose": 0,
        "save_artifacts": engine == "legacy",
        "save_charts": False,
        "random_state": 42,
        "refit": True,
    }
    if engine == "legacy":
        kwargs["workspace_path"] = artifacts_dir / "legacy-workspace"
    if engine == "dag-ml":
        kwargs["results_path"] = str(artifacts_dir / "native-results")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(_load_saved_pipeline(saved_pipeline), dataset_path("regression"), **kwargs)
    warning_messages = [str(warning.message) for warning in caught]
    native_results_dir = None
    if engine == "dag-ml":
        native_results_dir = _assert_no_dagml_fallback(result, warning_messages, artifacts_dir / "native-results")
    return result, warning_messages, native_results_dir


def test_reopen_rerun_parity(artifacts_dir: Path) -> None:
    saved_pipeline = artifacts_dir / "saved-pipeline.json"
    _write_json(saved_pipeline, PIPELINE_DESCRIPTOR)
    dataset_config = Path(dataset_path("regression"))

    legacy, legacy_warnings, _ = _run_reopened("legacy", saved_pipeline, artifacts_dir)
    bundle_path = legacy.export(artifacts_dir / "reopened-refit.n4a")

    dataset = DatasetConfigs(dataset_config, task_type="regression").get_dataset_at(0)
    x_train = dataset.x({"partition": "train"}, include_augmented=False)
    web_upload_dataset = _write_web_upload_dataset(artifacts_dir, dataset)
    reopened_bundle = NIRSPipeline.from_bundle(bundle_path)
    bundle_pred = _vector(reopened_bundle.predict(x_train))
    legacy_final_pred = _vector(legacy.final["y_pred"])
    assert bundle_pred.shape == legacy_final_pred.shape

    dagml, dagml_warnings, native_results_dir = _run_reopened("dag-ml", saved_pipeline, artifacts_dir)

    legacy_best_pred = _vector(legacy.best["y_pred"])
    dagml_best_pred = _vector(dagml.best["y_pred"])
    dagml_final_pred = _vector(dagml.final["y_pred"])
    assert legacy_best_pred.shape == dagml_best_pred.shape
    assert legacy_final_pred.shape == dagml_final_pred.shape
    assert legacy_best_pred.size > 0
    assert legacy_final_pred.size > 0

    tolerance = 1e-8
    parity = {
        "best_rmse_abs": abs(float(legacy.best_rmse) - float(dagml.best_rmse)),
        "best_score_abs": abs(float(legacy.best_score) - float(dagml.best_score)),
        "best_prediction_abs_max": float(np.max(np.abs(legacy_best_pred - dagml_best_pred))),
        "final_prediction_abs_max": float(np.max(np.abs(legacy_final_pred - dagml_final_pred))),
        "bundle_reopen_prediction_abs_max": float(np.max(np.abs(bundle_pred - legacy_final_pred))),
        "best_prediction_rows": int(legacy_best_pred.shape[0]),
        "final_prediction_rows": int(legacy_final_pred.shape[0]),
        "tolerance": tolerance,
    }
    assert parity["best_rmse_abs"] <= 1e-9
    assert parity["best_score_abs"] <= 1e-9
    assert parity["best_prediction_abs_max"] <= tolerance
    assert parity["final_prediction_abs_max"] <= tolerance
    assert parity["bundle_reopen_prediction_abs_max"] <= tolerance

    evidence = {
        "schema_version": "n4a.e2e.python_reopen_paper_repository/v1",
        "status": "passed",
        "scenario": "e2e-python-reopen-paper-repository-refit",
        "repo": "nirs4all",
        "git_head": _git_head(),
        "nirs4all_version": getattr(nirs4all, "__version__", "unknown"),
        "saved_pipeline": {
            "path": saved_pipeline.name,
            "sha256": _sha256(saved_pipeline),
            "schema_version": PIPELINE_DESCRIPTOR["schema_version"],
        },
        "dataset": {
            "source": "tests.integration.parity._datasets:regression",
            "config_path": str(dataset_config),
            "config_sha256": _sha256(dataset_config),
            "train_rows": int(x_train.shape[0]),
            "feature_count": int(x_train.shape[1]),
        },
        "web_upload_dataset": web_upload_dataset,
        "bundle_reopen": {
            "path": Path(bundle_path).name,
            "sha256": _sha256(Path(bundle_path)),
            "prediction_rows": int(bundle_pred.shape[0]),
        },
        "repository_refit_recipe": _repository_refit_recipe(),
        "runs": {
            "legacy": {
                "warnings": legacy_warnings,
                "num_predictions": int(legacy.num_predictions),
                "best_rmse": float(legacy.best_rmse),
                "best_score": float(legacy.best_score),
            },
            "dag_ml": {
                "warnings": dagml_warnings,
                "native_results_dir": str(native_results_dir),
                "num_predictions": int(dagml.num_predictions),
                "best_rmse": float(dagml.best_rmse),
                "best_score": float(dagml.best_score),
            },
        },
        "parity": parity,
        "repository_handoff": {
            "producer_repo": "nirs4all-papers",
            "expected_artifact": "repository-best-pipeline.json",
            "note": "The paired nirs4all-papers step exports the reproducible paper and repository descriptor; this Python step proves the reopened runtime pipeline and dag-ml refit parity.",
        },
    }
    _write_json(artifacts_dir / "reopened-result.json", evidence)
