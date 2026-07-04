from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression

import nirs4all
from nirs4all.data import SpectroDataset
from nirs4all.operators.splitters import KennardStoneSplitter

SCENARIO_ID = "e2e-multimodal-python-r-wasm-roundtrip"
PIPELINE_NAME = "multimodal_portable_roundtrip"
PREDICTION_TOLERANCE = 1e-8


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _build_multimodal_fixture(rows: int = 48, spectral_cols: int = 28) -> dict[str, Any]:
    wavelengths = np.linspace(950.0, 1750.0, spectral_cols, dtype=np.float64)
    nir = np.empty((rows, spectral_cols), dtype=np.float64)
    metadata_numeric = np.empty((rows, 4), dtype=np.float64)
    metadata_rows: list[dict[str, Any]] = []
    y = np.empty(rows, dtype=np.float64)

    for row in range(rows):
        batch_code = row % 3
        cultivar_code = (row // 3) % 2
        temperature_c = 18.5 + 0.11 * row + 0.8 * math.sin(row / 7.0)
        humidity_pct = 42.0 + 6.0 * math.cos(row / 8.0) + batch_code * 1.4
        dry_matter_pct = 31.0 + 0.18 * row - cultivar_code * 1.2 + 0.5 * math.sin(row / 5.0)

        phase = row / 6.0
        for col, wavelength in enumerate(wavelengths):
            peak_a = math.exp(-0.5 * ((wavelength - 1185.0) / 75.0) ** 2)
            peak_b = math.exp(-0.5 * ((wavelength - 1510.0) / 105.0) ** 2)
            nir[row, col] = (
                0.55 * math.sin(phase + col / 8.0)
                + 0.24 * math.cos(row / 9.0 - col / 10.0)
                + 0.32 * peak_a
                - 0.22 * peak_b
                + 0.0022 * wavelength
                + batch_code * 0.035
                - cultivar_code * 0.045
            )

        metadata_numeric[row] = [
            (temperature_c - 20.0) / 4.0,
            (humidity_pct - 45.0) / 10.0,
            float(batch_code),
            float(cultivar_code),
        ]
        y[row] = (
            0.055 * np.sum(nir[row, : spectral_cols // 2])
            - 0.032 * np.sum(nir[row, spectral_cols // 2 :])
            + 0.18 * metadata_numeric[row, 0]
            - 0.12 * metadata_numeric[row, 1]
            + 0.07 * batch_code
            - 0.05 * cultivar_code
            + 0.015 * math.sin(row / 4.0)
        )
        metadata_rows.append(
            {
                "sample_id": f"mm-{row:03d}",
                "batch": f"batch-{batch_code + 1}",
                "cultivar": "alpha" if cultivar_code == 0 else "beta",
                "temperature_c": round(temperature_c, 6),
                "humidity_pct": round(humidity_pct, 6),
                "dry_matter_pct": round(dry_matter_pct, 6),
            }
        )

    metadata_columns = ["temperature_scaled", "humidity_scaled", "batch_code", "cultivar_code"]
    fused = np.column_stack([nir, metadata_numeric]).astype(np.float64)
    headers = [f"nir_{int(round(wavelength))}_nm" for wavelength in wavelengths] + metadata_columns
    dataset = {
        "schema_version": "n4a.e2e.multimodal.v1",
        "name": "synthetic_nir_metadata_roundtrip",
        "description": "Deterministic NIRS spectra fused with sample-level metadata for portable cross-language parity.",
        "rows": int(rows),
        "cols": int(fused.shape[1]),
        "spectral_cols": int(spectral_cols),
        "metadata_cols": int(metadata_numeric.shape[1]),
        "feature_headers": headers,
        "target_name": "quality_score",
        "samples": metadata_rows,
        "sources": [
            {
                "name": "nir",
                "kind": "spectra",
                "feature_slice": [0, int(spectral_cols)],
                "wavelength_unit": "nm",
                "wavelengths": wavelengths.tolist(),
            },
            {
                "name": "sample_metadata",
                "kind": "tabular_metadata",
                "feature_slice": [int(spectral_cols), int(fused.shape[1])],
                "columns": metadata_columns,
                "categorical_encoding": {
                    "batch_code": {"batch-1": 0, "batch-2": 1, "batch-3": 2},
                    "cultivar_code": {"alpha": 0, "beta": 1},
                },
            },
        ],
        "portable_view": {
            "name": "fused_features",
            "fusion": "column_concatenate",
            "note": "Current nirs4all-core R/WASM execution accepts this explicit dense matrix view, not a native web UI multimodal step.",
            "X": fused.reshape(-1).tolist(),
            "y": y.tolist(),
            "rows": int(rows),
            "cols": int(fused.shape[1]),
        },
    }
    dataset["sha256"] = _stable_hash(dataset["portable_view"])
    return dataset


def _pipeline_definition() -> dict[str, Any]:
    return {
        "name": PIPELINE_NAME,
        "description": "Portable PLS pipeline over an explicit fused NIRS plus metadata feature matrix.",
        "random_state": 42,
        "multimodal_contract": {
            "dataset_view": "fused_features",
            "source_count": 2,
            "native_multimodal_runtime": False,
            "runtime_note": "R and JavaScript/WASM use nirs4all-core portable dense-matrix execution for this scenario.",
        },
        "pipeline": [
            {
                "class": "nirs4all.operators.splitters.KennardStoneSplitter",
                "params": {"test_size": 0.25},
            },
            {
                "model": {"class": "sklearn.cross_decomposition.PLSRegression"},
                "_range_": [1, 5, 2],
                "param": "n_components",
                "name": "PLS-fused-multimodal-sweep",
            },
        ],
    }


def _component_values(step: dict[str, Any]) -> list[int]:
    if "_range_" not in step:
        return [int(step.get("model", {}).get("params", {}).get("n_components", 2))]
    if step.get("param") != "n_components":
        raise ValueError("Only n_components sweeps are portable in this scenario.")
    start, stop, stride = [int(value) for value in step["_range_"]]
    return list(range(start, stop + 1, stride))


def _execute_python_oracle(pipeline: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    portable = dataset["portable_view"]
    X = np.asarray(portable["X"], dtype=np.float64).reshape(portable["rows"], portable["cols"])
    y = np.asarray(portable["y"], dtype=np.float64)

    split_step = pipeline["pipeline"][0]
    splitter = KennardStoneSplitter(**split_step["params"])
    train_indices, test_indices = next(splitter.split(X, y))
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    model_step = pipeline["pipeline"][-1]

    variants: list[dict[str, Any]] = []
    for n_components in _component_values(model_step):
        model = PLSRegression(n_components=n_components)
        model.fit(X[train_indices], y[train_indices])
        predictions = model.predict(X[test_indices]).reshape(-1)
        targets = y[test_indices]
        residuals = predictions - targets
        variants.append(
            {
                "n_components": int(n_components),
                "rmse": float(np.sqrt(np.mean(residuals * residuals))),
                "predictions": predictions.tolist(),
            }
        )

    selected = min(variants, key=lambda item: item["rmse"])
    return {
        "name": PIPELINE_NAME,
        "source": "full Python nirs4all operators + sklearn.cross_decomposition.PLSRegression",
        "split": {
            "kind": "KennardStone",
            "trainIndices": train_indices.tolist(),
            "testIndices": test_indices.tolist(),
        },
        "targets": y[test_indices].tolist(),
        "variants": variants,
        "selected": selected,
    }


def _prediction_frame(dataset: dict[str, Any], oracle: dict[str, Any]) -> pd.DataFrame:
    test_indices = oracle["split"]["testIndices"]
    predictions = oracle["selected"]["predictions"]
    targets = oracle["targets"]
    samples = dataset["samples"]
    return pd.DataFrame(
        {
            "scenario_id": SCENARIO_ID,
            "runtime": "python_oracle",
            "sample_index": test_indices,
            "sample_id": [samples[index]["sample_id"] for index in test_indices],
            "target": targets,
            "prediction": predictions,
            "residual": [prediction - target for prediction, target in zip(predictions, targets)],
            "n_components": oracle["selected"]["n_components"],
        }
    )


def _spectro_dataset(dataset: dict[str, Any]) -> SpectroDataset:
    portable = dataset["portable_view"]
    X = np.asarray(portable["X"], dtype=np.float64).reshape(portable["rows"], portable["cols"])
    y = np.asarray(portable["y"], dtype=np.float64)
    spectro = SpectroDataset(dataset["name"])
    spectro.add_samples(X, headers=dataset["feature_headers"], header_unit="text")
    spectro.add_metadata(pd.DataFrame(dataset["samples"]))
    spectro.add_targets(y)
    return spectro


def test_generate_oracle(artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _pipeline_definition()
    dataset = _build_multimodal_fixture()
    oracle = _execute_python_oracle(pipeline, dataset)

    pipeline_path = artifacts_dir / "multimodal-pipeline.n4a.json"
    dataset_path = artifacts_dir / "multimodal-dataset.json"
    oracle_path = artifacts_dir / "python-oracle.json"
    predictions_path = artifacts_dir / "python-predictions.parquet"
    predictions_json_path = artifacts_dir / "python-predictions.json"
    workspace_dir = artifacts_dir / "python-workspace"

    _write_json(pipeline_path, pipeline)
    _write_json(dataset_path, dataset)
    _write_json(
        oracle_path,
        {
            "scenario_id": SCENARIO_ID,
            "schema_version": "n4a.e2e.multimodal_oracle.v1",
            "dataset_sha256": dataset["sha256"],
            "pipeline_sha256": _stable_hash(pipeline),
            "metadata": {
                "prediction_abs_tolerance": PREDICTION_TOLERANCE,
                "parity_candidate_runtimes": ["nirs4all-core-python", "r", "javascript_wasm"],
            },
            "case": oracle,
        },
    )

    predictions = _prediction_frame(dataset, oracle)
    predictions.to_parquet(predictions_path, index=False)
    _write_json(predictions_json_path, {"rows": predictions.to_dict(orient="records")})

    run_summary: dict[str, Any] = {"status": "not_run"}
    result = nirs4all.run(
        pipeline,
        _spectro_dataset(dataset),
        name=PIPELINE_NAME,
        verbose=0,
        save_charts=False,
        save_artifacts=True,
        workspace_path=workspace_dir,
        show_spinner=False,
        random_state=42,
        refit=True,
    )
    export_summary: dict[str, Any]
    export_path = artifacts_dir / "python-best-model.n4a"
    try:
        exported = result.export(export_path)
        export_summary = {"status": "created", "path": exported.name, "sha256": _file_hash(exported)}
    except Exception as exc:  # pragma: no cover - export availability is reported, not required for oracle generation
        export_summary = {"status": "blocked", "reason": str(exc)}

    run_summary = {
        "status": "passed",
        "workspace_path": str(workspace_dir),
        "workspace_exists": workspace_dir.exists(),
        "num_predictions": int(result.num_predictions),
        "best_rmse": _finite_or_none(result.best_rmse),
        "best_r2": _finite_or_none(result.best_r2),
        "export": export_summary,
    }

    evidence = {
        "scenario_id": SCENARIO_ID,
        "status": "python_oracle_ready",
        "dataset": {
            "path": dataset_path.name,
            "sha256": dataset["sha256"],
            "rows": dataset["rows"],
            "cols": dataset["cols"],
            "sources": [{"name": source["name"], "kind": source["kind"], "feature_slice": source["feature_slice"]} for source in dataset["sources"]],
        },
        "pipeline": {
            "path": pipeline_path.name,
            "sha256": _stable_hash(pipeline),
            "name": pipeline["name"],
        },
        "python_oracle": {
            "path": oracle_path.name,
            "prediction_rows": int(len(predictions)),
            "selected_n_components": int(oracle["selected"]["n_components"]),
            "selected_rmse": float(oracle["selected"]["rmse"]),
            "predictions_parquet": predictions_path.name,
        },
        "workspace_save": run_summary,
        "remaining_blockers": [
            "Run nirs4all-core/scripts/e2e/run_multimodal_roundtrip.py to execute available core/R/WASM parity checks.",
            "No web roundtrip step exists in this harness yet; the portable R/WASM contract is the explicit fused dense matrix view.",
        ],
    }
    _write_json(artifacts_dir / "roundtrip-evidence.json", evidence)

    assert pipeline_path.exists()
    assert dataset_path.exists()
    assert predictions_path.exists()
    assert len(oracle["selected"]["predictions"]) == len(oracle["targets"])
    assert run_summary["status"] == "passed"
