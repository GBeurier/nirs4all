"""Four raw modalities, grouped DAG tuning and a complete prediction archive.

Run: python examples/user/02_data_handling/U07_multimodal.py --output /tmp/multimodal

This deterministic synthetic example qualifies software behavior, not a
scientific benefit on a real corpus. No external data or pretrained weights
are needed. Images and time series reach their encoders as raw N-D arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import time as timer
import tracemalloc
from pathlib import Path
from typing import Any, cast

import numpy as np
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.operators.models.multimodal import MultimodalRegressor, TensorPCA
from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped


def make_cohort(seed: int = 17, *, prediction: bool = False) -> MultimodalDataset:
    """Create unequal repetitions of independent units, aligned by observation ID."""
    rng = np.random.default_rng(seed)
    n_groups = 24 if not prediction else 6
    group_rows = np.repeat(np.arange(n_groups), 1 + np.arange(n_groups) % 3)
    latent = rng.normal(size=(n_groups, 3))[group_rows]
    n = len(group_rows)
    ids = [f"{'new' if prediction else 'sample'}.{seed}.{i}" for i in range(n)]
    wavelengths = np.linspace(900, 1700, 24)
    spectral_basis = np.stack([np.sin(wavelengths / 170), np.cos(wavelengths / 240), wavelengths / 1700])
    nir = latent @ spectral_basis + rng.normal(0, 0.03, (n, 24))
    pixels = np.linspace(-1, 1, 8)
    image_basis = np.stack(np.broadcast_arrays(pixels[:, None], pixels[None, :], np.ones((8, 8))), axis=-1)
    images = latent[:, None, None, :] * image_basis + rng.normal(0, 0.04, (n, 8, 8, 3))
    time = np.linspace(0, 2 * np.pi, 16)
    series = np.stack([latent[:, 0, None] * np.sin(time), latent[:, 1, None] * np.cos(time)], axis=-1)
    series += rng.normal(0, 0.03, series.shape)
    metadata = np.column_stack([latent[:, 2] + rng.normal(0, 0.05, n), np.where(group_rows % 2, "B", "A")])
    y = latent @ np.array([1.2, -0.8, 0.5]) + rng.normal(0, 0.04, n)
    raw = {"nir": (nir, "signal_1d"), "image": (images, "rgb_image"),
           "series": (series, "series_mv"), "metadata": (metadata, "tabular_mixed")}
    sources = {}
    axes: dict[str, dict[str, Any]] = {
        "nir": {"axis_units": {"wavelength": "nm"}, "axis_coordinates": {"wavelength": wavelengths}},
        "image": {"axis_units": {"height": "px", "width": "px"}, "axis_coordinates": {"channel": ["R", "G", "B"]}},
        "series": {"axis_units": {"time": "s"}, "axis_coordinates": {"time": time, "variable": ["sensor_a", "sensor_b"]}},
        "metadata": {"feature_names": ["measurement", "category"]},
    }
    for name, (values, representation) in raw.items():
        order = rng.permutation(n)
        sources[name] = TensorSource(values[order], [ids[i] for i in order], representation_id=representation, **axes[name])
    return MultimodalDataset(
        sources, sample_ids=ids, y=None if prediction else y,
        groups=[f"unit.{seed}.{g}" for g in group_rows],
        partitions=["predict" if prediction else "train" if g < 18 else "test" for g in group_rows],
        name="four_modalities",
    )


def make_pipeline() -> list:
    """Tune an encoder, fusion weight and regressor with the same grouped folds."""
    model = MultimodalRegressor(
        transformers={
            "nir": StandardScaler(),
            "image": TensorPCA(2, random_state=17),
            "series": TensorPCA(2, random_state=17),
            "metadata": ColumnTransformer([
                ("numeric", StandardScaler(), [0]),
                ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1]),
            ]),
        },
        model=Ridge(alpha=1.0),
    )
    return [GroupKFold(3), {"model": model, "_grid_": {
        "model__alpha": [0.1, 1.0],
        "source_weights__image": [0.5, 1.0],
        "transformers__image__n_components": [2, 4],
    }}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(os.environ["NIRS4ALL_WORKSPACE"]) / "multimodal_demo" if "NIRS4ALL_WORKSPACE" in os.environ else Path("multimodal_demo")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--search", choices=("random", "grid"), default="random")
    parser.add_argument("--resume", action="store_true", help="Continue the native random search checkpoint")
    parser.add_argument("--stop-after", type=int, help="Stop between trials; resume with the same output directory")
    parser.add_argument("--replay", type=Path, help="Predict new raw inputs using this archive, without training")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 0:
        parser.error("--stop-after must be non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.replay:
        prediction = nirs4all.predict(args.replay, make_cohort(29, prediction=True), verbose=0)
        report = {"new_predictions": np.asarray(prediction.y_pred).tolist(),
                  "training_performed_on_reload": prediction.metadata["training_performed"]}
        (args.output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if args.search == "grid" and (args.resume or args.stop_after is not None):
        parser.error("--resume and --stop-after apply to --search random")
    pipeline = make_pipeline()
    tuning: dict[str, Any] | None = None
    if args.search == "random":
        space = pipeline[-1].pop("_grid_")
        tuning = {"engine": "n4m", "sampler": "random", "seed": 17, "n_trials": 8,
                  "space": space, "storage": args.output.resolve().as_uri(), "study_name": "multimodal", "resume": args.resume}
    # Train from the round-tripped public declarations, retaining raw tensors.
    pipeline_file = args.output / "pipeline.json"
    dataset_file = args.output / "dataset.json"
    pipeline_file.write_text(json.dumps(serialize_component(pipeline), indent=2) + "\n")
    dataset_file.write_text(json.dumps(make_cohort().to_dict(), indent=2, allow_nan=False) + "\n")
    (args.output / "prediction_dataset.json").write_text(json.dumps(make_cohort(29, prediction=True).to_dict(), indent=2, allow_nan=False) + "\n")
    if tuning is not None:
        (args.output / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n")
        if args.stop_after is not None:
            tuning["progress_callback"] = lambda event: len(event["checkpoint"]["trials"]) < args.stop_after
    started = timer.perf_counter()
    tracemalloc.start()
    try:
        result = cast(RunResult, nirs4all.run(
            pipeline=deserialize_component(json.loads(pipeline_file.read_text())),
            dataset=MultimodalDataset.from_dict(json.loads(dataset_file.read_text())), engine="dag-ml", tuning=tuning,
            workspace_path=args.output / "workspace", save_charts=False, verbose=0, random_state=17,
        ))
    except MultimodalTuningStopped as exc:
        tracemalloc.stop()
        print(json.dumps({"status": "cancelled", "completed_trials": len(exc.evidence["checkpoint"]["trials"]),
                          "resume": f"--output {args.output} --resume"}, indent=2))
        return
    try:
        archive = args.output / "multimodal.n4a"
        result.export(archive)
        prediction = nirs4all.predict(archive, make_cohort(29, prediction=True), verbose=0)
        report = {
            "fixture": "deterministic synthetic data", "engine": result.execution_engine,
            "raw_shapes": {name: list(source.values.shape) for name, source in make_cohort().sources.items()},
            "cv_rmse": float(result.cv_best_score), "test_rmse": float(result.best_rmse),
            "archive": archive.name, "new_predictions": np.asarray(prediction.y_pred).tolist(),
            "training_performed_on_reload": prediction.metadata.get("training_performed"),
            "search": args.search, "trial_budget": 8,
            "tuning": result.tuning_result.to_dict() if result.tuning_result is not None else None,
            "elapsed_seconds": timer.perf_counter() - started,
            "python_peak_bytes": tracemalloc.get_traced_memory()[1], "archive_bytes": archive.stat().st_size,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))
    finally:
        result.close()
        tracemalloc.stop()


if __name__ == "__main__":
    main()
