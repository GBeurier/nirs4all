"""Generate variable-length series, fit grouped models and replay new lengths.

Run: python examples/user/02_data_handling/U12_multimodal_ragged_series.py --output /tmp/mm-ragged
Use --fusion intermediate for MBPLS instead of early fusion with Ridge.
All observations are synthetic; no real corpus, padding or charts are needed.
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from nirs4all_io import DataProvider, MultimodalDataset, RaggedSeriesSource, TensorSource
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.operators.models import MultimodalRegressor
from nirs4all.operators.models.sklearn.mbpls import MBPLS
from nirs4all.operators.transforms import SequenceSummary
from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component


def generate_cohort(*, seed: int, params: dict[str, Any], context: dict[str, Any]) -> MultimodalDataset:
    """Return a finite cohort with packed observations and explicit boundaries."""
    count = int(params["n_samples"])
    min_length, max_length = int(params["min_length"]), int(params["max_length"])
    prediction = context.get("role") == "predict"
    rng = np.random.default_rng(seed)
    ids = [f"{'new' if prediction else 'sample'}-{index:03d}" for index in range(count)]
    latent = rng.normal(size=(count, 2))
    wavelengths = np.linspace(950, 1150, 12)
    spectra = (latent[:, :1] * np.exp(-((wavelengths - 1000) / 35) ** 2)
               + latent[:, 1:] * np.exp(-((wavelengths - 1100) / 40) ** 2)
               + rng.normal(scale=0.05, size=(count, len(wavelengths))))
    lengths = min_length + np.arange(count) % (max_length - min_length + 1)
    series_values, time_coordinates = [], []
    for row, length in enumerate(lengths):
        times = np.cumsum(rng.uniform(0.1, 0.5, size=int(length)))
        values = np.column_stack([
            latent[row, 0] + 0.1 * np.sin(times),
            latent[row, 1] + 0.1 * np.cos(times),
        ]) + rng.normal(scale=0.03, size=(int(length), 2))
        series_values.append(values)
        time_coordinates.append(times)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)])
    sources: dict[str, TensorSource | RaggedSeriesSource] = {
        "nir": TensorSource(spectra, ids, representation_id="signal_1d", axis_units={"wavelength": "nm"},
                            axis_coordinates={"wavelength": wavelengths}),
        "series": RaggedSeriesSource(
            np.concatenate(series_values), offsets, ids,
            time_coordinates=np.concatenate(time_coordinates),
            channel_names=["temperature", "intensity"], time_unit="s",
        ),
    }
    # Prediction input order differs deliberately; the archive binds sources by name.
    if prediction:
        sources = dict(reversed(list(sources.items())))
    n_train = 2 * ((3 * count // 4) // 2)
    return MultimodalDataset(
        sources, sample_ids=ids,
        y=None if prediction else 2 * latent[:, 0] - 1.5 * latent[:, 1],
        target_names=["synthetic_response"], task_type="regression",
        groups=None if prediction else [f"subject-{index // 2:02d}" for index in range(count)],
        partitions=["predict"] * count if prediction else ["train"] * n_train + ["test"] * (count - n_train),
        name="ragged-prediction" if prediction else "ragged-provider",
    )


def pipeline_for(fusion: str) -> list[Any]:
    """Keep series packed until their explicit encoder runs inside each fold."""
    return [GroupKFold(3), MultimodalRegressor(
        {"nir": StandardScaler(), "series": make_pipeline(
            SequenceSummary(channel_names=["temperature", "intensity"]), StandardScaler(),
        )},
        model=Ridge(alpha=1.0) if fusion == "early" else MBPLS(n_components=2, standardize=False),
        fusion=fusion,
    )]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_demo(output: Path, *, fusion: str = "early") -> dict[str, Any]:
    """Exercise native preparation, grouped CV, typed JSON and archive inference."""
    output.mkdir(parents=True, exist_ok=True)
    provider = DataProvider(
        generate_cohort, provider_id="nirs4all.example.ragged-series", provider_version="1",
        params={"n_samples": 32, "min_length": 3, "max_length": 9}, seed=17,
    )
    recipe = serialize_component(pipeline_for(fusion))
    _write_json(output / "provider_recipe.json", provider.recipe())
    _write_json(output / "pipeline.json", recipe)
    with threadpool_limits(limits=1):
        result = cast(RunResult, nirs4all.run(
            deserialize_component(recipe), provider, engine="dag-ml", refit=True, random_state=17,
            workspace_path=output / "workspace", save_artifacts=True, save_charts=False, verbose=0,
        ))
        try:
            cohort = provider.cohort
            training_series = cohort.sources["series"]
            assert isinstance(training_series, RaggedSeriesSource)
            partitions = np.asarray(cohort.partitions)
            _write_json(output / "dataset.json", cohort.to_dict())
            archive = result.export(output / "multimodal.n4a")
            report = {
                "fixture": "deterministic synthetic data", "engine": result.execution_engine,
                "fusion": fusion, "metric": "rmse", "cv_best_score": result.cv_best_score,
                "test_rmse": result.best_rmse, "target_names": list(cohort.target_names),
                "provider": result.per_dataset[cohort.name]["data_provider_evidence"],
                "partition_lengths": {partition: training_series.lengths[partitions == partition].tolist() for partition in ("train", "test")},
                "sequence_encoding": {"statistics": ["mean", "std", "min", "max"],
                                      "include_length": True, "time_weighting": "uniform_observations"},
                "archive": archive.name,
            }
        finally:
            result.close()

        new = generate_cohort(seed=29, params={"n_samples": 8, "min_length": 11, "max_length": 18}, context={"role": "predict"})
        prediction_series = new.sources["series"]
        assert isinstance(prediction_series, RaggedSeriesSource)
        _write_json(output / "prediction_dataset.json", new.to_dict())
        restored = MultimodalDataset.from_dict(json.loads((output / "prediction_dataset.json").read_text(encoding="utf-8")))
        with ExitStack() as guards:
            guards.enter_context(patch.object(DataProvider, "materialize", side_effect=AssertionError("Replay must not regenerate training data")))
            for estimator in (SequenceSummary, StandardScaler, MultimodalRegressor, Ridge, MBPLS):
                for method in ("fit", "fit_transform", "partial_fit"):
                    if hasattr(estimator, method):
                        guards.enter_context(patch.object(estimator, method, side_effect=AssertionError("Replay must not fit")))
            expected = nirs4all.predict(archive, new)
            replay = nirs4all.predict(archive, restored)
        np.testing.assert_array_equal(replay.y_pred, expected.y_pred)
        assert replay.y_pred.shape == (len(new),)
        assert np.isfinite(replay.y_pred).all()
        assert replay.metadata["training_performed"] is False
        assert replay.metadata["scores"] is None
        assert min(prediction_series.lengths) > max(training_series.lengths)
        report.update({"new_lengths": prediction_series.lengths.tolist(),
                       "new_predictions": replay.y_pred.tolist(), "training_performed_on_reload": False,
                       "replay_with_fit_and_generation_forbidden": True})
        _write_json(output / "report.json", report)
    return {"status": "complete", "fusion": fusion, "archive": str(archive), "report": str(output / "report.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("multimodal_ragged_demo"))
    parser.add_argument("--fusion", choices=("early", "intermediate"), default="early")
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output, fusion=args.fusion), indent=2))
