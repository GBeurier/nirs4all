"""Run a fold-scoped data provider on a four-modality synthetic test fixture.

The NIR source is supplied again for each native train/validation/refit view;
image, series, metadata, identities and targets stay fixed. This is an example
of the provider interface, not a multimodal synthetic-data product generator.

Run: python examples/user/02_data_handling/U14_multimodal_fold_provider.py --output /tmp/mm-fold-provider
Use --hpo for native random search, --n-jobs 2 for candidate workers, or
--worker to run the outer DAG-ML job in an isolated Python worker.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from nirs4all_io import DataProvider, MultimodalDataset, TensorSource
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalRegressor, TensorPCA


def fixture_cohort(*, prediction: bool = False) -> MultimodalDataset:
    """Create only deterministic fixture rows for this runnable example."""
    rng = np.random.default_rng(181 if prediction else 79)
    rows = 5 if prediction else 16
    ids = [f"{'new' if prediction else 'train'}-{index:02d}" for index in range(rows)]
    latent = rng.normal(size=rows)
    sources = {
        "nir": TensorSource(
            latent[:, None] + rng.normal(scale=0.2, size=(rows, 6)), ids,
            representation_id="signal_1d", axis_units={"wavelength": "nm"},
            axis_coordinates={"wavelength": np.linspace(900, 1700, 6)},
        ),
        "image": TensorSource(
            latent[:, None, None, None] + rng.normal(scale=0.1, size=(rows, 2, 2, 3)),
            ids, representation_id="rgb_image",
        ),
        "series": TensorSource(
            latent[:, None, None] + rng.normal(scale=0.15, size=(rows, 5, 2)),
            ids, representation_id="series_mv", axis_units={"time": "s"},
            axis_coordinates={"time": np.arange(5)},
        ),
        "metadata": TensorSource(
            np.array([[index / rows, "a" if index % 2 else "b"] for index in range(rows)], dtype=object),
            ids, representation_id="tabular_mixed", feature_names=["age", "batch"],
        ),
    }
    return MultimodalDataset(
        sources, sample_ids=ids,
        y=None if prediction else 2.1 * latent + np.arange(rows) * 0.03,
        groups=None if prediction else [f"plant-{index // 2}" for index in range(rows)],
        partitions=["predict"] * rows if prediction else ["train"] * 12 + ["test"] * 4,
        name="new-fixture-rows" if prediction else "fold-provider-fixture",
    )


def make_provider() -> DataProvider:
    base = fixture_cohort()

    def prepare(**_: Any) -> dict[str, Any]:
        return {"sample_ids": list(base.sample_ids), "sources": {"nir": base.sources["nir"]}}

    def generate_view(*, sample_ids: list[str], seed: int, **_: Any) -> dict[str, Any]:
        source = base.take(sample_ids).sources["nir"]
        values = np.asarray(source.values) + float(seed % 7)
        return {"sample_ids": sample_ids, "sources": {"nir": TensorSource(
            values, sample_ids, representation_id=source.representation_id,
            axis_units=source.axis_units, axis_coordinates=source.axis_coordinates,
        )}}

    return DataProvider(
        prepare, generate_view=generate_view, provider_id="nirs4all.example.fold-provider",
        base=base, replace_sources=["nir"],
    )


def make_model() -> MultimodalRegressor:
    return MultimodalRegressor(
        {
            "nir": StandardScaler(),
            "image": TensorPCA(n_components=2, random_state=19),
            "series": TensorPCA(n_components=2, random_state=19),
            "metadata": ColumnTransformer([
                ("numeric", StandardScaler(), [0]),
                ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1]),
            ]),
        },
        model=Ridge(alpha=0.2),
    )


def run_demo(output: Path, *, hpo: bool = False, n_jobs: int = 1, worker: bool = False) -> dict[str, Any]:
    if type(n_jobs) is not int or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    if n_jobs > 1 and not hpo:
        raise ValueError("n_jobs > 1 requires --hpo")
    output.mkdir(parents=True, exist_ok=True)
    tuning = None
    if hpo:
        tuning = {
            "engine": "n4m", "sampler": "random", "seed": 19, "metric": "rmse",
            "direction": "minimize", "n_trials": 3, "n_jobs": n_jobs,
            "space": {"model__alpha": [0.1, 1.0]},
            "storage": (output / "study").as_uri(), "study_name": "fold-provider",
        }
    with patch.dict(os.environ, {"N4A_DAGML_INPROCESS": "0"}) if worker else nullcontext():
        result = nirs4all.run(
            [GroupKFold(3), {"model": make_model()}], make_provider(),
            tuning=tuning, engine="dag-ml", refit=True, save_artifacts=False,
            save_charts=False, random_state=19, verbose=0,
            results_path=output / "results",
        )
    try:
        archive = result.export(output / "fold-provider.n4a")
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("dagml_generated_view_manifest.json"))
        if not manifest["views"]:
            raise AssertionError("No generated train/validation/refit views were consumed")
        with patch.object(DataProvider, "materialize", side_effect=AssertionError("Replay regenerated training data")):
            prediction = nirs4all.predict(archive, fixture_cohort(prediction=True))
        values = np.asarray(prediction.y_pred)
        if prediction.metadata["training_performed"] is not False or values.size != 5 or not np.isfinite(values).all():
            raise AssertionError("Prediction archive did not replay five finite rows without training")
        report = {
            "mode": "worker" if worker else "in-process",
            "cv_rmse": float(result.best_rmse),
            "generated_views": len(manifest["views"]),
            "hpo_trials": 0 if tuning is None else len(result.tuning_result.trials),
            "prediction_rows": len(values),
            "prediction_without_training": True,
            "archive": str(archive),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report
    finally:
        result.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fold-provider-demo"))
    parser.add_argument("--hpo", action="store_true", help="Tune the multimodal estimator with native N4M random search")
    parser.add_argument("--n-jobs", type=int, default=1, help="Independent HPO candidates to evaluate concurrently")
    parser.add_argument("--worker", action="store_true", help="Run the outer DAG-ML job in a separate Python worker")
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output, hpo=args.hpo, n_jobs=args.n_jobs, worker=args.worker), indent=2))
