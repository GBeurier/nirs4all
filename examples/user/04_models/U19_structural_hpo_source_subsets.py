"""U19 — grouped native HPO over ordered source subsets and preprocessing.

Three aligned numeric blocks are a small deterministic test fixture, not a
multimodal data generator. Existing source-merge declarations choose block 0,
blocks 0 then 2, or blocks 2 then 0. Native DAG-ML enumerates those choices with
raw/scaled features and Ridge/PLS; Methods owns the conditional model axes.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data import SpectroDataset


def make_dataset(seed: int = 17) -> SpectroDataset:
    """Build 48 aligned fixture rows, 36 train rows and 12 external test rows."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(12), 4)
    latent = rng.normal(size=(48, 3)) + groups[:, None] * np.array([0.12, -0.04, 0.07])
    first = (latent @ np.array([[1, 0, 0.8, -0.4, 1.2, 0.2], [0, 1, 0.3, 0.9, -0.2, 0.6], [0.3, -0.2, 1, 0.1, 0.5, 1]])
             + rng.normal(scale=0.04, size=(48, 6))) * np.array([0.2, 1, 3, 15, 0.7, 8])
    unused = rng.normal(loc=25.0, scale=7.0, size=(48, 3))
    third = (latent @ np.array([[0.6, -0.2, 1.1, 0.7], [1.3, 0.4, -0.5, 0.2], [-0.4, 0.9, 0.3, 1.0]])
             + rng.normal(scale=0.03, size=(48, 4))) * np.array([7, 0.5, 2, 12])
    y = latent @ np.array([2.0, -1.5, 0.7]) + rng.normal(scale=0.06, size=48)
    blocks = [first, unused, third]
    dataset = SpectroDataset("structural-source-subsets")
    dataset.add_samples([block[:36] for block in blocks], {"partition": "train"})
    dataset.add_samples([block[36:] for block in blocks], {"partition": "test"})
    dataset.add_targets(y)
    dataset.add_metadata(np.array([f"batch-{value}" for value in groups])[:, None], headers=["batch"])
    return dataset


def make_pipeline() -> list[Any]:
    """Keep each ordered source subset as one declared generator alternative."""
    return [
        {"_or_": [
            {"merge": {"sources": {"strategy": "concat", "sources": [0]}}},
            {"merge": {"sources": {"strategy": "concat", "sources": [0, 2]}}},
            {"merge": {"sources": {"strategy": "concat", "sources": [2, 0]}}},
        ]},
        {"_or_": [None, StandardScaler()]},
        {"split": GroupKFold(3), "group_by": "batch"},
        {"model": {"_or_": [Ridge(), PLSRegression(scale=False)]}},
    ]


def make_tuning(directory: Path, *, resume: bool = False) -> dict[str, Any]:
    """Declare only the conditional numeric axes; DAG-ML owns recipe identity."""
    return {
        "engine": "n4m", "sampler": "random", "seed": 17,
        "metric": "rmse", "direction": "minimize", "n_trials": 16,
        "storage": directory.resolve().as_uri(), "study_name": "structural-source-subsets", "resume": resume,
        "space": {
            "model.alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            "model.n_components": {"type": "int", "low": 1, "high": 3},
        },
    }


def main(output_path: str | Path | None = None) -> Path | None:
    """Search and replay a source-selecting winner after removing its workspace."""
    temporary = tempfile.TemporaryDirectory(prefix="n4a-structural-sources-") if output_path is None else None
    output = Path(temporary.name) if temporary is not None else Path(str(output_path))
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "training-workspace"
    dataset = make_dataset()
    X_new = np.asarray(dataset.x({"partition": "test"}, layout="2d"))
    try:
        with nirs4all.run(
            make_pipeline(), dataset, tuning=make_tuning(output / "study"),
            engine="dag-ml", workspace_path=workspace, random_state=17,
            refit=True, save_charts=False, verbose=0,
        ) as result:
            archive = result.export(output / "structural-sources-winner.n4a")
            expected = nirs4all.predict(archive, X_new, engine="dag-ml").y_pred
            print(f"Selected CV RMSE: {result.tuning_best_value:.6f}")
            print(f"Active winner parameters: {result.tuning_best_params}")
        shutil.rmtree(workspace)
        replayed = nirs4all.predict(archive, X_new, engine="dag-ml")
        np.testing.assert_array_equal(replayed.y_pred, expected)
        assert replayed.metadata["training_performed"] is False
        print("Selected source subset archive replayed without FIT or HPO.")
        if temporary is None:
            print(f"Archive: {archive}")
            return archive
        return None
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U19 native grouped HPO: ordered source subsets, preprocessing and Ridge versus PLS")
    parser.add_argument("--output", default=None, help="Directory for the retained study and .n4a archive")
    main(parser.parse_args().output)
