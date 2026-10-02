"""U17 — native structure search over optional scaling and Ridge versus PLS.

The public generators declare four recipes. DAG-ML chooses their structure and
owns grouped CV, scoring and selection; Methods proposes only the parameter
active for the chosen model. A fitted winner can be exported and replayed after
the training workspace has been removed.
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
    """Return a small dense cohort with disjoint training and test batches."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(12), 4)
    latent = rng.normal(size=(48, 3)) + groups[:, None] * np.array([0.12, -0.04, 0.07])
    loadings = np.array([[1, 0, 0.8, -0.4, 1.2, 0.2], [0, 1, 0.3, 0.9, -0.2, 0.6], [0.3, -0.2, 1, 0.1, 0.5, 1]])
    X = (latent @ loadings + rng.normal(scale=0.04, size=(48, 6))) * np.array([0.2, 1, 3, 15, 0.7, 8])
    y = latent @ np.array([2.0, -1.5, 0.7]) + rng.normal(scale=0.06, size=48)
    dataset = SpectroDataset("structural-ridge-pls")
    dataset.add_samples(X[:36], {"partition": "train"})
    dataset.add_samples(X[36:], {"partition": "test"})
    dataset.add_targets(y)
    dataset.add_metadata(np.array([f"batch-{value}" for value in groups])[:, None], headers=["batch"])
    return dataset


def make_pipeline() -> list[Any]:
    """Declare choices directly with the existing public pipeline syntax."""
    return [
        {"_or_": [None, StandardScaler()]},
        {"split": GroupKFold(3), "group_by": "batch"},
        {"model": {"_or_": [Ridge(), PLSRegression(scale=False)]}},
    ]


def make_tuning(directory: Path, *, resume: bool = False) -> dict[str, Any]:
    """Use one durable native search; inactive model axes are never fitted."""
    return {
        "engine": "n4m", "sampler": "random", "seed": 17,
        "metric": "rmse", "direction": "minimize", "n_trials": 8,
        "storage": directory.resolve().as_uri(), "study_name": "structural-ridge-pls", "resume": resume,
        "space": {
            "model.alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            "model.n_components": {"type": "int", "low": 1, "high": 3},
        },
    }


def main(output_path: str | Path | None = None) -> Path | None:
    """Search, export the winner and replay with no training workspace."""
    temporary = tempfile.TemporaryDirectory(prefix="n4a-structural-hpo-") if output_path is None else None
    output = Path(temporary.name) if temporary is not None else Path(str(output_path))
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "training-workspace"
    dataset = make_dataset()
    X_new = dataset.x({"partition": "test"}, layout="2d")
    try:
        with nirs4all.run(
            make_pipeline(), dataset, tuning=make_tuning(output / "study"),
            engine="dag-ml", workspace_path=workspace, random_state=17,
            refit=True, save_charts=False, verbose=0,
        ) as result:
            archive = result.export(output / "structural-winner.n4a")
            expected = nirs4all.predict(archive, X_new, engine="dag-ml").y_pred
            print(f"Selected CV RMSE: {result.tuning_best_value:.6f}")
            print(f"Active winner parameters: {result.tuning_best_params}")
        shutil.rmtree(workspace)
        replayed = nirs4all.predict(archive, X_new, engine="dag-ml")
        np.testing.assert_array_equal(replayed.y_pred, expected)
        assert replayed.metadata["training_performed"] is False
        print("Winner archive replayed without FIT or HPO.")
        if temporary is None:
            print(f"Archive: {archive}")
            return archive
        return None
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U17 grouped native structure HPO: Ridge versus PLS")
    parser.add_argument("--output", default=None, help="Directory for the retained study and .n4a archive")
    main(parser.parse_args().output)
