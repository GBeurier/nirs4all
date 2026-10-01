"""U16 — initialize full-train SGD REFIT from an explicitly named native CV fold.

Uses the public DAG-ML run/export/predict path with a deterministic, closed SGD
recipe. Weights transfer through sklearn's public initializers; optimizer state
does not continue. No feature or target transformation belongs to this profile.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold

import nirs4all


def main(output_path: str | Path | None = None) -> Path | None:
    """Train, export, remove the training workspace, and replay the archive."""
    temporary = tempfile.TemporaryDirectory(prefix="n4a-cv-weight-") if output_path is None else None
    output = Path(temporary.name) if temporary is not None else Path(str(output_path))
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "training-workspace"
    try:
        rng = np.random.default_rng(41)
        X = rng.normal(scale=0.5, size=(30, 3)).astype(np.float32)
        y = (3 * X[:, 0] - 2 * X[:, 1] + 0.7 * X[:, 2] + 0.25).astype(np.float32)
        X_new = rng.normal(scale=0.5, size=(6, 3)).astype(np.float32)
        with nirs4all.run(
            [KFold(3, shuffle=True, random_state=31), {
                "model": SGDRegressor(
                    loss="squared_error", penalty="l2", alpha=0.01,
                    learning_rate="constant", eta0=0.01,
                    random_state=19, shuffle=False, max_iter=2, tol=None,
                ),
                "refit_params": {"warm_start": True, "warm_start_fold": "fold1", "max_iter": 3},
            }],
            (X, y), engine="dag-ml", workspace_path=workspace,
            save_charts=False, verbose=0,
        ) as result:
            archive = result.export(output / "sgd-warm-start.n4a")
            expected = nirs4all.predict(archive, X_new, engine="dag-ml").y_pred
        shutil.rmtree(workspace)
        replayed = nirs4all.predict(archive, X_new, engine="dag-ml")
        np.testing.assert_array_equal(replayed.y_pred, expected)
        assert replayed.metadata["training_performed"] is False
        print("SGD REFIT initialized from fold1; archive replay completed without training.")
        print(f"Predictions: {replayed.y_pred.tolist()}")
        if temporary is None:
            print(f"Archive: {archive}")
            return archive
        return None
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U16 explicit CV-fold SGD weight transfer")
    parser.add_argument("--output", default=None, help="Directory for the retained .n4a archive")
    main(parser.parse_args().output)
