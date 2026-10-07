"""D08 - Documented Feature Fusion, Stacking, and Multisource Residual Learning.

Executable companion to docs/source/guide/pipelines.md. Uses synthetic data and
explicit engine="dag-ml" with no legacy fallback. Requires the installed DAG-ML
host runtime. Runs three qualified shapes: feature fusion, single-source OOF
stacking, and source-local multimodal preprocessing with residual learning.
The last two shapes export and replay their predictors in temporary directories.

Source-local branching followed by duplication stacking is currently a richer
unsupported DAG-ML shape; this script does not combine those two branch modes.

Run from the repository root:
    .venv/bin/python examples/developer/01_advanced_pipelines/D08_documented_multisource_stacking.py

Data are synthetic and scores demonstrate contracts, not scientific performance.
"""

import tempfile
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.api.result import RunResult
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.models import ResidualModel
from nirs4all.operators.transforms import MSC, SNV, SavitzkyGolay


def main() -> None:
    """Run the three documented training and replay checkpoints."""
    print("Checkpoint 1")
    rng = np.random.default_rng(17)
    X = rng.normal(size=(48, 31))
    y = 2 * X[:, 10] - X[:, 20] + rng.normal(scale=0.1, size=48)

    dataset = SpectroDataset("doc_branch_feature_join")
    dataset.add_samples(X[:40], {"partition": "train"},
                        headers=[str(i) for i in range(31)])
    dataset.add_samples(X[40:], {"partition": "test"})
    dataset.add_targets(y.reshape(-1, 1))

    pipeline = [
        KFold(n_splits=3, shuffle=True, random_state=17),
        {"branch": {
            "normalized": [SNV()],
            "smoothed": [SavitzkyGolay(window_length=7, polyorder=2, deriv=0)],
        }},
        {"merge": "features"},
        {"model": Ridge(alpha=1.0)},
    ]

    with tempfile.TemporaryDirectory() as workspace:
        result = nirs4all.run(
            pipeline=pipeline, dataset=dataset, engine="dag-ml", refit=True,
            workspace_path=workspace, save_charts=False, verbose=0,
        )
        assert isinstance(result, RunResult)
        assert result.execution_engine == "dag-ml"
        print(result.execution_engine, result.best_rmse)
        result.close()

    print("Checkpoint 2")
    stacking_data = SpectroDataset("doc_stacking")
    stacking_data.add_samples(X[:40], {"partition": "train"},
                              headers=[str(i) for i in range(31)])
    stacking_data.add_samples(X[40:], {"partition": "test"})
    stacking_data.add_targets(y.reshape(-1, 1))

    stacking = [
        KFold(n_splits=3, shuffle=True, random_state=17),
        {"branch": {
            "pls": [SNV(), {"model": PLSRegression(n_components=3)}],
            "ridge": [MSC(), {"model": Ridge(alpha=1.0)}],
        }},
        {"merge": "predictions"},
        {"model": Ridge(alpha=0.1)},
    ]

    with tempfile.TemporaryDirectory() as workspace:
        result = nirs4all.run(
            pipeline=stacking, dataset=stacking_data, engine="dag-ml", refit=True,
            workspace_path=workspace, save_charts=False, verbose=0,
        )
        assert isinstance(result, RunResult)
        assert result.execution_engine == "dag-ml"
        archive = result.export(Path(workspace) / "stacked.n4a")
        replay = nirs4all.predict(archive, X[40:])
        assert len(replay.y_pred) == 8
        print(result.execution_engine, result.best_rmse)
        result.close()

    print("Checkpoint 3")
    rng = np.random.default_rng(17)
    spectra = rng.normal(size=(48, 31))
    markers = rng.normal(size=(48, 3))
    y = (2 * spectra[:, 10] - spectra[:, 20] + markers[:, 0]
         + rng.normal(scale=0.1, size=48))

    dataset = SpectroDataset("doc_multimodal_residual")
    dataset.add_samples(
        [spectra[:40], markers[:40]], {"partition": "train"},
        headers=[[str(i) for i in range(31)],
                 ["marker_0", "marker_1", "marker_2"]],
    )
    dataset.add_samples([spectra[40:], markers[40:]], {"partition": "test"})
    dataset.add_targets(y.reshape(-1, 1))

    pipeline = [
        KFold(n_splits=3, shuffle=True, random_state=17),
        {"branch": {"by_source": True, "steps": {
            "source_0": [SNV()],
            "source_1": [StandardScaler()],
        }}},
        {"merge": {"sources": "concat"}},
        {"model": ResidualModel(
            base=PLSRegression(n_components=3),
            learner=Ridge(alpha=1.0), gate=False,
        )},
    ]

    with tempfile.TemporaryDirectory() as workspace:
        result = nirs4all.run(
            pipeline=pipeline, dataset=dataset, engine="dag-ml", refit=True,
            workspace_path=workspace, save_charts=False, verbose=0,
        )
        assert isinstance(result, RunResult)
        assert result.execution_engine == "dag-ml"
        archive = result.export(Path(workspace) / "multimodal-residual.n4a")
        # Supply raw columns in the original source order; replay owns preprocessing.
        raw_new = np.column_stack([spectra[40:], markers[40:]])
        replay = nirs4all.predict(archive, raw_new)
        assert len(replay.y_pred) == 8
        assert np.isfinite(replay.y_pred).all()
        print(result.execution_engine, result.best_rmse)
        result.close()


if __name__ == "__main__":
    main()
