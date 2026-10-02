"""Choose ordered typed modalities and declared early-fusion weights natively.

This reuses U07's deterministic four-modality fixture for software validation.
The public declarations select encoder subsets; excluded encoders never fit.
All four original raw sources remain part of the portable archive input contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalRegressor

_U07 = Path(__file__).resolve().parents[1] / "02_data_handling/U07_multimodal.py"
_SPEC = importlib.util.spec_from_file_location("typed_modalities_u07_fixture", _U07)
assert _SPEC is not None and _SPEC.loader is not None
_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_fixture)


def make_dataset(seed: int = 17, *, prediction: bool = False) -> Any:
    """Reuse the canonical U07 aligned NIR/image/series/mixed-metadata fixture."""
    return _fixture.make_cohort(seed, prediction=prediction)


def make_model(names: tuple[str, ...], weights: dict[str, float] | None = None) -> MultimodalRegressor:
    """Declare selected encoder families in the requested fusion order."""
    full = _fixture.make_pipeline(backend="methods")[-1]["model"]
    return MultimodalRegressor(
        transformers={name: clone(full.transformers[name]) for name in names},
        source_weights=weights,
        model=Ridge(alpha=1.0),
        backend="methods",
    )


def make_pipeline() -> list[Any]:
    """Let the native catalogue identify explicit modality/weight alternatives."""
    return [
        GroupKFold(3),
        {
            "model": {
                "_or_": [
                    make_model(("nir",)),
                    make_model(("nir", "image"), {"nir": 1.0, "image": 0.5}),
                    make_model(("image", "nir"), {"image": 0.5, "nir": 1.0}),
                    make_model(("image", "series", "metadata"), {"image": 0.5, "series": 1.0, "metadata": 1.0}),
                ]
            }
        },
    ]


def make_tuning(directory: Path, *, resume: bool = False) -> dict[str, Any]:
    """Tune Ridge alpha; each modality order/weight remains a declared recipe."""
    return {
        "engine": "n4m",
        "sampler": "random",
        "seed": 17,
        "n_trials": 8,
        "metric": "rmse",
        "direction": "minimize",
        "storage": directory.resolve().as_uri(),
        "study_name": "structural-typed-modalities",
        "resume": resume,
        "space": {"model__alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True}},
    }


def main(output_path: str | Path | None = None) -> Path | None:
    """Export the complete winner and replay new raw inputs without training."""
    temporary = tempfile.TemporaryDirectory(prefix="n4a-typed-modalities-") if output_path is None else None
    output = Path(temporary.name) if temporary is not None else Path(str(output_path))
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "training-workspace"
    workspace.mkdir(exist_ok=True)
    new = make_dataset(29, prediction=True)
    try:
        with nirs4all.run(make_pipeline(), make_dataset(), tuning=make_tuning(output / "study"), engine="dag-ml", workspace_path=workspace, refit=True, random_state=17, save_charts=False, verbose=0) as result:
            archive = result.export(output / "typed-modalities-winner.n4a")
            expected = nirs4all.predict(archive, new, engine="dag-ml").y_pred
            winner = result.structural_tuning_training_request["graph"]["nodes"][0]
            print(f"Selected modalities: {winner['operator']['recipe']['source_order']}")
            print(f"Selected CV RMSE: {result.tuning_best_value:.6f}")
        shutil.rmtree(workspace)
        actual = nirs4all.predict(archive, new, engine="dag-ml")
        np.testing.assert_array_equal(actual.y_pred, expected)
        assert actual.metadata["training_performed"] is False
        print("Typed modality archive replayed with its complete raw input contract, without FIT or HPO.")
        if temporary is None:
            print(f"Archive: {archive}")
            return archive
        return None
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Directory for the retained study and portable winner archive")
    main(parser.parse_args().output)
