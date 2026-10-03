"""Search declared early and learned late fusion with native grouped nested OOF.

U07's deterministic four-modality cohort is a software validation fixture.
Every portable winner retains all original raw inputs, including exclusions.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import nirs4all

_FIXTURE = Path(__file__).with_name("U20_structural_hpo_typed_modalities.py")
_SPEC = importlib.util.spec_from_file_location("early_late_typed_fixture", _FIXTURE)
assert _SPEC is not None and _SPEC.loader is not None
_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_fixture)


def make_dataset(seed: int = 17, *, prediction: bool = False) -> Any:
    """Reuse the complete canonical aligned U07 cohort."""
    return _fixture.make_dataset(seed, prediction=prediction)


def make_model(names: tuple[str, ...], weights: dict[str, float] | None = None) -> Any:
    """Declare one complete native encoder/predictor without learning state."""
    return _fixture.make_model(names, weights)


def make_late(names: tuple[str, ...]) -> list[Any]:
    """Keep public named branches and their prediction feature order explicit."""
    return [
        {"branch": {name: [{"model": make_model((name,), {name: 0.5 if name == "image" else 1.0})}] for name in names}},
        {"merge": "predictions"},
        {"model": Ridge(alpha=1.0)},
    ]


def make_pipeline() -> list[Any]:
    """Declare early fusion and late fusion with two, three and four branches."""
    return [GroupKFold(3), {"_or_": [
        [{"model": make_model(("nir", "image"), {"nir": 1.0, "image": 0.5})}],
        make_late(("nir", "image")),
        make_late(("image", "nir")),
        make_late(("image", "series", "metadata")),
        make_late(("nir", "image", "series", "metadata")),
    ]}]


def make_tuning(directory: Path, *, resume: bool = False) -> dict[str, Any]:
    """Tune only alpha axes active in the chosen native topology."""
    axes = ["early.alpha", "late.nir.alpha", "late.image.alpha", "late.series.alpha", "late.metadata.alpha", "late.meta.alpha"]
    return {
        "engine": "n4m", "sampler": "random", "seed": 17, "n_trials": 10,
        "metric": "rmse", "direction": "minimize", "n_jobs": 1,
        "storage": directory.resolve().as_uri(), "study_name": "structural-early-late", "resume": resume,
        "space": {axis: {"type": "float", "low": 0.01, "high": 10.0, "log": True} for axis in axes},
    }


def main(output_path: str | Path | None = None) -> Path | None:
    """Export the exact fitted winner and replay after deleting training state."""
    temporary = tempfile.TemporaryDirectory(prefix="n4a-early-late-") if output_path is None else None
    output = Path(temporary.name) if temporary is not None else Path(str(output_path))
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "training-workspace"
    new = make_dataset(29, prediction=True)
    try:
        with nirs4all.run(make_pipeline(), make_dataset(), tuning=make_tuning(output / "study"), engine="dag-ml",
                          workspace_path=workspace, refit=True, random_state=17, save_charts=False, verbose=0) as result:
            archive = result.export(output / "early-late-winner.n4a")
            expected = nirs4all.predict(archive, new, engine="dag-ml").y_pred
            models = [node for node in result._dagml_graph["nodes"] if node["kind"] == "model"]
            mode = "learned late" if any(node["operator"].get("type") == "N4mRolePipeline" for node in models) else "early"
            print(f"Selected fusion: {mode}; active alphas: {result.tuning_best_params}")
            print(f"Selected grouped CV RMSE: {result.tuning_best_value:.6f}")
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.rmtree(output / "study")
        actual = nirs4all.predict(archive, new, engine="dag-ml")
        np.testing.assert_array_equal(actual.y_pred, expected)
        assert actual.metadata["training_performed"] is False
        print("Complete raw-input archive replayed without FIT or HPO.")
        if temporary is None:
            print(f"Archive: {archive}")
            return archive
        return None
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Directory for the retained portable winner archive")
    main(parser.parse_args().output)
