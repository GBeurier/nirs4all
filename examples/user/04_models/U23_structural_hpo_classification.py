"""Search native PLS-logistic early and learned late classification.

U07's deterministic four-modality cohort is a software validation fixture.
Labels and probability columns retain a signed Train-only class vocabulary.
Classification is serial; this example makes no parallel classification claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from n4m.roles import PLSLogistic
from nirs4all_io import MultimodalDataset
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalClassifier

_FIXTURE = Path(__file__).with_name("U20_structural_hpo_typed_modalities.py")
_SPEC = importlib.util.spec_from_file_location("classification_typed_fixture", _FIXTURE)
assert _SPEC is not None and _SPEC.loader is not None
_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_fixture)


def make_dataset(seed: int = 17, *, prediction: bool = False, n_classes: int = 3, numeric: bool = False, train_only: bool = False) -> Any:
    """Relabel the deterministic U07 software fixture; no product generator."""
    cohort = _fixture.make_dataset(seed, prediction=prediction)
    names = np.asarray([-9, 17, 103][:n_classes] if numeric else ["amber", "blue", "violet"][:n_classes])
    labels = None if prediction else names[np.arange(len(cohort.sample_ids)) % n_classes]
    return MultimodalDataset(cohort.sources, sample_ids=cohort.sample_ids, y=labels,
        groups=cohort.groups, partitions=["train"] * len(cohort.sample_ids) if train_only else cohort.partitions,
        target_names=("y",), task_type="classification", name="typed_classification")


def make_model(names: tuple[str, ...], weights: dict[str, float] | None = None) -> MultimodalClassifier:
    """Declare the existing encoders and a genuine Methods PLS-logistic head."""
    raw = _fixture.make_model(names, weights)
    return MultimodalClassifier(transformers=raw.transformers, source_weights=weights,
                               model=PLSLogistic(n_components=1, max_iter=500), backend="methods")


def make_late(names: tuple[str, ...]) -> list[Any]:
    """Keep public named branches and their prediction feature order explicit."""
    return [
        {"branch": {name: [{"model": make_model((name,), {name: 0.5 if name == "image" else 1.0})}] for name in names}},
        {"merge": "predictions"},
        {"model": PLSLogistic(n_components=1, max_iter=500)},
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
    """Tune only component axes active in the chosen native topology."""
    axes = ["early.n_components", "late.nir.n_components", "late.image.n_components", "late.series.n_components", "late.metadata.n_components", "late.meta.n_components"]
    return {
        "engine": "n4m", "sampler": "random", "seed": 17, "n_trials": 10,
        "metric": "accuracy", "direction": "maximize", "n_jobs": 1,
        "storage": directory.resolve().as_uri(), "study_name": "structural-classification", "resume": resume,
        "space": {axis: [1] if axis == "late.metadata.n_components" else [1, 2] for axis in axes},
    }


def main(output_path: str | Path | None = None) -> Path | None:
    """Export the exact fitted winner and replay after deleting training state."""
    temporary = tempfile.TemporaryDirectory(prefix="n4a-classification-") if output_path is None else None
    output = Path(temporary.name) if temporary is not None else Path(str(output_path))
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "training-workspace"
    new = make_dataset(29, prediction=True)
    try:
        with nirs4all.run(make_pipeline(), make_dataset(), tuning=make_tuning(output / "study"), engine="dag-ml",
                          workspace_path=workspace, refit=True, random_state=17, save_charts=False, verbose=0) as result:
            archive = result.export(output / "classifier-winner.n4a")
            assert result.classes_.tolist() == ["amber", "blue", "violet"]
            expected = nirs4all.predict(archive, new, engine="dag-ml").y_pred
            models = [node for node in result._dagml_graph["nodes"] if node["kind"] == "model"]
            mode = "learned late" if any(node["operator"].get("type") == "N4mRoleClassifierPipeline" for node in models) else "early"
            print(f"Selected fusion: {mode}; active components: {result.tuning_best_params}")
            print(f"Selected grouped CV accuracy: {result.tuning_best_value:.6f}")
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
