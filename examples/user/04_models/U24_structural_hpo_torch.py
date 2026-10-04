"""Native structural HPO over real CPU Torch early and learned late regression.

The small deterministic arrays below are software fixtures, not a scientific
benchmark or a dataset generator API. Requires matching DAG/SDK host adapters.
The complete fitted closure uses Python joblib host sidecars, not portable
Methods/Core numerical states or an R/WASM classifier host.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalRegressor
from nirs4all.pipeline.dagml.torch_estimator import DagMLTorchEstimator

SOURCE_ORDER = ("nir", "image", "series", "metadata")
FACTORY = "nirs4all.operators.models.pytorch.mlp.structural_mlp"


def make_dataset(seed: int = 17, *, prediction: bool = False, train_only: bool = False, target_name: str = "y") -> Any:
    """Declare four complete aligned numeric matrices for software validation."""
    rng = np.random.default_rng(seed)
    count = 9 if prediction else 42
    ids = [f"{'new' if prediction else 'sample'}_{i}" for i in range(count)]
    sources = {}
    for index, name in enumerate(SOURCE_ORDER):
        width = (4, 3, 2, 2)[index]
        values = rng.normal(size=(count, width)).astype(np.float64)
        sources[name] = TensorSource(values, ids, representation_id="tabular_numeric", axes=("sample", "feature"),
                                     feature_names=[f"{name}:{i}" for i in range(width)])
    targets = 1.2 * sources["nir"].values[:, 0] - 0.6 * sources["image"].values[:, 1] + 0.2 * sources["series"].values[:, 0]
    return MultimodalDataset(sources, sample_ids=ids, y=None if prediction else targets,
        target_names=(target_name,), task_type="regression", groups=[f"group_{i // 3}" for i in range(count)],
        partitions=["train" if train_only or i < 36 else "test" for i in range(count)], name="torch-structural-fixture")


def make_model(names: tuple[str, ...]) -> MultimodalRegressor:
    """Declare one genuine fresh-module factory and its complete fit controls."""
    return MultimodalRegressor(transformers=dict.fromkeys(names), model=DagMLTorchEstimator(
        factory_path=FACTORY, factory_params={"hidden_units": 8}, force_layout="2d", task_type="regression",
        num_classes=None, device="cpu", epochs=3, batch_size=12, patience=3, optimizer="Adam", loss="MSELoss", lr=0.01))


def make_late(names: tuple[str, ...]) -> list[Any]:
    return [{"branch": {name: [{"model": make_model((name,))}] for name in names}},
            {"merge": "predictions"}, {"model": Ridge(alpha=1.0, solver="svd")}]


def make_pipeline(fusion: str = "search") -> list[Any]:
    """Use existing _or_/branch grammar; forced modes declare one real recipe."""
    early = [{"model": make_model(("image", "nir"))}]
    late = make_late(("image", "nir"))
    if fusion not in {"search", "early", "late"}:
        raise ValueError("fusion must be search, early or late")
    choices = [early] if fusion == "early" else [late] if fusion == "late" else [
        early, late, make_late(("nir", "image")), make_late(("image", "series", "metadata")), make_late(SOURCE_ORDER)]
    return [GroupKFold(3), {"_or_": choices}]


def make_tuning(directory: Path, *, fusion: str = "search", resume: bool = False) -> dict[str, Any]:
    axes = ["early.lr"] if fusion == "early" else ["late.image.lr", "late.nir.lr", "late.meta.alpha"] if fusion == "late" else [
        "early.lr", *(f"late.{name}.lr" for name in SOURCE_ORDER), "late.meta.alpha"]
    return {"engine": "n4m", "sampler": "random", "seed": 17, "n_trials": 5, "n_jobs": 1,
        "metric": "rmse", "direction": "minimize", "pruner": None, "resume": resume,
        "storage": directory.resolve().as_uri(), "study_name": "torch-structural",
        "space": {name: {"type": "float", "low": 0.001 if name.endswith("lr") else 0.01,
                         "high": 0.03 if name.endswith("lr") else 10.0, "log": True} for name in axes}}


def main(output_path: str | Path | None = None, *, fusion: str = "search") -> Path | None:
    temporary = tempfile.TemporaryDirectory(prefix="n4a-torch-structural-") if output_path is None else None
    output = Path(temporary.name) if temporary is not None else Path(str(output_path))
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "training-workspace"
    new = make_dataset(29, prediction=True)
    try:
        with nirs4all.run(make_pipeline(fusion), make_dataset(), tuning=make_tuning(output / "study", fusion=fusion),
            engine="dag-ml", workspace_path=workspace, refit=True, random_state=17,
            cpu_threads=1, gpu_devices=[], save_charts=False, verbose=0) as result:
            archive = result.export(output / "torch-winner.n4a")
            expected = nirs4all.predict(archive, new, engine="dag-ml").y_pred
            assert result.structural_tuning_training_request["options"]["artifacts"]["fitted_artifacts"] == "allow_host_sidecar"
            print(f"Grouped native CV RMSE: {result.tuning_best_value:.6f}; active params: {result.tuning_best_params}")
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.rmtree(output / "study")
        actual = nirs4all.predict(archive, new, engine="dag-ml")
        np.testing.assert_array_equal(actual.y_pred, expected)
        assert actual.metadata["training_performed"] is False
        print("Complete Torch/Ridge Python-sidecar closure replayed without FIT/HPO or training workspace.")
        return archive if temporary is None else None
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Retain the fitted Python host-sidecar archive here")
    parser.add_argument("--fusion", choices=("search", "early", "late"), default="search")
    arguments = parser.parse_args()
    main(arguments.output, fusion=arguments.fusion)
