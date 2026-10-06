"""Branch campaigns bind seeds before native compilation and selection."""

import copy
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.data import SpectroDataset


@pytest.mark.parametrize("seed", [None, 17])
@pytest.mark.parametrize("shape", ["metadata", "by_source", "stacking"])
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_branch_operator_compilation_and_selection_share_seed(monkeypatch, seed, shape, mechanism):
    import dag_ml

    if mechanism == "subprocess":
        cli = Path(os.environ.get("N4A_DAGML_CLI", Path(__file__).resolve().parents[5] / "dag-ml/target/debug/dag-ml-cli"))
        if not cli.is_file():
            pytest.skip("local dag-ml-cli build unavailable")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    rng = np.random.default_rng(19)
    features = rng.normal(size=(24, 6))
    dataset = SpectroDataset("branch-operator-seed")
    dataset.add_samples([features[:, :3], features[:, 3:]] if shape == "by_source" else features, {"partition": "train"})
    dataset.add_targets(features[:, 0] - features[:, -1])
    generator = {"_or_": [StandardScaler(), MinMaxScaler()]}
    if shape == "metadata":
        dataset.add_metadata(np.array(["A", "B"] * 12).reshape(-1, 1), headers=["site"])
        pipeline = [KFold(3), {"branch": {"by_metadata": "site", "steps": [generator, Ridge()]}}]
    elif shape == "by_source":
        pipeline = [KFold(3), {"branch": {"by_source": True, "steps": {
            "source_0": [generator, Ridge()], "source_1": [generator, Ridge()],
        }}}, {"merge": {"sources": "concat"}}]
    else:
        pipeline = [KFold(3), {"branch": [
            [StandardScaler(), {"model": Ridge(alpha=0.1)}],
            [MinMaxScaler(), {"model": Ridge(alpha=1.0)}],
        ]}, {"merge": "predictions"}, {"model": Ridge()}]
    original_compile = dag_ml.compile_pipeline_dsl_artifact_with_controllers
    compiled = []

    def compile_campaign(dsl, manifests):
        assert dsl["root_seed"] == (0 if seed is None else seed)
        before = copy.deepcopy(dsl)
        artifact = original_compile(dsl, manifests)
        assert dsl == before
        compiled.append(dsl["id"])
        return artifact

    monkeypatch.setattr(dag_ml, "compile_pipeline_dsl_artifact_with_controllers", compile_campaign)
    with nirs4all.run(
        pipeline, dataset, engine="dag-ml", random_state=seed, refit=True,
        save_artifacts=False, save_charts=False, verbose=0,
    ) as result:
        assert np.isfinite(result.cv_best_score)
        validation = result.predictions.filter_predictions(partition="val", fold_id="avg")
        assert validation
        if shape == "metadata":
            variants = {row["result_metadata"]["dagml_projection"]["variant_id"] for row in validation}
            assert len(variants) == 4
        if shape == "by_source":
            assert {row["branch_name"] for row in validation} == {"source_0", "source_1"}
    assert compiled
