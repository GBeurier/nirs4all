"""Augmentation keeps operator choices and campaign seeds native-valid."""

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
from nirs4all.operators.augmentation import GaussianAdditiveNoise
from nirs4all.pipeline.dagml import run_paths


@pytest.mark.parametrize("seed", [None, 0, 17])
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_augmentation_operator_campaign_uses_one_seed(monkeypatch, seed, mechanism):
    if mechanism == "subprocess":
        cli = Path(os.environ.get("N4A_DAGML_CLI", Path(__file__).resolve().parents[5] / "dag-ml/target/debug/dag-ml-cli"))
        if not cli.is_file():
            pytest.skip("local dag-ml-cli build unavailable")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    rng = np.random.default_rng(19)
    features = rng.normal(size=(24, 6))
    dataset = SpectroDataset("augmentation-operator-seed")
    dataset.add_samples(features, {"partition": "train"})
    dataset.add_targets(features[:, 0] - features[:, 1])
    original = run_paths.run_cv_refit_bundle
    seen = []

    def execute(**kwargs):
        dsl = kwargs["dsl"]
        assert dsl["root_seed"] == (0 if seed is None else seed)
        before = copy.deepcopy(dsl)
        # Actual native execution validates the choices-derived signed identity.
        outcome = original(**kwargs)
        assert dsl == before
        seen.append(outcome)
        return outcome

    monkeypatch.setattr(run_paths, "run_cv_refit_bundle", execute)
    with nirs4all.run(
        [{"sample_augmentation": {
            "transformers": [GaussianAdditiveNoise(sigma=0.01)],
            "count": 1, "selection": "all", "random_state": 42,
        }}, {"_or_": [StandardScaler(), MinMaxScaler()]}, KFold(3), {"model": Ridge()}],
        dataset, engine="dag-ml", random_state=seed, refit=True,
        save_artifacts=False, save_charts=False, verbose=0,
    ) as result:
        assert np.isfinite(result.cv_best_score)
        reports = result._dagml_score_set["reports"]
        validation_variants = {report["variant_id"] for report in reports
                               if report["partition"] == "validation" and report.get("variant_id")}
        assert len(validation_variants) == 2
        assert {report["variant_id"] for report in reports if report.get("variant_id")} == validation_variants
    assert len(seen) == 1
    assert seen[0]["returncode"] == 0
