"""Strict native operator selection uses the compiled campaign seed."""

import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.data import SpectroDataset


@pytest.mark.parametrize("seed", [None, 0, 17])
def test_operator_selection_binds_campaign_and_runtime_seed(monkeypatch, seed):
    import dag_ml._dag_ml as native

    rng = np.random.default_rng(19)
    features = rng.normal(size=(24, 6))
    dataset = SpectroDataset("operator-seed")
    dataset.add_samples(features, {"partition": "train"})
    dataset.add_targets(features[:, 0] - features[:, 1])
    original = native.run_cv_refit_in_process
    seen = []

    def execute(*args):
        dsl = json.loads(args[0])
        expected = 0 if seed is None else seed
        assert dsl["root_seed"] == expected
        assert args[9] == expected
        # Real native execution applies strict choices-derived identity checks.
        payload = original(*args)
        seen.append(json.loads(payload))
        return payload

    monkeypatch.setattr(native, "run_cv_refit_in_process", execute)
    with nirs4all.run(
        [{"_or_": [StandardScaler(), MinMaxScaler()]}, KFold(3), {"model": Ridge()}],
        dataset, engine="dag-ml", random_state=seed, refit=True,
        save_artifacts=False, save_charts=False, verbose=0,
    ) as result:
        assert np.isfinite(result.cv_best_score)
        rows = result.predictions.filter_predictions(partition="val", fold_id="0")
        assert len(rows) == 2
        assert len({row["config_name"] for row in rows}) == 2
    assert len(seen) == 1
    variants = seen[0]["variant_catalog"]
    assert variants and all(isinstance(variant["seed"], int) for variant in variants)
