import os
from pathlib import Path

import numpy as np
import pytest

from nicon_v2 import datasets as ds

SMOKE_DATASETS = ds.SMOKE_DATASETS


@pytest.mark.skipif(
    not ds.COHORT_REGRESSION_CSV.is_file(),
    reason="cohort CSV missing in this checkout",
)
def test_load_cohort_manifest_smoke_returns_three_specs():
    specs = ds.load_cohort_manifest("smoke")
    names = {s.dataset for s in specs}
    assert names == set(SMOKE_DATASETS)
    for s in specs:
        assert s.n_train > 0 and s.n_test > 0 and s.n_features > 0
        assert s.train_path.is_file()
        assert s.ytrain_path.is_file()


@pytest.mark.skipif(
    not ds.COHORT_REGRESSION_CSV.is_file(),
    reason="cohort CSV missing in this checkout",
)
def test_load_dataset_shapes():
    specs = ds.load_cohort_manifest("smoke")
    spec = next(s for s in specs if s.dataset == "ALPINE_P_291_KS")
    X_train, y_train, X_test, y_test = ds.load_dataset(spec)
    assert X_train.shape == (spec.n_train, spec.n_features)
    assert X_test.shape == (spec.n_test, spec.n_features)
    assert y_train.shape == (spec.n_train,)
    assert y_test.shape == (spec.n_test,)
    assert not np.isnan(X_train).any()
    assert not np.isnan(y_train).any()


def test_safe_float_handles_none_and_nan():
    assert ds._safe_float(None) is None
    assert ds._safe_float(float("nan")) is None
    assert ds._safe_float("not a number") is None
    assert ds._safe_float(1.5) == pytest.approx(1.5)
