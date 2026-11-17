"""The residual target of each ResidualModel execution has a unique, deterministic name."""

import numpy as np
import pytest
from sklearn.preprocessing import FunctionTransformer

from nirs4all.controllers.models.residual_model import ResidualModelController
from nirs4all.data.dataset import SpectroDataset


def _dataset() -> SpectroDataset:
    dataset = SpectroDataset("residual")
    dataset.add_samples(np.arange(24, dtype=float).reshape(8, 3), {"partition": "train"})
    dataset.add_targets(np.arange(8, dtype=float))
    return dataset


def test_repeated_executions_on_one_dataset_never_reuse_a_residual_target_name():
    dataset = _dataset()
    residual = np.zeros(8)
    names = []
    for _ in range(4):
        # Every execution resolves a new operator object; CPython may hand it a recycled address.
        name = ResidualModelController._residual_target_name(dataset, "PLSRegression")
        dataset.add_processed_targets(name, residual, ancestor_processing="numeric", transformer=FunctionTransformer(validate=False))
        names.append(name)
    assert names == ["residual__PLSRegression", "residual__PLSRegression_2", "residual__PLSRegression_3", "residual__PLSRegression_4"]
    assert ResidualModelController._residual_target_name(dataset, "Ridge") == "residual__Ridge"
    with pytest.raises(ValueError, match="already exists"):
        dataset.add_processed_targets(names[0], residual, ancestor_processing="numeric", transformer=FunctionTransformer(validate=False))


def test_residual_target_name_is_the_same_in_every_process():
    assert ResidualModelController._residual_target_name(_dataset(), "PLSRegression") == "residual__PLSRegression"
