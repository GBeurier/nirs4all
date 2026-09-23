"""Public oracle for the transfer-preprocessing pipeline step."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

import nirs4all
from nirs4all.pipeline.dagml.rt import RtError

from ._datasets import dataset_path


@pytest.mark.parity
def test_auto_transfer_preproc_legacy_contract_and_native_gap() -> None:
    pipeline = [
        {"auto_transfer_preproc": {"preset": "fast", "apply_recommendation": False, "verbose": 0}},
        PLSRegression(3),
    ]
    path = dataset_path("regression")
    legacy = nirs4all.run(pipeline, path, engine="legacy", save_artifacts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    assert legacy.num_predictions > 0
    legacy.close()

    with pytest.raises(RtError, match="auto_transfer_preproc"):
        nirs4all.run(pipeline, path, engine="dag-ml", save_artifacts=False, verbose=0)
