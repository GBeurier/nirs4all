"""Open source-aware replay contract for independent DAG outputs."""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.api.result import _DagmlExportedModel, _DagmlNativeIndependentSourceModels


class _FirstFeatureModel:
    n_features_in_ = 1

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features)[:, 0]


@pytest.mark.parity
@pytest.mark.xfail(strict=True, reason="portable by_source replay has no native sample-ID join for independently ordered named source blocks")
def test_independent_outputs_join_named_sources_by_sample_id() -> None:
    """The host adapter must delegate source alignment to DAG before prediction."""
    model = _DagmlNativeIndependentSourceModels([
        (0, "source_0", "output:source_0", _DagmlExportedModel(_FirstFeatureModel(), None)),
        (1, "source_1", "output:source_1", _DagmlExportedModel(_FirstFeatureModel(), None)),
    ])
    input_sources = {
        "sample_ids": ["s1", "s2"],
        "sources": {
            "source_0": {"sample_ids": ["s2", "s1"], "values": [[20.0], [10.0]]},
            "source_1": {"sample_ids": ["s1", "s2"], "values": [[100.0], [200.0]]},
        },
    }

    outputs = model.predict_outputs(input_sources)
    np.testing.assert_array_equal(outputs["output:source_0"], [10.0, 20.0])
    np.testing.assert_array_equal(outputs["output:source_1"], [100.0, 200.0])
