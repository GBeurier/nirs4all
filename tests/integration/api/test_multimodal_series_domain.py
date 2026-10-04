"""Native multimodal archives retain explicit length and channel constraints."""

from pathlib import Path

import numpy as np
import pytest
from dag_ml._dag_ml import DagMlRuntimeError
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.operators.transforms import SequenceSummary
from tests.integration.api.test_multimodal_ragged import _ragged_cohort, _ragged_model, _replace_series, _run, _shorten_series


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
def test_series_domain_survives_native_refit_export_and_replay(fusion: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *a, **k: pytest.fail("legacy scheduler executed"))
    model = _ragged_model(fusion=fusion)
    model.set_params(transformers__series=Pipeline([
        ("summary", SequenceSummary(min_observations=2, channel_bounds=((-1e6, 1e6), (-1e6, 1e6)))),
        ("scale", StandardScaler()),
    ]))
    result = _run(_ragged_cohort(), tmp_path / "workspace", model=model)
    try:
        prediction_cohort = _ragged_cohort(prediction=True)
        archive = result.export(tmp_path / "declared-series-domain.n4a")
        expected = nirs4all.predict(archive, prediction_cohort).values
    finally:
        result.close()
    monkeypatch.setattr(SequenceSummary, "fit", lambda *a, **k: pytest.fail("archive prediction fitted the encoder"))
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("archive prediction fitted a model"))
    np.testing.assert_array_equal(nirs4all.predict(archive, prediction_cohort).values, expected)
    # A shorter variable-length row remains schema-compatible, but violates the
    # explicitly fitted minimum. Eight removals leave just one observation.
    short = prediction_cohort
    for _ in range(8):
        short = _shorten_series(short, 0)
    with pytest.raises((ValueError, DagMlRuntimeError), match="at least 2 observations"):
        nirs4all.predict(archive, short)
    values = prediction_cohort.sources["series"].values.values.copy()
    values[0, 1] = 1e6 + 1
    outside = _replace_series(prediction_cohort, values=values)
    with pytest.raises((ValueError, DagMlRuntimeError), match="outside the declared channel_bounds"):
        nirs4all.predict(archive, outside)
