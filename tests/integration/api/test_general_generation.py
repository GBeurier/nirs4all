"""The general synthesis API remains usable without a legacy ML engine."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_execution_selectors(monkeypatch):
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.delenv("N4A_GENERATE_PLUGIN", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.__init__", lambda *args, **kwargs: pytest.fail("legacy runner created"))


@pytest.mark.parametrize("method", ["__call__", "regression", "classification"])
def test_general_generation_matches_existing_scientific_builder(method):
    import nirs4all

    function = getattr(nirs4all.generate, method)
    X, y = function(n_samples=18, as_dataset=False, random_state=17)
    expected_X, expected_y = function(n_samples=18, as_dataset=False, random_state=17, engine="legacy")
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)
    assert np.isfinite(X).all()
    decision = nirs4all.generate.preflight()
    assert decision.executable and decision.lane == "plugin"
    assert decision.contract == "nirs4all.python.synthesis.v1"


def test_named_builtin_host_and_fluent_builder_are_real():
    import nirs4all

    dataset = nirs4all.generate(n_samples=12, random_state=5, plugin="nirs4all.python.synthesis.v1")
    assert dataset.num_samples == 12
    fluent = nirs4all.generate.builder(n_samples=12, random_state=5).with_features().with_targets().build()
    assert fluent.num_samples == 12


def test_unknown_host_and_runtime_errors_do_not_retry(monkeypatch):
    import nirs4all
    from nirs4all.pipeline.dagml.rt import RtError

    with pytest.raises(RtError, match="no callable"):
        nirs4all.generate(plugin="uninstalled-synthesis")
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("scientific synthesis failed")

    monkeypatch.setattr("nirs4all.synthesis.SyntheticDatasetBuilder.build", fail)
    with pytest.raises(RuntimeError, match="scientific synthesis failed"):
        nirs4all.generate(n_samples=12)
    assert calls == [True]
