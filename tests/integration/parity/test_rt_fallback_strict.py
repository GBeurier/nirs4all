"""Gate: dag-ml fallback is explicit and structured (B-018 / L10).

The V1 "no silent fallback" boundary is now the default. A dag-ml coverage/backend refusal raises a
structured ``RtError`` unless the caller explicitly passes ``allow_fallback=True``; opt-in fallback still
attaches the same diagnostic to the returned legacy result.
"""

from __future__ import annotations

import warnings

import pytest
from sklearn.cross_decomposition import PLSRegression

import nirs4all
from nirs4all.pipeline.dagml.errors import DagMlUnavailable
from nirs4all.pipeline.dagml.rt import RtError

from ._dagml_cli import dagml_cli_path
from ._datasets import dataset_path

_DAGML_CLI = dagml_cli_path()


def _unsupported_pipeline() -> list[object]:
    return [{"model": PLSRegression(n_components=2)}]


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_default_dagml_refusal_raises_rterror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dag-ml unsupported shape RAISES ``RtError(cause='unsupported_shape')`` by default — no degrade."""
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    with pytest.raises(RtError) as excinfo:
        nirs4all.run(pipeline=_unsupported_pipeline(), dataset=dataset_path("regression"), verbose=0)
    assert excinfo.value.cause == "unsupported_shape"
    assert excinfo.value.verb == "run"
    assert excinfo.value.mitigation  # carries a remedy


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_allow_fallback_false_raises_rterror() -> None:
    """The explicit strict spelling remains equivalent to the V1 default."""
    with pytest.raises(RtError) as excinfo:
        nirs4all.run(pipeline=_unsupported_pipeline(), dataset=dataset_path("regression"), engine="dag-ml", allow_fallback=False, verbose=0)
    assert excinfo.value.cause == "unsupported_shape"
    assert excinfo.value.verb == "run"


@pytest.mark.skipif(not _DAGML_CLI.exists(), reason=f"dag-ml-cli binary not built at {_DAGML_CLI}")
def test_allow_fallback_true_degrades_and_attaches_diagnostic() -> None:
    """Explicit ``allow_fallback=True`` degrades to legacy and attaches the RtError diagnostic."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the "falling back to the legacy engine" warning still fires
        result = nirs4all.run(pipeline=_unsupported_pipeline(), dataset=dataset_path("regression"), engine="dag-ml", allow_fallback=True, verbose=0)

    # The fallback ran legacy, but the result now carries the structured "ran legacy because <cause>" envelope.
    rt = result.to_rt_result()
    assert rt.manifest["engine"] == "legacy"
    assert [d.cause for d in rt.diagnostics] == ["unsupported_shape"]


def test_allow_fallback_false_raises_unavailable_backend_rterror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing native backend is catchable as ``RtError(cause='unavailable_backend')`` by default."""
    import nirs4all.pipeline.dagml.run_backend as run_backend

    def _unavailable(_cli: str) -> None:
        raise DagMlUnavailable("simulated missing dag-ml backend")

    monkeypatch.setattr(run_backend, "preflight_dagml_backend", _unavailable)
    with pytest.raises(RtError) as excinfo:
        nirs4all.run(pipeline=[], dataset=object(), engine="dag-ml", verbose=0)

    assert excinfo.value.cause == "unavailable_backend"
    assert excinfo.value.verb == "run"
    assert "simulated missing dag-ml backend" in excinfo.value.message
