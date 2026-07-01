"""Gate: ``run(engine="dag-ml", allow_fallback=False)`` raises a structured ``RtError`` (B-018 / L10).

The strict, opt-in "no silent fallback" boundary. For each shape on the ``EXPECTED_FALLBACK`` allowlist
(the shapes the dag-ml path legitimately rejects today), the DEFAULT ``allow_fallback=True`` transparently
degrades to legacy AND attaches an ``RtError`` diagnostic; ``allow_fallback=False`` instead RAISES that
``RtError`` (``cause="unsupported_shape"``). This reuses the EXACT same allowlist + case registry as the
dual-engine conformance pack, so the two can never drift.
"""

from __future__ import annotations

import warnings

import pytest

import nirs4all
from nirs4all.pipeline.dagml.errors import DagMlUnavailable
from nirs4all.pipeline.dagml.rt import RtError

from . import _conformance_helpers as H
from ._registry import PipelineCase, all_cases
from .test_conformance_dual_engine import EXPECTED_FALLBACK


def _expected_fallback_cases() -> list:
    """The runnable cases on the EXPECTED_FALLBACK allowlist (skip-registry cases excluded)."""
    return [pytest.param(c, id=c.name) for c in all_cases() if c.name in EXPECTED_FALLBACK and not c.skip_reason]


@pytest.mark.parametrize("case", _expected_fallback_cases())
def test_allow_fallback_false_raises_rterror(case: PipelineCase) -> None:
    """A fallback shape under ``allow_fallback=False`` RAISES ``RtError(cause="unsupported_shape")`` — no degrade."""
    dataset = H.make_dataset(case)
    with pytest.raises(RtError) as excinfo:
        nirs4all.run(pipeline=case.pipeline, dataset=dataset, engine="dag-ml", allow_fallback=False, verbose=0)
    assert excinfo.value.cause == "unsupported_shape"
    assert excinfo.value.verb == "run"
    assert excinfo.value.mitigation  # carries a remedy


def test_allow_fallback_true_degrades_and_attaches_diagnostic() -> None:
    """The DEFAULT keeps the transparent fallback but now attaches the RtError diagnostic on the result."""
    case = next(c for c in all_cases() if c.name in EXPECTED_FALLBACK and not c.skip_reason)
    dataset = H.make_dataset(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the "falling back to the legacy engine" warning still fires
        result = nirs4all.run(pipeline=case.pipeline, dataset=dataset, engine="dag-ml", allow_fallback=True, verbose=0)

    # The fallback ran legacy, but the result now carries the structured "ran legacy because <cause>" envelope.
    rt = result.to_rt_result()
    assert rt.manifest["engine"] == "legacy"
    assert [d.cause for d in rt.diagnostics] == ["unsupported_shape"]


def test_allow_fallback_false_raises_unavailable_backend_rterror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing native backend is strict-mode catchable as ``RtError(cause='unavailable_backend')``."""
    import nirs4all.pipeline.dagml.run_backend as run_backend

    def _unavailable(_cli: str) -> None:
        raise DagMlUnavailable("simulated missing dag-ml backend")

    monkeypatch.setattr(run_backend, "preflight_dagml_backend", _unavailable)
    with pytest.raises(RtError) as excinfo:
        nirs4all.run(pipeline=[], dataset=object(), engine="dag-ml", allow_fallback=False, verbose=0)

    assert excinfo.value.cause == "unavailable_backend"
    assert excinfo.value.verb == "run"
    assert "simulated missing dag-ml backend" in excinfo.value.message
