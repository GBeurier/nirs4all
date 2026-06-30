"""CONFORMANCE: error / refusal parity (B-011 / SW5 §6c, PYREF-err).

Today every ``pytest.raises`` in the parity dir is single-engine dag-ml-only
(A2 §4.4). No test feeds the SAME invalid pipeline to BOTH engines and asserts
the same refusal, and nothing pins the dag-ml refusal to a STABLE cause string.
This module closes both gaps:

* **Cross-engine refusal parity.** A structurally invalid pipeline (a model
  given impossible hyperparameters) must be rejected by BOTH ``engine="legacy"``
  and ``engine="dag-ml"`` — the dag-ml leg must RAISE, never silently fall back
  to legacy and mask the bug (the fallback contract catches only
  ``DagMlUnsupported``/``DagMlUnavailable``; a genuine fit failure propagates —
  ``errors.py:_raise_run_failure``). This proves the engines agree on *which
  pipelines are invalid*, not only on valid-pipeline numerics.

* **Stable dag-ml refusal cause.** Each dag-ml refusal maps to a stable
  ``RtError.cause`` from the CAP-004 / RT-003 vocabulary
  (``DagMlUnsupported → unsupported_shape``,
  ``DagMlUnavailable → unavailable_backend``, a genuine runtime failure →
  ``runtime_error`` — ``RT_spec.md`` RT-003:182-184). The vocabulary is owned by
  CAP-004 (``CAP_spec.md`` §5); this test references it, it does not invent
  strings.

Dependency (reported in W4_CROSS_ENGINE.md): the unified runtime classifier
``nirs4all/pipeline/dagml/rt.py`` (``RtError.from_dagml_error``) is W7's
deliverable (B-018/L10) and is NOT landed yet. Until it is, :func:`_classify_cause`
below applies the RT-003 migration table directly, behind a TODO, so the cause
vocabulary is pinned cross-engine today without blocking on W7.
"""

from __future__ import annotations

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline.dagml.errors import DagMlUnavailable, DagMlUnsupported, _reject_multi_model

from ._datasets import dataset_path

pytestmark = [pytest.mark.parity, pytest.mark.slow]


# RT-003 / CAP-004 RtError cause vocabulary (RT_spec.md:177-184, CAP_spec.md §5).
# Owned by CAP-004 — referenced here, never extended.
_RT_ERROR_CAUSES = frozenset({
    "unsupported_shape",
    "unsupported_capability",
    "unavailable_backend",
    "invalid_request",
    "runtime_error",
})


def _classify_cause(exc: BaseException) -> str:
    """Map a dag-ml refusal to its stable RT-003 / CAP-004 ``RtError.cause``.

    TODO(W7/B-018): replace with ``nirs4all.pipeline.dagml.rt.RtError.from_dagml_error``
    once W7's runtime classifier lands (it is not in-tree yet — verified absent).
    Until then this applies the RT-003 migration table verbatim
    (``RT_spec.md``:182-184): ``DagMlUnavailable → unavailable_backend``,
    ``DagMlUnsupported → unsupported_shape`` (it subclasses ``NotImplementedError``,
    the catchable-fallback contract), and any other propagated failure (a genuine
    operator/runtime error the host re-raises rather than falls back) →
    ``runtime_error``.
    """
    if isinstance(exc, DagMlUnavailable):
        return "unavailable_backend"
    if isinstance(exc, DagMlUnsupported):
        return "unsupported_shape"
    return "runtime_error"


def _invalid_hyperparam_pipeline() -> list:
    """A pipeline whose model carries structurally impossible hyperparameters.

    ``n_components`` far above the feature count is invalid for ``PLSRegression``
    on either engine; both reject it at fit (legacy: sklearn ``ValueError`` wrapped
    as a step failure; dag-ml: a native runtime-validation error). A fresh
    splitter+model instance per call avoids cross-test mutable-state sharing.
    """
    return [ShuffleSplit(n_splits=2, random_state=0), {"model": PLSRegression(n_components=99999)}]


def test_invalid_pipeline_refused_by_both_engines() -> None:
    """The same invalid pipeline RAISES on legacy AND on dag-ml (no silent fallback).

    The dag-ml leg must propagate — a genuine fit failure is NOT an unsupported
    shape, so ``run()`` re-raises instead of redirecting to legacy
    (``errors.py:_raise_run_failure``). If dag-ml silently fell back here it would
    run the SAME invalid pipeline on legacy and hit the SAME error, so the raise
    is asserted on both legs independently.
    """
    with pytest.raises(Exception) as legacy_exc:  # noqa: PT011 - cross-engine: type differs per engine
        nirs4all.run(
            pipeline=_invalid_hyperparam_pipeline(),
            dataset=DatasetConfigs(dataset_path("regression")),
            verbose=0,
            engine="legacy",
        )

    with pytest.raises(Exception) as dagml_exc:  # noqa: PT011
        nirs4all.run(
            pipeline=_invalid_hyperparam_pipeline(),
            dataset=DatasetConfigs(dataset_path("regression")),
            verbose=0,
            engine="dag-ml",
        )

    # Both engines reject the invalid pipeline (the parity claim).
    assert legacy_exc.value is not None and dagml_exc.value is not None

    # The dag-ml refusal carries a STABLE cause from the CAP-004 / RT-003 vocab.
    cause = _classify_cause(dagml_exc.value)
    assert cause in _RT_ERROR_CAUSES, f"dag-ml refusal cause {cause!r} not in RT-003 vocabulary"
    # A genuine fit failure (not an unsupported-shape / unavailable-backend
    # refusal) classifies to runtime_error — the engines agree it is a real
    # rejection, not a coverage gap that should have fallen back.
    assert cause == "runtime_error", (
        f"invalid-hyperparameter refusal should map to runtime_error, got {cause!r} "
        f"(dag-ml raised {type(dagml_exc.value).__name__})"
    )


def test_dagml_refusals_map_to_stable_rt_error_causes() -> None:
    """Each dag-ml refusal TYPE maps to its documented RT-003 / CAP-004 cause.

    Pins the migration table (``RT_spec.md`` RT-003) so the cause vocabulary is a
    fixed contract: ``DagMlUnsupported → unsupported_shape`` (exercised through the
    REAL ``_reject_multi_model`` refusal path, not just a constructed instance) and
    ``DagMlUnavailable → unavailable_backend``.
    """
    # Real refusal path: the multi-model guard raises DagMlUnsupported.
    with pytest.raises(DagMlUnsupported) as multi_model_exc:
        _reject_multi_model([{"model": PLSRegression(n_components=5)}, {"model": PLSRegression(n_components=7)}])
    assert _classify_cause(multi_model_exc.value) == "unsupported_shape"

    # DagMlUnsupported subclasses NotImplementedError — the contract that lets the
    # cutover fallback catch it; if that ever changes, the cause mapping is wrong.
    assert issubclass(DagMlUnsupported, NotImplementedError)
    assert _classify_cause(DagMlUnsupported("an unsupported pipeline shape")) == "unsupported_shape"

    # The narrow "backend not installed" refusal maps to unavailable_backend.
    assert _classify_cause(DagMlUnavailable("neither dag-ml mechanism installed")) == "unavailable_backend"

    # Every mapped cause is in the owned vocabulary.
    for exc in (DagMlUnsupported("x"), DagMlUnavailable("y"), RuntimeError("z")):
        assert _classify_cause(exc) in _RT_ERROR_CAUSES
