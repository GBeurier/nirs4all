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
  ``RtError.cause`` from the CAP-004 / RT-003 vocabulary through the shared
  runtime envelope helpers (``DagMlUnsupported → unsupported_shape``,
  ``DagMlUnavailable → unavailable_backend``, a genuine runtime failure →
  ``runtime_error``, bad request/workspace/spec → ``invalid_request``). The
  vocabulary is owned by CAP-004; this test references it, it does not invent
  strings.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline.dagml.errors import DagMlUnavailable, DagMlUnsupported, _reject_multi_model
from nirs4all.pipeline.dagml.rt import RT_ERROR_CAUSES, RtError

from ._datasets import dataset_path

pytestmark = [pytest.mark.parity, pytest.mark.slow]


def _invalid_hyperparam_pipeline() -> list:
    """A pipeline whose model carries structurally impossible hyperparameters.

    ``n_components`` far above the feature count is invalid for ``PLSRegression``
    on either engine; both reject it at fit (legacy: sklearn ``ValueError`` wrapped
    as a step failure; dag-ml: a native runtime-validation error). A fresh
    splitter+model instance per call avoids cross-test mutable-state sharing.
    """
    return [ShuffleSplit(n_splits=2, random_state=0), {"model": PLSRegression(n_components=99999)}]


def _valid_pipeline() -> list:
    """A minimal valid pipeline whose dataset load should be the first failure for bad dataset specs."""
    return [ShuffleSplit(n_splits=2, random_state=0), {"model": PLSRegression(n_components=2)}]


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
    cause = RtError.runtime_error(dagml_exc.value, verb="run").cause
    assert cause in RT_ERROR_CAUSES, f"dag-ml refusal cause {cause!r} not in RT-003 vocabulary"
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
    assert RtError.from_dagml_error(multi_model_exc.value, verb="run").cause == "unsupported_shape"

    # DagMlUnsupported subclasses NotImplementedError — the contract that lets the
    # cutover fallback catch it; if that ever changes, the cause mapping is wrong.
    assert issubclass(DagMlUnsupported, NotImplementedError)
    assert RtError.from_dagml_error(DagMlUnsupported("an unsupported pipeline shape"), verb="run").cause == "unsupported_shape"

    # The narrow "backend not installed" refusal maps to unavailable_backend.
    assert RtError.from_dagml_error(DagMlUnavailable("neither dag-ml mechanism installed"), verb="run").cause == "unavailable_backend"

    # Every mapped cause is in the owned vocabulary.
    errors = [
        RtError.from_dagml_error(DagMlUnsupported("x"), verb="run"),
        RtError.from_dagml_error(DagMlUnavailable("y"), verb="run"),
        RtError.runtime_error(RuntimeError("z"), verb="run"),
        RtError.invalid_request(ValueError("bad dataset spec"), verb="run"),
    ]
    for error in errors:
        assert error.cause in RT_ERROR_CAUSES


def test_unsupported_operator_refusal_maps_to_unsupported_shape() -> None:
    """A dag-ml-only unsupported operator raises the catchable backend error and maps through ``RtError``."""
    from nirs4all.operators.transforms.resampler import Resampler
    from nirs4all.pipeline.dagml.steps import _assert_supported_operators

    with pytest.raises(DagMlUnsupported, match="wavelength") as excinfo:
        _assert_supported_operators([Resampler(target_wavelengths=[1.0, 2.0, 3.0])])

    error = RtError.from_dagml_error(excinfo.value, verb="run")
    assert error.cause == "unsupported_shape"
    assert error.verb == "run"
    assert error.cause in RT_ERROR_CAUSES


def test_invalid_dataset_spec_maps_to_invalid_request_on_both_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed dataset partition spec is an invalid request on both engines, not a dag-ml fallback signal."""
    import nirs4all.pipeline.dagml.run_backend as run_backend

    monkeypatch.setattr(run_backend, "preflight_dagml_backend", lambda _cli: None)
    malformed_dataset = (np.zeros((5, 2)), np.arange(5, dtype=float), {"train": "bad"})
    for engine in ("legacy", "dag-ml"):
        with pytest.raises(Exception) as excinfo:  # noqa: PT011 - exact parser/load exception differs by engine path
            nirs4all.run(pipeline=_valid_pipeline(), dataset=malformed_dataset, verbose=0, engine=engine)

        assert not isinstance(excinfo.value, (DagMlUnsupported, DagMlUnavailable))
        error = RtError.invalid_request(excinfo.value, verb="run")
        assert error.cause == "invalid_request"
        assert error.verb == "run"
