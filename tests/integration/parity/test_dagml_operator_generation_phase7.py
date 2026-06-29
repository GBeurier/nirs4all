"""Phase-7 (#23) native operator-``_or_`` host SAFETY: sanitization, routability, narrow fallback.

These guard the behaviours the Codex gates required for native operator-level generator SELECT:

1. **Strict whole-subsequence label sanitization + fallback-on-failure** — a NaN / Inf / non-str-key /
   non-JSON param ANYWHERE in the label sub-sequence (chosen branch + downstream model params, model
   siblings, and ``y_processing``) DEMOTES the run to the Python-expand path (still dag-ml-native), never
   crashes with a ``DagMlValidationError`` and never silently ``repr``-stringifies a non-JSON value.
2. **Only genuinely routable operators enter native** — a non-routable / wavelength-requiring /
   non-importable ``_or_`` choice keeps the pipeline OFF the native path (the flat-single predicate is
   ``False``), so it runs via Python-expand instead of failing at fit inside native operator-SELECT.
3. **Single-choice ``_or_`` is still validated** — a one-choice ``_or_`` (whose dag-ml ``config_name`` is
   blank, so the config map is empty) STILL fingerprints + strict-validates its choice label, so a bad
   label demotes during lowering instead of leaking past it into the native run.
4. **Narrow inner fallback by TYPE** — ONLY the lowering sentinel ``_OperatorLoweringUnsupported`` demotes
   to Python-expand; a RUNTIME error — a raw exception OR a runtime ``DagMlUnsupported`` from
   ``_raise_run_failure`` (a non-zero run classified ``error_kind == "unsupported"``) — PROPAGATES.

The e2e demotion tests assert the dag-ml leg stays NATIVE-engine (it never bubbles to legacy) while
producing a real ``RunResult`` — i.e. the inner fallback redirected to Python-expand, not a crash.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import Detrend, Resampler
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV

from ._datasets import dataset_path

pytestmark = [pytest.mark.parity]

pytest.importorskip("dag_ml", reason="dag-ml not importable (core dependency; broken install?)")

_FALLBACK_FRAGMENT = "falling back to the legacy engine"


def _dataset() -> DatasetConfigs:
    return DatasetConfigs(dataset_path("regression"))


def _run_dagml(pipeline: list[Any]) -> tuple[Any, bool]:
    """Run the dag-ml leg; return ``(result, dagml_native)`` (native == no legacy-fallback warning)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(pipeline=pipeline, dataset=_dataset(), verbose=0, engine="dag-ml")
        fell_back = any(_FALLBACK_FRAGMENT in str(w.message) for w in caught)
    return result, (not fell_back) and bool(result._is_dagml_engine())  # noqa: SLF001


# --------------------------------------------------------------------------- #
# MUST-FIX 1 — strict sanitizer (pure unit)
# --------------------------------------------------------------------------- #


def test_strict_json_safe_coerces_numpy() -> None:
    """numpy scalars/arrays (incl. nested) are coerced to JSON-native values."""
    from nirs4all.pipeline.dagml_bridge import _strict_json_safe

    assert _strict_json_safe(np.int64(5), "x") == 5
    assert isinstance(_strict_json_safe(np.int64(5), "x"), int)
    assert _strict_json_safe(np.float64(1.5), "x") == 1.5
    assert _strict_json_safe(np.array([1.0, 2.0]), "x") == [1.0, 2.0]
    assert _strict_json_safe({"a": np.float32(1.5), "b": [np.int32(3)]}, "x") == {"a": pytest.approx(1.5), "b": [3]}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_strict_json_safe_rejects_non_finite(bad: float) -> None:
    """A NaN / Inf float is REJECTED as lowering-unsupported (not passed through allow_nan)."""
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml_bridge import _strict_json_safe

    assert not math.isfinite(bad)
    with pytest.raises(DagMlUnsupported):
        _strict_json_safe(bad, "param")
    # also nested inside a container
    with pytest.raises(DagMlUnsupported):
        _strict_json_safe({"a": [1.0, bad]}, "param")


def test_strict_json_safe_rejects_non_json() -> None:
    """A value with no JSON-native form (a callable) is REJECTED (no default=repr divergence)."""
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml_bridge import _strict_json_safe

    with pytest.raises(DagMlUnsupported):
        _strict_json_safe(lambda value: value, "param")
    with pytest.raises(DagMlUnsupported):
        _strict_json_safe(object(), "param")


@pytest.mark.parametrize("bad_key", [object(), (1, 2), 1, 1.5, None, True])
def test_strict_json_safe_rejects_non_string_dict_key(bad_key: object) -> None:
    """A non-string dict KEY is REJECTED (JSON allows only str keys; do not stringify) — MUST-FIX 2."""
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml_bridge import _strict_json_safe

    with pytest.raises(DagMlUnsupported):
        _strict_json_safe({bad_key: 1}, "param")


def test_strict_json_safe_accepts_string_dict_key() -> None:
    """A string dict key passes (the legitimate JSON-object shape)."""
    from nirs4all.pipeline.dagml_bridge import _strict_json_safe

    assert _strict_json_safe({"a": 1, "b": [2, 3]}, "param") == {"a": 1, "b": [2, 3]}


def test_operator_choice_label_demotes_on_non_finite_param() -> None:
    """A choice carrying a NaN param raises DagMlUnsupported from the label helper (not a PyO3 crash)."""
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml_bridge import operator_choice_variant_label

    class NaNParam(BaseEstimator, TransformerMixin):
        def __init__(self, alpha: float = float("nan")) -> None:
            self.alpha = alpha

        def fit(self, X, y=None):  # noqa: N803, ANN001
            return self

        def transform(self, X):  # noqa: N803, ANN001
            return X

    with pytest.raises(DagMlUnsupported):
        operator_choice_variant_label(NaNParam(), [{"model": PLSRegression(n_components=2)}])


def test_operator_choice_label_demotes_on_downstream_non_json_param() -> None:
    """A DOWNSTREAM model whose params carry a non-JSON value DEMOTES (not repr'd before strict) — MUST-FIX 1.

    The downstream model params previously flowed through ``_step_to_dsl`` → ``_json_safe_params``
    (``default=repr``), so an ``object()`` param became a ``"<object …>"`` string instead of being
    rejected. The label helper now lowers downstream steps from their RAW params with strict sanitization.
    """
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml_bridge import operator_choice_variant_label

    class NonJsonModel(PLSRegression):
        def get_params(self, deep: bool = True) -> dict[str, Any]:
            params: dict[str, Any] = dict(super().get_params(deep))
            params["weird"] = object()  # non-JSON downstream param
            return params

    with pytest.raises(DagMlUnsupported):
        operator_choice_variant_label(SNV, [{"model": NonJsonModel()}])


def test_operator_choice_label_demotes_on_downstream_non_json_sibling() -> None:
    """A DOWNSTREAM model SIBLING hyperparameter that is non-JSON DEMOTES (strict over siblings) — MUST-FIX 1."""
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml_bridge import operator_choice_variant_label

    with pytest.raises(DagMlUnsupported):
        operator_choice_variant_label(SNV, [{"model": PLSRegression(n_components=2), "weird_sibling": object()}])


def test_operator_choice_label_demotes_on_downstream_non_string_param_key() -> None:
    """A DOWNSTREAM model whose ``get_params()`` returns a NON-STRING top-level key DEMOTES — MUST-FIX (top-level keys).

    ``_canonical_label_params`` previously strict-checked param VALUES but built the dict with the raw keys,
    so a non-string top-level param key (which breaks JSON validity / byte-identity) slipped through. The
    whole params dict now runs through ``_strict_json_safe``, which rejects non-string keys.
    """
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml_bridge import operator_choice_variant_label

    class NonStringKeyModel(PLSRegression):
        def get_params(self, deep: bool = True) -> dict[Any, Any]:
            params: dict[Any, Any] = dict(super().get_params(deep))
            params[1] = "x"  # non-string TOP-LEVEL param key
            return params

    with pytest.raises(DagMlUnsupported):
        operator_choice_variant_label(SNV, [{"model": NonStringKeyModel()}])


def test_operator_choice_label_demotes_on_downstream_non_string_sibling_key() -> None:
    """A DOWNSTREAM model STEP-LEVEL sibling with a NON-STRING key DEMOTES — MUST-FIX (top-level keys).

    A step-level sibling key (``{"model": op, 1: "x"}``) bypasses the reserved-key filter (``1`` is not a
    reserved keyword) and would be inserted into the params dict as a non-string key. ``_label_step_from_raw``
    now rejects a non-string sibling key explicitly.
    """
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml_bridge import operator_choice_variant_label

    with pytest.raises(DagMlUnsupported):
        operator_choice_variant_label(SNV, [{"model": PLSRegression(n_components=2), 1: "x"}])


def test_operator_choice_label_byte_identity_preserved() -> None:
    """The strict refactor preserves byte-identity: SNV/MSC/Detrend hash to the pinned dag-ml report labels.

    These are the labels the in-process operator-SELECT reports stamp for ``[{"_or_": [SNV, MSC, Detrend]},
    KFold, {"model": PLSRegression(n_components=10)}]`` (verified against the live reports). A drift here
    breaks the content-keyed config map, so they are pinned.
    """
    from nirs4all.pipeline.dagml_bridge import operator_choice_variant_label

    downstream = [{"model": PLSRegression(n_components=10)}]
    assert operator_choice_variant_label(SNV, downstream).startswith("90d63b51")
    assert operator_choice_variant_label(MSC, downstream).startswith("f761cbd1")
    assert operator_choice_variant_label(Detrend, downstream).startswith("87236525")


# --------------------------------------------------------------------------- #
# MUST-FIX 2 — only routable operators enter native (predicate)
# --------------------------------------------------------------------------- #


def test_flat_predicate_admits_routable_bare_or() -> None:
    """A bare ``_or_`` of routable transforms IS flat-single (eligible for native)."""
    from nirs4all.pipeline.dagml.detect import _is_flat_single_operator_generator

    pipeline = [{"_or_": [SNV, MSC]}, KFold(n_splits=3), {"model": PLSRegression(n_components=10)}]
    assert _is_flat_single_operator_generator(pipeline) is True


def test_flat_predicate_rejects_wavelength_requiring_choice() -> None:
    """A wavelength-requiring choice (configured Resampler) keeps the pipeline OFF native."""
    from nirs4all.pipeline.dagml.detect import _is_flat_single_operator_generator

    pipeline = [
        {"_or_": [SNV, Resampler(target_wavelengths=np.array([1.0, 2.0, 3.0]))]},
        KFold(n_splits=3),
        {"model": PLSRegression(n_components=10)},
    ]
    assert _is_flat_single_operator_generator(pipeline) is False


def test_flat_predicate_rejects_scalar_choice() -> None:
    """A non-operator scalar choice keeps the pipeline OFF native (not routable)."""
    from nirs4all.pipeline.dagml.detect import _is_flat_single_operator_generator

    pipeline = [{"_or_": [SNV, 42]}, KFold(n_splits=3), {"model": PLSRegression(n_components=10)}]
    assert _is_flat_single_operator_generator(pipeline) is False


def test_flat_predicate_rejects_non_importable_choice() -> None:
    """A function-local (non-FQN-importable) choice keeps the pipeline OFF native."""
    from nirs4all.pipeline.dagml.detect import _is_flat_single_operator_generator

    def make_local() -> object:
        class LocalTransform(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):  # noqa: N803, ANN001
                return self

            def transform(self, X):  # noqa: N803, ANN001
                return X

        return LocalTransform()

    pipeline = [{"_or_": [SNV, make_local()]}, KFold(n_splits=3), {"model": PLSRegression(n_components=10)}]
    assert _is_flat_single_operator_generator(pipeline) is False


@pytest.mark.slow
def test_routable_or_runs_native_e2e() -> None:
    """E2E sanity: an all-routable bare ``_or_`` runs NATIVE on dag-ml and produces a real result."""
    pipeline = [{"_or_": [SNV(), MSC()]}, KFold(n_splits=3), {"model": PLSRegression(n_components=5)}]
    result, native = _run_dagml(pipeline)
    assert native is True
    assert result.num_predictions >= 1


# --------------------------------------------------------------------------- #
# MUST-FIX 3 — single-choice _or_ still validates its label (lowering check not skipped)
# --------------------------------------------------------------------------- #


def test_single_choice_or_validates_label_even_with_empty_config_names() -> None:
    """A ONE-choice ``_or_`` (empty config names) STILL fingerprints + validates its label — MUST-FIX 3.

    The config map is empty for a single-choice ``_or_`` (dag-ml's ``config_name`` is blank), but the
    label MUST still be computed + strict-validated during lowering, so a bad-label single choice DEMOTES
    here instead of leaking past the lowering guard into the native run.
    """
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported
    from nirs4all.pipeline.dagml.result import _native_operator_config_by_label

    class NonJsonModel(PLSRegression):
        def get_params(self, deep: bool = True) -> dict[str, Any]:
            params: dict[str, Any] = dict(super().get_params(deep))
            params["weird"] = object()
            return params

    steps = [{"_or_": [SNV]}, {"model": NonJsonModel()}]
    # Empty ordered_config_names (single-choice) MUST still validate the choice label → demote on a bad one.
    with pytest.raises(DagMlUnsupported):
        _native_operator_config_by_label(steps, [])


def test_single_choice_or_good_label_returns_empty_map() -> None:
    """A single-choice ``_or_`` with a GOOD label + empty config names returns ``{}`` (validated, no crash)."""
    from nirs4all.pipeline.dagml.result import _native_operator_config_by_label

    steps = [{"_or_": [SNV]}, {"model": PLSRegression(n_components=5)}]
    assert _native_operator_config_by_label(steps, []) == {}


# --------------------------------------------------------------------------- #
# MUST-FIX 4 — narrow inner fallback BY TYPE (lowering sentinel demotes; runtime DagMlUnsupported propagates)
# --------------------------------------------------------------------------- #


def test_lowering_sentinel_is_distinct_dagml_unsupported_subclass() -> None:
    """The lowering sentinel is a DagMlUnsupported SUBCLASS, so a bare runtime DagMlUnsupported is distinct."""
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported, _OperatorLoweringUnsupported

    assert issubclass(_OperatorLoweringUnsupported, DagMlUnsupported)
    # A plain runtime DagMlUnsupported must NOT be an instance of the lowering sentinel (so it propagates).
    assert not isinstance(DagMlUnsupported("runtime"), _OperatorLoweringUnsupported)


@pytest.mark.slow
def test_runtime_error_propagates_not_reclassified(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw RUNTIME error from the run phase PROPAGATES — never silently demoted to Python-expand.

    Simulate a runtime failure AFTER the lowering guard (in ``run_cv_refit_bundle``). The contract is
    that only the lowering sentinel demotes; a real runtime error must surface, not be swallowed.
    """
    import nirs4all.pipeline.dagml.run_paths as run_paths

    sentinel = RuntimeError("simulated runtime failure in the run phase")

    def boom(**_kwargs: object) -> dict:
        raise sentinel

    monkeypatch.setattr(run_paths, "run_cv_refit_bundle", boom)

    pipeline = [{"_or_": [SNV, MSC]}, KFold(n_splits=3), {"model": PLSRegression(n_components=10)}]
    with pytest.raises(RuntimeError, match="simulated runtime failure"):
        nirs4all.run(pipeline=pipeline, dataset=_dataset(), verbose=0, engine="dag-ml")


@pytest.mark.slow
def test_runtime_dagml_unsupported_from_outcome_is_not_swallowed_by_inner_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A RUNTIME ``DagMlUnsupported`` (a non-zero outcome classified unsupported) is NOT masked — MUST-FIX 4.

    ``_raise_run_failure`` (called OUTSIDE the lowering guard) raises the BROAD ``DagMlUnsupported`` for a
    non-zero run whose adapter frame is ``error_kind == "unsupported"``. The routing branch catches ONLY the
    distinct ``_OperatorLoweringUnsupported`` sentinel, so this RUNTIME ``DagMlUnsupported`` must PROPAGATE
    past the inner branch — it must NOT be silently reclassified as a lowering gap and re-run on the dag-ml
    Python-expand path (which would report ``dagml_native=True`` and hide the runtime boundary). It is the
    OUTER ``run()`` fallback that then redirects it to LEGACY (the documented contract for ANY unsupported
    shape), so the run is LEGACY (``dagml_native=False``), NOT a dag-ml Python-expand success.

    The probe distinguishes the two outcomes: the monkeypatched ``run_cv_refit_bundle`` returns the
    nonzero-unsupported outcome ONLY for the native-operator workdir (``native_op``); the Python-expand path
    uses a different workdir, so if the inner branch WRONGLY swallowed the error and re-ran Python-expand,
    that leg would run the real bundle and the run would be (incorrectly) dag-ml-native. Asserting LEGACY
    (fallback warning fired) proves the runtime error propagated past the inner branch.
    """
    import nirs4all.pipeline.dagml.run_paths as run_paths

    real_bundle = run_paths.run_cv_refit_bundle

    def nonzero_unsupported_for_native_op(**kwargs: Any) -> dict[str, Any]:
        workdir = str(kwargs.get("workdir", ""))
        if "native_op" in workdir:
            return {
                "returncode": 1,
                "stdout": "Error: simulated runtime unsupported shape",
                "results": [{"type": "error", "error_kind": "unsupported"}],
                "scores": None,
            }
        return dict(real_bundle(**kwargs))

    monkeypatch.setattr(run_paths, "run_cv_refit_bundle", nonzero_unsupported_for_native_op)

    pipeline = [{"_or_": [SNV, MSC]}, KFold(n_splits=3), {"model": PLSRegression(n_components=10)}]
    result, native = _run_dagml(pipeline)
    # The runtime DagMlUnsupported propagated past the inner branch → the OUTER run() fallback redirected to
    # LEGACY (fallback warning fired), NOT a silent dag-ml Python-expand re-run (which would be native).
    assert native is False
    assert result.num_predictions >= 1


@pytest.mark.slow
def test_lowering_unsupported_demotes_to_python_expand(monkeypatch: pytest.MonkeyPatch) -> None:
    """A LOWERING refusal (raised inside the narrow guard) DEMOTES to Python-expand, not crash — MUST-FIX 4.

    Force the per-choice label fingerprinting (a LOWERING step inside the narrow guard) to raise
    ``DagMlUnsupported``; the guard converts it to the ``_OperatorLoweringUnsupported`` sentinel, the
    routing branch catches it, and the run completes on the dag-ml engine via the Python-expand fallback —
    proving lowering-unsupported demotes (vs. the runtime tests above where the error propagates).
    """
    import nirs4all.pipeline.dagml.result as result_mod
    from nirs4all.pipeline.dagml.errors import DagMlUnsupported

    def refuse(*_args: object, **_kwargs: object) -> dict:
        raise DagMlUnsupported("simulated lowering-unsupported (label fingerprint)")

    monkeypatch.setattr(result_mod, "_native_operator_config_by_label", refuse)

    pipeline = [{"_or_": [SNV, MSC]}, KFold(n_splits=3), {"model": PLSRegression(n_components=10)}]
    result, native = _run_dagml(pipeline)
    # Demoted to Python-expand but STAYS on the dag-ml engine (no legacy-fallback warning), real result.
    assert native is True
    assert result.num_predictions >= 1
