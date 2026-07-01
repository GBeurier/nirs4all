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
from nirs4all.operators.transforms import Detrend, FirstDerivative, Resampler
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.pipeline.dagml.rt import RtError

from ._datasets import dataset_path

pytestmark = [pytest.mark.parity]

pytest.importorskip("dag_ml", reason="dag-ml not importable (core dependency; broken install?)")

_FALLBACK_FRAGMENT = "falling back to the legacy engine"


def _dataset() -> DatasetConfigs:
    return DatasetConfigs(dataset_path("regression"))


def _ss() -> Any:
    """A fresh seeded ShuffleSplit — fresh per call so a factory can be re-invoked for the legacy + dag-ml legs."""
    from sklearn.model_selection import ShuffleSplit

    return ShuffleSplit(n_splits=3, random_state=42)


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


def test_constrained_predicate_admits_concrete_prefix() -> None:
    """A concrete preprocessing PREFIX before the constrained generator is still constrained-native.

    The constrained predicate does NOT require the generator to be first — a leading concrete transform
    (a shared upstream preprocessing step) is supported, lowered to ONE shared upstream node applied to
    every survivor branch (matching the flat-single path), never silently dropped.
    """
    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    pipeline = [
        Detrend(),
        {"_or_": [SNV, MSC, Detrend], "pick": 2, "_mutex_": [[SNV, MSC]]},
        KFold(n_splits=3),
        {"model": PLSRegression(n_components=5)},
    ]
    assert _is_constrained_operator_generator(pipeline) is True


@pytest.mark.parametrize(
    "modifier",
    [
        {"count": 3},  # MUST-FIX 1: legacy SAMPLES post-prune; dag-ml `count` TRUNCATES (a different set)
        {"_seed_": 42},  # seeded subsample — no native analogue
        {"_weights_": [1, 1, 1, 1]},  # weighted random selection — no native analogue
    ],
)
def test_constrained_predicate_demotes_sampling_modifier(modifier: dict[str, Any]) -> None:
    """A constrained generator carrying a SAMPLING modifier (`count`/`_seed_`/`_weights_`) DEMOTES (MUST-FIX 1).

    Legacy `expand_spec` samples the post-prune survivors; dag-ml's native `count` truncates a different set
    (and rejects `count == 0`, where legacy means "no limit"). So a sampling modifier must keep the generator
    on the Python expand path — `_is_constrained_operator_generator` returns False.
    """
    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    node = {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2, "_mutex_": [[SNV, MSC]], **modifier}
    pipeline = [node, KFold(n_splits=3), {"model": PLSRegression(n_components=5)}]
    assert _is_constrained_operator_generator(pipeline) is False


def test_constrained_predicate_demotes_duplicate_options() -> None:
    """A constrained generator with two EQUAL operator options DEMOTES to Python-expand (MUST-FIX 2).

    Legacy constraint matching builds a normalized SET from the selected items, so a duplicate option
    satisfies the SAME ref; the native per-branch member rule would only prune the first. Demote so the
    survivor set never silently diverges.
    """
    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    node = {"_or_": [SNV, SNV, MSC], "pick": 2, "_mutex_": [[SNV, MSC]]}
    pipeline = [node, KFold(n_splits=3), {"model": PLSRegression(n_components=5)}]
    assert _is_constrained_operator_generator(pipeline) is False


def test_constrained_predicate_demotes_non_pair_exclude() -> None:
    """A NON-pair `_exclude_` (>2 operators, an exact N-combo) DEMOTES — dag-ml `exclude` is a pair only (MUST-FIX 3).

    A 2-operator `_exclude_` lowers 1:1 to dag-ml's `[String; 2]` pair and stays native; a >2 exclude (the
    legacy exact-combo exclusion) has no native pair form, so it routes Python-expand.
    """
    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    non_pair = {"_or_": [SNV, MSC, Detrend], "pick": 3, "_exclude_": [[SNV, MSC, Detrend]]}
    pair = {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2, "_exclude_": [[SNV, Detrend]]}
    model = {"model": PLSRegression(n_components=5)}
    assert _is_constrained_operator_generator([non_pair, KFold(n_splits=3), model]) is False
    assert _is_constrained_operator_generator([pair, KFold(n_splits=3), model]) is True


def test_constrained_predicate_admits_multi_requires() -> None:
    """A multi-`_requires_` (>2) stays NATIVE — the host splits it into the `[first, each-subsequent]` pairs (MUST-FIX 3).

    `_requires_=[[A, B, C]]` is "A requires B AND A requires C"; the host lowers it to the two pairs dag-ml's
    `[String; 2]` `requires` accepts, so it routes native (no demote needed).
    """
    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    node = {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 3, "_requires_": [[SNV, MSC, Detrend]]}
    pipeline = [node, KFold(n_splits=3), {"model": PLSRegression(n_components=5)}]
    assert _is_constrained_operator_generator(pipeline) is True


def test_constrained_predicate_demotes_cross_stage_duplicate_option() -> None:
    """An operator identity in DIFFERENT `_cartesian_` stages DEMOTES (MUST-FIX 5).

    The bridge resolves a constraint ref through ONE global identity map to the FIRST branch of that identity,
    but legacy set-membership should match the ref from ANY stage carrying it. So `_cartesian_
    [[SNV,MSC],[SNV,Detrend]]` with `_requires_[[SNV,Detrend]]` (SNV in BOTH stages) mis-resolves natively
    (SNV pinned to stage 0) and must route Python-expand. A `_cartesian_` with NO cross-stage duplicate (the
    `cartesian_exclude` oracle shape) stays native.
    """
    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    model = {"model": PLSRegression(n_components=5)}
    cross_stage_dup = {"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [SNV, Detrend]}], "_requires_": [[SNV, Detrend]]}
    no_dup = {"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}], "_exclude_": [[SNV, Detrend]]}
    assert _is_constrained_operator_generator([cross_stage_dup, KFold(n_splits=3), model]) is False
    assert _is_constrained_operator_generator([no_dup, KFold(n_splits=3), model]) is True


def test_constrained_predicate_demotes_pair_exclude_when_survivor_cardinality_not_two() -> None:
    """A pair `_exclude_` DEMOTES when the survivor cardinality is not exactly 2 (MUST-FIX 6).

    Legacy `_exclude_` is EXACT-COMBO (drops a survivor only when its full set EQUALS the pair) while dag-ml
    native `exclude` is SUBSET (drops ANY survivor where both co-occur). They agree ONLY at survivor
    cardinality 2: `_or_` pick=2 / a 2-stage `_cartesian_` stay native; `_or_` pick=3 or a 3-stage
    `_cartesian_` (where legacy keeps the >2-combos containing both, but native would prune them) DEMOTE.
    """
    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    model = {"model": PLSRegression(n_components=5)}
    pick3_pair_exclude = {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 3, "_exclude_": [[SNV, Detrend]]}
    pick2_pair_exclude = {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2, "_exclude_": [[SNV, Detrend]]}
    three_stage_pair_exclude = {"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}, {"_or_": [SNV, MSC]}], "_exclude_": [[SNV, Detrend]]}
    assert _is_constrained_operator_generator([pick3_pair_exclude, KFold(n_splits=3), model]) is False
    assert _is_constrained_operator_generator([pick2_pair_exclude, KFold(n_splits=3), model]) is True
    # (the 3-stage cartesian ALSO has a cross-stage SNV/MSC duplicate, so it demotes on both MUST-FIX 5 and 6 —
    # either guard is sufficient; the point is it does NOT route native.)
    assert _is_constrained_operator_generator([three_stage_pair_exclude, KFold(n_splits=3), model]) is False


@pytest.mark.parametrize(
    "node",
    [
        # EDGE A — bare constrained `_or_` (no pick/arrange): legacy `_expand_basic` makes single-op (non-list)
        # variants and `_apply_constraints` SKIPS them (`not isinstance(results[0], list)`), so the constraint
        # is IGNORED legacy-side; native applies it. DEMOTE.
        {"_or_": [SNV, MSC, Detrend], "_requires_": [[SNV, MSC]]},
        {"_or_": [SNV, MSC, Detrend], "_mutex_": [[SNV, MSC]]},
        # EDGE B — degenerate / empty / dangling / repeated-ref constraint groups: legacy normalizes to sets and
        # no-ops a single-ref / self-pair / empty / repeated-ref group (or never matches a dangling ref), native
        # REJECTS (mutex <2 distinct OR a REPEATED ref, a repeated requires/exclude pair) or CRASHES the bridge
        # (empty requires). DEMOTE.
        {"_or_": [SNV, MSC, Detrend], "pick": 2, "_mutex_": [[SNV]]},
        {"_or_": [SNV, MSC, Detrend], "pick": 2, "_mutex_": [[SNV, SNV]]},
        {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 3, "_mutex_": [[SNV, MSC, MSC]]},  # 2 distinct but a REPEATED ref → native rejects ("mutex group repeats an operator")
        {"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}], "_mutex_": [[SNV, Detrend, Detrend]]},  # repeated ref on a _cartesian_
        {"_or_": [SNV, MSC, Detrend], "pick": 2, "_requires_": [[SNV, SNV]]},
        {"_or_": [SNV, MSC, Detrend], "pick": 2, "_exclude_": [[SNV, SNV]]},
        {"_or_": [SNV, MSC, Detrend], "pick": 2, "_requires_": [[]]},
        {"_or_": [SNV, MSC, Detrend], "pick": 2, "_mutex_": [[SNV, FirstDerivative]]},  # FirstDerivative not an option
        # EDGE C — zero / oversize / range / second-order selection: legacy skips oversize (`s > n: continue`)
        # and no-ops zero (`s == 0: append([])`), and `then_*` is a different survivor structure; native
        # rejects zero / oversize, and the multi-op-sequence shape does not model `then_*`. DEMOTE.
        {"_or_": [SNV, MSC, Detrend], "pick": 0, "_mutex_": [[SNV, MSC]]},
        {"_or_": [SNV, MSC, Detrend], "pick": 5, "_mutex_": [[SNV, MSC]]},
        {"_or_": [SNV, MSC, Detrend], "pick": (1, 2), "_mutex_": [[SNV, MSC]]},
        {"_or_": [SNV, MSC, Detrend], "pick": 2, "then_pick": 1, "_mutex_": [[SNV, MSC]]},
    ],
)
def test_constrained_predicate_fail_closed_demotes_divergent_shape(node: dict[str, Any]) -> None:
    """The fail-closed predicate DEMOTES every admit-then-diverge shape (EDGE A/B/C) to the Python expand path.

    Each of these is a valid legacy shape where legacy is LENIENT (skips constraints on single-op variants,
    no-ops degenerate groups, skips oversize / zero selections, second-order selection) but dag-ml native is
    STRICT — so the predicate must NOT route it native. (The result is still correct via Python-expand on the
    dag-ml engine; correctness for a representative deterministic edge is checked separately.)
    """
    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    assert _is_constrained_operator_generator([node, KFold(n_splits=3), {"model": PLSRegression(n_components=5)}]) is False


@pytest.mark.slow
def test_constrained_bare_or_edge_a_demotes_and_matches_legacy() -> None:
    """EDGE A end-to-end: a bare constrained `_or_` (legacy IGNORES the constraint) DEMOTES and matches legacy.

    `{"_or_": [SNV, MSC, Detrend], "_requires_": [[SNV, MSC]]}` — legacy keeps ALL THREE single-op variants
    (constraints skipped on non-list results); a naive native apply would prune SNV-without-MSC. The predicate
    demotes it, so the dag-ml engine runs the (correct, all-3) Python-expand and matches the legacy engine.
    """
    from sklearn.model_selection import ShuffleSplit

    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    node = {"_or_": [SNV, MSC, Detrend], "_requires_": [[SNV, MSC]]}
    model = {"model": PLSRegression(n_components=5)}
    assert _is_constrained_operator_generator([node, model]) is False
    pipe = [node, ShuffleSplit(n_splits=3, random_state=42), model]
    legacy = nirs4all.run(pipeline=pipe, dataset=_dataset(), verbose=0, engine="legacy")
    dagml, native = _run_dagml(pipe)
    assert native is True, "the dag-ml engine runs the demoted bare-_or_ via Python-expand (no legacy fallback)"
    assert dagml.best_score == pytest.approx(legacy.best_score, abs=1e-3, rel=1e-3)


@pytest.mark.slow
def test_constrained_mutex_repeated_ref_demotes_and_matches_legacy() -> None:
    """A `_mutex_` group with a REPEATED ref DEMOTES and runs correctly (no DagMlValidationError leak).

    `{"_or_": [...], "pick": 3, "_mutex_": [[SNV, MSC, MSC]]}` has 2 DISTINCT refs but a repeated one — legacy
    normalizes the group to the set `{SNV, MSC}` (a valid pair mutex), but dag-ml native REJECTS the
    repeated-ref group ("mutex group repeats an operator"). The fail-closed predicate demotes it, so the
    dag-ml engine runs the (correct) Python-expand and matches the legacy engine — instead of a compile crash.
    """
    from sklearn.model_selection import ShuffleSplit

    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    node = {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 3, "_mutex_": [[SNV, MSC, MSC]]}
    model = {"model": PLSRegression(n_components=5)}
    assert _is_constrained_operator_generator([node, model]) is False
    pipe = [node, ShuffleSplit(n_splits=3, random_state=42), model]
    legacy = nirs4all.run(pipeline=pipe, dataset=_dataset(), verbose=0, engine="legacy")
    dagml, native = _run_dagml(pipe)
    assert native is True, "the dag-ml engine runs the demoted repeated-ref mutex via Python-expand (no crash, no legacy fallback)"
    assert dagml.best_score == pytest.approx(legacy.best_score, abs=1e-3, rel=1e-3)


@pytest.mark.slow
def test_constrained_prefix_runs_native_and_applies_prefix() -> None:
    """A PREFIXED constrained generator runs NATIVE and APPLIES the prefix (no silent drop).

    `[FirstDerivative(), _or_-pick+_mutex_, model]` routes native operator-SELECT (the constrained path
    lowers the leading `FirstDerivative` to ONE shared upstream transform node feeding every survivor
    branch — NOT inside the survivor branches, NOT in the per-variant `variant_label`, exactly as the
    flat-single path lowers a prefix). PARITY vs legacy on the SAME prefixed pipeline proves the prefix is
    actually applied (a dropped prefix would diverge the scores); the prefix-FREE twin scores DIFFERENTLY,
    so the prefix is load-bearing here, not a no-op.
    """
    from sklearn.model_selection import ShuffleSplit

    def _constrained_node() -> dict[str, Any]:
        # pick=2 of 3 with `_mutex_[[SNV,MSC]]`: C(3,2)=3 -> the {SNV,MSC} combo is pruned -> 2 survivors.
        return {"_or_": [SNV, MSC, Detrend], "pick": 2, "_mutex_": [[SNV, MSC]]}

    def prefixed() -> list[Any]:
        return [FirstDerivative(), _constrained_node(), ShuffleSplit(n_splits=3, random_state=42), {"model": PLSRegression(n_components=5)}]

    def prefix_free() -> list[Any]:
        return [_constrained_node(), ShuffleSplit(n_splits=3, random_state=42), {"model": PLSRegression(n_components=5)}]

    legacy = nirs4all.run(pipeline=prefixed(), dataset=_dataset(), verbose=0, engine="legacy")
    dagml, native = _run_dagml(prefixed())
    assert native is True, "prefixed constrained generator must route NATIVE, not fall back"
    # The prefix is applied: native matches legacy on the SAME (prefixed) pipeline within score tolerance.
    assert dagml.best_score == pytest.approx(legacy.best_score, abs=1e-3, rel=1e-3)
    # And the prefix is load-bearing: dropping it changes the score, so a silent drop would NOT have matched.
    no_prefix = nirs4all.run(pipeline=prefix_free(), dataset=_dataset(), verbose=0, engine="legacy")
    assert dagml.best_score != pytest.approx(no_prefix.best_score, abs=1e-6)


@pytest.mark.slow
def test_constrained_count_demotes_to_python_expand_and_runs() -> None:
    """A `count`-bearing constrained generator DEMOTES off the constrained-native path but still RUNS on dag-ml (MUST-FIX 1).

    `count` would make dag-ml's native operator-SELECT generator TRUNCATE the survivor list, whereas legacy
    `expand_spec` takes a SEEDED post-prune SAMPLE — a different (and, unseeded, non-deterministic) subset. So
    the constrained predicate keeps a `count`-bearing generator OFF the constrained-native path; the dag-ml
    engine still runs it via the Python-expand path (NOT a legacy fallback), proving the demote is a routing
    guard, not a feature loss. (A cross-engine SCORE comparison is intentionally NOT made: the count sample is
    randomized, so two independent runs legitimately pick different survivor subsets — that randomness is the
    very reason this case is demoted.)
    """
    from sklearn.model_selection import ShuffleSplit

    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    # `_or_` over 4 ops, pick 2, `_mutex_[[SNV,MSC]]` -> 5 survivors, then `count` 3 (a post-prune sample) — the
    # MUST-FIX 1 demote case.
    node = {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2, "_mutex_": [[SNV, MSC]], "count": 3}
    model = {"model": PLSRegression(n_components=5)}
    # The constrained-NATIVE operator-SELECT path is NOT taken (count would diverge the native survivor set).
    assert _is_constrained_operator_generator([node, model]) is False, "count must keep the generator off the constrained-native path"

    dagml, native = _run_dagml([node, ShuffleSplit(n_splits=3, random_state=42), model])
    # The dag-ml engine still runs it (Python-expand, NOT a legacy fallback) and yields a finite score.
    assert native is True, "the dag-ml engine runs the demoted generator via Python-expand (no legacy fallback)"
    assert math.isfinite(dagml.best_score)


@pytest.mark.slow
@pytest.mark.parametrize(
    "node",
    [
        # MUST-FIX 6: `_or_` pick=3 + pair `_exclude_` — legacy keeps the 3-combos containing both (exact-2-combo
        # never equals a 3-combo); native subset would wrongly prune them. DEMOTE -> Python-expand must match legacy.
        {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 3, "_exclude_": [[SNV, Detrend]]},
        # MUST-FIX 5: cross-stage duplicate operator identity (SNV in both stages) with a `_requires_` ref to it.
        {"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [SNV, Detrend]}], "_requires_": [[SNV, Detrend]]},
    ],
)
def test_constrained_divergent_edge_demotes_and_matches_legacy(node: dict[str, Any]) -> None:
    """The MUST-FIX 5 / 6 divergent edges DEMOTE to Python-expand and match the legacy engine (deterministic, no sampling).

    Unlike the count case (a randomized sample), these survivor sets are DETERMINISTIC, so the demoted
    Python-expand run on the dag-ml engine must reproduce the legacy engine's best score exactly — proving the
    demote routes off the (divergent) constrained-native path WITHOUT losing correctness.
    """
    from sklearn.model_selection import ShuffleSplit

    from nirs4all.pipeline.dagml.detect import _is_constrained_operator_generator

    model = {"model": PLSRegression(n_components=5)}
    # Routes OFF the constrained-native path (the divergent edge is demoted).
    assert _is_constrained_operator_generator([node, model]) is False

    pipe = [node, ShuffleSplit(n_splits=3, random_state=42), model]
    legacy = nirs4all.run(pipeline=pipe, dataset=_dataset(), verbose=0, engine="legacy")
    dagml, native = _run_dagml(pipe)
    # Python-expand on dag-ml (no legacy fallback) reproduces the legacy best score exactly.
    assert native is True, "the dag-ml engine runs the demoted generator via Python-expand (no legacy fallback)"
    assert dagml.best_score == pytest.approx(legacy.best_score, abs=1e-3, rel=1e-3)


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
    OUTER ``run()`` boundary that raises a structured ``RtError`` by default (V1 strict fallback). With
    explicit ``allow_fallback=True`` it redirects to LEGACY, so the run is LEGACY (``dagml_native=False``),
    NOT a dag-ml Python-expand success.

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

    with pytest.raises(RtError) as excinfo:
        nirs4all.run(pipeline=pipeline, dataset=_dataset(), verbose=0, engine="dag-ml")
    payload = excinfo.value.to_dict()
    assert payload["cause"] == "unsupported_shape"
    assert "simulated runtime unsupported shape" in payload["message"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nirs4all.run(pipeline=pipeline, dataset=_dataset(), verbose=0, engine="dag-ml", allow_fallback=True)
    # The runtime DagMlUnsupported propagated past the inner branch → the explicit OUTER fallback redirected
    # to LEGACY (fallback warning fired), NOT a silent dag-ml Python-expand re-run (which would be native).
    fell_back = any(_FALLBACK_FRAGMENT in str(w.message) for w in caught)
    assert fell_back
    assert not result._is_dagml_engine()  # noqa: SLF001
    assert result.num_predictions >= 1
    fallback_payload = result.to_rt_result().to_dict()
    assert fallback_payload["manifest"]["engine"] == "legacy"
    diagnostics = fallback_payload.get("diagnostics", [])
    assert any(
        diagnostic.get("cause") == "unsupported_shape"
        and "simulated runtime unsupported shape" in diagnostic.get("message", "")
        for diagnostic in diagnostics
    )


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


# --------------------------------------------------------------------------- #
# ADR-17 item 5 slice C — UNCONSTRAINED operator generators (pick/arrange/cartesian) route native
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "node",
    [
        # pure `_or_` with an INTEGER `pick` in [1, n] — survivors are the C(n, pick) combinations.
        {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2},
        # pure `_or_` with an INTEGER `arrange` in [1, n] — survivors are the P(n, arrange) ORDERED permutations.
        {"_or_": [SNV, MSC, Detrend], "arrange": 2},
        # pure `_cartesian_` of >=2 `_or_` stages, no pick/arrange — survivors are the full stage product.
        {"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}]},
    ],
)
def test_unconstrained_predicate_admits_pick_arrange_cartesian(node: dict[str, Any]) -> None:
    """The unconstrained predicate ADMITS the pick/arrange/cartesian shapes whose survivor set has no constraint prune.

    dag-ml's `expand_or_generator_sequences` / `expand_cartesian_generator_sequences` produce EXACTLY the
    unconstrained survivor set (all combinations / permutations / stage products) in legacy
    `itertools.combinations`/`permutations` order, so `_native_constrained_generator` lowers them with an
    EMPTY constraints set and dag-ml scores the byte-identical set.
    """
    from nirs4all.pipeline.dagml.detect import _is_unconstrained_operator_generator

    pipeline = [node, KFold(n_splits=3), {"model": PLSRegression(n_components=5)}]
    assert _is_unconstrained_operator_generator(pipeline) is True


@pytest.mark.parametrize(
    "node",
    [
        # then_pick / then_arrange — SECOND-ORDER selection (a different survivor structure); DEMOTE.
        {"_or_": [SNV, MSC, Detrend], "pick": 1, "then_pick": 2},
        {"_or_": [SNV, MSC, Detrend], "pick": 1, "then_arrange": 2},
        # `_cartesian_` with `pick` — pipeline-PAIR selection (dag-ml's cartesian mode REFUSES pick/arrange); DEMOTE.
        {"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}], "pick": 2},
        # SAMPLING modifiers — legacy SAMPLES survivors; dag-ml `count` TRUNCATES a different set (and the
        # `_or_`-pick count path is non-deterministic run-to-run). DEMOTE on count / _seed_ / _weights_.
        {"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}], "count": 2, "_seed_": 7},
        {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2, "count": 3, "_seed_": 42},
        {"_or_": [SNV, MSC, Detrend], "pick": 2, "_weights_": [1, 1, 1]},
        # oversize / zero / range pick — legacy is lenient (skip oversize, no-op zero, mixed-cardinality range);
        # native rejects. DEMOTE.
        {"_or_": [SNV, MSC], "pick": 5},
        {"_or_": [SNV, MSC], "pick": 0},
        {"_or_": [SNV, MSC, Detrend], "pick": (1, 2)},
        # a single-stage `_cartesian_` (degenerate — not the >=2-stage product shape). DEMOTE.
        {"_cartesian_": [{"_or_": [SNV, MSC]}], "pick": 2},
        # multi-step list choices (the `generator_or_multistep_branch` shape) are NOT bare X-transforms. DEMOTE.
        {"_or_": [[SNV, FirstDerivative], [MSC, Detrend]], "pick": 1},
        # a DUPLICATE operator option — two equal-content survivors would fingerprint-collide. DEMOTE.
        {"_or_": [SNV, SNV, MSC], "pick": 2},
    ],
)
def test_unconstrained_predicate_demotes_divergent_shape(node: dict[str, Any]) -> None:
    """The fail-closed unconstrained predicate DEMOTES every shape that cannot reach native parity.

    `then_*` (second-order), a cartesian pick/arrange, a count/_seed_/_weights_ sample, an oversize/zero/range
    pick, a degenerate single-stage cartesian, a non-routable multi-step choice, or a duplicate option all stay
    on the proven Python-expand path (still dag-ml-native via that route).
    """
    from nirs4all.pipeline.dagml.detect import _is_unconstrained_operator_generator

    assert _is_unconstrained_operator_generator([node, KFold(n_splits=3), {"model": PLSRegression(n_components=5)}]) is False


def test_unconstrained_predicate_excludes_constrained_and_bare_or() -> None:
    """The unconstrained predicate does NOT claim a CONSTRAINED node (constrained predicate's job) or a BARE `_or_` (flat-single's).

    A `_mutex_`/`_requires_`/`_exclude_` node belongs to `_is_constrained_operator_generator`; a bare `_or_`
    with no pick/arrange is the single-op flat path. The three predicates partition the operator-generator
    space without overlap.
    """
    from nirs4all.pipeline.dagml.detect import (
        _is_constrained_operator_generator,
        _is_flat_single_operator_generator,
        _is_unconstrained_operator_generator,
    )

    model = {"model": PLSRegression(n_components=5)}
    constrained = {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2, "_mutex_": [[SNV, MSC]]}
    bare = {"_or_": [SNV, MSC, Detrend]}
    assert _is_unconstrained_operator_generator([constrained, KFold(n_splits=3), model]) is False
    assert _is_constrained_operator_generator([constrained, KFold(n_splits=3), model]) is True
    assert _is_unconstrained_operator_generator([bare, KFold(n_splits=3), model]) is False
    assert _is_flat_single_operator_generator([bare, KFold(n_splits=3), model]) is True


@pytest.mark.slow
@pytest.mark.parametrize(
    "factory",
    [
        lambda: [{"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2}, _ss(), {"model": PLSRegression(n_components=10)}],
        lambda: [{"_or_": [SNV, MSC, Detrend], "arrange": 2}, _ss(), {"model": PLSRegression(n_components=10)}],
        lambda: [{"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}]}, _ss(), {"model": PLSRegression(n_components=10)}],
    ],
)
def test_unconstrained_runs_native_at_full_parity(factory: Any) -> None:
    """E2E: an admitted unconstrained pick/arrange/cartesian generator runs NATIVE and matches legacy at FULL parity.

    Member-exact survivor set + same winner (via num_predictions + best_score parity). The ARRANGE case proves
    dag-ml's ordered-permutation enumeration agrees with legacy `itertools.permutations` (the survivor labels
    are content-keyed, so ordered survivors align by content) — arrange reaches parity, it does NOT demote.
    """
    legacy = nirs4all.run(pipeline=factory(), dataset=_dataset(), verbose=0, engine="legacy")
    dagml, native = _run_dagml(factory())
    assert native is True, "an admitted unconstrained generator must route NATIVE, not fall back"
    assert dagml.num_predictions == legacy.num_predictions
    assert dagml.best_score == pytest.approx(legacy.best_score, abs=1e-3, rel=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize(
    "factory",
    [
        # then_pick / then_arrange (second-order) — deterministic, must match legacy via Python-expand.
        lambda: [{"_or_": [SNV, MSC, Detrend], "pick": 1, "then_pick": 2}, _ss(), {"model": PLSRegression(n_components=10)}],
        lambda: [{"_or_": [SNV, MSC, Detrend], "pick": 1, "then_arrange": 2}, _ss(), {"model": PLSRegression(n_components=10)}],
        # `_cartesian_` pick — deterministic, must match legacy via Python-expand.
        lambda: [{"_cartesian_": [{"_or_": [SNV, MSC]}, {"_or_": [Detrend, FirstDerivative]}], "pick": 2}, _ss(), {"model": PLSRegression(n_components=10)}],
        # NOTE: a bare multi-step `_or_` (`[[SNV, 1stDer], [MSC, Detrend]]`) is NO LONGER a demote — ADR-17 item 5
        # slice D routes it NATIVE via the flat-single path (`test_multistep_or_runs_native_at_full_parity`).
    ],
)
def test_unconstrained_demoted_shape_matches_legacy_via_python_expand(factory: Any) -> None:
    """A DEMOTED unconstrained shape stays on the dag-ml engine (Python-expand, no legacy fallback) and matches legacy.

    These deterministic survivor sets (then_*, cartesian-pick) demote off native but still run correctly on
    dag-ml, reproducing the legacy best score — the demote is a routing guard, not feature loss.
    (count/_seed_/_weights_ are excluded here: their sampled subsets are randomized run-to-run, so no
    cross-engine score claim is made — that randomness is the very reason they demote.)
    """
    legacy = nirs4all.run(pipeline=factory(), dataset=_dataset(), verbose=0, engine="legacy")
    dagml, native = _run_dagml(factory())
    assert native is True, "the dag-ml engine runs the demoted generator via Python-expand (no legacy fallback)"
    assert dagml.num_predictions == legacy.num_predictions
    assert dagml.best_score == pytest.approx(legacy.best_score, abs=1e-3, rel=1e-3)


# --------------------------------------------------------------------------- #
# ADR-17 item 5 slice D — MULTI-STEP `_or_` choices (native) + NESTED generators (demote)
# --------------------------------------------------------------------------- #


def test_multistep_choice_helpers_partition_choice_shapes() -> None:
    """The choice-shape helpers partition single / multi-step / nested-or-None choices (the slice-D admit boundary)."""
    from nirs4all.pipeline.dagml_bridge import _is_bare_operator_choice, _is_multistep_operator_choice, _operator_choice_operators

    # single bare op
    assert _is_bare_operator_choice(SNV) and not _is_multistep_operator_choice(SNV)
    assert _operator_choice_operators(SNV) == [SNV]
    # multi-step list of bare ops
    assert _is_multistep_operator_choice([SNV, Detrend]) and not _is_bare_operator_choice([SNV, Detrend])
    assert _operator_choice_operators([SNV, Detrend]) == [SNV, Detrend]
    # NOT multi-step: empty list, a list nesting a list/dict, a dict (nested generator), None
    assert not _is_multistep_operator_choice([])
    assert not _is_multistep_operator_choice([SNV, [Detrend, MSC]])  # nested-list element
    assert not _is_multistep_operator_choice([SNV, {"_or_": [Detrend]}])  # nested-generator element
    assert not _is_multistep_operator_choice({"_or_": [SNV, MSC]})  # a nested-generator dict choice
    assert not _is_multistep_operator_choice(None)


def test_flat_predicate_admits_multistep_or() -> None:
    """A bare `_or_` of MULTI-STEP bare-operator choices IS flat-single (slice-D native admit)."""
    from nirs4all.pipeline.dagml.detect import _is_flat_single_operator_generator

    pipeline = [{"_or_": [[SNV, FirstDerivative], [MSC, Detrend], SNV]}, KFold(n_splits=3), {"model": PLSRegression(n_components=10)}]
    assert _is_flat_single_operator_generator(pipeline) is True


@pytest.mark.parametrize(
    "node",
    [
        # multi-step + pick → nested-list survivors the single-op-per-branch native fingerprint cannot key.
        {"_or_": [[SNV, FirstDerivative], MSC, Detrend], "pick": 2},
        # a list element that is itself a list (nested-list multi-step) — not a flat bare-op sub-pipeline.
        {"_or_": [[SNV, [Detrend, MSC]], FirstDerivative]},
        # a NESTED operator-generator choice (`{_or_: …}` dict) — SUB-PART 2 hard demote.
        {"_or_": [{"_or_": [SNV, MSC]}, Detrend]},
        # a multi-step choice nesting a generator dict.
        {"_or_": [[SNV, {"_or_": [Detrend, MSC]}], FirstDerivative]},
    ],
)
def test_flat_predicate_demotes_multistep_with_selector_or_nested(node: dict[str, Any]) -> None:
    """A multi-step `_or_` carrying a selector, a nested-list element, or a NESTED generator DEMOTES off every native path (slice-D fail-closed)."""
    from nirs4all.pipeline.dagml.detect import (
        _is_constrained_operator_generator,
        _is_flat_single_operator_generator,
        _is_unconstrained_operator_generator,
    )

    pipeline = [node, KFold(n_splits=3), {"model": PLSRegression(n_components=5)}]
    assert _is_flat_single_operator_generator(pipeline) is False
    assert _is_unconstrained_operator_generator(pipeline) is False
    assert _is_constrained_operator_generator(pipeline) is False


def test_nested_generator_demotes_across_all_native_paths() -> None:
    """A NESTED operator-generator anywhere (bare `_or_`, pick, `_cartesian_` stage, constrained) DEMOTES on EVERY native predicate (SUB-PART 2 fail-closed)."""
    from nirs4all.pipeline.dagml.detect import (
        _is_constrained_operator_generator,
        _is_flat_single_operator_generator,
        _is_unconstrained_operator_generator,
    )

    model = {"model": PLSRegression(n_components=5)}
    cv = KFold(n_splits=3)
    nested_nodes = [
        {"_or_": [{"_or_": [SNV, MSC]}, Detrend]},  # bare
        {"_or_": [{"_or_": [SNV, MSC]}, Detrend, FirstDerivative], "pick": 2},  # pick
        {"_cartesian_": [{"_or_": [{"_or_": [SNV, MSC]}, Detrend]}, {"_or_": [FirstDerivative, SNV]}]},  # cartesian stage
        {"_or_": [{"_or_": [SNV, MSC]}, Detrend, FirstDerivative], "pick": 2, "_mutex_": [[Detrend, FirstDerivative]]},  # constrained
    ]
    for node in nested_nodes:
        pipe = [node, cv, model]
        assert _is_flat_single_operator_generator(pipe) is False
        assert _is_unconstrained_operator_generator(pipe) is False
        assert _is_constrained_operator_generator(pipe) is False


@pytest.mark.slow
@pytest.mark.parametrize(
    "factory",
    [
        # bare multi-step `_or_` (mixed multi-step + single-op choices) — NATIVE operator-SELECT, full parity.
        lambda: [{"_or_": [[SNV, FirstDerivative], [MSC, Detrend], SNV, MSC]}, _ss(), {"model": PLSRegression(n_components=10)}],
        # all-multi-step choices.
        lambda: [{"_or_": [[SNV, FirstDerivative], [MSC, Detrend]]}, _ss(), {"model": PLSRegression(n_components=10)}],
    ],
)
def test_multistep_or_runs_native_at_full_parity(factory: Any) -> None:
    """E2E: a bare `_or_` of MULTI-STEP choices runs NATIVE operator-SELECT and matches legacy at FULL parity (slice D SUB-PART 1).

    dag-ml lowers each list choice to a MULTI-TRANSFORM branch and carries it as ONE survivor (no pick/arrange
    recombination), so the survivor set is member-exact and the content-keyed `variant_label` aligns the
    winner — `num_predictions` + `best_score` parity confirm same-winner + same-set.
    """
    legacy = nirs4all.run(pipeline=factory(), dataset=_dataset(), verbose=0, engine="legacy")
    dagml, native = _run_dagml(factory())
    assert native is True, "an admitted multi-step `_or_` must route NATIVE, not fall back"
    assert dagml.num_predictions == legacy.num_predictions
    assert dagml.best_score == pytest.approx(legacy.best_score, abs=1e-3, rel=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize(
    "factory",
    [
        # multi-step + pick → nested-list survivors → DEMOTE → Python-expand (deterministic, matches legacy).
        lambda: [{"_or_": [[SNV, FirstDerivative], MSC, Detrend], "pick": 2}, _ss(), {"model": PLSRegression(n_components=10)}],
        # a NESTED operator-generator → DEMOTE → Python-expand; legacy flattens `{_or_: [SNV,MSC]}` → SNV, MSC, Detrend.
        lambda: [{"_or_": [{"_or_": [SNV, MSC]}, Detrend]}, _ss(), {"model": PLSRegression(n_components=10)}],
    ],
)
def test_multistep_nested_demoted_shape_matches_legacy_via_python_expand(factory: Any) -> None:
    """A DEMOTED slice-D shape (multi-step+pick / nested generator) stays on dag-ml (Python-expand) and matches legacy.

    These deterministic survivor sets demote off native operator-SELECT but still run correctly on the dag-ml
    engine (NOT a legacy fallback), reproducing the legacy best score — the demote is a routing guard, not
    feature loss. The nested case proves legacy's FLATTEN (`{_or_: [SNV,MSC]}` → SNV, MSC) is reproduced.
    """
    legacy = nirs4all.run(pipeline=factory(), dataset=_dataset(), verbose=0, engine="legacy")
    dagml, native = _run_dagml(factory())
    assert native is True, "the dag-ml engine runs the demoted slice-D generator via Python-expand (no legacy fallback)"
    assert dagml.num_predictions == legacy.num_predictions
    assert dagml.best_score == pytest.approx(legacy.best_score, abs=1e-3, rel=1e-3)


# --------------------------------------------------------------------------- #
# Slice F (ADR-17 item 5) — _zip_ / _chain_ stay fail-closed off the DEDICATED
# in-engine generation paths; they run NATIVE via the host VARIANT-EXPAND path.
# --------------------------------------------------------------------------- #


def test_bridge_demotes_param_zip_off_dedicated_native_path() -> None:
    """A param-level `_zip_` model sweep is NOT lowered to a dedicated dag-ml `generators` form (slice F).

    The host bridge `_step_to_dsl` DELIBERATELY raises NotImplementedError for a `{"_zip_": {...}, "model": M}`
    step (it is not a native `param_generators`/`Grid` form) — so it stays on the proven Python variant-expand
    path (still dag-ml-native via `_expand_operator_generators`), never mis-routed to `_run_native_generation`.
    Promoting it to dag-ml's native `compat_zip_variants` would be a LARGE host change (a new `variants`
    emission form + zip-truncation/nested-value parity gaps) for no gain, so it is DEMOTED fail-closed.
    """
    from nirs4all.pipeline.dagml_bridge import _step_to_dsl

    step = {"_zip_": {"n_components": [5, 10, 15], "scale": [True, False, True]}, "model": PLSRegression}
    with pytest.raises(NotImplementedError, match="_zip_"):
        _step_to_dsl(step)


def test_bridge_demotes_operator_chain_off_dedicated_native_path() -> None:
    """An operator-level `_chain_` is NOT lowered to a dedicated dag-ml operator `Generator` step (slice F).

    The host bridge `_step_to_dsl` only lowers a flat single `_or_` operator generator natively; a `_chain_`
    raises NotImplementedError naming the `_chain_` keyword, so it stays on the Python variant-expand path
    (still dag-ml-native), never mis-routed to `_run_native_operator_generation` (which would flip
    num_predictions to winner-only-refit — a NEW divergence the expand path does not have). DEMOTED fail-closed.
    """
    from nirs4all.pipeline.dagml_bridge import _step_to_dsl

    step = {"_chain_": [SNV, MSC, Detrend]}
    with pytest.raises(NotImplementedError, match="_chain_"):
        _step_to_dsl(step)


@pytest.mark.slow
@pytest.mark.parametrize(
    "factory",
    [
        # param-level `_zip_` (with-model) → variant-expand → 3 paired variants, FULL parity.
        lambda: [SNV(), _ss(), {"_zip_": {"n_components": [3, 7, 11]}, "model": PLSRegression}],
        # operator-level `_chain_` of multi-step sub-pipelines → variant-expand → 2 variants, FULL parity.
        lambda: [{"_chain_": [[SNV(), FirstDerivative()], [MSC(), Detrend()]]}, _ss(), {"model": PLSRegression(n_components=10)}],
    ],
)
def test_zip_chain_run_native_via_expand_at_full_parity(factory: Any) -> None:
    """E2E: a param-level `_zip_` and an operator-level `_chain_` run NATIVE-via-expand and match legacy (slice F).

    Both demote off the dedicated in-engine paths (asserted above) but still run on the dag-ml engine via
    `_expand_operator_generators` (NOT a legacy fallback), reproducing legacy's variant set + winner — so
    `num_predictions` + `best_score` parity confirm same-set + same-winner. (The multi-model `_chain_` of
    distinct model classes — the winner-only-refit num_predictions divergence — is covered separately by
    the conformance oracle's `generator_chain_model_configs` parity-note.)
    """
    legacy = nirs4all.run(pipeline=factory(), dataset=_dataset(), verbose=0, engine="legacy")
    dagml, native = _run_dagml(factory())
    assert native is True, "an admitted `_zip_`/`_chain_` shape runs NATIVE-via-expand on dag-ml (no legacy fallback)"
    assert dagml.num_predictions == legacy.num_predictions
    assert dagml.best_score == pytest.approx(legacy.best_score, abs=1e-3, rel=1e-3)
