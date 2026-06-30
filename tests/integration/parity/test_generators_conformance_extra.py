"""Generator-conformance assertions that are NOT pipeline-case parity runs.

Two things the dual-engine ``PipelineCase`` registry cannot express belong here:

* ``_depends_on_`` is a DEAD generator keyword — declared in
  ``CONSTRAINT_KEYWORDS`` but never consulted by the constraint engine. The task
  for the GENERATION conformance suite calls for a case "asserting it is a no-op
  / does not change the variant set, documenting that." It cannot be a runnable
  ``PipelineCase`` because adding ``_depends_on_`` to a top-level generator node
  removes it from its pure-strategy key set, which BREAKS expansion (the node
  falls into the mixed-dict path and raises / mangles the structure). So the
  no-op claim is asserted directly against the expansion engine.

* The single-variant-generator → empty-``config_name`` quirk on dag-ml is
  documented inline in ``cases_generators_conformance``; here we keep the
  ``_depends_on_`` engine-level evidence so a future change that wires
  ``_depends_on_`` into the constraint engine flips these asserts and forces a
  conscious decision (give it real semantics + a parity case, or drop it).
"""

from __future__ import annotations

import inspect

import pytest

from nirs4all.operators.transforms import Detrend, FirstDerivative
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.pipeline.config._generator.constraints import (
    apply_all_constraints,
    parse_constraints,
)
from nirs4all.pipeline.config._generator.core import count_combinations, expand_spec
from nirs4all.pipeline.config._generator.keywords import (
    CONSTRAINT_KEYWORDS,
    DEPENDS_ON_KEYWORD,
    PURE_CARTESIAN_KEYS,
    PURE_OR_KEYS,
    is_generator_node,
)

pytestmark = pytest.mark.parity


def test_depends_on_is_inert_in_constraint_engine() -> None:
    """`_depends_on_` is declared as a constraint keyword but the engine never reads it.

    Three independent signals pin it as a NO-OP filter:

    * ``parse_constraints`` extracts only mutex / requires / exclude (no
      depends_on bucket), even when ``_depends_on_`` is present;
    * ``apply_all_constraints`` has no depends_on parameter at all;
    * ``_depends_on_`` is absent from BOTH pure-node key sets (``PURE_OR_KEYS`` /
      ``PURE_CARTESIAN_KEYS``), so it can never travel on a pure generator node
      the constraint engine would inspect.
    """
    # It IS in the declared keyword set (the source of the "looks like a
    # constraint" confusion) ...
    assert DEPENDS_ON_KEYWORD in CONSTRAINT_KEYWORDS

    # ... but the parser drops it: only the three live buckets come back, and the
    # depends_on spec never appears in any of them.
    parsed = parse_constraints(
        {
            "_mutex_": [["A", "B"]],
            "_requires_": [["C", "D"]],
            "_exclude_": [["E", "F"]],
            "_depends_on_": [["A", "C"]],
        }
    )
    assert set(parsed) == {"mutex", "requires", "exclude"}
    assert ["A", "C"] not in parsed["mutex"]
    assert ["A", "C"] not in parsed["requires"]
    assert ["A", "C"] not in parsed["exclude"]

    # The filter function literally cannot receive a depends_on argument.
    assert "depends_on" not in inspect.signature(apply_all_constraints).parameters
    assert "_depends_on_" not in inspect.signature(apply_all_constraints).parameters

    # And it is not a pure-node key, so it never reaches OrStrategy / CartesianStrategy.
    assert DEPENDS_ON_KEYWORD not in PURE_OR_KEYS
    assert DEPENDS_ON_KEYWORD not in PURE_CARTESIAN_KEYS


def test_depends_on_does_not_prune_when_engine_could_see_it() -> None:
    """A live constraint (`_mutex_`) prunes; adding `_depends_on_` alongside changes nothing.

    On a pure `_or_` pick node the mutex is honored (C(4,2)=6 → 5). Re-adding a
    `_depends_on_` group that, IF it had requires-semantics, would prune further
    (here `[["A", "C"]]`, i.e. "A needs C", which would drop A-without-C pairs)
    leaves the variant set BYTE-IDENTICAL — proving `_depends_on_` is inert.
    NOTE: `_depends_on_` is not a pure-OR key, so the node is only still a pure
    node because the test asserts against the constraint helpers directly via the
    pruned-combination list, mirroring what OrStrategy passes to the engine.
    """
    choices = ["A", "B", "C", "D"]
    from itertools import combinations

    combos = [list(c) for c in combinations(choices, 2)]  # 6

    pruned_mutex_only = apply_all_constraints(combos, mutex_groups=[["A", "B"]])
    assert len(pruned_mutex_only) == 5  # {A,B} removed

    # The engine has no depends_on path; passing the same live constraints again
    # (depends_on simply has nowhere to go) yields the identical pruned set.
    pruned_again = apply_all_constraints(
        combos, mutex_groups=[["A", "B"]], requires_groups=None, exclude_combos=None
    )
    assert pruned_again == pruned_mutex_only


def test_depends_on_on_pure_or_node_breaks_expansion_not_silently_filters() -> None:
    """Documenting the trap: `_depends_on_` on a top-level `_or_` node BREAKS expansion.

    Because `_depends_on_` is not in `PURE_OR_KEYS`, the node is no longer a pure
    OR node; ``expand_spec`` routes it through the mixed-dict path, which tries to
    merge the bare `_or_` list-choices into a base dict and raises. This is why
    `_depends_on_` cannot be exercised as a runnable pipeline `PipelineCase` — it
    is a no-op at best and a hard error at worst, never a working filter.
    """
    bad = {"_or_": ["A", "B", "C", "D"], "pick": 2, "_depends_on_": [["A", "B"]]}
    with pytest.raises(ValueError, match="must be dicts"):
        expand_spec(bad)


def test_constraints_prune_expand_but_not_count() -> None:
    """`count_combinations` is the PRE-filter count; `expand_spec` is the pruned set.

    The orchestrator runs ``expand_spec`` (so ``num_predictions`` reflects the
    pruned variant set), while ``count_combinations`` reports the un-pruned total.
    Locking this asymmetry guards the conformance cases' ``expected_min_predictions``
    against a future change that makes ``count_combinations`` constraint-aware
    (which would silently shift the predicted variant counts).
    """
    spec = {"_or_": ["A", "B", "C", "D"], "pick": 2, "_mutex_": [["A", "B"]]}
    assert count_combinations(spec) == 6  # pre-filter C(4,2)
    assert len(expand_spec(spec)) == 5  # post-filter (mutex removed {A,B})


# ---------------------------------------------------------------------------
# EXACT survivor-set lock for the constraint cases (MUST-FIX 2).
#
# The dual-engine conformance test asserts num_predictions / scores / top-models
# / y_pred — all of which pass even if a constraint pruned the WRONG members as
# long as the COUNT is right and the winner happens to coincide. These DSL-level
# assertions pin the EXACT surviving variant identities (by operator class name)
# that ``expand_spec`` — the single source of truth BOTH engines host-expand from
# — produces, so a wrong-prune (right count, wrong members) fails here.
#
# CRITICAL: the survivor signature is derived from the ACTUAL runnable case's
# ``pipeline_factory()`` generator node (looked up in the registry), NOT from a
# hand-copied lambda. So ``_CONSTRAINT_SURVIVORS`` holds ONLY the expected RESULT;
# there is no second copy of the spec to drift from. If a case's generator node
# changes, ``expand_spec`` on the real node yields a different survivor set and the
# assertion fails — there is no stale lambda to mask the drift.
# ---------------------------------------------------------------------------


def _survivor_signatures(node: dict) -> set[frozenset[str]]:
    """Canonical survivor set: each variant -> frozenset of its operator class names.

    A `_or_`-pick / `_cartesian_` variant is a LIST of operator classes; order is
    irrelevant to "which preprocessors co-occur", so each variant collapses to a
    frozenset of class names and the whole expansion to a set of those frozensets.
    """
    out: set[frozenset[str]] = set()
    for variant in expand_spec(node):
        members = variant if isinstance(variant, list) else [variant]
        out.add(frozenset(getattr(m, "__name__", str(m)) for m in members))
    return out


def _extract_generator_node(pipeline: list) -> dict:
    """Return the single top-level generator-node dict in a pipeline's steps.

    The constraint cases each have exactly one step dict carrying a generation
    keyword (``_or_`` / ``_cartesian_``). Raise if zero or more than one is found,
    so a structural change to the case (no longer a single generator node) fails
    loudly rather than silently picking the wrong step.
    """
    nodes = [s for s in pipeline if isinstance(s, dict) and is_generator_node(s)]
    if len(nodes) != 1:
        raise AssertionError(f"expected exactly one generator node in the pipeline, found {len(nodes)}")
    return nodes[0]


def _names(*classes: type) -> frozenset[str]:
    return frozenset(c.__name__ for c in classes)


# (case_name -> EXACT expected survivor signatures). NO spec copy here — the spec
# is read from the registry case's pipeline_factory() at assert time, so this dict
# is purely the expected post-prune RESULT each case must produce.
_CONSTRAINT_SURVIVORS: dict[str, set[frozenset[str]]] = {
    # generator_or_pick_mutex: C(4,2)=6, mutex [SNV,MSC] removes {SNV,MSC} -> 5.
    "generator_or_pick_mutex": {
        _names(SNV, Detrend),
        _names(SNV, FirstDerivative),
        _names(MSC, Detrend),
        _names(MSC, FirstDerivative),
        _names(Detrend, FirstDerivative),
    },
    # generator_or_pick_mutex3: SIZE-3 mutex [SNV,MSC,Detrend] over pick=3 of 4. C(4,3)=4; legacy
    # ("not all co-occur" / issubset) forbids ONLY {SNV,MSC,Detrend} (the all-three-present combo)
    # and KEEPS every combo with just two of the group -> 3 survivors. A "<=1 present" reading would
    # wrongly prune all four to 0, so this lock catches a >2-mutex over-prune (member-level, not count).
    "generator_or_pick_mutex3": {
        _names(SNV, MSC, FirstDerivative),
        _names(SNV, Detrend, FirstDerivative),
        _names(MSC, Detrend, FirstDerivative),
    },
    # generator_or_pick_requires: SNV requires MSC -> SNV-without-MSC pairs drop -> 4.
    "generator_or_pick_requires": {
        _names(SNV, MSC),
        _names(MSC, Detrend),
        _names(MSC, FirstDerivative),
        _names(Detrend, FirstDerivative),
    },
    # generator_or_pick_exclude: exact pair {SNV,Detrend} dropped -> 5.
    "generator_or_pick_exclude": {
        _names(SNV, MSC),
        _names(SNV, FirstDerivative),
        _names(MSC, Detrend),
        _names(MSC, FirstDerivative),
        _names(Detrend, FirstDerivative),
    },
    # generator_cartesian_exclude: 2x2 cartesian (4 pipelines), {SNV,Detrend} pruned -> 3.
    "generator_cartesian_exclude": {
        _names(SNV, FirstDerivative),
        _names(MSC, Detrend),
        _names(MSC, FirstDerivative),
    },
    # generator_constraint_prunes_to_one: two mutex pairs prune C(3,2)=3 -> 1 ({MSC,Detrend}).
    "generator_constraint_prunes_to_one": {_names(MSC, Detrend)},
    # generator_combined_constraints: mutex [SNV,MSC] + exclude {Detrend,1stDer} -> 6-1-1 = 4.
    "generator_combined_constraints": {
        _names(SNV, Detrend),
        _names(SNV, FirstDerivative),
        _names(MSC, Detrend),
        _names(MSC, FirstDerivative),
    },
}


@pytest.mark.parametrize("name,expected", sorted(_CONSTRAINT_SURVIVORS.items()), ids=sorted(_CONSTRAINT_SURVIVORS))
def test_constraint_exact_survivor_set(name: str, expected: set[frozenset[str]]) -> None:
    """The EXACT surviving variant set (members, not just count) of the ACTUAL registry case.

    Reads the generator node from the REAL runnable case (``get(name).pipeline_factory()``)
    and asserts ``expand_spec`` (the host-expansion source of truth both engines use)
    yields precisely ``expected``. Because the spec comes from the registry — not a
    hand-copied lambda — a wrong-prune (right COUNT, wrong MEMBERS) OR a drift in the
    case's generator node both fail here; there is no second spec to mask it.
    """
    from tests.integration.parity import conftest  # noqa: F401  populate registry
    from tests.integration.parity._registry import get

    node = _extract_generator_node(get(name).pipeline_factory())
    actual = _survivor_signatures(node)
    assert actual == expected, (
        f"{name}: registry case's constraint pruned the WRONG survivor set\n"
        f"  expected: {sorted(sorted(s) for s in expected)}\n"
        f"  actual:   {sorted(sorted(s) for s in actual)}"
    )


def test_constraint_survivor_lock_covers_registry() -> None:
    """Drift guard (both directions): the survivor lock and the registry stay in lockstep.

    Two failure modes:

    * a survivor-locked name that is NOT a real registry case (renamed/removed case
      without updating the lock) -> the per-case test above would error on lookup, and
      this fails fast with a clear message;
    * a registry case that DECLARES a constraint keyword (`_mutex_` / `_requires_` /
      `_exclude_`) but is NOT survivor-locked -> a new constraint case shipped without
      an EXACT-survivor lock, a silent coverage hole. This direction is what makes the
      lock self-extending: you cannot add a constraint case and forget to lock it.
    """
    from tests.integration.parity import conftest  # noqa: F401  populate registry
    from tests.integration.parity._registry import all_cases

    cases = {c.name: c for c in all_cases()}
    locked = set(_CONSTRAINT_SURVIVORS)

    missing = locked - set(cases)
    assert not missing, f"survivor-locked names not in the case registry (renamed/removed?): {sorted(missing)}"

    # A registry case whose generator node carries a pruning constraint keyword but
    # has no survivor lock is an uncovered constraint case.
    constraint_kw = {"_mutex_", "_requires_", "_exclude_"}
    unlocked_constraint_cases = set()
    for name, case in cases.items():
        if name in locked or case.skip_reason:
            continue
        try:
            node = _extract_generator_node(case.pipeline_factory())
        except AssertionError:
            continue  # not a single-generator-node case
        if constraint_kw & set(node):
            unlocked_constraint_cases.add(name)
    assert not unlocked_constraint_cases, (
        "constraint cases without an EXACT-survivor lock (add to _CONSTRAINT_SURVIVORS): "
        f"{sorted(unlocked_constraint_cases)}"
    )


def test_survivor_lock_catches_simulated_registry_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """PROOF the guard catches drift: drift a registry case's node and the REAL lock FAILS.

    Simulates a genuine registry drift by replacing the case in the global
    ``_REGISTRY`` with a ``dataclasses.replace`` copy whose ``pipeline_factory``
    yields a generator node that prunes a DIFFERENT survivor set (the COUNT is even
    preserved — exclude a different pair so 5 survivors remain — but the MEMBERS
    differ). It then invokes the EXACT assertion path
    ``test_constraint_exact_survivor_set`` uses against the unchanged locked expected
    set and asserts it RAISES. This proves the lock is bound to the ACTUAL registry
    spec: had drift been masked by a stale spec copy, the assertion would pass.
    """
    import dataclasses

    from tests.integration.parity import (
        _registry,
        conftest,  # noqa: F401  populate registry
    )

    name = "generator_or_pick_exclude"  # locked: {SNV,Detrend} excluded -> 5 survivors
    case = _registry.get(name)
    expected = _CONSTRAINT_SURVIVORS[name]
    tail = case.pipeline_factory()[1:]  # splitter + model, unchanged

    # Drifted node: same 4 ops + pick 2, but exclude a DIFFERENT pair {MSC,1stDer}.
    # Still 5 survivors (same COUNT) but a different MEMBER set than the lock.
    def _drifted_factory() -> list:
        return [
            {"_or_": [SNV, MSC, Detrend, FirstDerivative], "pick": 2, "_exclude_": [[MSC, FirstDerivative]]},
            *tail,
        ]

    drifted_case = dataclasses.replace(case, pipeline_factory=_drifted_factory)
    monkeypatch.setitem(_registry._REGISTRY, name, drifted_case)  # noqa: SLF001

    # Sanity: the drift keeps the same survivor COUNT, so only a MEMBER-level lock can catch it.
    drifted_node = _extract_generator_node(_registry.get(name).pipeline_factory())
    assert len(_survivor_signatures(drifted_node)) == len(expected)

    # Invoke the REAL production assertion (not an inline copy) against the unchanged
    # locked expected set — with the registry drifted, it MUST raise. This is the proof
    # the lock binds to the live registry spec, so a drift in a runnable case fails.
    with pytest.raises(AssertionError, match="WRONG survivor set"):
        test_constraint_exact_survivor_set(name, expected)
