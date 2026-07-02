"""Strict gate: the parity suite carries no UNTRACKED xfail / skip / tolerance debt.

Two halves:

* **Live-tree gate** — :func:`_marker_audit.audit_tree` scans every parity module
  and must find zero untracked markers. This is the enforcement that turns the
  ``810 passed / 30 skipped / 11 xfailed`` state into auditable debt: a new bare
  ``pytest.skip`` / ``pytest.mark.xfail`` / loose tolerance fails CI immediately.
* **Negative self-tests** — feed the auditor rogue source and prove it FLAGS the
  debt (otherwise a silently-broken gate would pass the live tree forever).

The auditor is pure ``ast`` over source text, so these tests are fast and need no
engine run.
"""

from __future__ import annotations

from . import _marker_audit as M

_ALLOWED = M.allowed_tolerance_bands()


# ---------------------------------------------------------------------------
# Live-tree gate.
# ---------------------------------------------------------------------------
def test_live_parity_tree_has_no_untracked_marker_debt() -> None:
    result = M.audit_tree()
    assert result.ok(), "untracked parity marker/tolerance debt:\n" + M.format_report(result)


def test_live_xfail_is_confined_to_the_sanctioned_builder() -> None:
    result = M.audit_tree()
    modules = {use.module for use in result.xfail_uses}
    assert modules == {M.SANCTIONED_XFAIL_MODULE}, (
        f"xfail leaked outside {M.SANCTIONED_XFAIL_MODULE}: {sorted(modules)}"
    )


def test_every_live_skip_classifies_into_a_sanctioned_category() -> None:
    result = M.audit_tree()
    untracked = result.untracked_skip()
    assert not untracked, "skips with no sanctioned category:\n" + "\n".join(
        f"  {u.module}:{u.lineno} {u.reason!r}" for u in untracked
    )
    # The tracked-debt (registry) skip category must stay non-empty — it is the
    # 4 fixture/unknown_semantics cases; losing it would mean the debt vanished
    # untracked rather than being fixed.
    assert result.skip_category_counts()["registry_skip"] > 0


def test_live_marker_policy_ledger_matches_enforcement() -> None:
    # Raises AssertionError if compatibility.json's marker_policy drifts from
    # _marker_audit's SANCTIONED_XFAIL_MODULE / SKIP_CATEGORY_IDS / kinds.
    M.validate_marker_policy()


# ---------------------------------------------------------------------------
# Negative self-tests — the gate must FAIL on injected debt.
# ---------------------------------------------------------------------------
def test_gate_flags_rogue_xfail_outside_the_sanctioned_module() -> None:
    src = "import pytest\n@pytest.mark.xfail(reason='TODO', strict=False)\ndef test_x():\n    assert True\n"
    result = M.audit_source("test_rogue_case.py", src, _ALLOWED)
    assert [u.module for u in result.untracked_xfail()] == ["test_rogue_case.py"]
    assert not result.ok()


def test_gate_allows_xfail_inside_the_sanctioned_module() -> None:
    src = "import pytest\nmarks = [pytest.mark.xfail(reason='documented', strict=True)]\n"
    result = M.audit_source(M.SANCTIONED_XFAIL_MODULE, src, _ALLOWED)
    assert result.xfail_uses and not result.untracked_xfail()


def test_gate_flags_bare_untracked_skip() -> None:
    src = "import pytest\ndef test_x():\n    pytest.skip('flaky, revisit later')\n"
    result = M.audit_source("test_x.py", src, _ALLOWED)
    untracked = result.untracked_skip()
    assert len(untracked) == 1 and untracked[0].category == M.UNTRACKED
    assert not result.ok()


def test_gate_classifies_each_sanctioned_skip_shape() -> None:
    # One representative reason per sanctioned category; every one must classify.
    src = (
        "import pytest\n"
        "def test_reg():\n"
        "    pytest.skip('[fixture] no variety column')\n"
        "def test_imp():\n"
        "    pytest.importorskip('dag_ml')\n"
        "@pytest.mark.skipif(not _DAGML_CLI.exists(), reason='dag-ml-cli binary not built at x')\n"
        "def test_cli():\n"
        "    pass\n"
        "def test_dep():\n"
        "    pytest.skip('example on engine=dagml: optional dependency not installed')\n"
        "def test_sib():\n"
        "    pytest.skip('sibling nirs4all-ecosystem runtime schemas not checked out')\n"
        "def test_na():\n"
        "    pytest.skip('c: dag-ml ran the legacy fallback on this build; N/A')\n"
        "def test_cap():\n"
        "    pytest.skip('no gold baseline for c; run with --parity-capture')\n"
        "def test_lock():\n"
        "    pytest.skip('EXPECTED_FALLBACK is empty; LOCK-DROP D1 is closed')\n"
        "def _require_methods_snv_available():\n"
        "    pytest.skip(message)\n"
    )
    result = M.audit_source("test_shapes.py", src, _ALLOWED)
    categories = {u.category for u in result.skip_uses}
    assert M.UNTRACKED not in categories
    assert categories == {
        "registry_skip",
        "optional_env_import",
        "optional_env_dagml_cli",
        "optional_env_dependency",
        "optional_env_sibling",
        "runtime_na",
        "baseline_capture",
        "lockdrop_empty",
        "optional_env_methods",
    }


def test_gate_flags_loose_positive_tolerance_but_exempts_negative_guard() -> None:
    src = (
        "import numpy as np\n"
        "def test_pos():\n"
        "    assert np.allclose(a, b, atol=1e-1)\n"  # off-band positive -> debt
        "def test_neg():\n"
        "    assert not np.allclose(a, b, atol=1e-1)\n"  # divergence floor -> exempt
    )
    result = M.audit_source("test_tol.py", src, _ALLOWED)
    untracked = result.untracked_tolerance()
    assert len(untracked) == 1
    assert untracked[0].value == 1e-1 and not untracked[0].negated
    assert not result.ok()


def test_gate_accepts_in_band_tolerances_and_named_constants() -> None:
    src = (
        "import numpy as np\n"
        "_FIDELITY_TOL = 1e-6\n"
        "def test_ok():\n"
        "    assert np.allclose(a, b, atol=1e-6)\n"
        "    assert dagml == pytest.approx(legacy, abs=1e-3, rel=1e-3)\n"
    )
    result = M.audit_source("test_ok.py", src, _ALLOWED)
    assert result.tolerance_uses  # they were seen
    assert not result.untracked_tolerance()  # and all in-band


def test_ledger_bands_are_the_tolerance_allowlist() -> None:
    # The allowlist is derived from the ledger, not hard-coded — every published
    # positive band is honored and 0.0 / null are excluded.
    assert _ALLOWED == frozenset({1e-12, 1e-9, 1e-6, 1e-3, 5e-3})
