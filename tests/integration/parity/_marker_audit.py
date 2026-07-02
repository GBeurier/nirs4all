"""Static xfail / skip / tolerance debt audit for the parity suite (RC-B gate).

``_authority.py`` reconciles ``docs/compatibility.json`` against the live parity
CONSTANTS (``KNOWN_DIVERGENCES`` / ``NUM_PREDICTIONS_DIVERGENCE`` /
``Y_PRED_TOL_OVERRIDES`` / registry ``skip_kind``). That proves the *ledgered*
debt is internally consistent, but it is blind to debt added DIRECTLY in a test
body — a bare ``pytest.mark.xfail(reason="TODO")``, a ``pytest.skip("flaky")``,
or an ``atol=1e-1`` — which never touches those constants. This module is the
missing half: a pure-``ast`` scan of every ``tests/integration/parity/*.py`` that
fails on the first *untracked* marker or tolerance.

Three closed policies (each fails on the first item it cannot place):

1. **xfail containment.** ``pytest.mark.xfail`` / ``pytest.xfail`` may appear
   ONLY in :data:`SANCTIONED_XFAIL_MODULE` — the two ``_params()`` marks driven
   by ``KNOWN_DIVERGENCES`` + the registry ``legacy_bug`` cases. Any xfail
   anywhere else is an untracked cross-engine divergence and the ``6 xfailed``
   headline can no longer be trusted.
2. **skip taxonomy.** Every ``pytest.skip`` / ``pytest.mark.skip`` /
   ``pytest.mark.skipif`` / ``pytest.importorskip`` must classify into exactly
   one sanctioned category (:data:`SKIP_CATEGORY_IDS`). An unclassifiable skip is
   untracked coverage loss — a blocker under the RC-B policy (skips are release
   blockers unless they are optional-environment or tracked registry debt).
3. **tolerance band.** Every explicit numeric ``atol`` / ``rtol`` / ``abs`` /
   ``rel`` kwarg, every ``*_TOL`` constant, and every ``metric_tolerances`` /
   ``Y_PRED_TOL_OVERRIDES`` value must equal a ledgered tolerance band
   (``docs/compatibility.json`` ``tolerance_bands[].abs_tol``) — UNLESS it sits
   in a NEGATIVE assertion (``assert not np.allclose(...)`` / ``!= approx(...)``),
   which is a divergence *floor*, not a tolerance *ceiling*, and so is exempt.

The audit is intentionally line-number-independent: it keys on module + reason
template + tolerance value, so ordinary edits never churn it, but a new class of
debt always trips it.

CLI (mirrors ``coverage_meter``)::

    python -m tests.integration.parity._marker_audit           # human report
    python -m tests.integration.parity._marker_audit --check   # exit 1 on debt
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_DIR = Path(__file__).resolve().parent
COMPATIBILITY_JSON = REPO_ROOT / "docs" / "compatibility.json"

#: The single module allowed to carry ``pytest.mark.xfail`` — the collection-time
#: ``_params()`` builder that turns ``KNOWN_DIVERGENCES`` + registry ``legacy_bug``
#: into strict xfails. Every strict xfail in the suite flows through here.
SANCTIONED_XFAIL_MODULE = "test_conformance_dual_engine.py"

#: Ordered, closed set of skip categories. ``registry_skip`` is tracked coverage
#: debt (the ledger ``coverage_skips``); the ``optional_env_*`` family is allowed
#: (optional dependency / binary / sibling checkout); ``runtime_na`` guards a
#: cross-engine comparison whose precondition (a single-artifact NATIVE dag-ml
#: run) is absent on this build; ``baseline_capture`` / ``lockdrop_empty`` are
#: workflow guards. Anything else is ``UNTRACKED``.
SKIP_CATEGORY_IDS: tuple[str, ...] = (
    "registry_skip",
    "optional_env_import",
    "optional_env_dagml_cli",
    "optional_env_dependency",
    "optional_env_sibling",
    "optional_env_methods",
    "runtime_na",
    "baseline_capture",
    "lockdrop_empty",
)

#: Coarse "kind" per category, surfaced to auditors so a reviewer sees at a glance
#: whether a skip is tracked debt, an optional-environment allowance, a runtime
#: precondition guard, or a workflow guard.
SKIP_CATEGORY_KIND: dict[str, str] = {
    "registry_skip": "tracked_debt",
    "optional_env_import": "optional_env",
    "optional_env_dagml_cli": "optional_env",
    "optional_env_dependency": "optional_env",
    "optional_env_sibling": "optional_env",
    "optional_env_methods": "optional_env",
    "runtime_na": "runtime_precondition",
    "baseline_capture": "workflow",
    "lockdrop_empty": "workflow",
}

UNTRACKED = "UNTRACKED"

# The one function whose skip carries a computed (non-literal) reason string; it
# guards the optional nirs4all-methods SNV binding, so it is an optional-env skip.
_METHODS_GUARD_FUNC = "_require_methods_snv_available"


# ---------------------------------------------------------------------------
# Records.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class XfailUse:
    """One ``pytest.mark.xfail`` / ``pytest.xfail`` call expression."""

    module: str
    lineno: int
    func: str
    dotted: str

    @property
    def sanctioned(self) -> bool:
        return self.module == SANCTIONED_XFAIL_MODULE


@dataclass(frozen=True)
class SkipUse:
    """One skip-family call expression, classified into a sanctioned category."""

    module: str
    lineno: int
    func: str
    kind: str  # "skip" | "mark.skip" | "skipif" | "importorskip"
    reason: str  # reconstructed literal template ("" when non-literal)
    category: str

    @property
    def tracked(self) -> bool:
        return self.category != UNTRACKED


@dataclass(frozen=True)
class ToleranceUse:
    """One explicit numeric tolerance literal."""

    module: str
    lineno: int
    func: str
    source: str  # "kwarg:atol" | "const" | "metric_tolerances" | "override"
    value: float
    negated: bool
    in_band: bool

    @property
    def ok(self) -> bool:
        # A negative assertion (assert not allclose / != approx) is a divergence
        # floor, not a tolerance ceiling, so any value is acceptable there.
        return self.in_band or self.negated


@dataclass(frozen=True)
class AuditResult:
    """The full static inventory + its pass/fail verdicts."""

    xfail_uses: tuple[XfailUse, ...] = field(default_factory=tuple)
    skip_uses: tuple[SkipUse, ...] = field(default_factory=tuple)
    tolerance_uses: tuple[ToleranceUse, ...] = field(default_factory=tuple)

    def untracked_xfail(self) -> list[XfailUse]:
        return [u for u in self.xfail_uses if not u.sanctioned]

    def untracked_skip(self) -> list[SkipUse]:
        return [u for u in self.skip_uses if not u.tracked]

    def untracked_tolerance(self) -> list[ToleranceUse]:
        return [u for u in self.tolerance_uses if not u.ok]

    def ok(self) -> bool:
        return not (self.untracked_xfail() or self.untracked_skip() or self.untracked_tolerance())

    def skip_category_counts(self) -> dict[str, int]:
        counts = dict.fromkeys((*SKIP_CATEGORY_IDS, UNTRACKED), 0)
        for use in self.skip_uses:
            counts[use.category] += 1
        return counts


# ---------------------------------------------------------------------------
# AST helpers.
# ---------------------------------------------------------------------------
def _dotted(node: ast.AST) -> str:
    """Reconstruct a dotted call target (``pytest.mark.xfail``, ``np.allclose``)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _tail(dotted: str) -> str:
    return dotted.rsplit(".", 1)[-1]


def _literal_template(node: ast.AST | None) -> str:
    """Best-effort literal text of a reason argument.

    A plain string yields itself; an f-string yields its literal segments with
    ``{}`` standing in for interpolations; anything else (a bare ``Name``) yields
    ``""`` so the classifier falls back to structural context.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out: list[str] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            else:
                out.append("{}")
        return "".join(out)
    return ""


def _numeric(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    return None


def _reason_kwarg(call: ast.Call) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == "reason":
            return kw.value
    return None


def classify_skip(*, kind: str, reason: str, func: str, condition_src: str) -> str:
    """Map a skip occurrence to exactly one :data:`SKIP_CATEGORY_IDS` or ``UNTRACKED``.

    Precedence is deliberate: ``importorskip`` is always optional-env; a dag-ml-cli
    guard is recognised by its condition OR its reason; the registry bracket
    (``[fixture] ...``) is the tracked-debt marker; the rest match on a stable
    reason fragment; the single non-literal skip is placed by its guard function.
    """
    if kind == "importorskip":
        return "optional_env_import"
    if "_DAGML_CLI" in condition_src or "dag-ml-cli binary not built" in reason:
        return "optional_env_dagml_cli"
    if reason.startswith("["):
        return "registry_skip"
    if "optional dependency not installed" in reason:
        return "optional_env_dependency"
    if "runtime schemas not checked out" in reason:
        return "optional_env_sibling"
    if "EXPECTED_FALLBACK is empty" in reason:
        return "lockdrop_empty"
    if "captured legacy baseline" in reason or "no gold baseline" in reason:
        return "baseline_capture"
    if any(
        fragment in reason
        for fragment in (
            "legacy fallback",
            "did not write native results",
            "not a single-artifact native run",
            "no native triple to read",
        )
    ):
        return "runtime_na"
    if func == _METHODS_GUARD_FUNC:
        return "optional_env_methods"
    return UNTRACKED


class _ModuleAuditor(ast.NodeVisitor):
    """Collect xfail / skip / tolerance uses from one parsed module."""

    def __init__(self, module: str, allowed_tolerances: frozenset[float]) -> None:
        self.module = module
        self.allowed = allowed_tolerances
        self.func_stack: list[str] = []
        self._negated = False
        self.xfail: list[XfailUse] = []
        self.skip: list[SkipUse] = []
        self.tol: list[ToleranceUse] = []

    @property
    def _func(self) -> str:
        return self.func_stack[-1] if self.func_stack else "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):
            self._visit_negated(node.operand)
        else:
            self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(isinstance(op, (ast.NotEq, ast.IsNot)) for op in node.ops):
            prev, self._negated = self._negated, True
            self.generic_visit(node)
            self._negated = prev
        else:
            self.generic_visit(node)

    def _visit_negated(self, node: ast.AST) -> None:
        prev, self._negated = self._negated, True
        self.visit(node)
        self._negated = prev

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._record_tolerance_assignment(target.id, node.value, node.lineno)
        self.generic_visit(node)

    def _record_tolerance_assignment(self, name: str, value: ast.AST, lineno: int) -> None:
        if name.endswith("_TOL"):
            number = _numeric(value)
            if number is not None:
                self._add_tolerance(lineno, "const", number, negated=False)
        elif name == "Y_PRED_TOL_OVERRIDES" and isinstance(value, ast.Dict):
            for item in value.values:
                number = _numeric(item)
                if number is not None:
                    self._add_tolerance(lineno, "override", number, negated=False)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted(node.func)
        tail = _tail(dotted)

        if dotted in {"pytest.mark.xfail", "pytest.xfail"}:
            self.xfail.append(XfailUse(self.module, node.lineno, self._func, dotted))
        elif dotted in {"pytest.skip", "pytest.mark.skip"}:
            self._record_skip(node, kind="skip" if dotted == "pytest.skip" else "mark.skip")
        elif dotted == "pytest.mark.skipif":
            self._record_skip(node, kind="skipif")
        elif dotted == "pytest.importorskip":
            self._record_skip(node, kind="importorskip")
        elif tail in {"allclose", "isclose"}:
            self._record_call_tolerances(node, ("atol", "rtol"))
        elif tail == "approx":
            self._record_call_tolerances(node, ("abs", "rel"))

        # metric_tolerances={...} passed to PipelineCase(...) etc.
        for kw in node.keywords:
            if kw.arg == "metric_tolerances" and isinstance(kw.value, ast.Dict):
                for item in kw.value.values:
                    number = _numeric(item)
                    if number is not None:
                        self._add_tolerance(node.lineno, "metric_tolerances", number, negated=False)

        self.generic_visit(node)

    def _record_skip(self, node: ast.Call, *, kind: str) -> None:
        if kind in {"skip", "mark.skip"}:
            arg = node.args[0] if node.args else _reason_kwarg(node)
            reason = _literal_template(arg)
            condition_src = ""
        elif kind == "skipif":
            condition_src = ast.unparse(node.args[0]) if node.args else ""
            reason = _literal_template(_reason_kwarg(node))
        else:  # importorskip
            reason = _literal_template(_reason_kwarg(node))
            condition_src = ""
        category = classify_skip(kind=kind, reason=reason, func=self._func, condition_src=condition_src)
        self.skip.append(SkipUse(self.module, node.lineno, self._func, kind, reason, category))

    def _record_call_tolerances(self, node: ast.Call, kwargs: tuple[str, ...]) -> None:
        for kw in node.keywords:
            if kw.arg in kwargs:
                number = _numeric(kw.value)
                if number is not None:
                    self._add_tolerance(node.lineno, f"kwarg:{kw.arg}", number, negated=self._negated)

    def _add_tolerance(self, lineno: int, source: str, value: float, *, negated: bool) -> None:
        self.tol.append(
            ToleranceUse(self.module, lineno, self._func, source, value, negated, self._in_band(value))
        )

    def _in_band(self, value: float) -> bool:
        return any(abs(value - band) <= 1e-18 or value == band for band in self.allowed)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def allowed_tolerance_bands(ledger: dict[str, Any] | None = None) -> frozenset[float]:
    """Positive ``abs_tol`` values from the ledger — the tolerance allowlist.

    The ledger's tolerance bands ARE the source of truth: a test tolerance is
    sanctioned iff it equals a published band. ``0.0`` (exact classification) and
    ``null`` (semantic / RNG) are not tolerance magnitudes and are excluded.
    """
    data = ledger if ledger is not None else _load_ledger()
    values: set[float] = set()
    for band in data["tolerance_bands"]:
        abs_tol = band.get("abs_tol")
        if isinstance(abs_tol, (int, float)) and not isinstance(abs_tol, bool) and abs_tol > 0:
            values.add(float(abs_tol))
    return frozenset(values)


def audit_source(module: str, source: str, allowed_tolerances: frozenset[float]) -> AuditResult:
    """Audit a single module given as source text (used by the negative self-tests)."""
    auditor = _ModuleAuditor(module, allowed_tolerances)
    auditor.visit(ast.parse(source))
    return AuditResult(tuple(auditor.xfail), tuple(auditor.skip), tuple(auditor.tol))


def audit_tree(paths: Iterable[Path] | None = None, allowed_tolerances: frozenset[float] | None = None) -> AuditResult:
    """Audit every ``*.py`` under the parity directory (or an explicit path list)."""
    allowed = allowed_tolerances if allowed_tolerances is not None else allowed_tolerance_bands()
    files = sorted(paths) if paths is not None else sorted(PARITY_DIR.glob("*.py"))
    xfail: list[XfailUse] = []
    skip: list[SkipUse] = []
    tol: list[ToleranceUse] = []
    for path in files:
        auditor = _ModuleAuditor(path.name, allowed)
        auditor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        xfail.extend(auditor.xfail)
        skip.extend(auditor.skip)
        tol.extend(auditor.tol)
    return AuditResult(tuple(xfail), tuple(skip), tuple(tol))


def validate_marker_policy(ledger: dict[str, Any] | None = None) -> None:
    """Assert the ledger's ``marker_policy`` face matches this module's enforcement.

    Keeps ``docs/compatibility.json`` from drifting away from the code that
    actually enforces it (the same contract ``_authority.py`` upholds for the
    numeric ledger).
    """
    data = ledger if ledger is not None else _load_ledger()
    policy = data.get("marker_policy")
    if not isinstance(policy, dict):
        raise AssertionError("compatibility.json is missing the 'marker_policy' object")

    modules = policy.get("xfail", {}).get("sanctioned_modules")
    if modules != [SANCTIONED_XFAIL_MODULE]:
        raise AssertionError(
            f"marker_policy.xfail.sanctioned_modules drift: ledger={modules} enforced=[{SANCTIONED_XFAIL_MODULE!r}]"
        )

    ledger_cats = [row["id"] for row in policy.get("skip_categories", [])]
    if ledger_cats != list(SKIP_CATEGORY_IDS):
        raise AssertionError(
            f"marker_policy.skip_categories drift: ledger={ledger_cats} enforced={list(SKIP_CATEGORY_IDS)}"
        )
    for row in policy["skip_categories"]:
        expected_kind = SKIP_CATEGORY_KIND[row["id"]]
        if row.get("kind") != expected_kind:
            raise AssertionError(
                f"marker_policy skip category {row['id']!r} kind drift: ledger={row.get('kind')!r} enforced={expected_kind!r}"
            )


def format_report(result: AuditResult) -> str:
    """Human-readable summary (CLI + assertion messages)."""
    lines = [
        "# parity marker & tolerance audit (RC-B)",
        "",
        f"xfail call sites: {len(result.xfail_uses)} (sanctioned module: {SANCTIONED_XFAIL_MODULE})",
        f"skip call sites:  {len(result.skip_uses)}",
        f"tolerance literals: {len(result.tolerance_uses)}",
        "",
        "## skip categories",
    ]
    counts = result.skip_category_counts()
    for cat in (*SKIP_CATEGORY_IDS, UNTRACKED):
        if counts[cat]:
            kind = SKIP_CATEGORY_KIND.get(cat, "untracked")
            lines.append(f"- {cat} ({kind}): {counts[cat]}")

    for label, rows in (
        ("UNTRACKED xfail", result.untracked_xfail()),
        ("UNTRACKED skip", result.untracked_skip()),
        ("UNTRACKED tolerance", result.untracked_tolerance()),
    ):
        if rows:
            lines.extend(["", f"## {label}"])
            for row in rows:
                detail = getattr(row, "reason", None) or getattr(row, "value", "")
                lines.append(f"- {row.module}:{row.lineno} in {row.func}() → {detail!r}")
    lines.append("")
    lines.append("VERDICT: " + ("OK" if result.ok() else "DEBT DETECTED"))
    return "\n".join(lines) + "\n"


def _load_ledger(path: Path = COMPATIBILITY_JSON) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return cast("dict[str, Any]", json.load(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static parity marker & tolerance debt audit (RC-B gate).")
    parser.add_argument("--check", action="store_true", help="exit 1 if any untracked debt is found")
    parser.add_argument("--json", type=Path, default=None, help="write the full inventory JSON to this path")
    args = parser.parse_args(argv)

    result = audit_tree()
    if args.json is not None:
        payload = {
            "xfail": [vars(u) for u in result.xfail_uses],
            "skip": [vars(u) for u in result.skip_uses],
            "tolerance": [vars(u) for u in result.tolerance_uses],
            "skip_category_counts": result.skip_category_counts(),
            "ok": result.ok(),
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(format_report(result))
    if args.check and not result.ok():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
