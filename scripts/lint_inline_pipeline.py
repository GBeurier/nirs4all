#!/usr/bin/env python3
"""Advisory, complexity-aware lint for the nirs4all docs/examples inline-pipeline style.

House style (NOT tyrannical): a *simple* pipeline should be written inline inside the
call — ``nirs4all.run(pipeline=[...], dataset=...)`` — rather than assigned to an
external variable first. A *complex* pipeline (nested branch/merge, large generator
sweeps, many steps) MAY be factored into a named variable; that is exempt.

This lint flags only the clear-cut case: a variable assigned a *simple* list literal
that is later passed as the ``pipeline`` argument (positionally OR by keyword) to
``run`` / ``session`` / ``Session`` / ``PipelineRunner``. It is advisory — a warning,
not a hard gate.

Scope:
    * ``examples/**/*.py``  — parsed as whole modules via AST.
    * ``docs/source/**/*.md`` — each ```python fenced block parsed via AST (best effort).

Exemptions (never flagged):
    * complex literals: > MAX_SIMPLE_STEPS top-level steps, or any step that is a dict
      whose key is a branching/generator keyword (branch, merge, _cartesian_, _or_,
      _grid_, _chain_, _zip_, _sample_, _range_, _log_range_), or nested lists.
    * list-of-pipelines reuse: ``run(pipeline=[a, b, c])`` where the elements are names.
    * a name passed to the pipeline arg of MORE THAN ONE call (deliberate reuse).
    * any line carrying ``# noqa: inline-pipeline``.

Usage:
    python scripts/lint_inline_pipeline.py [--list] [paths...]
Exit code is always 0 (advisory). Use --strict to exit 1 when violations exist.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_CALLERS = {"run", "session", "Session", "PipelineRunner"}
BRANCHING_KEYS = {
    "branch", "merge", "_cartesian_", "_or_", "_grid_", "_chain_",
    "_zip_", "_sample_", "_range_", "_log_range_", "concat_transform",
}
MAX_SIMPLE_STEPS = 6  # a flat list with <= this many steps is "simple"
NOQA = "# noqa: inline-pipeline"


@dataclass
class Violation:
    path: str
    line: int  # line of the assignment (within the file)
    name: str
    n_steps: int
    detail: str


def _is_pipeline_name(name: str) -> bool:
    return "pipeline" in name.lower()


def _literal_is_complex(node: ast.AST) -> tuple[bool, int]:
    """Return (is_complex, n_top_level_steps) for a list-literal pipeline value."""
    if not isinstance(node, ast.List):
        return True, 0  # not a plain list literal -> treat as complex/skip
    n = len(node.elts)
    if n > MAX_SIMPLE_STEPS:
        return True, n
    for el in node.elts:
        # nested list (e.g. cartesian stage groups) -> complex
        if isinstance(el, (ast.List, ast.Tuple)):
            return True, n
        # dict step with a branching/generator key -> complex
        if isinstance(el, ast.Dict):
            for k in el.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value in BRANCHING_KEYS:
                    return True, n
    return False, n


def _list_of_names(node: ast.AST) -> bool:
    """True if a list literal whose elements are all bare names (list-of-pipelines)."""
    return isinstance(node, ast.List) and len(node.elts) > 0 and all(
        isinstance(e, ast.Name) for e in node.elts
    )


def _pipeline_arg_names(call: ast.Call) -> list[ast.Name]:
    """Names passed as the `pipeline` argument of a run-like call (positional or keyword)."""
    names: list[ast.Name] = []
    func = call.func
    fname = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
    if fname not in RUN_CALLERS:
        return names
    # keyword pipeline=
    for kw in call.keywords:
        if kw.arg == "pipeline" and isinstance(kw.value, ast.Name):
            names.append(kw.value)
    # positional first arg (run(pipeline, dataset, ...) / PipelineRunner(pipeline))
    if call.args and isinstance(call.args[0], ast.Name):
        names.append(call.args[0])
    return names


def _analyze_module(tree: ast.Module, path: str, line_offset: int, noqa_lines: set[int]) -> list[Violation]:
    # 1) collect simple pipeline-named list-literal assignments: name -> (lineno, node, n_steps)
    assigns: dict[str, tuple[int, ast.AST, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if _is_pipeline_name(name) and isinstance(node.value, ast.List) and not _list_of_names(node.value):
                complex_, n = _literal_is_complex(node.value)
                if not complex_:
                    assigns[name] = (node.lineno, node.value, n)
    if not assigns:
        return []
    # 2) count how many run-like calls consume each name as the pipeline arg
    consumed: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for nm in _pipeline_arg_names(node):
                if nm.id in assigns:
                    consumed[nm.id] = consumed.get(nm.id, 0) + 1
    out: list[Violation] = []
    for name, count in consumed.items():
        if count != 1:  # reuse across multiple calls -> exempt
            continue
        lineno, _node, n = assigns[name]
        abs_line = line_offset + lineno - 1
        if abs_line in noqa_lines:
            continue
        out.append(Violation(path, abs_line, name, n, f"simple {n}-step pipeline assigned then passed once"))
    return out


_FENCE = re.compile(r"^([ \t]*)```+\s*(python|py)\s*$", re.IGNORECASE)
_FENCE_END = re.compile(r"^[ \t]*```+\s*$")


def _iter_md_python_blocks(text: str):
    """Yield (start_line_1based, code) for each ```python fenced block."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if m:
            indent = len(m.group(1))
            start = i + 1
            body = []
            j = start
            while j < len(lines) and not _FENCE_END.match(lines[j]):
                body.append(lines[j][indent:] if lines[j][:indent].strip() == "" else lines[j])
                j += 1
            yield start + 1, "\n".join(body)
            i = j + 1
        else:
            i += 1


def lint_file(path: Path) -> list[Violation]:
    rel = str(path.relative_to(REPO))
    text = path.read_text(encoding="utf-8", errors="replace")
    noqa_lines = {n for n, ln in enumerate(text.splitlines(), 1) if NOQA in ln}
    violations: list[Violation] = []
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        violations += _analyze_module(tree, rel, 1, noqa_lines)
    elif path.suffix == ".md":
        for start_line, code in _iter_md_python_blocks(text):
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue  # partial snippet — skip silently
            violations += _analyze_module(tree, rel, start_line, noqa_lines)
    return violations


def collect_targets(paths: list[str]) -> list[Path]:
    if paths:
        out: list[Path] = []
        for p in paths:
            pp = Path(p).resolve()
            if pp.is_dir():
                out += [q for q in sorted(pp.rglob("*.py")) if "__pycache__" not in q.parts]
                out += sorted(pp.rglob("*.md"))
            else:
                out.append(pp)
        return out
    targets: list[Path] = []
    targets += sorted((REPO / "docs" / "source").rglob("*.md"))
    ex = REPO / "examples"
    for p in sorted(ex.rglob("*.py")):
        if "__pycache__" in p.parts or "_internal" in p.parts:
            continue
        targets.append(p)
    return targets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="files to lint (default: docs/source + examples)")
    ap.add_argument("--list", action="store_true", help="print each violation as file:line")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any violations (default: advisory, exit 0)")
    args = ap.parse_args()

    all_viol: list[Violation] = []
    for path in collect_targets(args.paths):
        if path.exists():
            all_viol += lint_file(path)

    by_file: dict[str, int] = {}
    for v in all_viol:
        by_file[v.path] = by_file.get(v.path, 0) + 1
        if args.list:
            print(f"{v.path}:{v.line}: inline-pipeline: '{v.name}' ({v.detail})")

    print(f"\nadvisory: {len(all_viol)} simple-case external-variable pipeline(s) in {len(by_file)} file(s)")
    if by_file and not args.list:
        for f, n in sorted(by_file.items(), key=lambda x: -x[1])[:15]:
            print(f"  {n:3}  {f}")
    return 1 if (args.strict and all_viol) else 0


if __name__ == "__main__":
    sys.exit(main())
