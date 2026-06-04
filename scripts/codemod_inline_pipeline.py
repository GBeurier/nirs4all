#!/usr/bin/env python3
"""Codemod: inline *simple* external-variable pipelines in .py example files.

Transforms the simple-case anti-pattern detected by ``lint_inline_pipeline.py``:

    pipeline = [SNV(), {"model": PLSRegression(10)}]      # assignment (deleted)
    result = nirs4all.run(pipeline=pipeline, dataset=ds)  # -> pipeline=[SNV(), {"model": PLSRegression(10)}]

It ONLY touches a name that (a) has a pipeline-ish name, (b) is bound once to a *simple*
list literal (per the lint's complexity rule), and (c) is passed exactly once as the
``pipeline`` arg (positional or keyword) of a run-like call. Everything else is left
untouched. The literal's original source text (including multi-line layout) is preserved.

Semantics are identical: the same list object is passed to ``run``. Usage:
    python scripts/codemod_inline_pipeline.py [--apply] [paths...]   # default: dry-run diff
"""
from __future__ import annotations

import argparse
import ast
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_inline_pipeline import (  # noqa: E402
    NOQA,
    REPO,
    RUN_CALLERS,
    _is_pipeline_name,
    _list_of_names,
    _literal_is_complex,
    _pipeline_arg_names,
)


def _plan_file(text: str):
    """Return list of edits: (assign_node, call_node, name, value_node). Empty if none."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    assigns: dict[str, ast.Assign] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if _is_pipeline_name(name) and isinstance(node.value, ast.List) and not _list_of_names(node.value):
                complex_, _ = _literal_is_complex(node.value)
                if not complex_:
                    # only the FIRST binding; if rebound, bail on that name
                    assigns[name] = None if name in assigns else node
    assigns = {k: v for k, v in assigns.items() if v is not None}
    # find single-consumer run-like calls
    consumers: dict[str, list[tuple[ast.Call, ast.Name]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for nm in _pipeline_arg_names(node):
                if nm.id in assigns:
                    consumers.setdefault(nm.id, []).append((node, nm))
    edits = []
    for name, uses in consumers.items():
        # also count ANY reference to the name to ensure it's used ONLY as this pipeline arg
        all_refs = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
        if len(uses) == 1 and len(all_refs) == 1:
            call, nm = uses[0]
            edits.append((assigns[name], call, name, assigns[name].value, nm))
    return edits


def _noqa_lines(text: str) -> set[int]:
    return {i for i, ln in enumerate(text.splitlines(), 1) if NOQA in ln}


def _reindent_literal(literal_src: str, shift: int) -> str:
    """Re-indent continuation lines of a multi-line list literal by `shift` columns."""
    lit_lines = literal_src.split("\n")
    if len(lit_lines) == 1:
        return literal_src
    out = [lit_lines[0]]
    for ln in lit_lines[1:]:
        if shift >= 0:
            out.append(" " * shift + ln)
        else:
            strip = min(-shift, len(ln) - len(ln.lstrip()))
            out.append(ln[strip:])
    return "\n".join(out)


def transform(text: str) -> tuple[str, int]:
    edits = _plan_file(text)
    if not edits:
        return text, 0
    noqa = _noqa_lines(text)
    lines = text.splitlines(keepends=False)
    ops = []  # ('sub', line, c0, c1, payload) | ('del', start, end)
    count = 0
    for assign, _call, _name, value, nm in edits:
        if assign.lineno in noqa:
            continue
        literal_src = ast.get_source_segment(text, value)
        if literal_src is None or nm.lineno != nm.end_lineno:
            continue
        call_line = lines[nm.lineno - 1]
        arg_indent = len(call_line) - len(call_line.lstrip())
        shift = arg_indent - assign.col_offset
        payload = _reindent_literal(literal_src, shift)
        ops.append(("sub", nm.lineno, nm.col_offset, nm.end_col_offset, payload))
        ops.append(("del", assign.lineno, assign.end_lineno))
        count += 1
    if count == 0:
        return text, 0
    # Apply bottom-up so earlier (lower-line) edits keep valid line numbers.
    ops.sort(key=lambda o: -o[1])
    for op in ops:
        if op[0] == "sub":
            _, ln, c0, c1, payload = op
            line = lines[ln - 1]
            lines[ln - 1] = line[:c0] + payload + line[c1:]
        else:
            _, start, end = op
            del lines[start - 1:end]
    new_text = "\n".join(lines)
    if text.endswith("\n"):
        new_text += "\n"
    return new_text, count


_FENCE_RE = __import__("re").compile(r"^```+\s*(python|py)\s*$", __import__("re").IGNORECASE)
_FENCE_END_RE = __import__("re").compile(r"^```+\s*$")


def transform_md(text: str) -> tuple[str, int]:
    """Inline simple pipelines inside top-level ```python fenced blocks of a markdown file."""
    lines = text.splitlines(keepends=False)
    out: list[str] = []
    total = 0
    i = 0
    while i < len(lines):
        if _FENCE_RE.match(lines[i]):  # only indent-0 fences (common case)
            start = i
            j = i + 1
            while j < len(lines) and not _FENCE_END_RE.match(lines[j]):
                j += 1
            body = "\n".join(lines[start + 1:j])
            new_body, n = transform(body)
            out.append(lines[start])
            out.extend(new_body.split("\n") if n else lines[start + 1:j])
            if j < len(lines):
                out.append(lines[j])
            total += n
            i = j + 1
        else:
            out.append(lines[i])
            i += 1
    new_text = "\n".join(out)
    if text.endswith("\n"):
        new_text += "\n"
    return new_text, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run diff)")
    args = ap.parse_args()
    if args.paths:
        targets = [Path(p).resolve() for p in args.paths]
    else:
        targets = [
            p for p in sorted((REPO / "examples").rglob("*.py"))
            if "__pycache__" not in p.parts and "_internal" not in p.parts
        ]
        targets += sorted((REPO / "docs" / "source").rglob("*.md"))
    total = 0
    changed_files = 0
    for path in targets:
        if not path.exists() or path.suffix not in (".py", ".md"):
            continue
        src = path.read_text(encoding="utf-8")
        new, n = (transform_md(src) if path.suffix == ".md" else transform(src))
        if n and new != src:
            total += n
            changed_files += 1
            if args.apply:
                path.write_text(new, encoding="utf-8")
                print(f"inlined {n} in {path.relative_to(REPO)}")
            else:
                diff = difflib.unified_diff(
                    src.splitlines(True), new.splitlines(True),
                    fromfile=str(path.relative_to(REPO)), tofile=str(path.relative_to(REPO)),
                )
                sys.stdout.writelines(diff)
    print(f"\n{'APPLIED' if args.apply else 'DRY-RUN'}: {total} inlining(s) across {changed_files} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
