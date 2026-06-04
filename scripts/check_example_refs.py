#!/usr/bin/env python3
"""Broken-example-reference gate: every example file named in the docs must exist.

Doc prose frequently points at ``examples/.../U0X_name.py`` or bare ``U0X_name.py`` /
``D0X_name.py`` / ``R0X_name.py`` filenames. A wrong *path* in prose is not executable,
so doc-snippet execution can never catch it — this gate does.

It scans ``docs/source/**/*.md`` for tokens that look like example filenames or
``examples/...`` paths and verifies each resolves to a real file under ``examples/``.

Exit 1 if any reference is dangling. By default advisory output lists them; pass
``--strict`` to fail (intended to flip on once the P3 backlog of dead pages is removed).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
DOCS = REPO / "docs" / "source"

# bare example filenames: U01_hello_world.py, D05_meta_stacking.py, R04_visualization.py
FNAME = re.compile(r"\b([UDR]\d{2}_[a-zA-Z0-9_]+\.py)\b")
# explicit examples/ paths
PATH = re.compile(r"\bexamples/([A-Za-z0-9_./-]+\.py)\b")


def real_basenames() -> set[str]:
    return {p.name for p in EXAMPLES.rglob("*.py")}


def real_relpaths() -> set[str]:
    return {str(p.relative_to(EXAMPLES)) for p in EXAMPLES.rglob("*.py")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="exit 1 on dangling refs (default: advisory exit 0)")
    args = ap.parse_args()

    names = real_basenames()
    relpaths = real_relpaths()
    dangling: list[str] = []

    for md in sorted(DOCS.rglob("*.md")):
        rel = str(md.relative_to(REPO))
        for i, line in enumerate(md.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for m in PATH.finditer(line):
                cand = m.group(1)
                # accept dir-or-file; normalise trailing nothing
                if cand not in relpaths and not (EXAMPLES / cand).exists():
                    dangling.append(f"{rel}:{i}: examples/{cand} does not exist")
            for m in FNAME.finditer(line):
                fn = m.group(1)
                if fn not in names:
                    dangling.append(f"{rel}:{i}: referenced example '{fn}' does not exist")

    # de-dup
    seen, uniq = set(), []
    for d in dangling:
        if d not in seen:
            seen.add(d)
            uniq.append(d)

    if uniq:
        print(f"broken-example-ref: {len(uniq)} dangling reference(s):")
        for d in uniq:
            print("  " + d)
        return 1 if args.strict else 0
    print("broken-example-ref OK: every example referenced in docs exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
