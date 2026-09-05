#!/usr/bin/env python3
"""Metadata gate: release docs must not drift from package/runtime metadata.

Checks (all sourced from the repo, no hard-coded truth):
  1. VERSION — every ``version = {X}`` citation field and any ``Version: X`` header
     in the checked doc set equals ``nirs4all.__version__`` (parsed from __init__.py).
  2. LICENSE — first-contact pages must NOT state a *single* license that contradicts
     the dual-license statement in ``LICENSE``. The project is dual-licensed (default
     AGPL-3.0-or-later); the legitimate ``CeCILL`` token is NOT banned — only a
     "licensed under the CeCILL-2.1 License" *sole-license* phrasing is rejected.
  3. RELEASE — ``.zenodo.json`` carries the package version and a non-empty title.
  4. DOCKER — README invocations respect the image's exec-form ``python`` entrypoint.

Exit 1 on any violation (suitable as a blocking CI gate). ``--list`` prints details.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# files whose version/license strings are user-facing first-contact surfaces
VERSION_FILES = [
    "docs/source/index.md",
    "docs/source/ai_onboarding.md",
    "README.md",
    "CLAUDE.md",
    "docs/source/conf.py",
]
LICENSE_FILES = [
    "docs/source/index.md",
    "docs/source/ai_onboarding.md",
    "CLAUDE.md",
    "README.md",
]
# sole-license phrasings that contradict the dual license (CeCILL as the ONLY license)
SOLE_CECILL = re.compile(r"licensed under the\s+CeCILL[- ]?2\.1\s+License", re.IGNORECASE)
VERSION_TOKEN = re.compile(
    r"(?:version\s*=\s*\{|[*_]*Version[*_]*\s*[:=]\s*)"
    r"\s*v?(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?)",
    re.IGNORECASE,
)


def package_version() -> str:
    src = (REPO / "nirs4all" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', src)
    if not m:
        raise SystemExit("could not parse nirs4all.__version__")
    return m.group(1)


def check_release_metadata(ver: str, problems: list[str]) -> None:
    """Validate machine-readable release metadata against the package."""
    rel = ".zenodo.json"
    try:
        metadata = json.loads((REPO / rel).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"{rel}: cannot read valid JSON: {exc}")
        return

    if not isinstance(metadata, dict):
        problems.append(f"{rel}: top-level JSON value must be an object")
        return
    if metadata.get("version") != ver:
        problems.append(
            f"{rel}: version '{metadata.get('version')}' != package __version__ '{ver}'"
        )
    title = metadata.get("title")
    if not isinstance(title, str) or not title.strip():
        problems.append(f"{rel}: title must be a non-empty string")


def check_docker_usage(problems: list[str]) -> None:
    """Reject README commands that repeat the image's Python entrypoint."""
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    if 'ENTRYPOINT ["python"]' not in dockerfile:
        return

    rel = "README.md"
    for i, line in enumerate((REPO / rel).read_text(encoding="utf-8").splitlines(), 1):
        if "docker run" in line and "ghcr.io/gbeurier/nirs4all python " in line:
            problems.append(
                f"{rel}:{i}: repeats 'python' already supplied by the Docker ENTRYPOINT"
            )


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ver = package_version()
    problems: list[str] = []

    check_release_metadata(ver, problems)
    check_docker_usage(problems)

    for rel in VERSION_FILES:
        p = REPO / rel
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in VERSION_TOKEN.finditer(line):
                found = m.group(1)
                if found != ver:
                    problems.append(f"{rel}:{i}: version '{found}' != package __version__ '{ver}'")

    for rel in LICENSE_FILES:
        p = REPO / rel
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if SOLE_CECILL.search(line):
                problems.append(
                    f"{rel}:{i}: states CeCILL-2.1 as the *sole* license; project is dual "
                    f"(default AGPL-3.0-or-later). State the dual license (CeCILL token itself is fine)."
                )

    if problems:
        print(f"metadata gate FAILED: {len(problems)} problem(s) (package version = {ver})")
        for pr in problems:
            print("  " + pr)
        return 1
    print(
        f"metadata gate OK: release/docs version == {ver}; Docker usage and license wording valid."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
