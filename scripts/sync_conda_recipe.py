#!/usr/bin/env python3
"""Sync and validate conda-forge meta.yaml against pyproject.toml dependencies.

Usage:
    python scripts/sync_conda_recipe.py              # Report discrepancies
    python scripts/sync_conda_recipe.py --update      # Update meta.yaml in place
    python scripts/sync_conda_recipe.py --sha VERSION # Fetch SHA256 from PyPI
"""

import argparse
import json
import re
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
META_YAML = ROOT / "conda-forge" / "meta.yaml"

# PyPI package name -> conda-forge package name
PYPI_TO_CONDA = {
    "duckdb": "python-duckdb",
    "matplotlib": "matplotlib-base",
    "pywavelets": "pywavelets",  # PyPI: PyWavelets, conda: pywavelets
}


def parse_pyproject() -> dict:
    """Parse pyproject.toml and extract version + dependencies."""
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    project = data["project"]
    return {
        "version": project["version"],
        "dependencies": project.get("dependencies", []),
    }


def parse_dep(dep_str: str) -> tuple[str, str]:
    """Parse 'package>=1.0.0' into ('package', '>=1.0.0')."""
    match = re.match(r"^([a-zA-Z0-9_-]+)\s*(.*)", dep_str.strip())
    if not match:
        return dep_str.strip(), ""
    return match.group(1), match.group(2).strip()


def normalize_name(name: str) -> str:
    """Normalize package name for comparison (lowercase, hyphens to underscores)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def conda_name(pypi_name: str) -> str:
    """Map a PyPI package name to its conda-forge equivalent."""
    normalized = normalize_name(pypi_name)
    return PYPI_TO_CONDA.get(normalized, normalized)


def parse_meta_yaml_deps() -> list[tuple[str, str]]:
    """Extract run dependencies from meta.yaml (simple regex-based parser)."""
    text = META_YAML.read_text(encoding="utf-8")
    in_run = False
    deps = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("run:"):
            in_run = True
            continue
        if in_run:
            if stripped.startswith("- "):
                dep = stripped[2:].strip()
                if dep.startswith("python"):
                    continue
                name, version = parse_dep(dep)
                deps.append((name, version))
            elif stripped and not stripped.startswith("#"):
                # New section
                if not stripped.startswith("-"):
                    break
    return deps


def parse_meta_yaml_version() -> str:
    """Extract version from meta.yaml Jinja2 template."""
    text = META_YAML.read_text(encoding="utf-8")
    match = re.search(r'{%\s*set\s+version\s*=\s*"([^"]+)"', text)
    return match.group(1) if match else "unknown"


def compare(pyproject: dict) -> list[str]:
    """Compare pyproject.toml deps against meta.yaml and report issues."""
    issues = []

    # Version check
    meta_version = parse_meta_yaml_version()
    if meta_version != pyproject["version"]:
        issues.append(f"Version mismatch: pyproject.toml={pyproject['version']}, meta.yaml={meta_version}")

    # Dependency check
    meta_deps = {normalize_name(name): ver for name, ver in parse_meta_yaml_deps()}
    pyproject_deps = {}
    for dep in pyproject["dependencies"]:
        name, version = parse_dep(dep)
        pyproject_deps[normalize_name(name)] = version

    # Check for missing deps in meta.yaml
    for pypi_name, version in pyproject_deps.items():
        conda = conda_name(pypi_name)
        if conda not in meta_deps:
            issues.append(f"Missing in meta.yaml: {conda} {version} (PyPI: {pypi_name})")

    # Check for version mismatches
    for pypi_name, version in pyproject_deps.items():
        conda = conda_name(pypi_name)
        if conda in meta_deps and meta_deps[conda] != version:
            issues.append(f"Version mismatch for {conda}: pyproject.toml={version}, meta.yaml={meta_deps[conda]}")

    # Check for extra deps in meta.yaml not in pyproject.toml
    pyproject_conda_names = {conda_name(normalize_name(n)) for n in pyproject_deps}
    for meta_name in meta_deps:
        if meta_name not in pyproject_conda_names:
            issues.append(f"Extra in meta.yaml (not in pyproject.toml): {meta_name}")

    return issues


def update_meta_yaml(pyproject: dict) -> None:
    """Update meta.yaml version to match pyproject.toml."""
    text = META_YAML.read_text(encoding="utf-8")
    text = re.sub(
        r'{%\s*set\s+version\s*=\s*"[^"]+"\s*%}',
        f'{{% set version = "{pyproject["version"]}" %}}',
        text,
    )
    META_YAML.write_text(text, encoding="utf-8")
    print(f"Updated meta.yaml version to {pyproject['version']}")


def fetch_sha256(version: str) -> str | None:
    """Fetch SHA256 hash for a given version from PyPI."""
    url = f"https://pypi.org/pypi/nirs4all/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        for release_file in data.get("urls", []):
            if release_file["filename"].endswith(".tar.gz"):
                return release_file["digests"]["sha256"]
    except Exception as e:
        print(f"Error fetching from PyPI: {e}", file=sys.stderr)
    return None


def update_sha256(sha: str) -> None:
    """Update SHA256 in meta.yaml."""
    text = META_YAML.read_text(encoding="utf-8")
    text = re.sub(r"sha256:\s*\S+", f"sha256: {sha}", text)
    META_YAML.write_text(text, encoding="utf-8")
    print(f"Updated meta.yaml SHA256 to {sha}")


def main():
    parser = argparse.ArgumentParser(description="Sync conda-forge recipe with pyproject.toml")
    parser.add_argument("--update", action="store_true", help="Update meta.yaml version in place")
    parser.add_argument("--sha", metavar="VERSION", help="Fetch SHA256 from PyPI for VERSION and update meta.yaml")
    args = parser.parse_args()

    if args.sha:
        sha = fetch_sha256(args.sha)
        if sha:
            print(f"SHA256 for nirs4all {args.sha}: {sha}")
            update_sha256(sha)
        else:
            print(f"Could not fetch SHA256 for version {args.sha}", file=sys.stderr)
            sys.exit(1)
        return

    pyproject = parse_pyproject()
    issues = compare(pyproject)

    if args.update:
        update_meta_yaml(pyproject)
        # Re-check after update
        issues = compare(parse_pyproject())

    if issues:
        print("Discrepancies found:")
        for issue in issues:
            print(f"  - {issue}")
        if not args.update:
            print(f"\nRun with --update to fix version. Run with --sha {pyproject['version']} to update SHA256.")
        sys.exit(1 if not args.update else 0)
    else:
        print("meta.yaml is in sync with pyproject.toml")


if __name__ == "__main__":
    main()
