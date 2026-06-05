#!/usr/bin/env bash
# Fetch SHA256 from PyPI for a given nirs4all version and update meta.yaml.
#
# Usage:
#   ./scripts/update_conda_sha.sh          # Uses version from nirs4all/__init__.py
#   ./scripts/update_conda_sha.sh 0.8.0    # Explicit version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Version is single-sourced from nirs4all/__init__.py (pyproject declares it dynamically).
VERSION="${1:-$(grep -E '^__version__' "$ROOT_DIR/nirs4all/__init__.py" | cut -d'"' -f2)}"

echo "Fetching SHA256 for nirs4all $VERSION from PyPI..."

SHA256=$(curl -sL "https://pypi.org/pypi/nirs4all/$VERSION/json" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print([u['digests']['sha256'] for u in d['urls'] if u['filename'].endswith('.tar.gz')][0])")

if [[ -z "$SHA256" ]]; then
  echo "ERROR: Could not fetch SHA256 from PyPI for version $VERSION" >&2
  exit 1
fi

echo "Version: $VERSION"
echo "SHA256:  $SHA256"

META_YAML="$ROOT_DIR/conda-forge/meta.yaml"
sed -i "s/{% set version = .*/{% set version = \"$VERSION\" %}/" "$META_YAML"
sed -i "s/sha256: .*/sha256: $SHA256/" "$META_YAML"

echo "Updated $META_YAML"
