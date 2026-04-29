#!/usr/bin/env bash
# Build a self-contained arXiv submission archive.
#
# Output: bench/AOM_v0/publication/arxiv/aompls_arxiv.zip
#
# The archive contains:
#   - main.tex (renamed to aompls_arxiv.tex on demand)
#   - references.bib
#   - figures/*.pdf (and *.png if any)
#   - supplement/supplement.tex
#   - tables/*.tex referenced by main.tex
#   - README_arxiv.md (instructions for arXiv reviewers)
#
# The script does not run pdflatex; arXiv builds the document itself.
# It only ensures every referenced asset is available with a relative
# path that resolves inside the archive.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUB_ROOT="$(cd "${HERE}/.." && pwd)"
OUT_DIR="${PUB_ROOT}/arxiv"
STAGING="$(mktemp -d -t aompls-arxiv-XXXXXXXX)"
trap 'rm -rf "${STAGING}"' EXIT

echo "[export_arxiv] staging: ${STAGING}"

mkdir -p "${OUT_DIR}"
mkdir -p "${STAGING}/figures"
mkdir -p "${STAGING}/tables"
mkdir -p "${STAGING}/supplement"

# Copy main TeX assets
cp "${PUB_ROOT}/manuscript/main.tex" "${STAGING}/main.tex"
cp "${PUB_ROOT}/manuscript/references.bib" "${STAGING}/references.bib"

# Copy supplement
if [ -f "${PUB_ROOT}/supplement/supplement.tex" ]; then
    cp "${PUB_ROOT}/supplement/supplement.tex" "${STAGING}/supplement/supplement.tex"
fi

# Copy figures (PDF and PNG)
shopt -s nullglob
for f in "${PUB_ROOT}/figures"/*.pdf "${PUB_ROOT}/figures"/*.png; do
    cp "$f" "${STAGING}/figures/"
done
shopt -u nullglob

# Copy tables
for f in "${PUB_ROOT}/tables"/*.tex; do
    if [ -f "$f" ]; then
        cp "$f" "${STAGING}/tables/"
    fi
done

# Rewrite figure / table paths so they resolve inside the archive root.
# main.tex uses ../figures and ../tables relative to manuscript/; in the
# arXiv archive everything is at the same level as main.tex.
sed -i 's|../figures/|figures/|g' "${STAGING}/main.tex"
sed -i 's|../tables/|tables/|g'   "${STAGING}/main.tex"
if [ -f "${STAGING}/supplement/supplement.tex" ]; then
    sed -i 's|../manuscript/references|../references|g' \
        "${STAGING}/supplement/supplement.tex"
fi

# Drop a small README inside the archive
cat > "${STAGING}/README_arxiv.md" <<'EOF'
# Operator-Adaptive PLS - arXiv submission package

Build instructions:

    pdflatex main
    bibtex   main
    pdflatex main
    pdflatex main

Files:

- main.tex / references.bib: manuscript and bibliography.
- supplement/supplement.tex: supplementary material.
- figures/*.pdf: precompiled figures (regenerate with
  `bench/AOM_v0/publication/scripts/make_figures.py`).
- tables/*.tex: tables included via \input.

This archive contains no shell escapes and no external paths; pdflatex
should run without -shell-escape.
EOF

# Pack
ARCHIVE="${OUT_DIR}/aompls_arxiv.zip"
rm -f "${ARCHIVE}"
(cd "${STAGING}" && zip -r "${ARCHIVE}" .)

echo "[export_arxiv] wrote: ${ARCHIVE}"
