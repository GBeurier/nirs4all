#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 scripts/make_figures.py

build_tex() {
  local stem="$1"
  pdflatex -interaction=nonstopmode -halt-on-error "${stem}.tex"
  bibtex "$stem"
  pdflatex -interaction=nonstopmode -halt-on-error "${stem}.tex"
  pdflatex -interaction=nonstopmode -halt-on-error "${stem}.tex"
}

build_tex main
build_tex supplement

cp main.pdf AOM-paper.pdf
cp supplement.pdf AOM-supplement.pdf

echo "Built paper_aom/main.pdf and paper_aom/supplement.pdf"
