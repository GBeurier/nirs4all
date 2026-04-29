# arXiv submission package

This directory holds the assets needed for an arXiv preprint of the
manuscript *Operator-Adaptive Partial Least Squares for Near-Infrared
Spectroscopy: Unified AOM/POP Selection, Covariance-Space SIMPLS, and
PLS-DA*.

## Building the archive

Run the export script from the repository root (paths are absolute
inside the script, so any working directory works):

```bash
bash bench/AOM_v0/publication/scripts/export_arxiv.sh
```

The script produces `bench/AOM_v0/publication/arxiv/aompls_arxiv.zip`,
a flat archive containing:

```
main.tex
references.bib
figures/             # PDF figures from publication/figures/
tables/              # *.tex include files used by main.tex
supplement/          # supplement.tex
README_arxiv.md      # short build instructions for arXiv reviewers
```

Path rewrites (`../figures/` -> `figures/`, `../tables/` -> `tables/`)
are applied during packaging so that pdflatex can resolve every
include from the archive root without further changes.

## Building locally

After unzipping, build the manuscript with:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

No `-shell-escape`, no `--write18`, no external Python or shell
helpers are required at build time. The figures are precompiled PDFs;
to regenerate them, run:

```bash
python bench/AOM_v0/publication/scripts/make_figures.py
python bench/AOM_v0/publication/scripts/make_tables.py
```

before invoking `export_arxiv.sh`.

## arXiv categories

Suggested primary category: `stat.ML` (statistical machine learning).
Secondary: `cs.LG`, `physics.data-an`.

## Disclosure to the journal

The arXiv preprint must be disclosed to the journal at submission
time. The cover letter (`journal/cover_letter.md`) already states
this explicitly. When updating the manuscript on arXiv after the
journal review, refresh the package by re-running
`export_arxiv.sh` and uploading the resulting zip to arXiv via
`arxiv replace`.

## Sanity checks

Before uploading, run:

```bash
bash bench/AOM_v0/publication/scripts/check_submission.sh
```

This validates that every figure and table referenced by `main.tex`
exists, that `references.bib` parses cleanly, and that `pdflatex` and
`bibtex` are available on `PATH`. The script exits non-zero on any
failure.
