# AOM NIRS Paper Draft

Draft manuscript for:

**Reframing preprocessing selection as model-internal calibration in
near-infrared spectroscopy: a large-scale benchmark of operator-adaptive PLS
and Ridge models**

Target journal: `Talanta`.

## Build

From this directory:

```bash
./build.sh
```

The script regenerates figures and runs `pdflatex`, `bibtex`, and two final
`pdflatex` passes for both the manuscript and the supplement. The current PDFs
are:

- `main.pdf` and `AOM-paper.pdf`
- `supplement.pdf` and `AOM-supplement.pdf`

## Main Files

- `main.tex`: manuscript.
- `supplement.tex`: supplementary material with derivations, claim ledger,
  source ledger, audit controls and experiment backlog.
- `cover_letter_talanta.md`: draft cover-letter positioning for Talanta.
- `references.bib`: bibliography.
- `tables/`: LaTeX tables used in the manuscript.
- `figures/`: generated PDF figures.
- `scripts/make_figures.py`: figure generation script.
- `review/experiments_needed.md`: independent review of missing experiments
  and reviewer risks.
- `review/results_inventory.md`: inventory of usable benchmark numbers and
  exact local sources.
- `review/figure_plan_opus.md`: figure-planning notes derived from the Opus
  assistant pass.

## Current Caveat

The draft is intentionally conservative: AOM-Ridge deployable results and
oracle-envelope results are separated. Before journal submission, the priority
is to freeze a single regression cohort and rerun all headline baselines and
AOM variants under one manifest.
