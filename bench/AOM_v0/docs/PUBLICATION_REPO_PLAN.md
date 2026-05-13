# Publication Repository Plan

## Journal Decision

Primary target: **Chemometrics and Intelligent Laboratory Systems**.

Reason:

- The journal scope explicitly includes original software publications and
  development of novel statistical, mathematical, and chemometrical methods for
  chemistry and related disciplines.
- Operator-Adaptive PLS is primarily a chemometric/mathematical method with
  software and benchmark validation.
- It is more naturally aligned than broad analytical chemistry journals where
  the novelty might be judged as too computational.

Preprint: submit to **arXiv** before journal submission once the benchmark
tables and full manuscript compile. Elsevier policy allows public sharing of
preprints; disclose the arXiv preprint in the cover letter.

Fallback journals:

1. **Journal of Chemometrics**: strong chemometrics audience; use if reviewers
   want a more statistics-focused venue.
2. **Analytica Chimica Acta**: higher analytical chemistry reach; use only if
   empirical NIRS results are strong enough and the method is framed as enabling
   analytical workflows.
3. **SoftwareX**: fallback only if the manuscript becomes primarily a software
   description rather than a methodological paper.

Sources checked on 2026-04-27:

- Chemometrics and Intelligent Laboratory Systems guide for authors:
  https://www.sciencedirect.com/journal/chemometrics-and-intelligent-laboratory-systems/publish/guide-for-authors
- Elsevier copyright/sharing policy:
  https://www.elsevier.com/about/policies-and-standards/copyright
- arXiv submission process documentation:
  https://arxiv.github.io/arxiv-submission-core/announcement_process.html

## Repository Layout

Use this exact layout under `bench/AOM_v0/publication`:

```text
publication/
  manuscript/
    main.tex
    references.bib
    response_to_reviewers.md
  supplement/
    supplement.tex
    extended_math.tex
    dataset_cards.tex
    extended_tables.tex
  figures/
    fig_framework.pdf
    fig_operator_paths.pdf
    fig_regression_cd.pdf
    fig_classification_cd.pdf
    fig_probability_calibration.pdf
  tables/
    table_variants.tex
    table_regression_main.tex
    table_classification_main.tex
    table_ablation.tex
  scripts/
    make_figures.py
    make_tables.py
    export_arxiv.sh
    check_submission.sh
  arxiv/
    README.md
  journal/
    cover_letter.md
    highlights.md
    graphical_abstract.md
```

## Manuscript Title

Use this title unless benchmark results force a narrower claim:

**Operator-Adaptive Partial Least Squares for Near-Infrared Spectroscopy:
Unified AOM/POP Selection, Covariance-Space SIMPLS, and PLS-DA**

## Required Paper Sections

1. Abstract.
2. Introduction.
3. Related work.
4. Operator-Adaptive PLS framework.
5. Algorithms.
6. Classification and probability calibration.
7. Experimental protocol.
8. Regression benchmark results.
9. Classification benchmark results.
10. Ablations and equivalence validation.
11. Discussion.
12. Limitations.
13. Reproducibility statement.
14. Conclusion.

## Required Tables

- Variant matrix: selection policy x engine x backend x task.
- Operator bank definitions.
- Regression benchmark summary vs PLS, TabPFN-Raw, TabPFN-opt.
- Classification benchmark summary vs PLS-DA.
- Ablation table for criteria, orthogonalization, bank size, and probability mode.

## Required Figures

- Framework diagram showing operator bank, selection policy, PLS core, and task
  heads.
- AOM vs POP operator selection traces on synthetic examples.
- Regression critical-difference diagram and per-dataset delta RMSEP.
- Classification critical-difference diagram and balanced accuracy gains.
- Probability calibration reliability plot for AOM-PLS-DA and POP-PLS-DA.

## arXiv Package

`publication/scripts/export_arxiv.sh` must create a zip containing:

- `main.tex`
- `references.bib`
- all required figure PDFs/PNGs
- supplement source if submitted together
- no raw benchmark data unless small and required for figure compilation

The arXiv source must compile without shell escape and without external files
outside the zip.

## Journal Package

Prepare:

- manuscript PDF
- source `.tex`
- highlights, 3 to 5 bullets
- cover letter
- graphical abstract if requested
- data/code availability statement
- conflict of interest statement
- author contributions

## Claims Discipline

The manuscript may claim:

- mathematical unification of AOM and POP as Operator-Adaptive PLS;
- covariance-space SIMPLS implementation for strict linear operators;
- leakage-safe benchmark protocol;
- reproducible comparison against TabPFN benchmark results.

The manuscript must not claim:

- universal superiority over TabPFN;
- guaranteed generalization improvement over PLS;
- strict linearity of SNV/MSC without qualification;
- calibration quality without log-loss/ECE evidence.
