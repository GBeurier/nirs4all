# Codex Publication Review Prompt — nicon_v2

Review the publication artefacts under `bench/nicon_v2/publication/`.

## What to check

1. **Manuscript skeleton (`publication/manuscript/PAPER_DRAFT.md`).**
   * Title is informative; abstract states problem, method, datasets, headline numbers, and contribution.
   * Sections: Introduction, Related Work, Method, Experimental Setup, Results, Ablations, Discussion, Conclusion, Reproducibility.
   * Method section maps 1-to-1 to `MATH_SPEC.md`.
   * Results table cites the curated cohort and includes the reference baselines (Ridge, PLS, AOM-PLS, AOM-Ridge, TabPFN, DeepSpectra, NICON, DECON).

2. **LaTeX (`publication/manuscript/main.tex`).**
   * Compiles via `xelatex` without errors.
   * Bibliography uses `references.bib` from `source_materials/literature_review/`.
   * Figures sourced from `publication/figures/*.pdf`.
   * Tables sourced from `publication/tables/*.tex`.

3. **Figures (`publication/figures/`).**
   * `fig_critical_difference.pdf` — Friedman/Nemenyi or Holm-corrected pairwise.
   * `fig_per_dataset_delta_vs_pls.pdf` — log-relative RMSEP across datasets.
   * `fig_cumulative_rmsep.pdf` — cumulative-distribution-style overlay.
   * `fig_cost_vs_precision.pdf` — fit-time vs RMSEP scatter.
   * `fig_framework.pdf` — schematic of nicon_v2 architecture.

4. **Tables (`publication/tables/`).**
   * Main regression table — RMSEP per (dataset, variant), with reference baselines and best-of-method bolded.
   * Ablation table — per-hypothesis Δ-RMSEP.
   * Variants table — list of every nicon_v2 variant with hyper-parameters.

5. **Reproducibility.**
   * Each table cites the CSV that produced it.
   * The publication scripts (`publication/scripts/make_*.py`) are deterministic and runnable from a clean checkout.

## Output format

Per-finding severity + location + issue + fix; final one-line verdict.
