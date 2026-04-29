# AOM_v0 Workspace

This directory is the working area for the one-shot implementation of the
Operator-Adaptive PLS project. The implementation must stay self-contained
inside `bench/AOM_v0` unless the prompt explicitly asks to read from the
existing `nirs4all` package for reference.

## Source Material Copied Here

- `source_materials/AOM/`: current AOM roadmap, publication plan, backlog,
  prototype benchmark report, and advanced architecture notes.
- `source_materials/fck_pls/FCK_PLS_README.md`: Torch FCK-PLS prototype notes.
- `source_materials/tabpfn/SPECTRAL_LATENT_FEATURES.md`: TabPFN spectral feature
  transformer notes.
- `source_materials/tabpfn/MASTER_RESULTS_PROFILE.md`: local profile of the
  regression benchmark oracle.
- `source_materials/tabpfn/TABPFN_PAPER_PROTOCOL_NOTES.md`: extracted protocol
  notes from the TabPFN paper draft.
- `source_materials/references_nirs4all.bib`: existing bibliography from the
  nirs4all paper.

## Driver Documents

- `docs/CONTEXT_REVIEW.md`: reviewed conclusions from AOM, FCK-PLS, TabPFN, and
  existing classifiers.
- `docs/AOMPLS_MATH_SPEC.md`: mathematical conventions and formulas to implement.
- `docs/IMPLEMENTATION_PLAN.md`: exact package, module, API, and phase plan.
- `docs/BENCHMARK_PROTOCOL.md`: benchmark plan against `master_results.csv` and
  the classification cohort.
- `docs/PUBLICATION_REPO_PLAN.md`: paper repository, journal strategy, arXiv
  workflow, and manuscript deliverables.
- `publication/manuscript/PAPER_DRAFT.md`: full manuscript scaffold to complete
  from benchmark outputs.
- `publication/manuscript/references.bib`: seed bibliography for the paper.

The master executable prompt is `Prompt.md`.
