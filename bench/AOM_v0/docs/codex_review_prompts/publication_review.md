# Codex Publication Review Prompt

You are reviewing the publication package under `bench/AOM_v0/publication`.

Read:

- `docs/PUBLICATION_REPO_PLAN.md`
- `publication/manuscript/main.tex`
- `publication/manuscript/references.bib`
- generated tables under `publication/tables`
- generated figures under `publication/figures`
- `docs/AOMPLS_VALIDATION.md`

Check:

1. Whether the central scientific claim is supported by benchmark evidence.
2. Whether the paper distinguishes measured results from guaranteed properties.
3. Whether SNV/MSC and fitted linear-at-apply operators are described honestly.
4. Whether regression comparisons against PLS, TabPFN-Raw, and TabPFN-opt are
   fair and reproducible.
5. Whether classification conclusions are appropriately scoped.
6. Whether the math in the paper matches the implementation and tests.
7. Whether the arXiv and journal package are complete.

Return findings ordered by severity, then list missing manuscript pieces.
