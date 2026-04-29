# Codex Code Review Prompt

You are reviewing `bench/AOM_v0` as an external senior Python reviewer.

Read the full implementation under:

- `aompls/`
- `benchmarks/`
- `tests/`

Focus on:

1. Incorrect shapes or silent broadcasting bugs.
2. sklearn API incompatibilities.
3. Mutable default arguments or non-cloneable estimators.
4. Leakage in CV, fitted operators, preprocessors, or calibrators.
5. Divergence between numpy and torch backends.
6. Non-determinism despite `random_state`.
7. Experimental modes that are not clearly marked.
8. Benchmark schema mismatches with `bench/tabpfn_paper/master_results.csv`.
9. Places where code duplicates logic instead of sharing the core path.

Return only actionable findings. For each finding, include a severity, file,
line, observed problem, and concrete fix.
