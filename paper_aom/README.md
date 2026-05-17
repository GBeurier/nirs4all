# paper_aom — moved

The Talanta paper (manuscript, supplement, figures, tables, review dossier,
benchmark stats) has moved to the new dedicated repository:

```
/home/delete/nirs4all/aom_nirs/paper/
```

(GitHub: `gbeurier/aom`, package `aom-nirs`.)

The migration also moved the AOM-PLS / AOM-Ridge / FastAOM source code
out of `bench/AOM_v0/` into the same repository. See:

- `aom_nirs/aom_nirs/pls/` — AOM-PLS package (formerly `bench/AOM_v0/aompls/`)
- `aom_nirs/aom_nirs/ridge/` — AOM-Ridge package (formerly `bench/AOM_v0/Ridge/aomridge/`)
- `aom_nirs/aom_nirs/fast/` — FastAOM package (formerly `bench/AOM_v0/FastAOM/`)
- `aom_nirs/paper/` — manuscript, supplement, figures, tables, review

Migration rationale and plan: `aom_nirs/paper/review/aom_lib_migration_plan.md`.

`nirs4all/operators/models/sklearn/aom_pls.py` and `pop_pls.py` are now
thin re-exports of the canonical implementations vendored into
`nirs4all/operators/models/_aom_nirs/` (so `nirs4all` keeps working
without a hard `aom-nirs` install). Once `aom-nirs` is on PyPI, the
vendored copy will be replaced by runtime imports.
