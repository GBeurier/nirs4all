# Phase 4 — Execution Log (P0: correctness fixes)

**Branch:** `docs/overhaul-p0-correctness` (off `main`). **Started:** 2026-06-04. **Not committed** (awaiting user go-ahead).
**Scope of this batch:** roadmap **P0** — metadata + copy-paste-fatal wrong-API triage. No IA restructuring (that is P3). Style sweep is P1.

## Proposal refinements applied first (user feedback 2026-06-04)

1. **Inline style is non-tyrannical** — `03_restructuring_proposal.md` §5/§10/§11 reworked: inline is the default for *simple/moderate* pipelines; *complex* (nested branch/merge, large sweeps, reuse) MAY use a named variable; the lint is an **advisory, complexity-aware warning**, not a hard gate. Saved to memory `feedback_inline_pipeline`.
2. **Dataset config + repetitions/aggregation = first-class topic (NIRS-critical)** — new §5b plus Explanation (`datasets-&-configuration`, new `repetitions-&-aggregation`), How-to Data recipes, a curated `Reference → DatasetConfigs` page, and Tutorial L3 coverage.

## P0 edits (19 files, +61/−52)

### Metadata (license + version, single coherent fix)
- `docs/source/index.md` — citation `0.2.1`→`0.9.1`/`2025`→`2026`; license CECILL-2.1-only → **dual** (default AGPL-3.0-or-later + GPL/CeCILL variants + commercial), per `LICENSE`.
- `docs/source/ai_onboarding.md` — `0.8.6`→`0.9.1`, license → AGPL-3.0-or-later (dual).
- root `CLAUDE.md` — license CeCILL-2.1 → dual AGPL-3.0-or-later.
- `README.md` — citation `0.9.0`→`0.9.1`.
- `docs/source/conf.py` — `release` now **parsed from `nirs4all/__init__.py`** (single source of truth, no heavy import); `copyright` 2025→2026. Verified resolves to `0.9.1`.
- `docs/source/reference/pipeline_syntax.md` — removed stale "Last Updated: December 2025 / Version 1.1 (Phase 3)" footer.

### Copy-paste-fatal wrong-API (survivor pages + source docstring)
- `getting_started/index.md` — `result.predict(X_new)` → `nirs4all.predict(model="…", data=X_new)` + `.y_pred` (#3).
- `user_guide/deployment/index.md` — `predict(bundle=…)` → `model=` (#4).
- `troubleshooting/faq.md` (×2) — `DatasetConfigs(y_column=…)` → `target_column` inside `global_params` config dict, matching `loading_data.md` (#11; the fix is NOT a kwarg rename — Codex/round-2 nuance honored).
- `visualization/index.md` — phantom `plot_predictions/plot_residuals/plot_calibration` → real `plot_top_k/plot_confusion_matrix/plot_histogram/plot_candlestick/plot_heatmap`; "Available Charts" table corrected; `PredictionAnalyzer(result)` → `PredictionAnalyzer(result.predictions)` (constructor wants a `Predictions`) (#9).
- `visualization/in_pipeline_charts.md` — `GaussianNoise`→`GaussianAdditiveNoise`, `SpectrumShift`→`WavelengthShift` (+ phantom `max_shift` kwarg → real `shift_range`), `OutlierExclusion`→`XOutlierFilter` (#10).
- `augmentation/synthetic_nirs_generator.md` — `from examples.synthetic` → `from nirs4all.synthesis` (#13).
- `nirs4all/visualization/predictions.py:76` (SOURCE) — docstring `Predictions.load('predictions.json')` → `Predictions.from_file(...)` so autodoc stops inheriting a stale call (#14).
- `reference/predictions_api.md` — `run()` doc completed with `refit`/`cache`/`project`/`report_naming` (#23).
- `reference/cli.md` — removed fabricated `runs/` (and other unsupported dirs) from `init` output; now matches real workspace layout (#24).
- `reference/models.md` — added `from nirs4all.operators.transforms import SNV` so the example doesn't NameError (#27).

### Examples (import-path correctness)
- `examples/developer/04_transfer_learning/D01_transfer_analysis.py` — `nirs4all.operators.transfer` → `nirs4all.analysis` (real home of `TransferPreprocessingSelector`); fixed both the real import and the printed snippet string (#20).
- `examples/developer/04_transfer_learning/D03_pca_geometry.py` — `nirs4all.operators.analysis` → `nirs4all.visualization.analysis` (real home of `PreprocPCAEvaluator`); both occurrences (#20).

## Audit corrections found *during* execution

- **#25 `PredictionResultsList` was a FALSE POSITIVE.** It is a real, exported class (`nirs4all/data/_predictions/result.py:328`, via `nirs4all.data`, documented at `predictions_api.md:203`) — just not in top-level `__all__`. The original doc reference was accurate. `02_audit.md` #25 corrected. (Third audit false-positive caught by adversarial verification + execution — consistent with the audit's own §6 transparency.)
- Bonus real fix: `in_pipeline_charts.md` used a phantom `max_shift=` kwarg on the shift augmenter; corrected to the real `shift_range=`.

## Verification

- `conf.py` parses; version-parse resolves to `0.9.1`. ✓
- `ruff check` on the 3 changed `.py` files: **All checks passed.** ✓
- Post-edit sweeps (native grep): **0** remaining `result.predict(` / `bundle=` / `y_column=` / `plot_predictions|residuals|calibration` / `GaussianNoise|SpectrumShift|OutlierExclusion` / `examples.synthetic` / `Predictions.load(` in docs; **0** stale `0.2.1|0.8.6|CECILL` in first-contact pages. ✓
- Could NOT run nirs4all itself (no project venv; system py is 3.10, lib needs 3.11+). Doc/import edits verified statically against source by each agent (cited `__init__.py` lines).

## Deferred (with rationale)

| Item | Why deferred | Target phase |
|---|---|---|
| `predictions.ipynb` → `R04_visualization.py` (#14/#34) | Writing a *runnable* example needs a py3.11 env to validate; won't ship an untested "runnable" file | needs venv → P0.1 |
| `developer/05` `D01/D03` must-call-`run()` (#32) | Making them execute needs validation; the gate that enforces this is P2 | P2 |
| Full inline-style sweep (~648 simple-case sites) | P1; now *non-tyrannical* (simple only) | P1 |
| Front-door collapse, Explanation home, api/→reference, gallery, dataset-config pages | structural; needs the IA build | P3/P4 |
| `persona_paths.md` / `getting_started/concepts.md` internal fixes | both are DELETE targets in P3 — not worth polishing doomed pages | P3 (delete) |

## Update — venv set up; P1, P0.1, P2 executed & validated

A `uv` venv (py3.11) with `nirs4all -e .[docs,dev]` was created, so everything below is **runtime-validated**, not just static. All P0 symbols re-confirmed against the installed package (`WavelengthShift(shift_range=)`, no `Predictions.load`, the 5 real chart methods, all transfer/synthesis imports).

### P1 — inline-style sweep (non-tyrannical)
- Built `scripts/lint_inline_pipeline.py` — **advisory, complexity-aware AST lint**: flags only *simple* (flat, ≤6 steps, no nested branch/merge) `pipeline`-named vars passed (positionally OR by keyword) to run/session/Session/PipelineRunner. Authoritative count: **149 simple-case sites in 59 files** (not the naive grep's 648 — the rest are complex/reuse/inline and correctly exempt).
- Built `scripts/codemod_inline_pipeline.py` — semantics-preserving inliner (AST source-segment, re-indents the literal) for `.py` and `.md` fenced blocks.
- Applied: **145/149 inlined (97%)** across examples + docs; 4–5 edge cases left (advisory). Fixed 6 codemod-induced `I001` import-sorts via `ruff --fix`; collapsed double-blank artifacts in 14 md files.
- Validated: examples ruff-clean; `U01_hello_world` + `U01_preprocessing_basics` **run (EXIT 0)**; docs build clean.

### P0.1 — predictions.ipynb → R04_visualization.py
- Wrote `examples/reference/R04_visualization.py` — self-contained (synthetic data), plot-free by default, exercises the `Predictions` + `PredictionAnalyzer` API. **Runs end-to-end (EXIT 0)**; ruff clean.
- Wired into `run.sh`, `run_ci_examples.sh`, README inventory. **Deleted the 1.8 MB stale notebook.**

### P2 — safety-net gates + CI
- `scripts/check_doc_metadata.py` (**blocking**, passes): version strings == `__version__`; no sole-CeCILL phrasing. 
- `scripts/check_example_refs.py` (**blocking**, passes): every `examples/*.py` named in docs exists. Surfaced + fixed 16 dangling refs (persona_paths ×5 + its fake `load_dataset()`→`DatasetConfigs(repetition=,aggregate=)`; export_bundles U21/U22; metadata.md; cli.md; `01_branching/` dirs).
- `.github/workflows/docs-quality.yml`: gates (blocking) + inline lint (advisory) + Sphinx build + **fast canonical example subset on every push/PR** (the weekly-only `examples.yml` gap).
- `docs/source/developer/documentation_style.md`: the house style page, wired into the developer toctree.

### Proposal refinements (user feedback, folded into 03 + memory)
- Inline rule made non-tyrannical; dataset-config + repetitions/aggregation elevated to a first-class topic (§5b).

### Commits on branch `docs/overhaul-p0-correctness`
1. `docs(internal)` — trace + tooling
2. `docs+examples` — P0 wrong-API/metadata + P1 inline sweep
3. `fix(viz)` — source docstring
4. `examples` — remove stale notebook
5. `examples` — add R04 + wiring
6. `ci(docs)` — P2 gates + style guide + workflow

## Remaining: P3 (structural IA migration) + P4 (reference unification)

These are the large, interconnected phases (the proposal estimates ~2–3 weeks): collapse the 4 front doors to 1, build the single Explanation concept home + the 13-lesson Tutorial rail, dissolve `onboarding/`, delete `getting_started/concepts.md` + `persona_paths.md`, re-home `user_guide/**`→How-to and `developer/**`→Contributing, dissolve `api/`→`reference/` (autosummary + `autodoc_mock_imports` for TF/Torch/JAX), author the new `repetitions-&-aggregation` Explanation page + curated `DatasetConfigs` reference, then flip nitpick/`-W` strict after the broken-ref backlog burns down.

**Recommendation:** execute P3/P4 as a *reviewed, incremental* migration (each step build-validated) rather than one blind burst — it rewrites the whole navigation spine and changes the docs UX, so it benefits from the user's eye on the resulting structure. P0–P2 already make the docs materially more correct, consistent, and regression-protected.

## P3 increment — front-door collapse (DONE, build-validated)

Executed the defining structural fix (audit #6 → success metric "exactly one front door"):

- **onboarding/ dissolved** as a competing front door: `mental_models`, `data_workflow`, `pipeline_workflow`, `workspace_intro` re-homed into `concepts/` (the single Explanation home); `controllers_intro` → `developer/`; `onboarding/index.md` deleted.
- **Deleted** `persona_paths.md` (duplicate front door) and `getting_started/concepts.md` (duplicate of concepts/).
- **Landing toctree** reduced to one path: `getting_started → concepts → user_guide → reference → developer → examples → api`. `ai_onboarding` dropped from nav (now `:orphan:`).
- All inbound/cross links repointed (controllers_intro `../developer/`, concepts pages' `../developer/controllers_intro`, 3 stale `{doc}/getting_started/concepts` → `/concepts/*`). Bonus: wired 3 pre-existing toctree orphans (`migration/*`, `scoring_and_refit`).
- **Result:** Sphinx warnings 4 → 1–2 (the lone remaining is a pre-existing autodoc duplicate, POPPLSRegressor exported via two module paths — a P4 `:no-index:`/autodoc-unification item). Metadata + broken-ref gates green; 0 refs to deleted pages.
- NIRS dataset-config/repetitions topic: confirmed `aggregation.md` already comprehensive (covers `repetition`/`aggregate`/`aggregate_method`/`aggregate_exclude_outliers`, constructor + config-dict forms) and toctree-surfaced; made its `data/index` card NIRS-explicit.

Commit: `docs: collapse competing front doors into one canonical entry (P3)`.

## P4 increment — reference de-duplication (DONE, build-validated)

- **Deleted 356 stale `api/nirs4all.*.rst` stubs** (−3525 lines): a committed apidoc dump explicitly excluded from the build (`conf.py exclude_patterns`) and superseded by the gitignored, regenerated `_generated_api/`. Removed the dead exclude pattern too. Zero build impact.
- **Merged the duplicate `api/storage.md` + `api/workspace.md` into their `reference/` twins** (#18): confirmed `reference/workspace.md` is a strict superset (has the "no nested runs/" principle + Common Workflows) and `reference/storage.md` is canonical (has PipelineLibrary, no "Version 5.0" defect); preserved the one unique bit (the SQLite tables schema) into `reference/storage.md`; deleted both api copies and dropped them from `api/modules.rst`. `api/` now holds only the unique `module_api` + `sklearn_integration` pages beside the generated autodoc — no more reference/ duplication.
- Build clean (1 warning), gates green, no dangling links.

## Still remaining (large, next sessions)
- **P4:** the lone autodoc warning (`POPPLSRegressor` re-exported via two module paths → wants `:no-index:`); then flip `-W`/`fail_on_warning` strict after the nitpick burn-down; doc-snippet `literalinclude` execution; optional fold of `module_api`/`sklearn_integration` into curated reference pages.
- **P3 rest:** the 13-lesson progressive Tutorial rail (authoring); deeper concept dedup (`data_workflow` vs `datasets`).

## Success-metric scorecard (proposal §11)
- ✅ Exactly one front door · ✅ 0 stale versions (gate) · ✅ license correct (gate) · ✅ wrong-API copy-paste defects fixed · ✅ simple pipelines inline (advisory lint) · ✅ per-PR example execution + R04, notebook gone · ✅ example refs resolve (gate)
- ✅ **One Reference home** — `api/` storage/workspace duplicates merged into `reference/`; 356 stale stubs gone (#18 resolved for the duplicated surface)
- ⏳ all doc snippets execute in CI (literalinclude — P4) · ⏳ nitpick strict + the lone autodoc re-export warning (P4) · ⏳ mean health ≥4/5 (needs the Tutorial rail + remaining P4)
