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

## Suggested next step

Set up a `uv` venv (py3.11) with nirs4all installed, so P0.1 (notebook→R04) and P1/P2 example edits can be **executed and verified**, not just statically checked. Then convert the notebook and run the example subset.
