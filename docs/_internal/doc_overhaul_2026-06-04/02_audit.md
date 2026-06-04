# Phase 2 — Exhaustive Audit of nirs4all Documentation & Examples

**Date:** 2026-06-04 · **Library version of record:** 0.9.1 · **License of record:** AGPL-3.0
**Method:** ground-truth-first, 22-agent grounded fan-out, adversarial completeness critic, manual verification pass, Codex-gated.
**Status:** complete. Raw data in `data/` (`audit_segments.json`, `audit_synthesis.json`, `audit_completeness_critic.json`). Ground-truth card in `GROUND_TRUTH.md`.

---

## 0. Verdict

The corpus is **broad, conceptually strong, and NOT publication-ready.** Mean segment health is **2.80 / 5** across 20 audited segments. The content (concepts, reference tables, scientific prose) is largely *accurate and complete*; what drags it down is **mechanical rot and structural sprawl**, not wrong ideas. Three systemic failures compound to mislead a newcomer within the first ten minutes:

1. **Metadata drift on first-contact surfaces** — the license is **mis-stated** (the project is *dual-licensed*: default `AGPL-3.0-or-later`, with `GPL-3.0`/`CeCILL-2.1` optional variants + a commercial license per `LICENSE:1-13`; but the landing page, AI-onboarding page, and root `CLAUDE.md` present `CeCILL-2.1` as the **sole** license) and the version is wrong (0.2.1 / 0.7.1 / 0.8.6 / 0.9.0 / "5.0" instead of 0.9.1) on the landing page, the citation block, the AI-onboarding page, and `conf.py`. A mis-stated license on the landing page is a compliance issue, not a typo.
2. **The inline-pipeline style mandate is violated 648 times across 140 files** (authoritative grep, §3). *No segment is fully clean.* The docs teach, at scale, the exact anti-pattern the project wants forbidden.
3. **Copy-paste-fatal wrong-API references** scattered through the on-ramp and how-to layers (`result.predict()`, `nirs4all.load_dataset()`, `bundle=`, `y_column=`, `PredictionAnalyzer.plot_predictions()`, `GaussianNoise`, `SklearnWrapper`, `examples.synthetic` import, `Predictions.load()` in the shipped notebook) — every one verified against source as non-existent.

Underlying all three is one root cause: **nothing that executes documentation exists in the safety net.** Doc `.md` code blocks are never run; the `examples/*.py` suite (which *does* run) is on a **weekly** schedule, not per-PR; and the Sphinx build actively **suppresses** broken cross-references (`nitpick_ignore_regex` blankets entire `nirs4all.*` namespaces). So every defect above survived precisely because nothing could catch it.

The good news, and the reason a rewrite is tractable: the failures are **overwhelmingly mechanical and high-reach/low-effort**. A metadata one-commit fix, one mechanical inline-pipeline sweep, a ~15-item wrong-API triage, and a CI gate to stop regression clear the majority. The structural work (collapsing the competing front doors) is design, not archaeology. This audit is the input to the Phase 3 restructuring proposal.

---

## 1. Methodology

Why this much machinery: the user asked for an audit that is *detailed, structured, exhaustive, and justified*, reviewed by Codex at each step. Documentation audits fail in two classic ways — (a) the auditor hallucinates the API and "corrects" things that are actually fine, and (b) coverage is uneven so whole areas are silently skipped. We defended against both:

- **Ground-truth-first.** Before any agent ran, the real public surface was extracted *directly from source* (`nirs4all/__init__.py`, `nirs4all/api/*.py`, `api/result.py`) into `GROUND_TRUTH.md` — exact `__all__`, signatures, result-object members, the canonical style form. Every agent was required to Read it and to cite the source/ground-truth fact proving each defect (the `evidence` field). No claim rests on memory.
- **Deterministic fan-out.** The doc+example surface was partitioned into **20 segments with explicit, non-overlapping file lists** (12 documentation · 6 examples · 2 root/build) so coverage is provable, not best-effort. One `Explore` (read-only) reader per segment, run in parallel, each grounding against source.
- **Cross-cutting synthesis** (1 agent, barrier): merged duplicate findings, quantified repo-wide patterns, built the prioritized register.
- **Adversarial completeness critic** (1 agent): tasked only with finding what the audit *missed or over-claimed*.
- **Manual verification pass** (this orchestrator, against source): resolved every dispute the critic raised, ran the authoritative style-violation grep, and audited the surfaces the fan-out had missed (the notebook, the hand-written `api/*.md` pages, build hygiene).
- **Codex gate:** the finished audit is handed to Codex (`codex exec`, CLI 0.136.0) for independent adversarial review before Phase 3. Transcript archived under `codex_reviews/`.

**Run cost (audit workflow):** 22 agents, ~1.73M agent tokens, 1011 tool calls, ~11 min wall-clock. 20/20 segment readers returned. Plus 3 gap-closing agents and the manual verification pass.

### 1.1 The completeness critic earned its keep — corrections applied

The critic caught **two false "critical release blockers"** and a **false "clean segment"** claim in the raw synthesis. All three were independently re-verified against source and **corrected** in this document (full transparency in §6):

| Raw claim | Verdict after verification | Source proof |
|---|---|---|
| "Uninstantiated transform classes → runtime crash" (U01_shap_basics.py) | **FALSE — downgraded.** Bare class types are an *intentional, supported* pipeline form. | `nirs4all/pipeline/steps/parser.py:231` "Class types: returns as-is (controller will instantiate)"; the file is in both CI runners and passes. |
| "Two broken transfer-learning imports → ImportError crash, release blocker" | **PARTLY FALSE — downgraded to medium.** Both imports are inside `try/except ImportError`; they don't crash. But the module paths *are* wrong, so the feature is silently never demonstrated. | `examples/developer/04_transfer_learning/D01_transfer_analysis.py:130-152` (try/except). Real homes: `nirs4all.analysis.selector.TransferPreprocessingSelector`, `nirs4all.visualization.analysis.transfer.PreprocPCAEvaluator`. |
| "Only 2 segments are style-clean; ~250+ violations" | **FALSE.** predictions/deployment is *not* clean; the real count is **648** (§3). | grep of `making_predictions.md:246`, `session_api.md:30,217`; authoritative count in §3. |

This is the audit doing its job: the multi-agent pass surfaced ~216 candidate defects fast, and the adversarial+manual layer pruned the wrong ones before they could waste a maintainer's time. **Act on the corrected register (§4), not the raw per-segment files.**

---

## 2. Scope & coverage

### What was audited (provably complete surface)

| Layer | Count | Coverage |
|---|---|---|
| Documentation narrative `.md` (`docs/source/**`, excl. `api/`) | ~119 pages | 12 segments, explicit file lists |
| Hand-written `api/*.md` + `reference/api/session.md` | 5 pages | gap-closing pass (§5.2) |
| Auto-generated `api/*.rst` autodoc stubs | 357 | assessed structurally (build segment + §5.3) |
| Executable examples `examples/**/*.py` | ~80 user/dev/reference + infra | 6 segments |
| Shipped notebook `examples/predictions.ipynb` | 1 (1.8 MB) | gap-closing pass (§5.1) |
| Root/meta (`README`, `INSTALLATION`, `CLAUDE.md`, `AGENTS.md`, `CONTRIBUTING`, `Roadmap`, `CHANGELOG`) | 7 | 1 segment |
| Build/RTD config (`conf.py`, `Makefile`, `build_docs.sh`, `.readthedocs.yaml`, CI workflows) | — | build segment + §5.3 |
| Assets (`docs/source/assets/`) | 28 | reference-integrity check (§5.3) |

### Genuinely out of scope (stated, not skipped)

- **i18n:** there is no FR/EN duplication (no `locale/`, no `*.fr.md`, single-language `conf.py`) — nothing to audit. The root *repo* ships FR variants (`CLA_FR.md`, `CONTRIBUTING_FR.md`, `THIRD_PARTY_NOTICES_FR.md`) but the *docs site* is English-only.
- **Versioned-docs divergence** (stable vs latest RTD builds) beyond the single stale `release` string — not examined; flagged as a follow-up.

---

## 3. The inline-pipeline style mandate — authoritative measurement

> **Mandate (user, verbatim intent):** "1 pipeline, des datasets, run()." Pipelines must be written **inline, in one shot**, inside `run()`, with **no external variable**, for readability:
> ```python
> result = nirs4all.run(pipeline=[ ...inline... ], dataset=...)   # ✅
> ```
> ```python
> pipeline = [ ... ]                 # ❌ external variable
> result = nirs4all.run(pipeline, dataset)
> ```

The raw synthesis estimated "250+ violations, only 2 clean segments." That was wrong (the critic flagged it; one reader's grep was non-uniform). A single uniform grep across `docs/source/**.md` and `examples/**.py` (variable name contains `pipeline`, assigned a list literal at statement level) gives the authoritative figure:

### **~648 candidate sites across 140 files. No segment is clean. Only 55 compliant inline `run(pipeline=[` usages exist.**

> **Precision note (Codex round 2):** 648 is a *candidate* count from a uniform `pipeline-named-var = [` grep, not a hand-verified violation total. Codex's independent grep found 655 (or 605 requiring whitespace around `=`) — same ballpark. The large majority are genuine violations, but the pattern also catches a few *legitimate* cases: compliant inline calls split across lines, and deliberate list-of-pipelines reuse (e.g. `user_guide/pipelines/writing_pipelines.md:1242-1252`). Treat 648 as the upper-bound work-list for the sweep; the documented exceptions (below) carve out the legitimate ones.

| Area | Violations | Notes |
|---|---:|---|
| `docs/source/user_guide/**` | **224** | epicenter of the how-to layer |
| `examples/user/**` | 132 | the on-ramp the README points to |
| `examples/developer/**` | 99 | branching/generator examples |
| `docs/source/reference/**` | 69 | incl. `pipeline_syntax.md` (47), the canonical syntax page |
| `docs/source/examples/**` (doc pages) | 42 | |
| root `docs/source/*.md` (incl. `ai_onboarding.md` ×16) | 18 | |
| `docs/source/getting_started/**` | 12 | the literal first tutorial |
| `docs/source/concepts/**` | 12 | |
| `docs/source/developer/**` | 10 | |
| `docs/source/onboarding/**` | 7 | |
| `examples/_internal`, `pipeline_samples`, `reference`, `aom_paper` | 23 | mostly non-user-facing tooling |

**Worst single files:** `user_guide/pipelines/writing_pipelines.md` (51), `reference/pipeline_syntax.md` (47), `user_guide/pipelines/generators.md` (23), `user_guide/models/training.md` (20), `ai_onboarding.md` (16), `examples/user/03_preprocessing/U01_preprocessing_basics.py` (9).

**Interpretation.** This is a *propagated template defect*, not isolated authoring lapses — the same `pipeline = [...]` skeleton was copied everywhere. That is good news for remediation: a single mechanical sweep clears the overwhelming majority. **But** a documented exception is warranted for two legitimate cases, and the proposal must name them so the pattern does not regenerate:
- **Deliberate reuse / comparison** examples that run the *same* pipeline through several `run()` calls or `PipelineRunner`/`Session` deep-dives (a handful in `06_deployment`, `06_internals`).
- **Synthesis/utility** scripts (`developer/02_generators/D05–D09`, `examples/_internal/*`) that are not pipeline tutorials.

Even those should be re-examined; most can still be inlined. **A `ruff`/grep CI lint enforcing the inline form on `docs/` and `examples/` is the only durable fix** — without it, 648 becomes 700 by the next release.

---

## 4. Prioritized defect register (corrected)

Ranked by severity × reach × how-misleading-to-a-newcomer. Severities reflect the verification pass (the two raw "critical" example bugs are correctly downgraded). Every item cites source-verified evidence.

### CRITICAL — release blockers (fix before any further doc work)

| # | Defect | Where | Evidence / fix |
|---|---|---|---|
| 1 | **License mis-stated as CeCILL-2.1-only** on first-contact pages (project is dual-licensed, default AGPL-3.0-or-later) | `docs/source/index.md:284`; `ai_onboarding.md:14`; root `CLAUDE.md:3` | Legal/compliance. `LICENSE:1-13` = dual (default `AGPL-3.0-or-later`; `GPL-3.0`/`CeCILL-2.1` optional; + commercial); `README.md:430-431` states this correctly. Fix the pages to match `README`. **Do NOT** add a CI gate banning the `CeCILL` token (it is a valid optional variant) — gate on the license string matching the `LICENSE`/README dual-license statement. |
| 2 | **Stale version** (0.2.1 / 0.7.1 / 0.8.6 / 0.9.0) across landing, citation, Sphinx config | `index.md:277` (0.2.1); `conf.py:50` (release 0.7.1); `ai_onboarding.md:14` (0.8.6); `README.md:411` (0.9.0 citation) | Real = 0.9.1 (`__init__.py:45`). Source `conf.py` release from `nirs4all.__version__`; bump `conf.py:48` copyright 2025→2026; drop the "Dec 2025 / Phase 3" footer in `reference/pipeline_syntax.md`. |
| 3 | **`result.predict()` shown on `RunResult`** (no such method) — on the core getting-started on-ramp | `getting_started/index.md:192` | `RunResult` (`api/result.py:214`) has methods `top/filter/get_datasets/get_models/export/...` (`:605-903`), no `predict()`. Replace with `nirs4all.predict(model="exports/best_model.n4a", data=X_new)`. |
| 4 | **`nirs4all.predict(bundle=...)`** wrong kwarg on the primary deploy workflow | `user_guide/deployment/index.md:84-86` | Verified signature `predict(model=None, data=None, *, chain_id=...)` (`api/predict.py:56-67`). Note `bundle=` is *absorbed by `**runner_kwargs`* (no bind-time `TypeError`); it fails the "neither model nor chain_id supplied" validation at `predict.py:161-165`. Change to `model=`. |

### HIGH — copy-paste-fatal or structurally damaging

| # | Defect | Where | Evidence / fix |
|---|---|---|---|
| 5 | **Inline-pipeline mandate violated 648×/140 files** (§3) | corpus-wide | Mechanical sweep + CI lint + documented exceptions. Single largest-reach defect. |
| 6 | **Competing front doors, no canonical "start here"** | `index.md` toctree (8 top-level paths); overlap among `getting_started/`, `concepts/`, `onboarding/`, `ai_onboarding.md`; `developer/index.md` overlaps `onboarding/` | Foundational concepts taught 3–4× with no canonical source. Flagged by 4 readers + GROUND_TRUTH #4. This is the central Phase-3 design problem. |
| 7 | **`nirs4all.load_dataset()` documented as public API** (does not exist) | `onboarding/persona_paths.md:83-87` | Not in `__all__`. Remove or replace with path-string / `DatasetConfigs`. |
| 8 | **5 broken example-file refs** | `onboarding/persona_paths.md` (U01_minimal→U01_hello_world, U02_workspace→U02_basic_regression, `01_branching/`→`01_advanced_pipelines/`, D04_stacking→D05_meta_stacking, U02_multi_source→U02_multi_datasets) | Files don't exist on disk. Correct each path. |
| 9 | **`PredictionAnalyzer.plot_predictions/plot_residuals/plot_calibration`** don't exist; the "Available Charts" table lists phantom charts; **and the quickstart constructs `PredictionAnalyzer(result)` but the constructor expects a `Predictions`** | `user_guide/visualization/index.md:73,76-88` | Real public methods: `plot_top_k/plot_confusion_matrix/plot_histogram/plot_candlestick/plot_heatmap` (`visualization/predictions.py:533,641,755,847,994`; plus branch-plot methods at `:1235+`). Constructor is `PredictionAnalyzer(predictions_obj: Predictions)` (`predictions.py:97-100`) — passing `result` is wrong too. Replace table + quickstart and pass `result.predictions`. |
| 10 | **Phantom operator class names** (import errors) | `user_guide/visualization/in_pipeline_charts.md:260` (GaussianNoise), `:304` (SpectrumShift), `:351` (OutlierExclusion) | Real: `GaussianAdditiveNoise`, `WavelengthShift`, `XOutlierFilter`. Rename. |
| 11 | **FAQ uses non-existent `y_column=`** on `DatasetConfigs` | `user_guide/troubleshooting/faq.md:55,73` | The FAQ is where stuck users go. **Fix is NOT a simple rename:** `DatasetConfigs.__init__` (`data/config.py:24-32`) takes `configurations, task_type, signal_type, repetition, aggregate, ...` — neither `y_column` nor `target_column` is a direct kwarg. `target_column` belongs inside the config dict / `global_params` (see the correct usage at `user_guide/data/loading_data.md:47-50`). Rewrite the snippet to put `target_column` in `global_params`. |
| 12 | **`SklearnWrapper`** referenced; class is `NIRSPipeline` | `docs/source/examples/user/explainability.md:189,192` | `nirs4all.sklearn` exports `NIRSPipeline` / `NIRSPipelineClassifier`. Won't import. |
| 13 | **Synthetic generator import path wrong** (`examples.synthetic` → `nirs4all.synthesis`) | `user_guide/augmentation/synthetic_nirs_generator.md:405` | `SyntheticNIRSGenerator` lives in `nirs4all.synthesis`. |
| 14 | **Stale `Predictions.load()` in the shipped notebook AND in a source docstring** that autodoc surfaces | `examples/predictions.ipynb`; **`nirs4all/visualization/predictions.py:76`** | `Predictions` has no `.load()` — real: `from_parquet`/`from_file`/`from_workspace` (`data/predictions.py:497,530,548`); the notebook also touches non-existent `predictions._storage._df` (real internal: `_store`/`_buffer`). The `PredictionAnalyzer` class docstring's example *also* calls `Predictions.load('predictions.json')`, so the **generated API reference inherits the stale call** — fix the docstring too. Notebook health 1/5: convert to a runnable `R04_*.py` in CI, or remove. (§5.1) |
| 15 | **Broken refs to non-existent bundles U21/U22** | `user_guide/deployment/export_bundles.md:373-374` | Only U01–U04 exist in `06_deployment`. Repoint. |
| 16 | **Example-count claims inconsistent** (50+/60+/66/67/72/30) | `index.md:131`, `quickstart.md:178`, `examples/index.md:3,338`, README | Derive programmatically; stamp one number everywhere. |
| 17 | **4 CI-tested example files undocumented** (U05/U06 in models & CV) | `docs/source/examples/user/models.md`, `cross_validation.md` | `U05_advanced_finetuning`, `U06_tabpfn_nirs`, `U05_tagging_analysis`, `U06_exclusion_strategies` exist but aren't in their doc pages. Add sections. |
| 18 | **`api/storage.md` & `api/workspace.md` duplicate `reference/storage.md` & `reference/workspace.md`** and contradict the "no `runs/` directory" design principle; carry "Version 5.0" | `docs/source/api/storage.md`, `api/workspace.md` | ~66–100% duplication (PipelineLibrary section identical). Merge into the `reference/` versions; one source of truth. (§5.2) |
| 19 | **Examples run weekly, not per-PR; doc snippets never run; broken refs suppressed** | `.github/workflows/examples.yml` (cron Sunday), `docs.yml`, `conf.py` `nitpick_ignore_regex` | The safety-net root cause (§5.3). A broken example sits on `main` up to 7 days; `.md` API drift is never caught; `nirs4all.{data,pipeline,api}.*` xrefs are blanket-ignored. |

### MEDIUM — corrected-down from raw "critical", plus real-but-contained issues

| # | Defect | Where | Evidence / fix |
|---|---|---|---|
| 20 | **Transfer-learning examples import wrong module paths** (downgraded: `try/except`-guarded, no crash, but feature silently never demonstrated) | `developer/04_transfer_learning/D01_transfer_analysis.py:131`, `D03_pca_geometry.py:88` | Real: `from nirs4all.analysis import TransferPreprocessingSelector`; `from nirs4all.visualization.analysis import PreprocPCAEvaluator`. Fix path so the demo actually runs. |
| 21 | **Bare-class style inconsistency** (NOT a bug — downgraded) | `U01_shap_basics.py:81-86`, `U03_feature_selection.py` | Bare classes are valid (`parser.py:231`). No fix needed for correctness; optionally standardize on instances for teaching consistency. |
| 22 | **`module_api.md` predict() doc names first param `source`** (real: `model`) | `docs/source/api/module_api.md:117-124` | Source: `predict(model, data, ...)` (`api/predict.py:56`). *(Correction: the raw audit also claimed `ExplainResult.get_feature_importance()` is fake — it is NOT; it exists at `api/result.py:1237-1241`. That subclaim is dropped.)* |
| 23 | **`run()` reference omits 4 real params** (`refit`, `cache`, `project`, `report_naming`) | `reference/predictions_api.md:35-48` | Add with defaults. |
| 24 | **CLI `init` doc vs workspace design disagree on `runs/`** | `reference/cli.md:40-48` vs `reference/workspace.md:16` | Workspace = "no nested runs/". Remove `- runs/` from cli example. |
| 25 | ~~**`PredictionResultsList`** referenced; no such class~~ **— FALSE POSITIVE (found during P0 execution).** | `user_guide/predictions/index.md:126` | `PredictionResultsList` **does exist** (`nirs4all/data/_predictions/result.py:328`, re-exported via `nirs4all.data`, documented at `reference/predictions_api.md:203`). It is just not in the top-level `__all__`. The original reference was accurate; no defect. (The P0 pass rephrased the line to `PredictResult`/`RunResult` for primary-API clarity — harmless, not a fix.) |
| 26 | **`reference/api/session.md` is orphaned** (in no toctree) | — | Wire into `reference/index.md` or remove. |
| 27 | **`models.md` reference example uses `SNV()` un-imported** | `reference/models.md:12-18` | Add import. |
| 28 | **`DependencyGraph.topological_sort()` example mismatches source** | `developer/artifacts_internals.md:136-175` | Real surface: `add_dependency/get_dependencies/check_cycle`. Relabel illustrative or fix. |
| 29 | **`metadata.md` imports a fictional `browse_folder`** | `developer/metadata.md:467-517` | Module/function doesn't exist. Replace with real config entry point. |
| 30 | **Stale "Coming Soon"** for already-written content | `user_guide/data/index.md:74-76` | Delete. |
| 31 | **`aggregation.md` conflates repetition with aggregation** | `user_guide/data/aggregation.md` | Primarily about repetition; retitle/split. |
| 32 | **Two `developer/05` examples define pipelines but never call `run()`** (validate nothing) | `D01_metadata_branching.py`, `D03_repetition_transform.py` | Add execution or relabel syntax-only — they give false coverage on a weekly-only CI. |
| 33 | **Internal Phase/roadmap leakage** into user-facing docs | `developer/caching.md`, `synthetic.md`; `examples/developer/02_generators/D07-D09`; `Roadmap.md` (public brain-dump) | Move to `docs/_internal` / `ROADMAP.md`; rewrite intros problem-statement-first. |
| 34 | **R04 reference example missing**, no explanation; example inventory split across `run.sh`/`run_ci_examples.sh`/README (drift) | `examples/reference/` | Renumber or document; derive one inventory consumed by all runners + README. |

### LOW

| # | Defect | Where |
|---|---|---|
| 35 | `example_utils.py:134` emoji output ignores `DISABLE_EMOJI` set by `run.sh:304` | examples infra |
| 36 | SNV naming drift (`StandardNormalVariate` vs `SNV` alias) without an equivalence note | preprocessing docs |
| 37 | tabpfn example dataset path inconsistency (`examples/sample_data/...` vs `sample_data/...`) | `U06_tabpfn_nirs.py:119` |
| 38 | Generator syntax documented in 3 reference files with drift | `generator_keywords.md`, `combination_generator.md`, `pipeline_syntax.md` |
| 39 | 3 orphaned assets never referenced | `assets/models_preprocessing.pdf`, `pipeline.jpg`, `stacking.png` |
| 40 | `augmentations.md` "Developer guidelines" header inside `user_guide/` (audience mislabel) | `user_guide/augmentation/augmentations.md` |

---

## 5. Gap-closing pass (surfaces the fan-out missed)

### 5.1 `examples/predictions.ipynb` — the single largest coverage hole

A 1.8 MB Jupyter notebook ships as an example but is **referenced by no runner and no doc page**, and is executed by **no CI**. It is **stale and unrunnable** against 0.9.1:
- `Predictions.load(path=…)` — **no such method**. `Predictions` has `from_workspace()` / `from_file()` / `from_parquet()` (`nirs4all/data/predictions.py:497,530,548`) and a path-detecting constructor.
- `predictions._storage._df` — **no such attribute**. Internal state is `self._store` / `self._buffer` (`predictions.py:236,240`). The notebook reaches into a private Polars frame that doesn't exist.

**Health 1/5.** Recommendation: either delete it, or rewrite as a runnable `examples/reference/R04_visualization.py` (conveniently filling the missing-R04 gap, #34) using the public `PredictionAnalyzer` API, and add it to `run_ci_examples.sh`. A shipped, un-CI'd, broken artifact is worse than no artifact.

### 5.2 Hand-written `api/*.md` pages

| Page | Health | Key issue |
|---|---|---|
| `api/module_api.md` | 3/5 | predict() doc names first param `source` (real `model`); 1 style violation. *(The `get_feature_importance()` "fake method" flagged by the reader is actually real — `result.py:1237`.)* |
| `api/sklearn_integration.md` | 4/5 | clean; imports verified (`NIRSPipeline`, `NIRSPipelineClassifier` both exported) |
| `api/storage.md` | 2/5 | "Version 5.0"; duplicates `reference/storage.md` (~800 vs ~400 LOC); legacy `runs/` framing |
| `api/workspace.md` | 2/5 | missing "no `runs/`" principle; PipelineLibrary section 100% duplicate of `reference/workspace.md` |
| `reference/api/session.md` | 4/5 | **orphaned** (no toctree); minor unverified property docs |

**Pattern:** the `api/` tree (4 hand-written pages + 357 autodoc stubs) and the `reference/` tree document the same surface twice. This is a Phase-3 consolidation target: pick one API-reference home.

### 5.3 Build / RTD hygiene — the root cause of undetected drift

- **(A) Broken cross-refs are actively suppressed.** `conf.py` `nitpick_ignore_regex` blankets entire namespaces — `nirs4all.data.*`, `nirs4all.pipeline.*`, `nirs4all.api.*` — and `suppress_warnings` includes `myst.xref_missing` and `ref.*`. Any typo'd `:class:` ref in those namespaces is invisible. **And** CI `docs.yml` builds with `-W --keep-going` (strict) while `.readthedocs.yaml` sets `fail_on_warning: false` — CI and the published build disagree on strictness.
- **(B) Asset integrity: clean.** All referenced images exist with alt text. 3 orphaned asset files (#39).
- **(C) Zero doc-snippet execution.** No `doctest` / `nbval` / `myst_nb` / `literalinclude`-exec anywhere. The only executed code is `examples/*.py`, and `examples.yml` runs **weekly (cron Sunday) + manual only — never on push/PR** (the workflow header says so explicitly). This is *why* `result.predict()`, `bundle=`, `plot_predictions()`, `SklearnWrapper`, and the notebook's `Predictions.load()` all survived: nothing executes them, and the one thing that could runs once a week.

**This is the highest-leverage systemic fix.** A doc-snippet execution gate (or at minimum moving `examples.yml` to per-PR for a fast subset) plus a tightened `nitpick_ignore` converts the entire class of wrong-API defects from "ships silently" to "fails the build."

---

## 6. Transparency: false positives caught and corrected

To keep the register trustworthy, the audit's own mistakes are recorded rather than hidden (full reasoning in §1.1). The raw 22-agent pass produced ~216 candidate defects; the completeness critic + manual verification **downgraded 2 of the 9 raw "criticals"** (bare classes #21, transfer imports #20) and **corrected the headline style count** from "~250, 2 clean segments" to "~648 candidate sites, 0 clean segments."

**Codex review round 2 (verdict: SOLID-WITH-FIXES; transcript `codex_reviews/phase2_audit_review.txt`)** independently spot-checked the register against source — confirming #3, #4, #5 scale (its grep: 605–655), #7, #9, #10, #12/#13/#15, #19 — and forced six accuracy fixes, all re-verified against source and applied above:

1. **License is dual, not "CeCILL = wrong."** `LICENSE:1-13` = default AGPL-3.0-or-later + optional GPL-3.0/CeCILL-2.1 + commercial; `README:430-431` says so. Defect reframed (#1); the "ban CeCILL token" CI gate idea is removed.
2. **`bundle=`** is absorbed by `**runner_kwargs` (no bind-time `TypeError`); fails the model/chain_id validation instead (#4).
3. **`ExplainResult.get_feature_importance()` exists** (`result.py:1237`) — the reader's "fake method" subclaim was wrong; dropped (#22, §5.2).
4. **`PredictionAnalyzer(result)`** is a second bug — constructor wants a `Predictions` (#9).
5. **`target_column` is not a `DatasetConfigs` kwarg** — belongs in `global_params`; the `y_column→target_column` rename was wrong (#11).
6. **Stale `Predictions.load()` also lives in `visualization/predictions.py:76` docstring** → autodoc inherits it (#14).

Net: the register above is the trustworthy, twice-reviewed artifact; the per-segment JSON in `data/audit_segments.json` is raw input and still contains the un-pruned claims.

---

## 7. Health by segment (corrected)

| Segment | Score | Headline |
|---|:--:|---|
| `concepts/` | 4 | Accurate, well-sequenced; only style-block violations. |
| `user_guide/pipelines/` + ug root | 4 | Comprehensive, API-accurate; style + minor redundancy. |
| `user_guide/preprocessing/` + augmentation/ | 4 | Scientifically grounded; one wrong synthesis import. |
| `user_guide/predictions/` + deployment/ | 4 | (corrected) `bundle=` bug, U21/U22 refs, phantom `PredictionResultsList`, *some* style violations. |
| examples `reference/` + infra | 4 | Robust CI infra; R01 style, R04 missing, split inventory. |
| `user_guide/data/` + troubleshooting/ | 3 | FAQ `y_column`, repetition/aggregation naming, style. |
| `reference/` (pipeline-centric) | 3 | Gold-standard coverage; 69 style instances, stale footer. |
| `reference/` (operators/storage) | 3 | Accurate tables; style, `run()` missing 4 params, cli/runs disagreement. |
| `developer/` + migration/ | 3 | Solid internals; duplicated architecture docs, `topological_sort` mismatch. |
| docs `examples/` pages | 3 | Good learning paths; stale counts, undocumented U05/U06, `SklearnWrapper`. |
| README + root meta | 3 | Stale CeCILL in `CLAUDE.md`, 0.9.0 citation; style; competing quick-starts. |
| Landing + getting_started | 2 | Wrong license+version, `result.predict()`, pervasive style on the core on-ramp. |
| `onboarding/` | 2 | Strong prose undercut by 5 broken refs, non-existent `load_dataset()`, style. |
| `user_guide/models/` + visualization/ | 2 | 44 style blocks, 3 phantom classes, visualization landing documents non-existent methods. |
| examples user/01+02 | 2 | API-correct, clean teaching, but every pipeline file violates inline style. |
| examples user/03+04 | 2 | Highest style-violation density; inconsistent cross-refs. |
| examples user/05+06+07 | 2 | Every file external-pipeline; bare-class *style* inconsistency (not a bug). |
| examples developer/01+02 | 2 | Comprehensive but ~heavy style violations, wrong-dir cross-refs. |
| examples developer/03–06 | 2 | Wrong transfer import paths (guarded), heavy style, 2 demos never run. |
| docs build config + RTD | 2 | Sound scripts; stale `conf.py` release/copyright; suppression masks broken refs. |

**Mean: 2.80 / 5.**

---

## 8. What this means for Phase 3 (restructuring)

The audit converges on a clear mandate for the restructuring proposal:

1. **One canonical learning path, one source of truth per concept.** The single biggest *structural* defect is the 3–4 competing front doors (`getting_started/` vs `concepts/` vs `onboarding/` vs `ai_onboarding.md`) and the duplicated `api/` ↔ `reference/` trees. Phase 3 must pick a target information architecture (a Diátaxis-style split is the obvious candidate) and assign every existing page a home or a deletion.
2. **The style mandate becomes infrastructure, not aspiration.** "1 pipeline inline + `run()`" must be enforced by a CI lint over `docs/` + `examples/`, with the two documented exceptions, or 648 will regrow.
3. **Documentation must enter the safety net.** Per-PR execution of a fast example subset + a doc-snippet doctest gate + a tightened `nitpick_ignore`. This is what makes the rewrite *stay* correct.
4. **Metadata is sourced from one place.** Version/license/copyright derived from `nirs4all.__version__` and `LICENSE`, with a CI grep gate.

Phase 3 will deliver the target IA, a page-by-page migration map (keep / merge / rewrite / delete), the enforced style guide, and the CI gates — justified against these findings.

---

### Appendix — raw data & reproducibility

- `GROUND_TRUTH.md` — the source-verified API surface every agent grounded against.
- `data/audit_segments.json` — 20 raw segment reports (un-pruned; contains the false positives corrected in §6).
- `data/audit_synthesis.json` — cross-cutting synthesis (raw).
- `data/audit_completeness_critic.json` — adversarial critic output.
- `audit_workflow.js` — the workflow script (re-runnable / resumable).
- Authoritative style count reproducible via the grep in §3 (variable-name-contains-`pipeline`, list-literal assignment, over `docs/source/**.md` + `examples/**.py`).
