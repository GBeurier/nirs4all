# Phase 3 — Complete Restructuring Proposal: nirs4all Documentation & Examples

**Date:** 2026-06-04 · **Inputs:** `02_audit.md` (Codex-signed-off), `GROUND_TRUTH.md`
**Method:** 3 competing IA candidates (distinct philosophies) → 4-lens judge panel → grounded synthesis → page-by-page migration map (9 agents). Codex-gated.
**Raw data:** `data/proposal_*.json`. **Status:** complete; Codex review round 3 pending.

---

## 0. Executive summary

The audit proved the corpus is *conceptually strong but structurally sprawling and mechanically rotten*: 4 competing front doors, `api/`↔`reference/` documenting the same surface twice, ~648 inline-style violations, ~15 copy-paste-fatal wrong-API snippets, and — the root cause — **nothing executes documentation**, so all of it ships silently.

This proposal adopts a **hybrid architecture** that the judge panel converged on, because the three philosophies each won a different decisive lens and the hybrid takes each winner where it is strongest:

> **A strict-Diátaxis single-source-of-truth backbone, carrying a FastAPI-style progressive Tutorial spine as its Tutorials arm, with the Reference and Examples estates generated from source/disk, and verb/goal routing demoted to navigation-only.**

- **Diátaxis** (Tutorials / How-to / Reference / Explanation as four non-overlapping *modes*) is the backbone because it is the only candidate that makes single-source-of-truth a **structural invariant** — *"Explanation owns the why, Reference owns the signature; neither restates the other, the only permitted relationship is a hyperlink"* — and gives a **mechanical placement rule** for any new page. This makes the audit's #6 (competing front doors) and #18 (api↔reference duplication) *hard to recreate* — but not magic: residual drift (handwritten contract pages, docstrings, allowlists, generated pages exposing internals) is held by the §8 gates + the per-estate ownership rules in §4, not by the IA shape alone (Codex round 3).
- **FastAPI's single linear Tutorial rail** is grafted as the Tutorials arm because strict Diátaxis scored *lowest* on newcomer onboarding (four equal mode cards reproduce today's confusing grid). The rail gives one CTA per on-ramp page, one new concept per lesson, every step a runnable inline `run()`, and a structurally pre-decided "Next →".
- **Generate-from-source** (autosummary-recursive over the 357 API symbols; sphinx-gallery over `examples/*.py`) is adopted for the drift-prone Reference and Examples estates — *generated-not-handwritten is the only true zero-drift mechanism*, and it self-heals the audit's stale counts (#16), undocumented examples (#17), and missing R04 (#34).
- **Verb/goal/persona routing** (the task-first candidate's idea) is kept as **navigation cards + "Next goals →" links**, never as duplicated content tracks — which is exactly what kept that candidate lowest on maintainability.

The work is sequenced so **correctness lands before structure**: P0 metadata + wrong-API triage (1–2 days, no IA change), P1 style sweep + lint, P2 safety-net CI, then P3 the structural IA migration, P4 reference unification. Definition of done is measurable (§9): 0 style violations, 0 stale versions, exactly one front door, one Reference home, all snippets execute in CI, mean health ≥ 4/5.

---

## 1. How the architecture was chosen (panel result)

Three complete IAs were designed independently, then scored 1–10 by four judges, each through one lens:

| Candidate | Newcomer onboarding | Maintainability/SSOT | Fit to API surface | Enforceability | Outcome |
|---|:--:|:--:|:--:|:--:|---|
| **1 — Strict Diátaxis** | 6 | **9** | 8 | **9** | Won SSOT + enforceability; *lost* onboarding (4 equal cards fragment topics). |
| **2 — Tutorial-first (FastAPI)** | **9** | 7 | **9** | **9** | Won onboarding + fit + enforceability; risk of teach-vs-explain duplication. |
| **3 — Task/persona-first** | 8 | 6 | 8 | 8 | Best anti-drift *generation* idea; but 7 task tracks re-invite concept duplication. |

The lenses split cleanly, so the synthesis is a **hybrid, not a single winner**: Candidate 1's SSOT skeleton + placement law as the spine; Candidate 2's linear rail + executable gallery + IA-agnostic CI gates as the Tutorials arm and safety net; Candidate 3's generate-from-source as the *implementation* of Reference and Examples; its task routing demoted to navigation. This is the standard best-practice resolution (it is, in effect, how Django, FastAPI, scikit-learn, and Polars docs actually compose these patterns).

**Patterns borrowed, named:** Diátaxis four-mode split + SSOT law (backbone); FastAPI tutorial-first single rail, one-CTA-per-page, one-concept-per-page, every-snippet-runnable (Tutorials arm); scikit-learn User-Guide(How-to) + API-Reference + executable-gallery separation; Polars/Pydantic mkdocs-material clarity + per-page mode badges; scikit-learn/sphinx generate-from-source for zero drift.

---

## 2. Target information architecture

```
Start here  (index.md — THIN ROUTER; teaches nothing)
├─ "Start here → first run"  (the single primary CTA)
└─ "I already have a goal →"  (nav links into Tutorial/How-to/Reference — no content)

Get Started / first run   (install + ZERO-DATA generate.regression() inline run(); only forward link: "Continue to the Tutorial →")

Tutorial   (learning-oriented · single linear FastAPI rail · prev/next · one new concept/lesson · every step a runnable inline run() · mode badge · sideways "Why it works →" Explanation + "Run this yourself →" Gallery)
   L1 Your first model (swap synthetic → your own folder)   L8  Visualization
   L2 Preprocessing (SNV)                                    L9  Hyperparameter tuning (generators)
   L3 Datasets & loading                                     L10 Branching & merging
   L4 Cross-validation                                       L11 Augmentation
   L5 Models & scoring/refit                                 L12 Export & deploy (.n4a, predict reuse)
   L6 Reading results & predict()                            L13 Retrain & transfer (full|transfer|finetune)
   L7 Classification

How-to guides   (task-oriented recipes · goal-titled · self-contained · NO concept-teaching, link to Explanation · NO "why")
   Data (loading · dataset configuration · repetitions & aggregation · multi-source · signal types · synthetic) · Preprocessing · Models · Pipelines · Augmentation · Predictions · Deployment · Visualization · Operations · Troubleshooting

Reference   (information-oriented · GENERATED where possible · ONE home — api/ dissolved into it)
   Syntax & keywords · Operator catalog · Public-API contract pages · reference/api/ (357 autosummary-recursive stubs) · DatasetConfigs (full param table: configurations, task_type, signal_type, repetition, aggregate, aggregate_method, aggregate_exclude_outliers, global_params) · Configuration · Storage & workspace · Session · Metrics · CLI

Explanation   (understanding-oriented · the SINGLE canonical concept home · read not run · links DOWN to Reference, never restates a signature)
   pipelines · datasets-&-configuration · repetitions-&-aggregation (NIRS-critical) · cross_validation · branching_and_merging · generators · augmentation · predictions_and_deployment · mental_models/overview · workspace & reproducibility

Examples gallery   (executable · sphinx-gallery over examples/*.py · tagged by lesson/recipe · counts derived from disk)
   User (U) track → lessons/recipes   ·   Reference (R) track → Reference pages   ·   (Developer D track lives under Contributing)

Contributing & internals   (separate audience · NEVER a user front door · internally split by the same four modes)
   Architecture · Extending (controllers, custom operators) · Internals (caching, artifacts, metadata, synthetic) · Testing & dev gallery · Documentation style (STYLE.md) · Migration

About   (all metadata source-derived)   changelog · license (dual, default AGPL-3.0-or-later) · citation · roadmap pointer

llms.txt   (non-nav machine on-ramp; ex ai_onboarding.md; regenerated from canonical anchors)
```

---

## 3. The single front door (resolving audit #6)

Today: **four** competing front doors (`getting_started/`, `concepts/`, `onboarding/`, `ai_onboarding.md`) plus a 6-card landing grid; foundational concepts taught 3–4×. Resolution:

1. **`index.md` becomes a thin router** — one prominent *"Start here → first run"* CTA, four secondary Diátaxis mode cards + one Examples card, and a small *"I already have a goal →"* nav block (the only surviving trace of persona routing — navigation, not content). It teaches nothing.
2. **One Get-Started page** whose only forward link is *"Continue to the Tutorial →"*: `pip install nirs4all`, then the literal first copy-paste, which **succeeds with zero user data** because the dataset is synthetic. It must be *complete* (imports included) so it actually pastes-and-runs:
   ```python
   import nirs4all
   from sklearn.preprocessing import MinMaxScaler
   from sklearn.model_selection import ShuffleSplit
   from sklearn.cross_decomposition import PLSRegression

   result = nirs4all.run(
       pipeline=[MinMaxScaler(), ShuffleSplit(n_splits=3), {"model": PLSRegression(n_components=10)}],
       dataset=nirs4all.generate.regression(n_samples=500, random_state=0),
       random_state=0,
   )
   print(result.best_rmse)
   ```
   Each token is GROUND_TRUTH-verified against source: `generate.regression` (`api/generate.py:914` namespace), the `{"model": ...}` dict step (`parser.py:146`), `run(pipeline, dataset, ...)` (`api/run.py:192`), `RunResult.best_rmse` (`api/result.py:333`); `random_state` makes the beginner's output deterministic. The snippet is the **whole backing example file** rendered via `literalinclude` (imports and all), and that file runs in CI — so it can never rot the way `result.predict()` (#3) did.
3. **Tutorial L1 changes exactly one thing** — swap the synthetic dataset for the user's own folder path — and the prev/next rail makes every "second task" structurally pre-decided.
4. **Concrete deletions that make the single door real:** `getting_started/concepts.md` DELETED (residue → Explanation); `persona_paths.md` DELETED (its fake `load_dataset()` #7 and 5 broken refs #8 die with it); `onboarding/` DISSOLVED into Explanation + Contributing; `ai_onboarding.md` removed from the human toctree → `llms.txt`.

Net: **exactly one Start-here, exactly one conceptual home (Explanation), zero front-door overlap.**

---

## 4. The governing law (so it stays fixed)

Two rules, adopted verbatim from the maintainability-winning candidate, are what convert this from a one-off cleanup into durable governance:

- **SSOT law:** *Explanation owns the "why"; Reference owns the signature/parameters; neither restates the other — the only permitted relationship is a hyperlink.* (This single sentence is the most drift-proof artifact in the exercise; it makes #6 and #18 *much harder* to recreate — enforced by the link-check + nitpick gates of §8, not by wishful adherence.)
- **Placement rule (Contributing checklist):** a new page is filed by the *reader's posture* — learning → Tutorial, doing-a-task → How-to, looking-up → Reference, understanding → Explanation. Topic-agnostic, so a future contributor knows where a new page goes without judgment.

---

## 5. The style guide (the 648-violation defect → infrastructure)

**THE RULE — inline by default, NOT tyrannically** (user directive, 2026-06-04: *"je veux pas une application tyrannique de l'inlining… quand le pipeline n'est pas ultra complexe, je veux que ça reste inline pour la lisibilité"*). Readability is the goal, not a blanket ban on variables:

- A **simple/moderate pipeline** — a flat list of a handful of steps, no deeply nested branch/merge/generator structure — **SHOULD be written inline** inside `run()`. This is the house style and the form every tutorial/how-to/reference snippet uses.
  ```python
  result = nirs4all.run(pipeline=[MinMaxScaler(), ShuffleSplit(n_splits=3), {"model": PLSRegression(10)}], dataset=...)   # ✅ inline
  pipeline = [MinMaxScaler(), {"model": PLSRegression(10)}]; nirs4all.run(pipeline, ds)                                  # ✗ should be inlined
  ```
- A **genuinely complex pipeline** — nested branch/merge trees, large generator sweeps, or a multi-line structure that would bury the `run()` call — **MAY be assigned to a named variable.** Cramming a 40-line nested structure into the call args is *less* readable, which defeats the purpose. Factoring it out (ideally with a descriptive name) is fine and sometimes better.
  ```python
  branching_pipeline = [                       # ✓ acceptable: complex, named for clarity
      {"branch": {"by_source": [...]}},
      {"merge": "predictions"},
      {"_cartesian_": [...]}, ...
  ]
  result = nirs4all.run(pipeline=branching_pipeline, dataset=...)
  ```
- **Reuse/comparison** (the same pipeline through several `run()` calls, or a list-of-pipelines) legitimately uses variables and is never flagged.

**THE LINT — advisory and complexity-aware, not a hard wall.** It flags only the clear-cut case: an external-variable pipeline that is **simple** (a flat list, no nested `branch`/`merge` dict, below a small step-count threshold ≈ 6) later passed as the `pipeline` arg — positionally OR by keyword — to `run()`/`session()`/`Session()`/`PipelineRunner()`. The discriminator is the *value*: a bare **name reference** bound to a *simple* literal is the suggestion target; a **list literal written inline** is compliant; a **complex** literal (nested branch/merge or above threshold) is **auto-exempt** so the rule is never tyrannical. **Codex round 3 corrected the spec:** the dominant shape is the *keyword* pass `run(pipeline=<var>)` (~281 sites, e.g. `logging.md:21`, `U01_hello_world.py:50`), not positional — flag both. An **AST-based** check (resolve the name to its assignment, measure complexity) is required for this — grep cannot judge complexity. Per-site `# noqa: inline-pipeline` with a reason remains available, and `STYLE.md` is the auditable allowlist.

> **Severity:** this is a **warning**, not a build-breaking error — at most a soft CI annotation. The metadata/version and wrong-API gates are hard; the inline-style lint nudges toward readability without blocking a PR over a judgment call about "complex enough." This is the non-tyrannical posture the user asked for.

**Order:** one mechanical sweep inlines the *simple* external-variable pipelines (the bulk of the ~648 candidate sites) for consistency; complex ones are left factored. Thereafter the advisory lint discourages regrowth of the simple-case anti-pattern.

---

## 5b. Dataset configuration & repetitions/aggregation — a first-class topic (NIRS-critical)

*(Added on user feedback, 2026-06-04: the original draft scattered this across `data/aggregation.md`, `loading_data.md`, and `reference/configuration.md` with no prominent home. In NIRS it is central — multiple scans per physical sample, replicate handling, and how `y`/metadata map to spectra are everyday concerns — so it gets dedicated coverage in every relevant mode.)*

The library already supports this richly: `DatasetConfigs(configurations, task_type, signal_type, repetition, aggregate, aggregate_method, aggregate_exclude_outliers, …)` (`data/config.py:24`), the `repetition` controller (`controllers/data/repetition.py`), and the `rep_to_sources` / `rep_to_pp` pipeline keywords. The docs must surface it at first-class altitude:

- **Explanation → `datasets-&-configuration`** (canonical *why*): what a `DatasetConfigs` is, how a dataset is assembled, how `task_type`/`signal_type` are inferred, and how `global_params` (`header_unit`, `signal_type`, `delimiter`, `na_policy`, …) shape loading. Links DOWN to the Reference param table; never restates signatures.
- **Explanation → `repetitions-&-aggregation`** (NEW dedicated concept page): the NIRS replicate model — multiple spectra per physical sample — and how `repetition` groups them, how `aggregate` (`True`/column/`None`) + `aggregate_method` + `aggregate_exclude_outliers` collapse or carry them, and how this interacts with CV (don't leak replicates across folds), with `rep_to_sources` / `rep_to_pp`. This is the page the audit's #31 (`aggregation.md` "conflates repetition with aggregation") is really asking for — done right, the two concepts are distinguished, not merged.
- **How-to → Data**: discrete recipes — *"Configure a dataset"* (the `DatasetConfigs` params + `global_params`, fixing the FAQ #11 `target_column`-in-`global_params` confusion at its source), *"Handle replicate scans (repetitions)"*, *"Aggregate replicates"* (`aggregate`/`aggregate_method`/exclude-outliers), backed by `U04_aggregation.py` and `D03_repetition_transform.py`.
- **Reference → `DatasetConfigs`** (curated contract page): the full, source-derived param table — the single lookup home, with `repetition`/`aggregate`/`aggregate_method`/`aggregate_exclude_outliers` and the `global_params` keys documented exhaustively. `reference/configuration.md` (the 392-LOC duplicate) folds in here.
- **Tutorial → L3 "Datasets & loading"** explicitly demonstrates loading a real folder *with replicates*, configuring it, and choosing an aggregation — so a NIRS newcomer meets repetitions on the canonical learning path, not buried in a how-to.

This is additive to the migration map (§9): the `data/aggregation.md` REWRITE becomes the seed of the new `repetitions-&-aggregation` Explanation page + the Data how-to recipes, and a new `Reference → DatasetConfigs` page is authored.

## 6. Examples plan (the executable backbone)

- **Gallery, not prose.** The ~80 `examples/*.py` are the single source of truth and are rendered *from* the scripts. The **8 hand-written `docs/source/examples/*.md` wrappers are DELETED and regenerated** — they had drifted (stale counts #16, undocumented U05/U06 #17, `SklearnWrapper` #12, U21/U22 #15). Because the rendered snippet *is* the CI-run file, the documented-vs-tested gap is closed for everything the gate covers (see the caveats below — not "by construction", but held by gates 1–2 of §8).
  - **Feasibility caveats (Codex round 3 — this is net-new tooling, not a reconfig):** `sphinx-gallery` is **not** in the docs deps (`pyproject.toml`, `docs/readthedocs.requirements.txt`) — it must be added. The scripts are run only after `cd examples` (`examples.yml:70`) and use relative `sample_data/...` paths (`U01_hello_world.py:71`), and several pull in heavy lazy-loaded backends (TF/Torch/JAX). So the gallery cannot *execute* all ~80 scripts in the docs build without ballooning CI. **Decision:** the gallery renders example *source* via `literalinclude` / sphinx-gallery in **no-execute mode** during the docs build; execution stays in the separate (now per-PR) examples CI with the correct cwd. Heavy/backend examples are skip-listed from any in-build execution. This keeps the docs build fast while preserving the "rendered snippet == tested file" guarantee.
- **"Doubles as integration tests" — corrected.** The examples *are intended* as integration tests, but the runner manifests are **hand-maintained and already drift**: `run.sh` is a hard-coded list that omits at least `examples/user/04_models/U06_tabpfn_nirs.py` (`run.sh:83`). So a **disk-derived manifest is a prerequisite (P2), not a current fact** — until it lands, "every example is tested" is false.
- **Embed, don't duplicate.** Each Tutorial lesson links *"Run this yourself →"* to its backing U-file; each How-to recipe links to its script; each Reference page links to its canonical R-example.
- **R04 fix (#34/#14):** convert the stale 1.8 MB health-1/5 `examples/predictions.ipynb` into a runnable `examples/reference/R04_visualization.py` on the public `PredictionAnalyzer` API — fills the missing-R04 gap and deletes the broken un-CI'd notebook in one move.
- **Weekly → per-PR CI (#19):** replace `examples.yml`'s Sunday cron (`0 0 * * 0`) with a **fast tagged U+R subset on every push/PR** (~7–10 files); the full ~80-file suite stays nightly/weekly. No gallery script sits broken on `main` for 7 days.
- **Must-call-`run()` gate (#32):** every `examples/**.py` either calls `run()`/`PipelineRunner`/`Session` **or** carries an explicit `# syntax-only` marker (verified: `D01_metadata_branching`, `D03_repetition_transform` have 0 `run()` calls — a "runs without error" check would pass them silently).
- **One disk-derived manifest** feeds `run.sh`, `run_ci_examples.sh`, README, and the gallery — the count is stamped once everywhere (#16).

---

## 7. Reference unification (resolving #18)

`api/` is **deleted, not maintained beside `reference/`** — acceptance criterion: *after the merge there is exactly one Reference tree, and a CI check fails if a second `api/` toctree reappears.* (Residual drift can still enter through the handwritten curated contract pages and source docstrings — those are owned by Reference and covered by gates 1/4 of §8, not by deletion alone.)

- **Autodoc (corrected scope, Codex round 3):** the 357 `api/*.rst` files are **module stubs** (356 `automodule` directives), and the current build generates them with **`sphinx.ext.apidoc`** (`conf.py:12`), **not** autosummary. The plan is to replace the hand-touched stubs with a **single generated root** (apidoc-regenerated on build, or migrated to `autosummary --recursive`) mounted under `reference/api/`, so adding a public symbol updates the reference with zero manual edits. **Critical constraint:** several optional backend modules import **TensorFlow/Torch/JAX at module-import time** (e.g. `operators/models/tensorflow/generic.py`), so the autodoc config **must keep `autodoc_mock_imports` / skip patterns** for those backends or the generated reference build will be fragile/fail. This is a real engineering task, not a one-line switch.
- **Hand-written `api/*.md` merged then deleted:** `api/storage.md` (~794 LOC, "Version 5.0", legacy `runs/`) → `reference/storage.md`; `api/workspace.md` (100%-duplicate PipelineLibrary) → `reference/workspace.md`; `api/module_api.md` → curated public-API pages (fixing predict() first param `source`→`model` #22); `api/sklearn_integration.md` → `reference/` sklearn page.
- **`reference/api/session.md`** wired into `reference/index` (fixes orphan #26).
- **Generator syntax** (triplicated #38) collapses to ONE canonical `generator_keywords.md`; `combination_generator.md` + the flat `configuration.md` merged away.
- **Curated contract pages** for the stable public surface own narrative + signatures; `run()` completed with the 4 missing params `refit/cache/project/report_naming` (#23); `cli.md` drops the `- runs/` line (#24).
- **Docstring fixes at source** so autodoc stops inheriting stale calls: `Predictions.load()` in `visualization/predictions.py:76` → `from_parquet`/`from_file`/`from_workspace` (#14).

---

## 8. Safety-net CI (the root-cause fix — five per-PR gates)

The audit's §5.3 root cause: *nothing that executes documentation exists.* Five gates convert the entire "ships silently" class into build failures:

1. **Doc-snippet execution via `literalinclude`-from-tested-examples (primary).** The rendered snippet is a slice of a CI-run `examples/*.py`, so doc/example drift cannot survive a green CI for any snippet routed this way (defeats #3/#4/#9/#12/#14). Snippets *not* backed by a tested file (prose-only) remain a residual drift surface — minimize them and cover with doctest where possible. `myst_nb`/doctest with tiny `generate.regression()` fixtures + `plots_visible=False` is reserved only for snippets that genuinely cannot be a whole runnable file.
2. **Per-PR examples (#19) + must-call-`run()` gate (#32).**
3. **Nitpick tightening (#19/§5.3A):** remove the blanket `nitpick_ignore_regex` masking `nirs4all.{data,pipeline,api}.*` and the `myst.xref_missing`/`ref.*` suppressions so a typo'd `:class:` ref **fails** the build; reconcile the verified contradiction — `docs.yml` builds `-W --keep-going` (strict) while `.readthedocs.yaml` sets `fail_on_warning: false` — to the same strictness (RTD → `true`). Expect a backlog of real broken refs to burn down first (staged after P4).
4. **Metadata-from-source + grep gates:** `conf.py` `release` read from `nirs4all.__version__`; license/citation templated from `LICENSE` in one place; a CI grep asserts the rendered license **string matches** the `LICENSE` dual-license statement **without banning the legitimate `CeCILL` token** (audit #1, GROUND_TRUTH); fails if any version string ≠ `__version__`.
5. **Precise inline-style lint (§5) + broken-example-ref gate + example-count stamp.** The ref-gate greps that every `examples/*.py` path referenced in docs exists on disk (catches #8, #15 — a wrong *path* in prose isn't executable, so doc-execution can't catch it).

---

## 9. Page-by-page migration map

All **140 dispositions** from the grounded walk of the real tree. Organized by destination estate; every row carries the audit defect it fixes. Legend: KEEP · MOVE · MERGE · REWRITE · SPLIT · DELETE · INTERNAL (move to `docs/_internal`/non-nav).

### Start here / Tutorial (the on-ramp)

| Current | Disp. | → Destination | Fixes / note |
|---|---|---|---|
| `index.md` | REWRITE | Start here (thin router) | strip 6-card grid + 8-path toctree; #1, #2, #16 |
| `getting_started/index.md` | REWRITE | Get Started / first run | zero-data inline `run()`; **#3** `result.predict()`→`nirs4all.predict()` |
| `getting_started/installation.md` | MOVE | Get Started → Install section | folded into the one on-ramp page |
| `getting_started/quickstart.md` | MERGE | Tutorial L1 + Get Started | kill competing quickstart; #16 |
| `getting_started/tutorial.md` | MERGE | Tutorial L1–L6 spine | seed the linear rail, one concept/lesson |
| `getting_started/concepts.md` | **DELETE** | residue → Explanation | **#6** competing front door |

### Explanation (the single concept home)

| Current | Disp. | → Destination | Fixes / note |
|---|---|---|---|
| `concepts/index.md` | REWRITE | Explanation/index | promoted to SOLE concept home; absorbs `onboarding/*` + `scoring_and_refit` |
| `concepts/pipelines.md` | REWRITE | Explanation/pipelines | absorbs `onboarding/pipeline_workflow` + `getting_started/concepts` residue |
| `concepts/datasets.md` | REWRITE | Explanation/datasets | absorbs `onboarding/data_workflow` |
| `concepts/cross_validation.md` | KEEP | Explanation/cross_validation | strong (health 4); add mode badge |
| `concepts/branching_and_merging.md` | KEEP | Explanation/branching_and_merging | keep mechanism whole |
| `concepts/generators.md` | KEEP | Explanation/generators | links DOWN to the ONE generator ref (#38) |
| `concepts/augmentation.md` | KEEP | Explanation/augmentation | — |
| `concepts/predictions_and_deployment.md` | REWRITE | Explanation/predictions_and_deployment | absorbs `understanding_predictions` concept half + `scoring_and_refit` |
| `onboarding/index.md` | **DELETE** | dissolved | **#6** front door |
| `onboarding/mental_models.md` | MOVE | Explanation/overview | strong prose |
| `onboarding/data_workflow.md` | MERGE | Explanation/datasets | concept taught once |
| `onboarding/pipeline_workflow.md` | MERGE | Explanation/pipelines | concept taught once |
| `onboarding/workspace_intro.md` | MOVE | Explanation/workspace_and_reproducibility | the "why", not the schema |
| `onboarding/controllers_intro.md` | MOVE | Contributing/Extending: controllers | dev audience |
| `onboarding/persona_paths.md` | **DELETE** | nav cards only | **#7** `load_dataset()` + **#8** 5 broken refs |
| `user_guide/scoring_and_refit.md` | MOVE | Explanation/predictions_and_deployment | SSOT for scoring/refit |
| `user_guide/predictions/understanding_predictions.md` | SPLIT | Explanation (why) + How-to (task) | misfiled concept page |

### How-to guides (task recipes; concepts stripped out to Explanation)

| Current (group) | Disp. | Fixes / note |
|---|---|---|
| `user_guide/index.md` | REWRITE | How-to estate landing |
| `pipelines/writing_pipelines.md` | SPLIT | worst style file (51); concept→Explanation, syntax→Reference; **allowlist :1242** |
| `pipelines/{branching,merging,stacking,multi_source,cache_optimization,generators}.md` | MOVE | recipes; concepts→Explanation; `generators.md` 23 violations swept |
| `models/{training,hyperparameter_tuning,deep_learning,tabpfn_nirs}.md` | MOVE | `training.md` 20 violations; tabpfn path #37 |
| `data/{loading_data,sample_filtering,signal_types,synthetic_data}.md` | KEEP | `loading_data` is the canonical `target_column`-in-`global_params` ref for #11 |
| `data/aggregation.md` | REWRITE | **#31** retitle repetition+aggregation |
| `data/index.md` | MERGE | delete stale "Coming Soon" **#30** |
| `preprocessing/{handbook,cheatsheet,snv,resampler,transfer_…cheatsheet}.md` | KEEP | health 4; SNV alias note #36 |
| `preprocessing/{index,overview}.md` | MERGE/REWRITE | collapse duplicate landings; technique residue → handbook |
| `augmentation/{sample_augmentation_guide}.md` | KEEP | backs Tutorial L11 |
| `augmentation/augmentations.md` | **SPLIT** | **#40** — not just a stray header: it carries implementation details, factory/config guidance, and testing guidelines (`augmentations.md:978`). Split three ways — augmenter catalog → Reference/augmentations, how-to recipe → How-to/Augmentation, factory/testing internals → Contributing. |
| `augmentation/synthetic_nirs_generator.md` | REWRITE | **#13** `examples.synthetic`→`nirs4all.synthesis` |
| `predictions/{making_predictions,advanced_predictions,analyzing_results,session_api}.md` | KEEP/REWRITE/MOVE | sweep style (`:246`,`:30`,`:217`); `analyzing_results` passes `result.predictions` **#9** |
| `predictions/index.md` | MERGE | remove phantom `PredictionResultsList` **#25** |
| `predictions/exporting_models.md` | MOVE | merge w/ `deployment/export_bundles`; **#15** |
| `deployment/index.md` | REWRITE | **#4** `predict(bundle=)`→`model=` |
| `deployment/{export_bundles,prediction_model_reuse,retrain_transfer}.md` | MOVE | `export_bundles` **#15** U21/U22; backs L12/L13 |
| `visualization/index.md` | REWRITE | **#9** phantom methods + `PredictionAnalyzer(result)` |
| `visualization/prediction_charts.md` | MOVE | **#9** real `plot_*` methods; links R04 |
| `visualization/in_pipeline_charts.md` | MOVE | **#10** phantom operator classes |
| `visualization/{shap,pipeline_diagram}.md` | MOVE | recipes |
| `logging.md` | MOVE | How-to/Operations |
| `troubleshooting/faq.md` | REWRITE | **#11** `y_column`→`target_column` in `global_params` |
| `troubleshooting/dataset_troubleshooting.md` | KEEP | — |
| `troubleshooting/migration.md` | MOVE | → Contributing/Migration |
| all `user_guide/*/index.md` | MERGE | folded into one How-to index |

### Reference (one generated home; api/ dissolved)

| Current | Disp. | Fixes / note |
|---|---|---|
| `reference/index.md` | REWRITE | the ONE reference home; mounts autosummary api/ tree |
| `reference/pipeline_syntax.md` | REWRITE | 47 violations; drop "Dec 2025/Phase 3" footer #2 |
| `reference/pipeline_keywords.md` | KEEP | backed by R03 |
| `reference/generator_keywords.md` | KEEP | **THE** canonical generator page (#38) |
| `reference/combination_generator.md` | **DELETE** | merged into generator_keywords (#38) |
| `reference/operator_catalog.md` + `{transforms,models,splitters,filters,augmentations}.md` | KEEP | `models.md` add SNV import #27 |
| `reference/predictions_api.md` | REWRITE | **#23** add `refit/cache/project/report_naming`; curated run/predict pages |
| `reference/configuration.md` | **DELETE/MERGE** | 392-LOC dup → `configuration/index.md` |
| `reference/configuration/{index,cache_config}.md` | KEEP | one Configuration home |
| `reference/storage.md` | REWRITE | **#18** absorb `api/storage.md`; drop "Version 5.0" |
| `reference/workspace.md` | REWRITE | **#18** absorb `api/workspace.md`; keep "no runs/" |
| `reference/metrics.md` | KEEP | — |
| `reference/cli.md` | REWRITE | **#24** remove `- runs/` |
| `reference/api/session.md` | MOVE | **#26** wire into reference/index |
| `api/module_api.md` | MERGE | curated public-API; **#22** `source`→`model` |
| `api/sklearn_integration.md` | MOVE | reference sklearn page (health 4) |
| `api/storage.md` | **DELETE** | **#18** merged → reference/storage |
| `api/workspace.md` | **DELETE** | **#18** merged → reference/workspace |
| `api/*.rst` (357 stubs) | REWRITE | autosummary-recursive under reference/api/ |

### Contributing & internals (walled-off audience)

| Current | Disp. | Fixes / note |
|---|---|---|
| `developer/index.md` | REWRITE | Contributing estate landing; **#6** (was overlapping onboarding) |
| `developer/{architecture,pipeline_architecture,outputs_vs_artifacts}.md` | KEEP | de-dup architecture overlap |
| `developer/controllers.md` | KEEP | absorbs `onboarding/controllers_intro` |
| `developer/caching.md`, `synthetic.md` | REWRITE | **#33** strip Phase/roadmap leakage |
| `developer/artifacts.md` | KEEP | — |
| `developer/artifacts_internals.md` | REWRITE | **#28** `topological_sort()` mismatch |
| `developer/metadata.md` | REWRITE | **#29** fictional `browse_folder` |
| `developer/testing.md` | KEEP | links D-track gallery |
| `migration/{duckdb_to_sqlite,storage_migration_webapp}.md` | MOVE | Contributing/Migration |

### Examples & root meta

| Current | Disp. | Fixes / note |
|---|---|---|
| `examples/index.md` | REWRITE | generated gallery landing; #16 |
| `docs/.../examples/user/{getting_started,data_handling,preprocessing,models,cross_validation,deployment,explainability}.md` | **DELETE×7** | drifted wrappers → sphinx-gallery (#16/#17/#12/#15) |
| `docs/.../examples/developer.md` | **DELETE** | 522-LOC wrapper → generated D gallery |
| `examples/user/**.py` (U) | KEEP | gallery + tests; sweep style; per-PR subset |
| `examples/developer/04_transfer_learning/*.py` | REWRITE | **#20** wrong import paths |
| `examples/developer/05_advanced_features/*.py` | REWRITE | **#32** add `run()` or `# syntax-only` |
| `examples/developer/{01,02,03,06}/*.py` | KEEP | D05–D09 = style exception (b) |
| `examples/reference/R0{1,2,3,5,6,7}.py` | KEEP | wired to Reference pages |
| `examples/predictions.ipynb` | **DELETE** | **#14/#34** → `R04_visualization.py` |
| `examples/{aom_paper,pipeline_samples,scripts,_internal,…}` | INTERNAL | tooling; **#35** emoji fix |
| `ai_onboarding.md` | INTERNAL | **#6** out of human toctree → `llms.txt` machine file; fix license #1 / version #2 in its regen source |
| `Roadmap.md` (root) | INTERNAL | **#33** → `docs/_internal` + About pointer |
| `README.md` / root `CLAUDE.md` / `INSTALLATION.md` | REWRITE | **#1** CLAUDE.md CeCILL-only→dual; **#2** README citation 0.9.0→0.9.1; README is the templated license source |
| `CHANGELOG.md` | KEEP | → About/changelog; verify current to 0.9.1 (audit root-meta scope) |
| `CONTRIBUTING.md` / `CONTRIBUTING_FR.md` | KEEP | → Contributing landing; add the doc placement-rule + `STYLE.md` pointer |
| `AGENTS.md` | KEEP | repo-root agent guide; unchanged, not a docs-site page |

**Totals (reconciled — Codex round 3).** The underlying migration data in `data/proposal_migration.json` is genuinely **page-by-page: 140 disposition entries** (KEEP 46, MOVE 30, REWRITE 31, MERGE 14, DELETE 14, MOVE_TO_INTERNAL 3, SPLIT 2). The §9 table above **condenses those 140 into ~92 rows** for readability (grouping e.g. all `examples/user/01_*/*.py` into one row) — the table is a digest, the JSON is the exhaustive list. **File removals:** 14 explicit `DELETE` dispositions, **plus** 4 sources that are merged-then-deleted (`api/module_api.md`, `reference/combination_generator.md`, `reference/configuration.md`, and `api/sklearn_integration.md` whose original is removed after the move) → **~18 files deleted**, the 357 `.rst` stubs replaced by a generated root, and `ai_onboarding.md` + `Roadmap.md` *relocated* (not deleted). **New pages to author (13 work-items, several plural):** thin-router `index.md`; Get-Started; the 13-lesson Tutorial rail (one work-item, 13 pages); Explanation index + overview + workspace_and_reproducibility; curated Reference public-API contract pages (one work-item, ~9 pages); `reference/api/` generated root; `R04_visualization.py`; Examples gallery index; Contributing/index; `STYLE.md`; About estate; `llms.txt`; the placement-rule checklist; **(added on user feedback)** the `repetitions-&-aggregation` Explanation page and the curated `Reference → DatasetConfigs` page (NIRS-critical — see §5b).

---

## 10. Execution roadmap (correctness before structure)

| Phase | Goal | Key tasks | Effort | Depends on |
|---|---|---|---|---|
| **P0 — Quick wins** | Stop misleading newcomers in 10 min; clear all CRITICALs. No IA change. | Dual-license fix (#1); version/copyright from source (#2); ~15-item wrong-API triage (#3,#4,#7–#15); notebook→R04 (#14/#34); transfer imports (#20) | **1–2 days**, 1 reviewer | none — start now |
| **P1 — Style sweep + advisory lint** | Inline the *simple* external-variable pipelines for consistency; leave complex ones factored; discourage regrowth. | sweep **survivor pages + ALL examples only** (skip the ~18 pages marked DELETE); inline only simple/moderate pipelines (flat, ≤~6 steps, no nested branch/merge) — **leave complex/nested pipelines as named variables** (non-tyrannical); author `STYLE.md` + allowlist; AST-based complexity-aware positional-OR-keyword **warning** (not a hard gate). Migrated content swept as it lands in P3. | **2–3 days** | P0 |
| **P2 — Safety-net CI** | Bring docs into the safety net. | sphinx-gallery + literalinclude + fixtures; `examples.yml` off cron → per-PR subset + must-call-`run()` gate (#32); broken-ref + count-stamp gates (#8/#15/#16); metadata grep gates | **4–6 days** | P0, P1 |
| **P3 — Structural IA migration** | 4 front doors → 1; build Explanation + Tutorial rail; delete duplicates. | thin-router `index.md`; Get-Started + 13-lesson rail; Explanation as sole concept home (dissolve `onboarding/`, delete `concepts.md`+`persona_paths.md`); re-home `user_guide/**`→How-to, `developer/**`→Contributing; repoint all cross-links (per the NO-SEMANTIC-SEARCH rule) | **1.5–2 weeks** (119 pages; ≤5-file sub-agent phases) | P0, P1, P2 |
| **P4 — Reference unification + nitpick burn-down** | One generated Reference home; eliminate duplicate trees; go strict. | merge `api/*`→`reference/*` then delete; 357 stubs→autosummary-recursive; complete `run()` ref (#23); fix cli `runs/` (#24), imports (#27), `PredictionResultsList` (#25); remove blanket `nitpick_ignore_regex`, burn down broken refs, flip RTD `fail_on_warning:true`; clear LOW #33,#31,#35–#40 | **~1 week** | P2, P3 |

The **≤5-files-per-phase / sub-agent-swarm** discipline (per the repo's CLAUDE.md agent directives) applies to P3/P4 especially.

---

## 11. Success metrics (definition of done)

1. **Simple pipelines are inline; complex ones may be factored** — the advisory complexity-aware lint reports 0 *simple-case* external-variable pipelines across `docs/source/**.md` + `examples/**.py` (from ~648 candidate sites); complex/nested pipelines are exempt by design. This is a soft annotation, not a hard gate (per the non-tyrannical directive).
2. **0 stale-version tokens** — every rendered version == `nirs4all.__version__` (0.9.1); `conf.py release` sourced from it.
3. **License correct & not over-gated** — landing/citation/About match the `LICENSE` dual statement (default AGPL-3.0-or-later); grep gate green; `CeCILL` still permitted as a valid optional variant.
4. **0 wrong-API references** — all ~15 copy-paste-fatal defects fixed and *held* by doc-snippet execution.
5. **Exactly ONE front door** — `index.md` is a thin router; `concepts.md`/`persona_paths.md`/`onboarding/` gone; `ai_onboarding.md` out of human toctree.
6. **ONE Reference home** — `api/*.md` deleted, 357 stubs replaced by a generated root under `reference/api/`, session wired; a CI check fails if a second `api/` toctree reappears.
7. **All doc snippets execute in CI** — a wrong-API edit fails the build.
8. **Per-PR example execution live** — off the Sunday cron; fast subset + must-call-`run()` green; R04 exists, notebook gone.
9. **Example count derived programmatically**, identical across `run.sh`/`run_ci_examples.sh`/README/gallery; U05/U06 documented.
10. **Nitpick strict** — blanket `nitpick_ignore_regex` removed, RTD `fail_on_warning:true` == CI `-W`; 0 broken cross-refs.
11. **Mean segment health ≥ 4/5** (from audited 2.80), no segment below 3.

---

## 12. Codex review round 3 — corrections applied

Verdict: **SOLID-WITH-FIXES** — *"The IA direction is good, but the proposal is not executable as written."* Codex confirmed the front-door diagnosis, the canonical snippet's API validity (`run`/`generate.regression`/dict-model-step/`best_rmse` all real), and several migration choices (`scoring_and_refit` is genuinely Explanation; `controllers_intro` is Contributing). It forced 10 executability fixes, all verified against source and applied above:

1. **Sphinx-gallery is net-new tooling, not a reconfig** → §6 now adds the dep, renders source in *no-execute* mode in the docs build, keeps execution in the per-PR examples CI, and skip-lists heavy backend examples (`sphinx-gallery` absent from deps; scripts need `cd examples` + relative `sample_data/` paths).
2. **"Examples double as integration tests" overclaimed** → §6 corrected: manifests are hand-maintained and drift (`run.sh:83` omits `U06_tabpfn_nirs`); disk-derived manifest is a P2 prerequisite.
3. **Lint missed the dominant shape** → §5 now flags `pipeline=<var>` *keyword* passes (the majority, ~281 sites) as well as positional, with an AST-based name-resolution check.
4. **Get-Started snippet not copy-paste-complete** → §3 now includes the imports and `random_state`.
5. **Autosummary mislabeled/underspecified** → §7 corrected: 357 are *module stubs* under `sphinx.ext.apidoc` (not autosummary); `autodoc_mock_imports` for TF/Torch/JAX is mandatory.
6. **Migration count contradictions** → §9 reconciled: the JSON holds the exhaustive **140 page-by-page dispositions**; the table is a ~92-row digest; ~18 files actually deleted (14 DELETE + 4 merge-then-delete); "13 new pages" are work-items (several plural).
7. **Omitted existing pages** → §9 adds `ai_onboarding.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `AGENTS.md` rows.
8. **`augmentations.md` disposition too coarse** → changed from REWRITE to a three-way SPLIT (Reference / How-to / Contributing) — it carries factory/config/testing internals (`:978`).
9. **Roadmap sequencing conflict** → §10 P1 now scopes the sweep to *survivor pages + all examples*; migrated content is swept as it lands in P3 (nothing swept twice, nothing wasted on deleted pages).
10. **"By construction" / "structurally impossible" overstated** → §0/§1/§4/§7/§8/§11 reworded to "made hard, held by gate X"; each absolute claim now names the concrete gate + ownership rule that actually prevents the residual drift.

Transcript: `codex_reviews/phase3_proposal_review.txt`.

### Appendix — provenance

- `proposal_workflow.js` — the workflow (3 designs → 4-lens panel → synthesis → migration map); re-runnable/resumable.
- `data/proposal_designs.json`, `proposal_judgments.json`, `proposal_synthesis.json`, `proposal_migration.json` — raw artifacts.
- Grounded in `02_audit.md` (Codex round 2, SOLID-WITH-FIXES) and `GROUND_TRUTH.md`.
