# nirs4all — Technical Debt Audit

**Date:** 2026-06-04
**Scope:** `/home/delete/nirs4all/nirs4all` — the Python library only (`nirs4all/` package, `tests/`, packaging/CI). Excludes the webapp and sibling repos.
**Method:** 7 parallel evidence-based sub-audits (architecture, code health, lint/type gates, tests, public API, data/storage, deps/packaging/CI). Every claim below was verified with commands or file reads; line numbers are `file:line` relative to the package root.
**Why now:** preparing a "clean version" with large planned changes. This document is the baseline to refactor against.

> Companion docs (read alongside, do not duplicate):
> - `docs/_internal/python-library-beta-audit.md` (2026-02-20) — beta readiness.
> - `docs/_internal/god_classes_modularization.md` (2026-04-01) — the >2000 LOC files and proposed splits.
> This audit **supersedes their status claims** where they disagree with today's measurements.

---

## 1. Executive summary

The library is scientifically strong and broad, but it has **regressed on its own quality gates since February** and carries **structural debt that was identified months ago and never paid down**. None of the 12 god classes flagged on 2026-04-01 were refactored; most grew. The single most important finding is not any one bug — it is that **the quality gate stopped being enforced on at least one merge**, and large unverified code entered the package.

### The headline: gate regression, one dominant cause

| Gate | Feb-2026 claim | Today (2026-06-04) | Cause |
|---|---|---|---|
| `ruff check nirs4all` | All checks passed | **453 errors** | **450 inside `operators/models/_aom_nirs/`**; 3 elsewhere (`aom_pls.py:18` I001 + `transforms/nirs.py:271,395` F811 `scale`-redefinition) |
| `mypy nirs4all` | 0 issues / 434 files | **140 errors / 36 files** (496 files) | **135 inside `_aom_nirs/`**; 5 elsewhere (`residual.py:86,97,100`; `torch_model.py:374`; `residual_model.py:52`) |
| `# type: ignore` | 0 | **28** across 15 files | drift |
| `print()` in lib | 0 | **158 executable** (423 raw incl. doctests) | drift |
| `pytest tests` (collect) | green | **1 collection error** (39 tests dead since 2026-02-19) | renamed operator never cleaned up |

**Root cause:** commit `404cd67e` "Migration to gbeurier/aom-nirs" (2026-05-17) vendored **57 files / 17,683 LOC** into `nirs4all/operators/models/_aom_nirs/` (AOM-PLS/Ridge/Fast/POP-PLS paper code) that was **never linted or typed to project standards**. The CI gate (`CI.yaml`: `ruff check .` + `mypy nirs4all`, with `tests` gated on both) and `pre-publish.sh` are correctly configured — so this code should not have been mergeable. **The gate definition is fine; enforcement on that merge failed.** This is the load-bearing lesson for the "clean version": *the config is not the problem, merge discipline is.*

Good news from this framing: outside `_aom_nirs/` the package is nearly clean — **3 ruff errors** (1 import-sort, 2 `scale`-redefinition F811 in `transforms/nirs.py` — a real shadowing smell worth a look) and **5 mypy errors** in 3 files. The debt is concentrated, not diffuse rot.

> **Codex review correction (2026-06-04):** the original draft mis-stated "1 ruff / 2 mypy elsewhere" and placed the 2× F811 inside `_aom_nirs/`. Verified counts: 3 ruff + 5 mypy outside; the F811s are in `transforms/nirs.py:271,395`, not aom. `_aom_nirs` duplicate-definition issues are mypy `no-redef` (`pls/operators.py:374,380`), not ruff F811.

### Debt scorecard

| Dimension | State | Trend since Feb/Apr |
|---|---|---|
| Lint/type gate health | 🔴 Regressed (concentrated in vendored code) | ↓ |
| Architecture / god classes | 🔴 12 god classes, 0 refactored, most grew | ↓ |
| Layering & coupling | 🟠 `data → pipeline` inversion; registry priority collisions | ↓ |
| Code health (logging, error handling) | 🟠 158 prints, 42 silent `except: pass`, real compat shims | ↓ |
| Test suite | 🟠 broad (1.8:1) but 1 broken module, no coverage floor, big modules untested | → |
| Public API / stability contract | 🟠 declared stable, **largely unenforced**; no deprecation policy | → |
| Data/storage durability | 🟠 no SQLite schema version; non-transactional cross-store writes; unbounded Parquet growth | → |
| Packaging/metadata | 🟢 mostly fixed (py.typed, license) — 🟠 version not single-sourced, bloated core deps, conda drift | ↑ |
| Governance/CI scaffolding | 🟢 improved (CODEOWNERS, SECURITY, codeql, dependency-review) | ↑ |

---

## 2. Priorities for the clean version

Ordered by **leverage for the planned refactor**, not just severity. Effort: S(<½d) M(1–3d) L(1–2wk) XL(>2wk).

### P0 — do before any large refactor (these are the foundation)

1. **Restore gate enforcement** *(S)* — make `ruff check .` + `mypy nirs4all` + `pytest` required status checks on the default branch so a red gate blocks merge. Audit why `404cd67e` landed red. Without this, every fix below re-rots. **This is the #1 action.**
2. **Decide `_aom_nirs/`'s standard** *(S→XL)* — either (a) auto-fix + type-clean it to library standard (`ruff --fix` clears 373/453 mechanically; remaining ~80 + 138 mypy = M–L), or (b) formally carve it out with documented `per-file-ignores` and narrow the `Typing :: Typed` claim accordingly. Pick one explicitly; today it's an unmanaged exception.
3. **Fix the broken test module** *(S)* — `tests/unit/operators/models/test_new_operators.py:14` imports a removed `FFTBandpassOperator`; 39 tests dead since 2026-02-19 and `pytest tests` exits non-zero. Fix or delete.
4. **Add the missing contract enforcement** *(M)* — before changing things, freeze what must not silently change:
   - `inspect.signature` pins for the 6 public functions + `__all__` snapshots.
   - SQLite **schema golden test** + introduce `PRAGMA user_version`/`SCHEMA_VERSION`.
   - `.n4a` manifest golden test against `BUNDLE_FORMAT_VERSION`.
   These let you refactor aggressively and *know* when you break the contract.
5. **Single-source the version** *(S)* — `0.9.1` is hard-coded in `pyproject.toml:7` and `nirs4all/__init__.py:45`; conda recipe is stuck at `0.8.11`. Use `dynamic = ["version"]`.

### P1 — pay down during the refactor (high-value structural)

6. **Split the top god classes** *(XL total)* — execute the 2026-04-01 plan, starting with `controllers/data/merge.py` (5,402 LOC) and consolidating the **triplicated OOF reconstruction** logic. Then `workspace_store.py` (3,084, 84 methods) and `data/predictions.py` (2,740, 69 methods, `top()` = 293 LOC).
7. **Fix the `data → pipeline` layering inversion** *(S/M)* — `data/predictions.py:286,2355,2696` do in-method `from nirs4all.pipeline.storage import WorkspaceStore` to dodge a circular import. Invert the dependency (inject the store). Unblocks treating `data` as a true lower layer.
8. **Storage durability** *(M–L)* — wrap cross-store flush/cascade in transactions (`workspace_store.py` autocommit + separate Parquet I/O → crash leaves orphans); add a schema-version gate; auto-compact ArrayStore tombstones (Parquet grows unboundedly today).
9. **Establish a deprecation policy** *(M)* — there is **no** `DeprecationWarning` infrastructure in the public API. For a library about to make big changes with a webapp depending on it (118 import sites, many into internals), this is the soft-landing mechanism you'll need.

### P2 — hygiene (do continuously)

10. Convert the 158 library `print()` to the existing `core/logging` stack; audit the 42 silent `except Exception: pass` (esp. `dataset.py:1985,2002` silent incomplete metadata; `stacking_refit.py:346-416` masking failures as `0`/`{}`).
11. Delete real backward-compat shims (`utils/model_utils.py`, `splitters/split.py:355-369`) and the **18 unimported dead model files** in `operators/models/tensorflow/legacy/` (~190 KB; only 4 of 22 are used; dir is also a misnomer for live code).
12. Slim core deps: `umap-learn` (0 imports) and `seaborn` (0 functional imports) are dead core deps; move `shap`/`optuna`/`matplotlib` to extras.
13. Add a coverage `fail_under` floor and run `--cov` on the PR gate (today coverage is informational, never gates).
14. Consolidate ~15 ad-hoc hashing/fingerprint helpers and 3 overlapping SHA256 functions around `utils/hashing.py`.

---

## 3. Findings by dimension

### 3.1 Architecture & god classes

**Zero of the 12 god classes from 2026-04-01 were refactored; most grew.** The proposed split packages (e.g. `controllers/data/merge/`) do not exist on disk.

| File | LOC (Apr → now) | Classes | Severity | Issue |
|---|---|---|---|---|
| `controllers/data/merge.py` | 5402 → 5402 | 7 | CRITICAL | `MergeController` monolith; `_execute_branch_merge()` 355 LOC |
| `synthesis/fitter.py` | 5203 → 5203 | 24 | CRITICAL | 24 classes / 8 domains; `RealDataFitter` ~1400 LOC |
| `pipeline/storage/workspace_store.py` | 2833 → **3084** | 1 | CRITICAL | 84 methods, grew +251 |
| `data/predictions.py` | 2503 → **2740** | 2 | HIGH | 69 methods; `top()` 293 LOC |
| `pipeline/execution/orchestrator.py` | 2117 → **2433** | 1 | HIGH | `execute()` **659 LOC**, grew +46 |
| `controllers/data/branch.py` | 2320 → **2418** | 1 | HIGH | parallel/sequential dispatch duplicated |
| `controllers/models/base_model.py` | 2272 → 2272 | 1 | HIGH | `launch_training()` 252 LOC |
| `data/dataset.py` | 2083 → 2094 | 1 | HIGH | facade over accessors (less alarming than LOC) |
| + `meta_model.py`, `nirs.py`, `pipeline_diagram.py`, `_constants.py` | — | — | MED | unchanged |

**New debt not in the Apr doc:** `synthesis/` alone is **34,625 LOC** across 9 files >1000 LOC (the doc flagged only 2). Also `pipeline/resolver.py` (1601), `stacking_refit.py` (1543), `controllers/models/stacking/reconstructor.py` (1426).

**Coupling / boundary violations:**
- **`data → pipeline` layering inversion** — `data/predictions.py:286,2355,2696` import `WorkspaceStore` inside method bodies (deferred import to dodge a circular dependency). `data` should sit *below* `pipeline`.
- **Split-brain merge feature** — controller in `controllers/data/merge.py` (5402), its config dataclasses in `operators/data/merge.py` (1262). One feature, two packages.
- **OOF reconstruction triplicated** — the Apr doc asked for one `OOFReconstructor`; logic now lives in 3 places (`merge.py`, `stacking/reconstructor.py`, `stacking/branch_validator.py`). Debt *increased*.
- **Controller registry priority collisions** — `CONTROLLER_REGISTRY` (`controllers/registry.py:8`) is a global list sorted by `priority`; 10 controllers share priority 10, 4 share 5, 3 share 4. Ties break by **import order** (non-deterministic w.r.t. intent). The router picks the first matching `matches()`, so two same-priority matching controllers resolve by accident. Latent dispatch-ambiguity bug. *(S to fix: deterministic secondary tie-break.)*
- **Healthy:** `operators/` and `core/` have no upward imports. `__init__.py` is **not** bloated (114 lines, 22 `__all__`; the "44 commits" churn is version bumps). Leave it.

### 3.2 Lint & type gate health

Covered in §1. Additional detail:
- **ruff breakdown:** 372/450 aom errors auto-fixable (safe); the non-auto-fixable in aom include **8× B905** (`zip` without `strict`) + **7× B007** (unused loop vars) + duplicate definitions (mypy `no-redef`, `pls/operators.py:374,380`). Treat as possible real bugs — review individually, do not blanket-fix.
- **Non-aom ruff (3):** `aom_pls.py:18` (I001 import-sort, fixable) + **2× F811** in `operators/transforms/nirs.py:271,395` — `scale` (imported at `nirs.py:8`) is shadowed by a parameter/local. A genuine shadowing smell in production transform code; verify it isn't masking a bug before "fixing" the lint.
- **5 non-aom mypy bugs:** `operators/models/residual.py:86,97,100` (a `dict` assigned to a `str`-typed var then `**`-unpacked — likely a runtime bug); `controllers/models/torch_model.py:374` and `residual_model.py:52` (`no-any-return`).
- **mypy config is lenient, not strict** (`[tool.mypy]`: only `warn_return_any`, `warn_unused_ignores`, `ignore_missing_imports=true`; no `disallow_untyped_defs`/`check_untyped_defs`/`strict`). The Feb "0 issues" pass was itself a *lenient* pass — precisely how 57 unchecked files could be added and trip only the few enabled rules. Recommend `check_untyped_defs = true` at minimum once `_aom_nirs` is resolved.
- **`Typing :: Typed`:** `py.typed` ships correctly (PEP 561 mechanically true), but with 140 mypy errors the *quality* behind the badge regressed.

### 3.3 Code health (logging, error handling, dead code)

| Marker | Count | Verdict |
|---|---|---|
| `print()` executable | 158 | regression (logging stack exists: 1013 logger call-sites) |
| `except Exception: pass` (silent) | 42 | swallows errors against the project's own no-defensive-handling rule |
| `except Exception` (total) | 290 | many legit at boundaries; ~50 swallow silently |
| `legacy` markers | 126 | mostly docstrings; one misnamed live dir + ~18 dead files |
| real backward-compat shims | several | violate the no-compat-shim rule |

- **Worst silent print sites** (not CLI/verbose): `pipeline/config/component_serialization.py:286,294,301,308,350` — unguarded error prints in the deserialization hot path; `data/targets.py:408` — unconditional task-type-change warning in core data logic.
- **Dangerous swallows:** `data/dataset.py:1985,2002` (metadata/wavelength compute → silent incomplete metadata); `pipeline/execution/refit/stacking_refit.py:346-416` (refit failure masked as `0`/`{}`).
- **Real compat shims (delete):** `utils/model_utils.py` (whole module re-exports with `DeprecationWarning` "for stale .pyc caches"); `controllers/splitters/split.py:355-369` (`group`→`group_by` alias); `store_queries.py:322,384` (functions marked `.. deprecated::` but **still actively called** from `workspace_store.py:1850,1926` — false/half-done deprecation).
- **Dead code:** `operators/models/tensorflow/legacy/` — 22 files, only 4 imported; **18 never imported** (~190 KB of UNet/segmentation code); dir name is a misnomer (holds live models).
- **Duplication:** 3 overlapping SHA256 helpers (`artifacts/utils.py:182`, `artifact_persistence.py:128`, `utils/hashing.py:17`); ~15 ad-hoc hash/fingerprint methods across executor/config/cache with no owning utility.

### 3.4 Test suite health

| Metric | Value |
|---|---|
| Test files / functions / classes | 395 / 7,311 / 1,643 |
| Lib : test LOC | 230,975 : 127,890 (**1.8:1**, healthy breadth) |
| Collection errors | **1** (39 tests dead since 2026-02-19) |
| Coverage gate | measured, **no `fail_under`**, not run on PR `CI.yaml` |
| Declared markers actually used | tiny (`gpu`/`torch`/`keras`/`jax`=0; `-m sklearn` selects ~6) |

- **Broken module:** `tests/unit/operators/models/test_new_operators.py:14` (`FFTBandpassOperator` removed). Hides 39 tests and makes a bare `pytest tests` non-zero; CI only survives by invoking subdirs separately.
- **Skips are legitimate** (dep-gating: 65× JAX, etc.) — low skip debt. No `assert True`, minimal sleeps — good.
- **Untested critical modules (ranked):** `controllers/models/base_model.py` (2272 LOC, **0 dedicated tests**); `visualization/pipeline_diagram.py` (2184, **0 tests, 0 references**); `orchestrator.py` (2433, 1 thin file); `synthesis/fitter.py` (5203, ~6:1 source:test). Well-covered for contrast: `merge.py`, `predictions.py`, `workspace_store.py`.
- **Over-mock hotspots** (validate wiring, not numerics): `test_training_set_reconstructor.py` (75 mocks), `stacking/test_source_model_selectors.py` (62), `data/test_merge_controller.py` (62).
- Stray hand-rolled runners in `tests/` root (`run_tests.py`, `run_runner_tests.py`) — duplicate harness debt.

### 3.5 Public API & stability contract

The 0.9.x "stable contract" (run/predict/explain/retrain/session/generate + RunResult/PredictResult/ExplainResult + SQLite/Parquet schema + `.n4a`) is **declared but largely unenforced**.

- **No signature-pin test** for the 6 public functions; no `__all__` snapshot. Only the *internal* `WorkspaceStore` has a contract test (`tests/unit/pipeline/storage/test_workspace_store_api.py`).
- **Signature inconsistencies:** `verbose` defaults `1` everywhere but `0` in `predict`; `plots_visible` defaults `False` in `run`, `True` in `explain`; catch-all kwargs are named `**runner_kwargs`/`**shap_params`/`**kwargs` inconsistently; `name` defaults are per-function magic strings.
- **Duplicated spec aliases** (`ModelSpec/DataSpec/SourceSpec/PipelineSpec`) redefined in 4 files (`run/predict/explain/retrain.py`) and **already diverged** (`DataSpec` differs between `predict` and `explain`). Needs one `api/types.py`.
- **`RunResult` is a leaky wrapper, not a frozen schema:** holds 8 private live runtime handles (`_runner`, `_refit_executor`, …), owns DB lifecycle, and `best` returns a bare `dict[str, Any]` accessed via `.get()`. Any internal refactor of those dict keys breaks downstream. Stale "Phase 1 / v0.6.0" provenance comments remain in `result.py`.
- **`__all__` leaks internals:** top-level exports `Run`, `RunStatus`, `RunConfig`, `generate_run_id`, `register_controller`, `CONTROLLER_REGISTRY` — execution/registry internals, not the documented contract. Docstring falsely claims `DatasetConfigs` is top-level.
- **No deprecation policy:** only 2 `DeprecationWarning` in the whole lib, both internal.
- **Webapp coupling is far wider than the declared surface:** `nirs4all-studio/api/` has **118 import sites** reaching into `pipeline.storage.WorkspaceStore` (un-versioned schema!), `pipeline.config.generator`, `data.parsers`/`loaders`/`detection`, `operators.*` class layout, `core.task_detection`/`metrics`. The webapp guards `import nirs4all` but **not** these internal paths — big changes will silently degrade UI features rather than fail loudly.

### 3.6 Data & storage layer

Hybrid SQLite (metadata) + Parquet (arrays) + content-addressed joblib. Decomposition is generally clean; SQL is injection-safe (parameterized `?` everywhere; the few f-string fragments interpolate only allowlist-validated identifiers, `store_queries.py:515-553`). Durability is the weak spot.

| # | Sev | Issue | Location | Effort |
|---|---|---|---|---|
| 1 | HIGH | **No SQLite schema version** (`PRAGMA user_version`/`schema_version` absent). Only additive `ALTER…ADD COLUMN` via `_migrate_schema`. A non-additive change bricks old workspaces with no diagnostic; an old lib opening a new workspace fails silently. | `store_schema.py:627-738` | M |
| 2 | HIGH | ArrayStore tombstones **never auto-compacted**; Parquet grows monotonically after deletes until manual `compact()`. | `array_store.py:418-447,468-539` | M |
| 3 | HIGH | Cross-store writes **non-transactional** (SQLite `isolation_level=None` autocommit + separate Parquet I/O). Crash mid-`flush`/cascade → orphaned metadata or dangling tombstones; `clean_dead_links` is a band-aid. `transaction()` exists but has **1 caller**. | `workspace_store.py:256,2408-2563`; `predictions.py:903-1012` | M–L |
| 4 | MED | `save_batch` rewrites the **entire** dataset Parquet on every flush (O(n²) I/O); no file lock → concurrent writers last-writer-wins data loss. | `array_store.py:238-289` | M |
| 5 | MED | `_safe_execute` silently swallows `OperationalError` after retries for log/error writes → silent loss. | `workspace_store.py:333-340` | S |
| 6 | MED | `Predictions` god class (69 methods); `top()` 293 LOC. | `predictions.py:1018-1311` | L |
| 7 | LOW | `SpectroDataset.folds` getter returns internal list by reference (mutable-state leak). | `dataset.py:1692-1694` | S |

### 3.7 Dependencies, packaging & CI

**Improved since Feb** (py.typed ships; license/classifier/LICENSE files all agree on AGPL-3.0-or-later dual; governance scaffolding added: CODEOWNERS, SECURITY, codeql, dependency-review). Remaining debt:

- **Core-dep bloat:** `umap-learn` (**0 imports** — dead), `seaborn` (**0 functional imports**, version string only — dead), `shap`, `optuna`, `matplotlib` are all **core** deps. A pure-inference user installs all of them. Move to `viz`/`explain`/`tune` extras; drop the two dead ones. *(Lazy-loading of TF/torch/jax verified PASS — they do not load on `import nirs4all`.)*
- **Version not single-sourced** (`pyproject.toml:7` + `__init__.py:45`); conda recipe drifted to `0.8.11`.
- **No lockfile**; 3 hand-maintained `requirements*.txt` duplicate pyproject and drift.
- **Release gate weaker than PR gate:** `publish.yml` builds + version-checks but **does not run `twine check`**; the strongest validation (`pre-publish.yml` / `pre-publish.sh`: twine + ruff + mypy + examples) is **manual-only** (`workflow_dispatch`). Docs build has no `-W`, so doc warnings are advisory despite the "0 warnings" goal.
- **`mypy .` vs `mypy nirs4all`:** the Feb `fck-pls` failure is sidestepped by scoping to the package (`bench/fck-pls/` is hyphenated, out of scope). Implicit and fragile — either rename `bench/fck-pls/` → `bench/fck_pls/` and re-enable `mypy .`, or document the carve-out.

---

## 4. How this maps onto "the clean version"

The planned big changes are an opportunity, but two things must be true *first* or the refactor will be flying blind:

1. **The gate must bite again (P0.1).** Every other improvement is reversible without it — `404cd67e` proves it.
2. **The contract must be pinned before you change it (P0.4).** Signature tests, a SQLite `user_version` + schema snapshot, and a `.n4a` manifest golden test convert "we think this is stable" into "CI tells us the moment it isn't." With 118 webapp import sites — most into *undeclared* internals — you otherwise cannot know what a rename breaks until the webapp does.

Then the structural work (god-class splits, layering inversion, storage durability) can proceed aggressively because regressions become visible. Sequence suggestion: **P0 (1 week) → contract pins → merge.py + OOF consolidation → workspace_store/predictions splits → storage durability → hygiene sweep**.

The encouraging part: the diffuse-looking metrics (453 ruff, 140 mypy, 158 prints) are mostly *concentrated and mechanical* — `_aom_nirs` vendoring + drift, not codebase-wide rot. The genuinely hard debt is the unrefactored god classes and the unenforced contract, both of which are well-scoped and already mapped.

---

*Generated 2026-06-04 from 7 parallel sub-audits. All findings verified against the working tree at branch `docs/overhaul-p0-correctness`. No source files were modified.*
