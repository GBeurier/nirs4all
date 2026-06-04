# nirs4all — Technical Debt Execution Plan (Codex-reviewed)

**Date:** 2026-06-04 · **Rev:** 2 (post-Codex review + fact-check)
**Derived from:** `technical-debt-audit-2026-06-04.md`.
**Overriding constraint:** **ZERO REGRESSION.** Before/after usage and numerical results must be *strictly identical*. The library is in production (consumed by `nirs4all-studio`).
**Process gate:** Codex reviews every action — the *plan* before, the *diff* after. No phase is "done" until Codex sign-off + green gate.

> **Rev-3 changelog (user directive 2026-06-04):** **No backward compatibility — pre-v1.** Remove ALL dead / deprecated / legacy code *and* BC tests. Consequences:
> - **Regression vector 2 collapses to within-version only.** Old artifacts/`.n4a` bundles created by *previous* code are **not** required to load after a refactor. Only the live, same-version fit→save→load→predict path must stay identical (already covered by golden). → **No `__module__`-preserving shims; god-class classes may move freely.** This resolves the Rev-2 decision #3 (no shim dilemma) and obeys the repo "no compat shim" rule.
> - **Dead/deprecated/legacy/BC removal is promoted to a first-class Tier-S workstream** (it is deletion of code with no *live* dependents). Items with live callers (e.g. `store_queries` "deprecated" fns still called, `stacking_refit` "Remove in 1.0.0" path still wired) are not BC — they are mislabeled live code: remove the misleading label or consolidate, but do not delete a live path. Per-item judgement, fanned out to agents, guarded by the golden net.
> - **BC tests are deleted**, not preserved (e.g. `test_new_operators.py` — already removed; it characterized the pre-May operator API: `'FiniteDifferenceOperator' object has no attribute 'initialize'`).
> - **Studio coupling** is now the user's to manage in tandem (they own the sibling repo); import-surface/studio-smoke checks stay as *awareness*, not a hard freeze.
>
> **Rev-2 changelog (what Codex's review + my fact-check changed):**
> - Corrected counts: ruff 450 in aom / **3** outside (`aom_pls.py:18` I001 + `transforms/nirs.py:271,395` F811); mypy 135 in aom / **5** outside. The 2× F811 are **not** in aom — they're `scale` shadowing in `transforms/nirs.py`.
> - **New top regression vector: serialization/module-path.** joblib artifacts + `.n4a` bundles persist class module paths → moving/renaming a persisted class breaks `joblib.load` of existing artifacts and bundles, *even with re-export shims*. This reclassifies god-class splits.
> - Moved god-class splits, dead-code removal, print→logging, `__all__` trimming, version single-sourcing **out of unconditional Tier S** into guarded/Tier-R.
> - Added: golden-run determinism controls, import-surface + studio smoke tests, artifact/`.n4a` compat tests, hash golden fixtures, the registry aliasing bug, and the `import nirs4all` → matplotlib eager-load constraint (blocks naive dep-slimming).
> - aom: Option A stays recommended but is **not "guaranteed identical" mechanically** — safe ruff fixes only (372), no unsafe fixes; B905→`strict=False`, B007 reviewed, duplicate-defs removed only if byte-identical, mypy via annotations/casts not restructuring, **module paths unchanged**. Option B is confirmed Tier R (sibling MRO differs: `(BaseEstimator, RegressorMixin)` vs `(RegressorMixin, BaseEstimator)`).

---

## 0. The seven regression vectors (what "identical" must cover)

A 231k-LOC production library can regress along more than numerics. Every change is checked against all seven:

| # | Vector | What breaks it | Guard (built in Phase 0) |
|---|---|---|---|
| 1 | **Numerical** | any logic/order/RNG change | end-to-end golden snapshots (scores + predictions) |
| 2 | **Serialization / module-path** | moving/renaming a class persisted via joblib or `.n4a` (`artifact_persistence.py:202,326`, `bundle/loader.py:333`) | artifact + `.n4a` roundtrip-load tests on **pre-change** fixtures; keep persisted classes' `__module__` stable |
| 3 | **Cache-key / hash** | changing any hash fn/serialization → flips StepCache hit/miss (`executor.py:830,840,855`) and cross-run artifact lookup (`workspace_store.py:1447,1466`) | hash golden fixtures == byte-identical |
| 4 | **Public surface / `__all__`** | studio introspects `__all__` for UI (`pipeline_service.py:397`) and imports internals directly (`WorkspaceStore` etc., ~78 import lines) | import-surface snapshot + studio smoke import test |
| 5 | **Import-time side effects** | `import nirs4all` eagerly loads matplotlib via `data/__init__.py:8 → visualization → transfer.py:2` (verified: `matplotlib in sys.modules == True`) | `import nirs4all` smoke test; **blocks** naive dep-slimming |
| 6 | **Observable stdout/stderr** | print→logging changes streams CI/studio may parse | documented as accepted; golden covers numerics |
| 7 | **Determinism / parallelism** | `random_state=None` defaults (`feature_selection.py:119,204`), joblib parallel (`orchestrator.py:396`, `branch.py:2353`), BLAS threads, `PYTHONHASHSEED` | pinned golden env + **both** `n_jobs=1` and parallel-mode matrices |

### 0.1 Phase-0 safety net (built BEFORE any change; all additive = Tier S0)
1. **Green baseline.** Fix the one broken collection (`test_new_operators.py`) so `pytest tests` is trustworthy.
2. **Golden snapshots** under a pinned env (`PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`/`OPENBLAS_NUM_THREADS=1`, fixed `random_state`): `examples/reference/R01–R07` + `pipeline_samples/test_all_pipelines.py` end-to-end scores+predictions; every re-exported AOM estimator's fit/predict/predict_proba/operator apply+adjoint/`get_params`; SQLite schema (tables/cols/indexes) + a sample `.n4a` manifest. Run **twice** to prove determinism; run once at `n_jobs=1` and once parallel.
3. **Compat fixtures:** persist a small set of artifacts (`.joblib`) and a `.n4a` bundle on the *current* code; a test reloads them after every change (vector 2).
4. **Hash fixtures:** snapshot outputs of every cache-key/hash helper on fixed inputs (vector 3).
5. **Import-surface + studio smoke:** snapshot top-level `__all__`, `operators.models.__all__`, and assert every symbol/path the studio imports still resolves (vectors 4, 5).

### 0.2 Refactor discipline
- Only changes that pass **all seven** vectors run in the automated campaign.
- Small, reversible, single-concern. After each: `ruff` + `mypy` + `pytest` + golden diff == 0 + compat reload green + hash fixtures green.
- Parallel agents on **disjoint file sets**, own worktree, integrated serially behind the gate.

### 0.3 Codex gate (per action)
- **Before:** Codex reviews step plan + exact edits; fact-check its review; reconcile before editing.
- **After:** Codex reviews the diff for behavior preservation + hidden coupling; fact-check; only then mark done.

---

## 1. Regression-risk tiering (revised)

### Tier S0 — Pure-additive safety net (no behavior surface) → do first
Contract/golden/compat/hash/import-surface tests; stale-comment cleanup; deprecation-policy *scaffolding that emits no warnings yet*.

### Tier S1 — Behavior-preserving **only if a named invariant holds** (guarded campaign)
| Item | Invariant that MUST be verified | Guard |
|---|---|---|
| `ruff --fix` safe-only (aom 372 + non-aom `aom_pls.py:18`) | no unsafe fixes; F811/B905/B007/no-redef reviewed as possible real bugs, not blanket-fixed | golden + per-fix diff |
| God-class splits (pure extraction) | **persisted classes keep their `__module__`/import path** (use shims that preserve identity); no logic change | vectors 1+2+3 (artifact/`.n4a` reload, hash, golden) |
| `print()` → `logging` | values unchanged; stdout change accepted+documented | golden (numeric) |
| Version single-sourcing (`dynamic`) | wheel/sdist metadata equals `__version__` | build + metadata equality test |
| Hashing consolidation | new hashes **byte-identical** to old | hash golden fixtures |

### Tier R — Behavior-changing → separate track, explicit new tests + user sign-off (NOT "identical results")
- Fix silent `except Exception: pass` (42 sites) — errors now surface.
- **Registry fixes** — both (a) deterministic priority tie-break and (b) the **aliasing bug** (`reset_registry()` rebinds `CONTROLLER_REGISTRY` to a new list at `registry.py:30`, but `router.py:5,26` captured the old list object → router sees a stale registry after reset). (a) can change which controller wins; (b) changes reset semantics. Both alter behavior.
- Storage transactionality / schema `user_version` / ArrayStore auto-compaction — on-disk + crash-path behavior.
- **Dep-slimming** (`shap`/`optuna`→extras; drop dead `umap-learn`/`seaborn`) — **blocked** until `import nirs4all` stops eagerly importing matplotlib (vector 5); otherwise moving matplotlib to an extra breaks plain `import nirs4all`. Sequence: make viz imports lazy first (itself Tier R), *then* slim.
- **Dead-code removal of importable public paths** (18 `tensorflow/legacy/` files) — unimported in-repo, but they are importable public module paths and potential pickled paths → needs a deprecation decision, not a silent delete.
- `__all__` trimming — public + studio UI introspection.
- Layering inversion (`data → pipeline` DI) — promote to S1 only if import-smoke + golden confirm identical import-time side effects.

---

## 2. P0.2 — Clean `_aom_nirs` (Codex-refined)

**Facts (verified):** 57 files / 17,683 LOC; 450 ruff + 135 mypy errors. 372 ruff fixes are *safe*; 20 are *unsafe* (do not apply mechanically). Re-exported as ~25 estimators via `operators/models/__init__.py:__all__`; 98 guarding tests. `aom_pls_aomlib.py` uses a separate external `aompls` package, not `_aom_nirs`.

**Option A (RECOMMENDED) — lint/type-fix in place, behavior-preserving but verified, not assumed:**
1. **AOM golden first** (part of Phase 0): for every re-exported estimator/operator — fit/predict, `predict_proba` where applicable, operator matrix apply + adjoint identity, `get_params`, pickle/joblib roundtrip, and *artifact* roundtrip.
2. Apply **only safe** `ruff --fix`; review diff; run AOM golden.
3. Residuals, reviewed individually: **B905** → add `strict=False` only; **B007** → `noqa` or prove the loop var is unused post-loop; duplicate defs (`pls/operators.py:374,380`) → remove only the byte-identical overwritten one.
4. **mypy** → prefer annotations / `cast` / targeted `# type: ignore`, never runtime restructuring.
5. **Keep all module paths unchanged** (vector 2).
Outcome: gate 453→~3, 140→~5, results provably identical, 98 tests green. The vendored tree stays but is gate-clean; externalizing to a real dependency is a later, deliberate migration.

**Option B (externalize to sibling `nirs4all-aom` now) — Tier R, NOT identical.** Confirmed structural divergence: `AOMRidgePLS` is `(BaseEstimator, RegressorMixin)` vendored vs `(RegressorMixin, BaseEstimator)` in the sibling (`ridge/aom_ridge_pls.py:157`; same for `ridge/classification.py:118`). Different MRO → different sklearn tags/`_estimator_type` resolution before any numeric difference. Defer; gate behind the AOM golden + user sign-off.

**Recommendation:** Option A now; Option B as a follow-up once nirs4all-methods integration lands.

---

## 3. Sequenced execution

**Phase 0 — Safety net (serial).** Green baseline → all Tier-S0 nets (golden, compat, hash, import-surface, studio smoke) under pinned determinism env, n_jobs=1 + parallel. Nothing else starts until these exist and are green twice.

**Phase 1 — aom lint/type fix (Option A).** 453→~3 ruff, 140→~5 mypy. Per-fix golden + AOM golden. Codex before/after.

**Phase 2 — Non-aom gate closure + version single-source.** The 3 non-aom ruff (incl. the `scale` F811 — review for a real bug first) + 5 mypy + `dynamic` version + conda sync. Additive contract snapshots wired into CI.

**Phase 3 — Tier-S1 structural (parallel, disjoint files).** God-class splits via pure extraction with `__module__`-preserving shims, starting `merge.py`; dead extraction verified by artifact/`.n4a` reload + hash + golden. prints→logging. Codex per agent.

**Phase 4 — Tier-S1 caution.** Hashing consolidation (byte-identical guard).

**Phase 5 — Tier-R track (separate, signed-off).** Each item presented with its behavior delta + new tests; executed only on approval. Includes the registry fixes, storage durability, lazy-viz-then-dep-slim, `__all__` trim, dead public-path removal, layering inversion, silent-except fixes.

**Gate after every phase:** `ruff check .` clean (modulo documented carve-outs) · `mypy nirs4all` clean · `pytest tests` green · golden diff == 0 · artifact/`.n4a` reload green · hash fixtures green · import-surface unchanged · Codex sign-off.

---

## 4. Decisions to confirm with the user
1. **aom:** Option A (in-place, recommended) now, Option B deferred — OK?
2. **Tier-R scope:** confirm silent-`except`, registry determinism, storage durability, dep-slimming, `__all__` trim, dead-path removal are **out** of the "identical results" campaign and handled as signed-off behavior changes later.
3. **God-class splits:** acceptable to ship `__module__`-preserving shims (a thin, *intentional* compatibility layer) to keep old artifacts/bundles loadable — even though the repo's stated rule is "no compat shims"? This is the one place zero-regression *requires* a shim. Need explicit OK.
4. **Determinism env** for golden (`PYTHONHASHSEED`, BLAS threads, n_jobs matrix) — confirm this is representative of how the user runs/ships.
