# nirs4all — Technical-Debt Handoff (remaining work)

**Date:** 2026-06-05
**Context:** Follow-up to the June 2026 debt campaign (15 commits `ed9244ac..131bf9dc`, all suite-green/zero-regression, Codex-gated). This doc hands off the items that were **deliberately deferred** because they are deep, behaviour-changing, or coupled — each warrants its own focused session. See `technical-debt-audit-2026-06-04.md` for the original audit and the project memory `nirs4all_debt_campaign_2026` for the campaign log.

**Golden rule for all of the below:** keep the campaign's discipline — fresh-context agent → Codex review (before + after) → full suite **0 fail** → commit. Tier-R items (behaviour-changing) additionally need new tests proving the new behaviour + user sign-off.

---

## 1. Cross-store transactionality (audit HIGH #3) — the flagship remaining item

### The problem
The workspace is **three independent stores** with no transaction spanning them:
- **SQLite** (`store.sqlite`): relational metadata + scores. Connection is **autocommit** (`isolation_level=None`, `workspace_store.py:254`); a `transaction()` context manager exists (`workspace_store.py:379`) but has ~1 caller.
- **Parquet** (`arrays/<dataset>.parquet` + `_tombstones.json`): dense prediction arrays.
- **joblib artifacts** (`artifacts/<hash>`): content-addressed model binaries.

Cross-store writes/deletes happen as **independent, separately-committed steps**, so a crash (or an outer-transaction rollback) mid-operation leaves the stores inconsistent.

### Concrete hazards (with file:line)
1. **Delete ordering.** Every delete does `array_store.delete_batch(...)` (writes a durable tombstone) **before** the SQLite `DELETE`: `workspace_store.py:1197,1207` (upsert), `:2430` (`delete_run`), `:2582` (`_delete_predictions_matching`). A crash between tombstone-write and SQLite-commit leaves **live SQLite rows + a tombstone**. Then **any** later `compact()` — including the existing **manual** `compact()` (`array_store.py:468`), which trusts the tombstone list with **no SQLite validation** — physically removes arrays that live SQLite rows still reference. (This is why the ArrayStore auto-compaction attempt was reverted; it made a *pre-existing* latent hazard far more likely to fire.)
2. **Flush ordering.** `Predictions.flush()` (`data/predictions.py:903-1012`) writes prediction metadata **row-by-row to SQLite** (each autocommitted) in a loop, then a **single** Parquet batch write at the very end. A crash between the loop and the Parquet write leaves SQLite prediction rows with **no arrays** — exactly the "missing_ids" orphans that `clean_dead_links` (`predictions.py:2521`) exists to repair *after the fact*.
3. **Cascade deletes** (`delete_run`, `delete_predictions`) delete Parquet then SQLite as separate autocommitted steps — dangling tombstones or orphaned metadata on crash.

### Recommended design (do these together — they unblock auto-compaction)
Aim for **"crash can only ever orphan ARRAYS, never leave live SQLite metadata pointing at missing arrays"** (orphaned arrays are harmless: reads go through SQLite, which no longer references them; a later compaction reclaims the space).

1. **Reorder delete: SQLite-first, then tombstone.** Delete the SQLite metadata rows **first** (committed), *then* tombstone the arrays. A crash in between → SQLite gone, arrays present (orphaned, safe). Touch the 4 call sites above.
2. **SQLite-validated compaction.** Give `compact()` / a new `auto_compact_if_needed()` the set of **live** `prediction_id`s queried from **committed** SQLite, and have it remove a tombstoned row **only if its id is NOT live**. This makes compaction self-correcting against stale tombstones (a tombstone for a still-live row never removes the array). It **must run on committed state** (outside any open `transaction()`), so it cannot be naively called inside a method that may be wrapped in `with store.transaction()` (Codex flagged exactly this — a rollback after compaction would resurrect SQLite rows whose arrays are gone).
3. **Flush:** wrap the SQLite metadata writes + the Parquet batch so a crash is recoverable; generalize `clean_dead_links` into a **startup reconciliation** run on `WorkspaceStore` open (reconcile SQLite ↔ Parquet ↔ tombstones), so any pre-existing inconsistency self-heals.
4. **Then, and only then,** re-add **auto-compaction** (threshold-gated, run post-commit, SQLite-validated) — the reverted feature is correct once 1+2 are in place.

### Constraints / gotchas
- `WorkspaceStore` is WAL + autocommit + thread-safe (RLock) with retry-on-lock. `_safe_execute` (`workspace_store.py:333`) silently swallows `OperationalError` for "non-critical" writes — those can vanish; surface them when adding the transactional path.
- ArrayStore has **no file lock**: two processes writing the same dataset's Parquet last-writer-wins (`array_store.py:238-289`); `save_batch` also does a full read-modify-write per flush (O(n²) I/O). Worth addressing in the same pass.
- Reads do **not** filter tombstones (`load_batch/load_single/load_dataset`) — safe today only because the facade reads via SQLite; keep that invariant or make reads tombstone-aware if any direct ArrayStore read path appears.

### Test strategy
Crash-injection tests (fail between step N and N+1 of delete/flush), a startup-reconciliation test (open a deliberately-inconsistent workspace, assert it self-heals), and a SQLite-validated-compaction test (stale tombstone for a live row → array kept). The schema contract test (`tests/regression/test_storage_schema_contract.py`) and `test_user_version_stamped` already guard the schema; bump `SCHEMA_VERSION` if the on-disk shape changes (the contract test fails loudly to force review).

---

## 2. Dependency slimming (audit packaging) — has an ordering subtlety

Core deps carry 4 heavy/dead libs a pure-inference user is forced to install.
- **`umap-learn`** — **0 imports** in the package (dead core dep). Safe to drop from `[project.dependencies]`.
- **`seaborn`** — **0 functional imports**; only a version string in `cli/installation_test.py:69`. Drop from core deps **but** first remove/guard that `installation_test` reference.
- **`shap` / `optuna`** — heavy, used lazily; belong in `explain`/`tune` extras.
- **`matplotlib`** — used, **but eagerly imported at `import nirs4all`** time: chain `data/__init__.py:8 → visualization.predictions → … → visualization/analysis/transfer.py:2 (import matplotlib)`. Verified: `import nirs4all` ⇒ `matplotlib in sys.modules`. **So matplotlib cannot be moved to an extra until the viz imports are made lazy.**

**Recommended order:** (1) make `data/__init__` (and the chain to `visualization`) import matplotlib/viz **lazily** (deferred imports), verify `import nirs4all` no longer loads matplotlib; (2) then move `matplotlib`/`shap`/`optuna`/`seaborn` to extras (`viz`/`explain`/`tune`) and drop `umap-learn`; (3) update `requirements*.txt` (3 hand-maintained files drift — generate from pyproject or add a CI check). This is a minor version bump (install-surface change) — coordinate with the studio/downstream.

---

## 3. Layering inversion: `data → pipeline`

`data/predictions.py:286,2355,2696` do in-method `from nirs4all.pipeline.storage.workspace_store import WorkspaceStore` to dodge a circular dependency (`data` is meant to sit *below* `pipeline`). Invert via dependency injection (inject the store, or move the store-reopening helpers up into `pipeline/`). Small/bounded; verify import-time side effects unchanged (golden + `import nirs4all` smoke). Promotes `data` to a true lower layer.

---

## 4. Remaining god-classes (under-tested — add tests FIRST)

The well-tested god-methods are done. What remains is in **shallowly-tested** files, so decomposing them blind is risky (cf. base_model, which needed its white-box tests modernized first):
- `synthesis/fitter.py` (5203 LOC, ~6:1 source:test) — the 2nd-largest file.
- `visualization/pipeline_diagram.py` (2184, ~0 tests), `visualization/predictions.py` (2115).
- `synthesis/_constants.py` (2844), `_aggregates.py`, `_bands.py`.
**Approach:** add behavioural/golden tests for the target methods **before** extraction (like the base_model unblock), then decompose with the standard loop. `PipelineOrchestrator._execute_refit_pass` is guarded by a `getsource` test (`test_refit_p2c.py:374`) — modernize that test first (whole-class for existence, helper-scope for ordering — see the base_model precedent in commit `4c828a9b`).

---

## 5. Smaller items
- **Registry priority tie-break:** same-`priority` controllers resolve by import order (non-deterministic). `controllers/registry.py:23` sorts only by `priority`; the router returns `matches[0]`. Add a deterministic secondary tie-break (e.g. specificity or an explicit ordinal). *(The separate `reset_registry` rebind bug is already fixed — commit `131bf9dc`.)*
- **mypy is lenient:** `[tool.mypy]` sets only `warn_return_any`/`warn_unused_ignores`/`ignore_missing_imports=true`; no `check_untyped_defs`/`strict`. Tighten (at least `check_untyped_defs = true`) once `_aom_nirs` is replaced, so the gate is meaningful.
- **`_aom_nirs`:** still carved out of ruff+mypy (pyproject `extend-exclude` + `[[tool.mypy.overrides]] ignore_errors`). Owner is replacing it (nirs4all-methods integration); remove the carve-out when it lands. 4 AOM estimators had their MRO fixed (`(Mixin, BaseEstimator)`) in commit `80256501` — re-apply if the replacement regresses sklearn tags.
- **Version single-sourcing / conda drift:** `pyproject.toml:7` + `__init__.py:45` duplicate `0.9.1`; conda recipe was at `0.8.11`. Use `dynamic = ["version"]`.

---

*Handoff produced 2026-06-05 after the 15-commit debt campaign. Branch `docs/overhaul-p0-correctness` merged to `main`. The contract snapshot tests (`tests/regression/`) now guard the public API + SQLite schema + `.n4a` manifest against silent drift while this work proceeds.*
