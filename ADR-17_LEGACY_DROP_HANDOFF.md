# ADR-17 — historical dag-ml handoff and current rollback status

**Date:** 2026-06-30 · **Branch:** `core/dagml` · **Tag:** `dagml-adr17-complete-2026-06-30`
**Companion engine branch:** dag-ml `feat/native-scoring` @ `f58d7bf` · **Migration war-room:** `dag-ml/docs/migration-nirs4all/`

> **Current R4 status (2026-09-02):** this is a historical ADR-17 handoff, not
> the current product architecture. The Python package default is `native`.
> `engine="legacy"` remains an explicit direct-Python rollback selector, but
> ambient `N4A_ENGINE=legacy|dual` and `allow_fallback=True` are not rollback
> routes. Studio's Rust-owned product boundary cannot select legacy execution.

---

## TL;DR

This document records the earlier dag-ml migration evidence. It is superseded
for current engine selection and product topology by
`docs/source/reference/public_interfaces.md` and the native capability preflight
page. Dag-ml remains explicitly selectable, the default portable Archive V2
route is `native`, and legacy remains only as `engine="legacy"` for bounded
rollback through R4. There is no automatic fallback, and legacy removal is not
a current product-path prerequisite.

---

## 1. Current V1 posture

| Aspect | State |
|---|---|
| Engine selector | `nirs4all/pipeline/engine.py` — `resolve_engine()`, precedence: explicit arg > `$N4A_ENGINE` > `DEFAULT_ENGINE`; ambient legacy/dual selection is refused. |
| **Default engine** | **`native`**. An unqualified `nirs4all.run()` selects the portable Archive V2 path. |
| dag-ml availability | **Default and explicitly selectable** — `nirs4all.run(..., engine="dag-ml")` or `N4A_ENGINE=dag-ml`. In-process (Mechanism B) by default; subprocess CLI available. |
| Dependencies | `dag-ml>=0.2.1`, `dag-ml-data>=0.2.1` are **hard deps** (installed, so dag-ml is selectable out-of-the-box). Not moved to extras. |
| Fallback | No automatic fallback. Native selectors reject `allow_fallback=True`; direct Python rollback requires `engine="legacy"`. The strict product profile rejects legacy entirely. |
| Public API contract | The 0.9.x surface (`run/predict/explain/retrain/session/generate`, `RunResult`, workspace SQLite+Parquet, `.n4a`) is preserved on **both** engines. |

The original 2026-06-30 default transitions below are retained only as migration
history. `nirs4all/pipeline/engine.py` now defines `DEFAULT_ENGINE = "native"`.

---

## 2. Historical ADR-17 inventory

All items below are **committed on `core/dagml`**, **Codex-reviewed (SHIP)**, and covered by the parity/test gates.

### 2.1 Cutover infrastructure
- **Flip skeleton** (e5ab1387): `engine.py` selector, in-process default, hard deps, `DagMlUnavailable` typed fallback, CLI-check relocation.
- **In-process execution** (Mechanism B): byte-exact, faster than legacy everywhere (0.25–0.83×); no subprocess import tax.
- **`.so` freshness guard** (`scripts/check_so_freshness.py`, acbf779): fails the green gate if the committed `dag_ml` `.abi3.so` predates its Rust sources. **Rule: every dag-ml-core/py Rust change must rebuild+commit the `.so`.**

### 2.2 Native execution + persistence (the dag-ml path)
- Native CV aggregation, score persistence (`ScoreSet`), variant SELECT + generation, OOF/leakage safety, branch/merge/stacking, exclude, augmentation, multi-source, multivariate-y, rep-fusion — all native in dag-ml-core.
- **P3 additive persistence:** per-sample `y_pred` populated (per-fold-val + final + per-variant, alignment-proven); native results store (`native_results.py`, a369ab5b); native model-artifact persistence + `export_model` (72c71c05, f17fab46), OFF by default. (`avg/w_avg` OOF surface = deferred, needs a native dag-ml block.)
- `.n4a` export on dag-ml (d122ec18, legacy-refit bridge for the multi-model/non-joblib cases).

### 2.3 ADR-17-finish native generator coverage (all Codex-gated)
| Item | What | Commit(s) |
|---|---|---|
| **2** | `_mutex_` group>2 reconciled (legacy "not all co-occur" = issubset) | dag-ml a6ce68a |
| **1a / 1b-cartesian** | constrained `_or_`-pick + `_cartesian_` operator generators native (7 cases) | dag-ml 3f3ef1b + host |
| **1c** | multi-model `{"model":{"_or_"}}` — winner-only 32-row contract, num_predictions exempt-pinned | dd1a56ea |
| **5-A** | `_grid_` param-sweep native (+ winner config_name content-keying fix) | 5fb4a91b |
| **5-B** | **catch-22 keystone** — model-terminated constrained generator native via the `tail` mechanism; host drops the survivor `expand_spec`; fail-closed predicate (6 Codex rounds) | dag-ml f58d7bf + host 502746b9 |
| **5-C** | unconstrained `_or_`-pick/arrange + `_cartesian_` native (arrange parity confirmed) | 6260f251 |
| **5-D** | multi-step `_or_` choices native (host-only); nested generators **demoted fail-closed** (large engine change, host-flatten later path) | b9a82f6c |
| **5-E** | multi-model native operator-SELECT — **documented no-op** (already 1c-native; native-SELECT is a structural mismatch for zero gain) | — |
| **5-F** | `_zip_`/`_chain_` — native-via-variant-expand (NOT a legacy fallback); dedicated in-engine promotion **demoted/deferred** | a92ea03a |

> **Key architectural fact:** the host `_expand_operator_generators` path is **NOT a legacy fallback** — it expands
> variants in Python but runs **each through dag-ml's single-variant CV+refit**. So generator shapes routed through it
> are still **dag-ml-native**. `_sample_`/`_weights_` (statistical RNG) are a **permanent acceptable residue** on this
> path — they never bubble to legacy and have no cross-language-deterministic native primitive.

### 2.4 nirs4all-io input bridge (item 3, D-io)
- `DatasetConfigs.from_io()` (opt-in) consumes the nirs4all-io MVP `nio.load` → a real `SpectroDataset`. No circular dep, default loader untouched.
- Slices 1/2/3: directory/config/JSON/YAML/glob/arrays + 6 constructor overrides (faithful `__init__` precedence) + multi-source **parity** + relational joins / `nio.infer` **golden** (`from_io_infer`). Commits c2d7b517, 35a80cfa, a3388b96.

### 2.5 dag-ml-data provider (item 4)
- Real `DagMlDataProvider` + working `InMemoryProvider` already exist; dag-ml consumes the vtable at ABI v2. The #56 builtin-models spike is **committed-local, NOT pushed** — a **deliberate lockstep push-hold** coupled to dag-ml `feat/native-scoring`'s un-merged-to-main conformance-pack mirrors. **Releases at the cutover** (coordinated dag-ml-data → dag-ml → nirs4all push). No in-scope slice remains.

### 2.6 Validation at the tag (`dagml-adr17-complete-2026-06-30`, HEAD 5481cfd5)
- **Full `tests/` suite on dag-ml default: 8220 passed / 164 skipped / 11 xfailed / 0 FAILED.**
- **Examples (72) on dag-ml: 71/72** — the 1 fail is `U01_shap_basics` (`shap` optional dep absent), engine-independent (identical `ImportError` on legacy) ⇒ full example parity. (Fixed 1 engine-independent example bug: D07 `plt.cm.get_cmap`, e542fe0d.)
- **Dual-engine numeric conformance: 273 / 0** (exact best_score + num_predictions + winner + per-sample y_pred).
- The **11 xfailed** = documented permanent parity-debt: unseeded `_sample_`/`_weights_` RNG, `rep_to_*` legacy double-count (dag-ml is the *correct* value), augmentation-RNG, **2 pre-existing legacy-0.9.1 bugs** (`branch by_tag` bool-keys; `by_filter` missing `deserializer`). They self-surface if ever fixed.

### 2.7 Studio (deferred, unmerged)
- nirs4all-studio `feat/native-results-reader` (additive `NativeResultsAdapter` over `read_native_results`) — **not merged/pushed**, dormant until the cutover points Studio at the `core/dagml` nirs4all.

---

## 3. Archived legacy-DROP proposal

The sequence below is preserved as the original irreversible removal proposal.
It is not the current product roadmap: R4 deliberately retains explicit
`engine="legacy"` rollback while keeping it unreachable from Studio. Any future
removal remains separately user-gated and must first account for legacy users
and archives.

1. **Historical default flip — superseded** — ADR-17 temporarily set
   `DEFAULT_ENGINE = "dag-ml"`; the current default is `native`. This did not
   remove the explicit rollback lane.
2. **Remove the legacy execution engine** — the legacy orchestrator path (`PipelineRunner` / `PipelineOrchestrator` / `PipelineExecutor` legacy branch) that `engine="legacy"` selects. Decide what (if any) of that machinery dag-ml still reuses (e.g. controllers, operators, the dataset layer **stay** — they are engine-agnostic; only the legacy *scheduling/execution* path is removed).
3. **Remove the engine selector & fallback** — collapse `engine.py` to a no-op (or remove `engine=`/`$N4A_ENGINE`); remove the `DagMlUnavailable` → legacy fallback (no fallback target once legacy is gone). Decide the public `run(engine=...)` contract (drop the kwarg, or keep it accepting only `"dag-ml"`).
4. **Collapse the dual-engine test layer** — `test_conformance_dual_engine.py`, `test_parity_baseline.py` (+ the `baselines/` legacy gold), `test_dagml_run_selector.py`, `test_engine_selector.py`, the `engine="legacy"` legs of export-roundtrip/examples-smoke. These exist **only** to prove legacy↔dag-ml parity; once legacy is gone they are dead weight. Keep the *single-engine* dag-ml oracle (`cases_generators_conformance.py`, `test_dagml_operator_generation_phase7.py`).
5. **Resolve the permanent residue** — `_sample_`/`_weights_` stay on the Python-expand-on-dag-ml path (fine, no legacy involved). The `rep_to_*` strict-xfails were a *legacy* double-count artifact — they become moot (dag-ml's value is correct). The **2 legacy-0.9.1 bugs** disappear with legacy.
6. **Release the lockstep push-hold** — push dag-ml-data (#56 spike + conformance mirrors) → dag-ml `feat/native-scoring` (merge to dag-ml main) → nirs4all, as one coordinated release-train (ADR-10 ordering). `validate_contracts.py` must stay green across the pair at every push.
7. **Merge the Studio reader** — nirs4all-studio `feat/native-results-reader`; point Studio at the dag-ml-default nirs4all.
8. **Dependency posture** — with legacy gone, dag-ml/dag-ml-data are *load-bearing* (already hard deps). Consider whether the public package documents the Rust-wheel requirement.
9. **`.n4a` export** — the P1c legacy-refit bridge is used for multi-model/non-joblib/fold cases. Without legacy, that bridge must be replaced by the native export path (2c-ii `export_model` already covers single-model joblib; extend it to the bridge's remaining cases) **before** legacy is removed.

### Gates / risks before the drop
- **Backups + legacy users** (the maintainer's explicit gate) — anyone pinned to the legacy engine / legacy-produced `.n4a` / workspace must be migrated or warned.
- **0.9.x public contract** — the drop must not silently change `RunResult` semantics. Known *intended* changes already shipped: `best_X` now anchors on the SELECTED model (a 0.9.x bugfix, CHANGELOG-noted); the 1c winner-only 32-row num_predictions for multi-model sweeps.
- **Global refactoring interplay** — the default flip has landed, but the destructive removal should still be coordinated with the broader refactoring and the R4 decision to avoid double-churn.
- **Performance** — dag-ml in-process is faster than legacy on the measured workloads; no perf regression expected, but re-benchmark the public hot paths post-drop.

---

## 4. Key artifacts / pointers

- **Tags:** `dagml-adr17-complete-2026-06-30` (this state, both repos) · `dagml-parity-allgreen-2026-06-30` (the slice-C checkpoint) · `dagml-parity-allgreen` (the P1 all-green milestone).
- **Branches:** nirs4all-core `core/dagml` · dag-ml `feat/native-scoring` · nirs4all-studio `feat/native-results-reader` (unmerged) · dag-ml-data local `main` ahead-4 (held).
- **War-room:** `dag-ml/docs/migration-nirs4all/` (README, PARITY_AND_PERF_HARNESS, TARGET_RESPONSIBILITY_SPLIT, WORKING_STRATEGY, NATIVE_PERSISTENCE_LAYER_REPORT) · ADR: `dag-ml/docs/adr/ADR-17-cutover-rollback.md`.
- **Parity oracle:** `tests/integration/parity/` — `test_conformance_dual_engine.py` (dual-engine exact equality), `cases_generators_conformance.py` (the generator registry), `test_parity_baseline.py` (legacy gold), `baselines/` (committed gold).
- **The current default lives in one place:** `nirs4all/pipeline/engine.py::DEFAULT_ENGINE`.

> **Not touched by this work:** `nirs4all-io` has pre-existing uncommitted Rust changes in the `nirs4all-io-dagml`
> crate (the `SpectroDataset → CoordinatorDataPlanEnvelope` bridge) from a separate effort — left strictly alone.
