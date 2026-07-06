# nirs4all ↔ dag-ml compatibility ledger

**schema_version:** 1
**owner:** `nirs4all compatibility ledger`
**consumer_of:** `dag-ml/docs/contracts/parity_oracle.v1.json` (`dag-ml.nirs4all.parity_oracle.v1`)
**machine-readable companion:** `docs/compatibility.json`
**last reconciled:** 2026-07-06 against `nirs4all 95e81280202488a0b7f9504a0b1baffde65a38f4` / `dag-ml f58d7bf`
**lock:** `LOCK-PYREF` (`DEC-PYREF-001`, `DEC-PYREF-002` accepted)

This is the ADR-01 tolerance ledger the dag-ml contract names as its
`consumer_ledger` with `required_before_bridge: true`
(`parity_oracle.v1.json:5-9`). It is the single authority for **which tolerance
governs which numeric path**, and **which engine is correct** when the legacy
(sklearn) and dag-ml (Rust) engines disagree.

The enforced source of truth is the live parity harness under
`tests/integration/parity/` — every row below cites the exact `file:line` that
enforces it. This document reconciles those scattered structures into one
table; it does not replace them.

---

## §A — Tolerance bands (resolves B-009: `1e-9` contract vs `1e-3` enforced)

### A.0 The reconciliation, in one paragraph

The dag-ml contract ships a single `regression.default` profile of
`absolute_tolerance = relative_tolerance = 1e-9`
(`parity_oracle.v1.json:13-20`) whose `owner` field is literally
`"nirs4all compatibility ledger"`. **That `owner` means the contract delegates
the authoritative number to this ledger** — `1e-9` is the contract's placeholder
default, not a measured cross-engine value. `1e-9` is real and achievable **only
on a same-implementation numeric path** (the n4m kernel vs sklearn, or a dag-ml
native export reproducing itself). It is **false** on the
cross-implementation pipeline path that the dual-engine PYREF oracle actually
exercises — legacy *sklearn*-PLS vs dag-ml *Rust*-PLS — where the measured float
noise floor is ~`1e-4` to ~`6e-4`, so the enforced band is `1e-3`
(`_conformance_helpers.py:60,65`). **A single global tolerance is the bug.** The
ledger replaces it with bands keyed by `(numeric_path, metric_class)`: `1e-9` is
the `kernel_pls` band, not the cross-engine default.

### A.1 Measured cross-engine noise (the evidence the bands are sized against)

| Numeric path | Measured Δ | Source |
|---|---|---|
| PLS rmse / r2 (pipeline score) | ~`7e-6` | `_conformance_helpers.py:57-60` |
| PLS per-sample y_pred | ~`1.1e-4` | `_conformance_helpers.py:62-65` |
| y_processing-inverse per-sample y_pred | ~`6e-4` | `_conformance_helpers.py:62-65` |
| FirstDerivative-amplified PLS y_pred | ~`3.45e-3` | `test_conformance_dual_engine.py:217-218` |
| n4m kernel PLS vs sklearn (same language) | <`1e-9` | `tests/unit/operators/methods/test_n4m_ops.py:169` |
| n4m kernel SNV vs sklearn (same language) | `1e-12` | `test_n4m_ops.py:120,130` |

### A.2 The band table

| `band_id` | numeric_path | metric_class | abs_tol | measured ceiling | enforced_at |
|---|---|---|---|---|---|
| `kernel_snv` | same-impl kernel | y_transform | **1e-12** | exact | `test_n4m_ops.py:120,130` |
| `kernel_pls` | same-impl kernel | prediction | **1e-9** | ~1e-10 | `test_n4m_ops.py:169` |
| `native_export_reproduce` | dag-ml native ↔ itself | prediction | **1e-6** | — | `test_conformance_export_roundtrip.py` |
| `per_case_tight` | cross-impl pipeline | score | **1e-6** | — | case `metric_tolerances` (`baseline_vertical_slice`) |
| `cross_impl_score` | cross-impl pipeline | score (rmse/r2/acc) | **1e-3** | ~7e-6 | `_conformance_helpers.py:60` |
| `cross_impl_ypred` | cross-impl pipeline | prediction (per-sample) | **1e-3** | ~6e-4 | `_conformance_helpers.py:65` |
| `cross_impl_ypred_firstderiv` | cross-impl pipeline | prediction (per-sample) | **5e-3** (guarded) | ~3.45e-3 | `test_conformance_dual_engine.py:244-251`, under `assert_same_winner` |
| `classification_label` | any | class_label | **0 / exact** | exact | `parity_oracle.v1.json:21-27` |
| `n/a_semantic` | — | — | — | — | Tier-2/3 permanent divergence; never compared by tolerance |
| `n/a_rng` | — | — | — | — | Tier-3 RNG nondeterministic; **never** masked by tolerance |

### A.3 Where each tolerance applies, and why (the B-009 answer)

- **`1e-9` (and `1e-12`) apply ONLY to the same-implementation kernel path.**
  In `test_n4m_ops.py` the *same* numerical core is on both sides
  (`libn4m` vs sklearn computing the identical PLS/SNV), so near-bit-exactness
  is correct and required. This is the **only** legitimate home of the
  contract's `1e-9` number.
- **`1e-6` applies to two deterministic paths:** (a) a dag-ml *native* single-model
  export reproducing its own final-(test) y_pred (`native_export_reproduce`), and
  (b) the no-preprocessing `baseline_vertical_slice` case whose author pinned
  `metric_tolerances` tight (`per_case_tight`).
- **`1e-3` applies to the cross-implementation pipeline path** — the legacy↔dag-ml
  RunResult parity the dual-engine oracle enforces. sklearn-PLS and Rust-PLS are
  *different implementations*; their measured divergence (~`1e-4`..`6e-4`) is float
  noise, not a correctness gap, so `1e-9` here would spuriously fail every PLS
  case. `cross_impl_score` (`:60`) governs scores; `cross_impl_ypred` (`:65`)
  governs per-sample y_pred.
- **`5e-3` (`cross_impl_ypred_firstderiv`)** is a *guarded* relaxation for 6
  cases (§C.3) whose winning pipeline carries FirstDerivative preprocessing that
  amplifies the same PLS Rust-vs-sklearn noise to ~`3.45e-3`. It is only valid
  **under `assert_same_winner`** — the engines must pick the identical winning
  variant first (`test_conformance_dual_engine.py:457-462`), so the relaxation can
  never mask a selection divergence.
- **`0` / exact** governs classification class labels (`classification_label`).
- **`n/a_*`** marks the paths that are **never** tolerance-compared: Tier-2/3
  permanent semantic divergences (`n/a_semantic`) and RNG-nondeterministic cases
  (`n/a_rng`). Per `DEC-PYREF-002`, an RNG case is xfail/skip, **never** relaxed
  into a wider tolerance band — tolerance bands cover float-noise divergence
  only, never stochastic-path divergence.

### A.4 Contract amendment owed to dag-ml (lockstep / L20)

`parity_oracle.v1.json.tolerance_profiles` should stop labelling `1e-9` as the
cross-engine `regression.default`. The agreed direction (SW5 §3b option B) is to
publish explicit profiles — `regression.cross_impl` (`1e-3`),
`regression.kernel` (`1e-9`), `regression.native_export` (`1e-6`), keep
`classification.default` (`0`) — each contract *case* referencing the profile
that matches its numeric path. Because the profile `owner` is *this* ledger, the
ledger is authoritative and the dag-ml JSON is the consumer; the amendment is a
paired dag-ml↔nirs4all change validated in lockstep (`LOCK-LOCKSTEP`/L20).
**Until the amendment lands, this ledger's bands are the authority and the
enforced cross-engine tolerance is `1e-3`.**

---

## §B — 3-tier authority registry

Each tier is a claim about **which engine is correct**, distinct from coverage,
skip, and fallback bookkeeping (those are §C). The five scattered structures —
`_registry.SkipKind`, and `KNOWN_DIVERGENCES` / `NUM_PREDICTIONS_DIVERGENCE` /
`Y_PRED_TOL_OVERRIDES` / `SAME_WINNER_CASES` / `EXPECTED_FALLBACK` in
`test_conformance_dual_engine.py` — are consolidated here.

### Tier 1 — Python (legacy) authoritative

**Default.** Every runnable case not listed in Tier 2 / Tier 3 and not in
`EXPECTED_FALLBACK` runs *native* on dag-ml and must equal the legacy oracle
within its matching `cross_impl_*` (or `per_case_tight`) band. No pytest marker;
**PASS = green parity**. This is the implicit majority tier (≈70 cases).
Authority: **Python (legacy)**, the oracle of record (ADR-01). Enforced by
`assert_score_parity` / `assert_y_pred_parity` / `assert_runresult_contract`
(`_conformance_helpers.py:170,355,253`).

### Tier 2 — dag-ml authoritative (legacy is wrong, or was deliberately changed)

| Case | Mechanism | Authority | Disposition | Band | Measured Δ | Evidence (`file:line`) |
|---|---|---|---|---|---|---|
| `rep_to_sources_basic` | legacy keeps a stale processing selector after repetition reshape; dag-ml follows corrected native reshape scoring | dag-ml | **PASS** parity-note; `cv_best_score` pinned | `n/a_semantic` | cv_best_score `6.673486795441247` (legacy) vs `6.190624012206827` (dag-ml) | `test_conformance_dual_engine.py:199-222` |
| `rep_to_pp_basic` | same stale-context reshape scalar divergence | dag-ml | **PASS** parity-note; `cv_best_score` pinned | `n/a_semantic` | `6.1427143240770405` vs `6.190624012206827` | `:199-222` |
| `feature_augmentation_replace_three_views` | legacy replace-mode context excludes `raw`, but the model matrix still includes it; dag-ml follows replacement semantics | dag-ml | **PASS** parity-note; public score scalars pinned | `n/a_semantic` | best_rmse `12.506075123140343` (legacy) vs `12.627259100347944` (dag-ml) | `test_conformance_dual_engine.py:224-241` |
| `generator_or_models_pls_ridge` | multi-model `_or_` operator-SELECT refits the **winner only**; legacy refits every loser and stores its `(train,final)`+`(test,final)` rows | dag-ml | **PASS** parity-note; `num_predictions` pinned | `cross_impl_score` (winner) | winner/score/winner-y_pred all match (Δ≈`2e-15`); only count differs **34**(legacy)/**32**(dag-ml) | `:190-195` |
| `generator_chain_model_configs` | same SELECT semantic over `_chain_` of distinct models | dag-ml | **PASS** parity-note; `num_predictions` pinned | `cross_impl_score` (winner) | **49**(legacy)/**47**(dag-ml) | `:196-201` |
| *(contract-wide)* `best_rmse` / `best_r2` / `best_accuracy` re-anchored on the **SELECTED** model | 0.9.x **bugfix**: the scalar shortcuts previously re-ranked per metric and could each report a *different, non-selected* CV fold; they now read from `best` | dag-ml / post-fix nirs4all | enforced (not xfail) by `assert_runresult_contract` (`best_score` = selected-metric value) | `cross_impl_score` | e.g. `best_r2` returned fold R² `0.5426` instead of selected `0.5499` | `CHANGELOG.md:44-57`; `_conformance_helpers.py:286-300` |

> **Parity-note discipline:** Tier-2 semantic deltas are not hidden behind
> `xfail(strict)`. The `num_predictions` notes pin exact `34/32` and `49/47`
> counts, while the repetition-fusion notes pin exact `cv_best_score` scalars.
> Any other count or score fails.

### Tier 3 — strict-xfail native comparisons

There are currently **no** strict-xfail native comparison rows. The former
`concat_transform_pca_svd_plsr` row is now an explicit `EXPECTED_FALLBACK`
boundary (§C.1): the native FeatureConcat path fits stateful PCA/SVD fold-locally,
while the Python oracle materializes `concat_transform` before CV.

---

## §C — Orthogonal axes (NOT authority tiers; tracked so they don't pollute §B)

### C.1 Native-coverage boundary — `EXPECTED_FALLBACK` (15)

Shapes the dag-ml host bridge does **not serialize yet**, so `engine="dag-ml"`
transparently re-runs legacy. These make **no parity claim** — they are pinned by
the never-xfailed `test_native_fallback_boundary` (`test_conformance_dual_engine.py:375`):
a fallback off the allowlist = native-coverage **regression → FAIL**; a native
case on the allowlist = **stale entry → FAIL**. **Owner: L5/A3** (host-bridge
serialization, runtime work — not a tolerance question). When L5 lands native
coverage, the entry leaves the allowlist and the boundary test then demands
native parity.

Source: `test_conformance_dual_engine.py:310-337`.

| Shape group | Cases |
|---|---|
| branch (duplication) + merge → multi-model | `branch_dup_three_way_merge_predictions`, `branch_dup_two_way_merge_features`, `branch_dup_named_with_metamodel`, `branch_dup_merge_all` |
| branch separation by metadata/tag/filter | `branch_separation_by_metadata_auto`, `branch_separation_by_tag`, `branch_separation_by_filter` |
| classification repetition + vote aggregation | `aggregation_classification_vote` |
| legacy Optuna finetuning | `generator_finetune_params_optuna` |
| stateful `concat_transform` pre-CV materialization | `concat_transform_pca_svd_plsr` |
| by-source / per-source multi-source | `multi_source_by_source_branch_shared_preproc`, `multi_source_per_source_models_stacking` |
| explicit `preprocessing` keyword + `fit_on_all` + `force_layout` | `preprocessing_explicit_keyword`, `preprocessing_fit_on_all`, `preprocessing_force_layout_2d` |

**`EXPECTED_FALLBACK == ∅` is the `LOCK-DROP` D1 gate, owned by L5 — not a
`LOCK-PYREF` gate.**

### C.2 Coverage-debt fixture skips (0)

All former fixture-debt rows are now runnable. The two shapes that still lack
native coverage moved to the explicit `EXPECTED_FALLBACK` boundary; the
multi-filter exclusion case now runs as a live legacy-oracle parity case.

| Case | Reason | Evidence |
|---|---|---|
| _none_ | _n/a_ | _n/a_ |

### C.3 `Y_PRED_TOL_OVERRIDES` (6) — band `cross_impl_ypred_firstderiv` (5e-3, guarded)

**Not divergences.** Same-winner FirstDerivative-amplified PLS Rust-vs-sklearn
per-sample noise relaxed to `5e-3`, valid **only** under `assert_same_winner`
(score parity still holds at the default `1e-3`). Source:
`test_conformance_dual_engine.py:244-251`.

`generator_or_with_pick`, `generator_cartesian_stages`,
`generator_cartesian_with_param_range`, `generator_or_pick_requires`,
`generator_cartesian_pick`, `generator_or_pick_mutex3`.

### C.4 `SAME_WINNER_CASES` (19) — selection-agreement guard

Multi-variant generator/constraint cases that must select the **identical**
winning variant (by `config_name`) on both engines — the engine-level companion
to the DSL-level survivor-set lock in `test_generators_conformance_extra.py`. Not
a divergence axis. Source: `test_conformance_dual_engine.py:265-300`.

### C.5 `NUM_PREDICTIONS_DIVERGENCE` (2)

Already recorded in **Tier 2** as `pass_parity_note`. Listed here only as a
reminder that the count is *pinned* (`assert_num_predictions_divergence`), never
merely exempted. Source: `test_conformance_dual_engine.py:189-202`.

### C.6 `CV_BEST_SCORE_DIVERGENCE` (2)

Already recorded in **Tier 2** as `pass_cv_best_score_note`. Listed here only as
a reminder that the repetition-fusion scalars are *pinned*
(`assert_cv_best_score_divergence`), never merely exempted. Source:
`test_conformance_dual_engine.py:199-222`.

### C.7 `RUNRESULT_SCORE_DIVERGENCE` (1)

Already recorded in **Tier 2** as `pass_runresult_score_note`. Listed here only
as a reminder that the public score scalars are *pinned*
(`assert_runresult_score_divergence`), never merely exempted. Source:
`test_conformance_dual_engine.py:224-241`.

---

## §D — Cross-engine surface ledger (EXISTS / PARTIAL / GAP)

The numerical pipeline parity above is proven. These five *cross-engine
surfaces* are tracked separately; GAPs are `LOCK-PYREF` G5–G9 work (see the A2
report §4 and SW5 §6 for the concrete test specs).

| Surface | Status | Owning test / evidence | Band | Owner lane |
|---|---|---|---|---|
| `.n4a` export round-trip (export→reload→predict, both engines; native single-model exact) | **PARTIAL** | `test_conformance_export_roundtrip.py` (native export reproduces final-test y_pred within `1e-6`) | `native_export_reproduce` | L17 |
| `.n4a` *cross-engine* (legacy-written bundle predicted via dag-ml runtime, and reverse) | **GAP** (PYREF-009a) | — | `cross_impl_ypred` (target) | L17 + L5 |
| Workspace cross-engine (legacy SQLite/Parquet/manifest read via runtime V1; native-results triple round-trip) | **GAP** (PYREF-009b) | engines write non-overlapping formats (`test_dagml_native_results.py:210-219` asserts legacy ignores the native dir) | `cross_impl_score` (target) | L17 + L5 |
| Error / refusal parity (same invalid pipeline → same refusal on both engines) | **GAP** (PYREF-err) | every `pytest.raises` in the parity dir is single-engine dag-ml-only | n/a | L17 |
| Studio rides the oracle (records resolved engine; one pipeline through both engines) | **GAP** (PYREF-008) | adapter reads native triple correctly, but Studio never passes/records `engine=`; 4 backend routes re-implement nirs4all logic | `cross_impl_score` (target) | L17 + L12 |
| methods-installed lane (n4m parity) | **PARTIAL** | `test_n4m_ops.py` exists (SNV `1e-12`, PLS `<1e-9`) but is `importorskip("n4m")` and not pinned in CI | `kernel_snv` / `kernel_pls` | L17 + L9 |
| nirs4all-side wheel / `.so` freshness | **GAP** (PYREF-011) | dag-ml has `scripts/check_so_freshness.py`; nirs4all tracks no artifact and has no consumer-side freshness gate | n/a | L17 + L9 |

---

## §E — Coverage meter (the LOCK-DROP instrument)

Counts verified against `tests/integration/parity/cases_*.py` (95 `register()`
calls) after the fixture-debt cleanup. The `EXPECTED_FALLBACK` shrink target is the LOCK-DROP D1
gate.

| Metric | Count | Note |
|---|---|---|
| Registered `PipelineCase`s | **95** | `cases_*.py` `register()` calls |
| Non-runnable (`skip_reason` set) | **0** | no fixture/unknown/legacy-bug skips in the registry |
| Runnable | **95** | 95 − 0 |
| → fall back to legacy (`EXPECTED_FALLBACK`) | **15** | boundary-asserted, no parity claim — **target → 0 (LOCK-DROP D1, L5)** |
| → run native on dag-ml | **80** | full parity asserted or parity-note pinned |
| Strict-xfail (documented divergence) | **0** | `KNOWN_DIVERGENCES` is empty; no `legacy_bug` rows in the current registry |
| `pytest.skip` (fixture) | **0** | fixture skips retired |
| `NUM_PREDICTIONS_DIVERGENCE` parity-notes (PASS) | **2** | counts pinned |
| `CV_BEST_SCORE_DIVERGENCE` parity-notes (PASS) | **2** | repetition-fusion scalars pinned |
| `RUNRESULT_SCORE_DIVERGENCE` parity-notes (PASS) | **1** | public score scalars pinned |

> **Correction to prior counts:** the current machine-readable ledger and live
> registry have no `legacy_bug`, `unknown_semantics`, or fixture skip rows. The
> verified meter is **0** non-runnable / **95** runnable. `EXPECTED_FALLBACK` is
> now **15** because two retired fixture-debt shapes are runnable but still not
> native, `finetune_params` is an explicit coverage boundary, and stateful
> `concat_transform` now falls back until dag-ml preserves the Python oracle's
> pre-CV materialization semantics.

---

## §F — Invariants this ledger fixes (the `LOCK-PYREF` contract)

1. **Bands, not a global number.** Every parity assertion binds to a `band_id`
   in §A.2. `1e-9` is `kernel_pls`, never the cross-engine default.
2. **RNG is never tolerance-masked** (`DEC-PYREF-002`). An `n/a_rng` case is
   xfail/skip; widening a tolerance to absorb a stochastic path is forbidden.
3. **XPASS = RED when strict-xfail rows exist.** There are no strict-xfail rows
   currently; any future `xfail(strict)` must flip the suite red the moment the
   engines reconverge, so a fixed divergence cannot silently leave coverage.
4. **The native/fallback boundary can never silently widen.**
   `test_native_fallback_boundary` is never xfailed; an off-allowlist fallback or
   a stale allowlist entry fails (`test_conformance_dual_engine.py:375`).
5. **num_predictions divergences are pinned, not exempted.** Only the documented
   `34/32` and `49/47` loser-refit-row deltas pass.
