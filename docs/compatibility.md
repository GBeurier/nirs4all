# nirs4all ↔ dag-ml compatibility ledger

**schema_version:** 1
**owner:** `nirs4all compatibility ledger`
**consumer_of:** `dag-ml/docs/contracts/parity_oracle.v1.json` (`dag-ml.nirs4all.parity_oracle.v1`)
**machine-readable companion:** `docs/compatibility.json`
**static debt gate:** `tests/integration/parity/_marker_audit.py` (§G)
**last reconciled:** 2026-07-02 against `nirs4all 3d568abe` / `dag-ml 7f86a9b` / `dag-ml-data e681685` (RC-C: full parity proof on the selected RC stack; §G marker/skip/tolerance gate still clean; `per_case_tight` band corrected `1e-6`→`1e-3`)
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
| `per_case_tight` | cross-impl pipeline | score | **1e-3** | ~7e-6 | case `metric_tolerances` (`baseline_vertical_slice`: rmse **and** r2), bound live by `_authority._validate_per_case_tight_band` |
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
- **`1e-6` applies to one deterministic path:** a dag-ml *native* single-model
  export reproducing its own final-(test) y_pred (`native_export_reproduce`).
- **`per_case_tight` is the per-case `metric_tolerances` override mechanism, not a
  tighter number.** Its sole live instance, `baseline_vertical_slice`, pins **both**
  `rmse` **and** `r2` at `1e-3` — the same `cross_impl_score` magnitude, adding the
  secondary-metric guard that caught the `best_r2` re-rank bug (§B Tier-2). No case
  currently pins a sub-`1e-3` tolerance. `_authority._validate_per_case_tight_band`
  binds this band's `abs_tol` to the live case value, so the two can never drift
  again (the ledger previously mis-claimed `1e-6` here while the case enforced `1e-3`).
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
  (`n/a_rng`). Per `DEC-PYREF-002`, an RNG case is either a ledgered run-only
  nondeterministic contract or explicit debt, **never** relaxed into a wider
  tolerance band — tolerance bands cover float-noise divergence only, never
  stochastic-path divergence.

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
skip, and fallback bookkeeping (those are §C). The scattered live structures —
`_registry.SkipKind`, and `KNOWN_DIVERGENCES` / `LEGACY_CV_SCORE_DIVERGENCE` /
`NUM_PREDICTIONS_DIVERGENCE` / `Y_PRED_TOL_OVERRIDES` / `SAME_WINNER_CASES` /
`EXPECTED_FALLBACK` in `test_conformance_dual_engine.py` — are consolidated here.

### Tier 1 — Python (legacy) authoritative

**Default.** Every runnable case not listed in Tier 2 / Tier 3 and not in
`EXPECTED_FALLBACK` runs *native* on dag-ml and must equal the legacy oracle
within its matching `cross_impl_*` (or `per_case_tight`) band. No pytest marker;
**PASS = green parity**. This is the implicit majority tier (≈65 cases).
Authority: **Python (legacy)**, the oracle of record (ADR-01). Enforced by
`assert_score_parity` / `assert_y_pred_parity` / `assert_runresult_contract`
(`_conformance_helpers.py:170,355,253`).

### Tier 2 — dag-ml authoritative (legacy is wrong, or was deliberately changed)

| Case | Mechanism | Authority | Disposition | Band | Measured Δ | Evidence (`file:line`) |
|---|---|---|---|---|---|---|
| `rep_to_sources_basic` | legacy **double-counts** overlapping rep OOF folds; dag-ml aggregates each reshaped sample once | dag-ml | **PASS** parity-note; `cv_best_score` non-equivalence pinned | `n/a_semantic` | cv_best_score `6.6735` (legacy bug) vs `6.1906` (dag-ml, correct) | `test_conformance_dual_engine.py:LEGACY_CV_SCORE_DIVERGENCE` |
| `rep_to_pp_basic` | same rep OOF double-count | dag-ml | **PASS** parity-note; `cv_best_score` non-equivalence pinned | `n/a_semantic` | `6.1427` (legacy bug) vs `6.1906` (correct) | `test_conformance_dual_engine.py:LEGACY_CV_SCORE_DIVERGENCE` |
| `generator_or_models_pls_ridge` | multi-model `_or_` operator-SELECT refits the **winner only**; legacy refits every loser and stores its `(train,final)`+`(test,final)` rows | dag-ml | **PASS** parity-note; `num_predictions` pinned | `cross_impl_score` (winner) | winner/score/winner-y_pred all match (Δ≈`2e-15`); only count differs **34**(legacy)/**32**(dag-ml) | `:190-195` |
| `generator_chain_model_configs` | same SELECT semantic over `_chain_` of distinct models | dag-ml | **PASS** parity-note; `num_predictions` pinned | `cross_impl_score` (winner) | **49**(legacy)/**47**(dag-ml) | `:196-201` |
| *(contract-wide)* `best_rmse` / `best_r2` / `best_accuracy` re-anchored on the **SELECTED** model | 0.9.x **bugfix**: the scalar shortcuts previously re-ranked per metric and could each report a *different, non-selected* CV fold; they now read from `best` | dag-ml / post-fix nirs4all | enforced (not xfail) by `assert_runresult_contract` (`best_score` = selected-metric value) | `cross_impl_score` | e.g. `best_r2` returned fold R² `0.5426` instead of selected `0.5499` | `CHANGELOG.md:44-57`; `_conformance_helpers.py:286-300` |

> **Parity-note discipline:** Tier-2 legacy-bug cases are **not** xfail when
> dag-ml is authoritative. The two rep-fusion `cv_best_score` values are pinned
> by `LEGACY_CV_SCORE_DIVERGENCE` / `_assert_legacy_cv_score_divergence`, and the
> two `num_predictions` parity-notes pin the exact `34/32` and `49/47` counts by
> `assert_num_predictions_divergence` (`_conformance_helpers.py:220-243`). Any
> unledgered value drift fails.

### Tier 3 — oracle non-executable / RNG / unknown-semantics (comparison invalid)

| Case | Sub-class | Authority | Disposition | Band | Evidence (`file:line`) |
|---|---|---|---|---|---|
| `generator_sample_log_uniform_alpha` | RNG — unseeded `_sample_` (`_seed_` not set): variant set / winner is intentionally stochastic | neither | **PASS** run-only native contract; no equality claim | `n/a_rng` | `UNSEEDED_NONDETERMINISTIC_CASES`; seeded twin `generator_sample_seeded_alpha` carries strict equality |

> **The `_or_ count/_weights_` path is now a live parity case.** `_seed_` and
> `_weights_` are threaded into `OrStrategy.sample_with_seed`, so
> `generator_or_count_seed` and `generator_or_weights_count_seed` run alongside
> the deterministic `_cartesian_` count path in `SAME_WINNER_CASES`.
>
> `branch_separation_by_tag`, `branch_separation_by_filter`, and
> `concat_transform_pca_svd_plsr` are no longer Tier-3 debt: the legacy oracle
> runs, dag-ml runs natively, and the targeted parity assertions are live.

---

## §C — Orthogonal axes (NOT authority tiers; tracked so they don't pollute §B)

### C.1 Native-coverage boundary — `EXPECTED_FALLBACK` (0)

Shapes the dag-ml host bridge does **not serialize yet**, so `engine="dag-ml"`
transparently re-runs legacy. These make **no parity claim** — they are pinned by
the never-xfailed `test_native_fallback_boundary` (`test_conformance_dual_engine.py:372`):
a fallback off the allowlist = native-coverage **regression → FAIL**; a native
case on the allowlist = **stale entry → FAIL**. **Owner: L5/A3** (host-bridge
serialization, runtime work — not a tolerance question). When L5 lands native
coverage, the entry leaves the allowlist and the boundary test then demands
native parity.

Source: `test_conformance_dual_engine.py:310-326`.

| Shape group | Cases |
|---|---|
| — | — |

`preprocessing_fit_on_all` and `preprocessing_force_layout_2d` now run native for the registered SNV cases: `fit_on_all=True` is equivalent for stateless transforms, and `force_layout='2d'` on a preprocessing step is not consumed by the legacy preprocessing controller.

`multi_source_by_source_branch_shared_preproc` now runs native with per-source shared preprocessing and legacy-compatible prediction bookkeeping.

`branch_dup_two_way_merge_features` now runs native by lowering the duplication branch to a fold-local feature-merge transformer before the downstream model.

Default stacking now runs native. List-branch stacking keeps the explicit full-coverage OOF/refit contract; named-dict default stacking (`branch_dup_three_way_merge_predictions`) uses dag-ml's explicit CV-only stacking policy and projects legacy's no-refit row surface.

`branch_dup_named_with_metamodel` now runs native with the legacy-compatible CV-only row surface: branch preprocessing is fit on the full train pool, branch-local base and `Ridge_MetaModel` rows are projected per branch, the structured best-by-RMSE prediction merge becomes OOF feature columns, and the downstream Ridge emits no `final` rows, matching legacy.

`branch_dup_merge_all` now runs native with combined branch feature blocks plus branch prediction columns for the downstream model, and legacy-compatible branch/downstream row projection.

`multi_source_sources_concat_then_rf` now runs native by preserving the legacy source-concat storage boundary: upstream stateless transforms run per source, the merged block replaces source 0, and non-zero sources remain visible to downstream materialization.

`multi_source_per_source_models_stacking` now runs native by replaying legacy's by-source source-layout contract for `{"merge": "predictions"}`: source branches mutate the layout cumulatively, the post-merge Ridge trains on the 10,755-column source layout, and the public result preserves legacy's CV-only/no-final rows.

**`EXPECTED_FALLBACK == ∅` is the `LOCK-DROP` D1 gate, owned by L5 — not a
`LOCK-PYREF` gate.**

### C.2 Coverage-debt fixture skips (0)

Closed on 2026-07-02:

| Case | Resolution |
|---|---|
| `branch_separation_by_metadata_auto` | moved to `with_metadata` and the `group` metadata column; dag-ml now projects stateless by-metadata preprocessing + concat + downstream model |
| `exclude_multi_any_y_and_x` | raised the Mahalanobis threshold to keep a viable two-filter UNION on `sample_data/regression` |
| `aggregation_classification_vote` | moved to the repeated multiclass `classification` fixture and aggregates by `Sample_ID` |
| `refit_params_use_all_partitions` | flipped to a live parity case with the existing `refit_params` key |

### C.3 `Y_PRED_TOL_OVERRIDES` (6) — band `cross_impl_ypred_firstderiv` (5e-3, guarded)

**Not divergences.** Same-winner FirstDerivative-amplified PLS Rust-vs-sklearn
per-sample noise relaxed to `5e-3`, valid **only** under `assert_same_winner`
(score parity still holds at the default `1e-3`). Source:
`test_conformance_dual_engine.py:244-251`.

`generator_or_with_pick`, `generator_cartesian_stages`,
`generator_cartesian_with_param_range`, `generator_or_pick_requires`,
`generator_cartesian_pick`, `generator_or_pick_mutex3`.

### C.4 `SAME_WINNER_CASES` (~22) — selection-agreement guard

Multi-variant generator/constraint cases that must select the **identical**
winning variant (by `config_name`) on both engines — the engine-level companion
to the DSL-level survivor-set lock in `test_generators_conformance_extra.py`. Not
a divergence axis. Source: `test_conformance_dual_engine.py:265-300`.

### C.5 `NUM_PREDICTIONS_DIVERGENCE` (2)

Already recorded in **Tier 2** as `pass_parity_note`. Listed here only as a
reminder that the count is *pinned* (`assert_num_predictions_divergence`), never
merely exempted. Source: `test_conformance_dual_engine.py:189-202`.

---

## §D — Cross-engine surface ledger (EXISTS / PARTIAL / GAP)

The numerical pipeline parity above is proven. These five *cross-engine
surfaces* are tracked separately; GAPs are `LOCK-PYREF` G5–G9 work (see the A2
report §4 and SW5 §6 for the concrete test specs).

| Surface | Status | Owning test / evidence | Band | Owner lane |
|---|---|---|---|---|
| `.n4a` export round-trip (export→reload→predict, both engines; native single-model exact) | **PARTIAL** | `test_conformance_export_roundtrip.py` (native export reproduces final-test y_pred within `1e-6`) | `native_export_reproduce` | L17 |
| `.n4a` *cross-engine* (legacy-written bundle predicted via dag-ml runtime, and reverse) | **EXISTS** (PYREF-009a) | `test_conformance_n4a_cross_engine.py::test_n4a_bundle_cross_engine_round_trip` (legacy & dag-ml `.n4a` interchange + reproduce dag-ml-native y_pred, both within `1e-3`; pins the transitional export bridge — tightens to `native_export_reproduce` when native `.n4a` (DML-008/W3) lands) | `cross_impl_ypred` | L17 + L5 |
| Workspace cross-engine (legacy SQLite/Parquet/manifest read via runtime V1; native-results triple round-trip) | **EXISTS** (PYREF-009b) | `test_conformance_workspace_cross_engine.py::test_native_results_triple_round_trips_and_agrees_cross_engine` (native triple reads back faithfully via `read_native_results` AND agrees with legacy within `cross_impl_*`; legacy workspace inspectable) | `cross_impl_score` | L17 + L5 |
| Error / refusal parity (same invalid pipeline → same refusal on both engines) | **EXISTS** (PYREF-err) | `test_conformance_error_parity.py` (invalid pipeline refused by BOTH engines; dag-ml refusal → stable `RtError.cause` from CAP-004/RT-003 — local helper until W7 `rt.py` lands) | `n/a_semantic` | L17 |
| Studio rides the oracle (records resolved engine; one pipeline through both engines) | **EXISTS** (Studio-side RC gate) | Studio RC `tests/test_runtime_engine.py`, `tests/test_studio_oracle_routes.py`, and `tests/test_runs_engine_routing.py` prove requested/default engine threading, fallback policy, actual-engine recording, and manifest round-trip; Python full parity remains the numerical oracle | `cross_impl_score` (target) | L17 + L12 |
| methods-installed lane (n4m parity) | **EXISTS** | `scripts/prove_installed_n4m.py` builds a fresh `nirs4all-methods` wheel, installs this checkout in a proof venv, strips dev overrides, requires `NIRS4ALL_REQUIRE_N4M=1`, and runs `test_n4m_ops.py` packaging/SNV/PLS slices | `kernel_snv` / `kernel_pls` | L17 + L9 |
| nirs4all-side wheel / `.so` freshness | **EXISTS** (PYREF-011 consumer-side) | `scripts/prove_installed_n4m.py` verifies SHA-256 identity from the source `libn4m` reported by the methods smoke, to the staged wheel payload, to the library loaded from the proof venv; source-to-binary freshness stays owned by `nirs4all-methods` | n/a | L17 + L9 |

---

## §E — Coverage meter (the LOCK-DROP instrument)

Counts verified against `tests/integration/parity/cases_*.py` (95 `register()`
calls) on `98c33788+working-tree`. The `EXPECTED_FALLBACK` shrink target is the LOCK-DROP D1
gate.

| Metric | Count | Note |
|---|---|---|
| Registered `PipelineCase`s | **95** | `cases_*.py` `register()` calls |
| Non-runnable (`skip_reason` set) | **0** | all registry debt closed |
| Runnable | **95** | all registered cases run |
| → fall back to legacy (`EXPECTED_FALLBACK`) | **0** | boundary-asserted, no parity claim — **target → 0 (LOCK-DROP D1, L5)** |
| → run native on dag-ml | **95** | native reach asserted; one unseeded `_sample_` is run-only |
| Strict-xfail (documented divergence) | **0** | no live strict xfail rows |
| `pytest.skip` (fixture + unknown-semantics) | **0** | registry skip debt closed |
| `NUM_PREDICTIONS_DIVERGENCE` parity-notes (PASS) | **2** | counts pinned |
| Run-only nondeterministic cases (PASS) | **1** | unseeded `_sample_`; seeded twin owns equality |

> **Correction to prior counts:** registry skips, legacy-bug xfails, branch
> separation xfails, concat PCA/SVD xfail, and repetition score xfails are closed
> or converted to passing pinned notes. The remaining RNG case is explicitly
> run-only and counted separately.

---

## §F — Invariants this ledger fixes (the `LOCK-PYREF` contract)

1. **Bands, not a global number.** Every parity assertion binds to a `band_id`
   in §A.2. `1e-9` is `kernel_pls`, never the cross-engine default.
2. **RNG is never tolerance-masked** (`DEC-PYREF-002`). An `n/a_rng` case is
   either a ledgered run-only contract or a skip/xfail; widening a tolerance to
   absorb a stochastic path is forbidden.
3. **No silent xfail debt.** Strict xfail is permitted only through the sanctioned
   collection-time builders and currently counts **0**; any future xfail must be
   ledgered and XPASS-red by construction.
4. **The native/fallback boundary can never silently widen.**
   `test_native_fallback_boundary` is never xfailed; an off-allowlist fallback or
   a stale allowlist entry fails (`test_conformance_dual_engine.py:372`).
5. **num_predictions divergences are pinned, not exempted.** Only the documented
   `34/32` and `49/47` loser-refit-row deltas pass.
6. **No untracked marker or tolerance (§G).** `pytest.mark.xfail` lives only in the
   sanctioned `_params()` builder; every skip maps to a sanctioned category; every
   tolerance literal equals a published §A.2 band (or is a negative-assertion
   divergence floor). `_marker_audit` fails on the first exception.

---

## §G — Marker & tolerance debt gate (RC-B / `LOCK-PYREF`)

§A–§F reconcile the machine-readable ledger against the live parity **constants**
(`KNOWN_DIVERGENCES` / `NUM_PREDICTIONS_DIVERGENCE` / `Y_PRED_TOL_OVERRIDES` /
registry `skip_kind`), enforced by `_authority.validate_compatibility_ledger`.
That proves the *ledgered* debt is self-consistent but is blind to debt added
**directly in a test body** — a bare `pytest.mark.xfail(reason="TODO")`, a
`pytest.skip("flaky")`, or a loosened `atol=1e-1` — which never touches those
constants. §G closes that hole with a **static `ast` scan** of every
`tests/integration/parity/*.py`.

**Enforcement:** `tests/integration/parity/_marker_audit.py` (scanner + CLI),
`test_marker_audit.py` (live-tree gate + negative self-tests that prove the gate
flags injected debt), and the JSON face `compatibility.json.marker_policy`
(validated against the scanner by `_authority.validate_marker_policy`, so the
published policy can never drift from the code that enforces it).

Three **closed** policies — each fails on the *first* item it cannot place:

1. **xfail containment.** `pytest.mark.xfail` / `pytest.xfail` may appear only in
   `test_conformance_dual_engine.py` and `test_parity_smoke.py`, where
   collection-time builders mirror ledgered `KNOWN_DIVERGENCES` / `legacy_bug`
   debt. The live count is **0**; an xfail elsewhere is untracked divergence debt
   and fails the gate.
2. **skip taxonomy.** Every `pytest.skip` / `pytest.mark.skip(if)` /
   `pytest.importorskip` must classify into exactly one sanctioned category below;
   an unclassifiable skip is untracked coverage loss (a blocker under RC-B: skips
   are release blockers unless optional-environment or tracked registry debt).
3. **tolerance band.** Every explicit numeric `atol`/`rtol`/`abs`/`rel` kwarg,
   every `*_TOL` constant, and every `metric_tolerances` / `Y_PRED_TOL_OVERRIDES`
   value must equal a published §A.2 band (`tolerance_bands[].abs_tol` — the ledger
   *is* the allowlist). A value inside a **negative** assertion (`assert not
   np.allclose(...)` / `!= approx(...)`) is a divergence *floor*, not a tolerance
   *ceiling*, and is exempt.

### G.1 Sanctioned skip categories (the closed taxonomy)

| category | kind | what it covers |
|---|---|---|
| `registry_skip` | tracked_debt | `fixture` / `unknown_semantics` registry skips (`[{skip_kind}] …`) — currently no live cases; still re-asserted across the compile / smoke / baseline / dual-engine modules |
| `optional_env_import` | optional_env | `pytest.importorskip(dag_ml / dag_ml_data / shap / jsonschema / referencing)` |
| `optional_env_dagml_cli` | optional_env | `skipif(not _DAGML_CLI.exists())` — the native `dag-ml-cli` binary is not built on this host |
| `optional_env_dependency` | optional_env | an example needs an optional dependency that is not installed |
| `optional_env_sibling` | optional_env | sibling `nirs4all-ecosystem` runtime schemas not checked out |
| `optional_env_methods` | optional_env | the `nirs4all-methods` SNV / n4m binding is unavailable (hard-fails only under `NIRS4ALL_REQUIRE_N4M=1`) |
| `runtime_na` | runtime_precondition | a cross-engine comparison is N/A on this build: dag-ml ran the legacy fallback, wrote no native results, or the case is not a single-artifact native run. With `coverage_meter.fallback == 0` the fallback branch does not fire for covered cases |
| `baseline_capture` | workflow | gold-baseline capture / absent-baseline guard in `test_parity_baseline` (`--parity-capture`) |
| `lockdrop_empty` | workflow | historical category for the old empty-allowlist skip; the live boundary test now asserts the empty allowlist without skipping |

### G.2 How skips / xfails map to the gate

- **0 xfailed** is exact and fully ledgered. `KNOWN_DIVERGENCES` and registry
  `legacy_bug` debt are empty; any new xfail must be added through the sanctioned
  collection-time builders and the compatibility ledger.
- **Skipped tests** are environment-dependent (which optional bins / the `dag-ml-cli`
  binary are present) and decompose entirely into the G.1 taxonomy. Static call
  sites are audited by `_marker_audit`; a single run realizes only the subset
  matching the local environment. The `registry_skip` category remains enforced,
  but there are no live registry-skip cases.
- **tolerance overrides**: 42 static tolerance literals, every one a published §A.2
  band value or a negative-assertion divergence floor. A new looser value fails.

### G.3 Current RC parity proof

The selected RC stack is `nirs4all 3d568abe`, `dag-ml 7f86a9b`, and
`dag-ml-data e681685`. The last full `pyref_oracle_full` run on that stack
reported `659 passed, 227 deselected, 1530 warnings` in `2037.46s`, with no
parity skips and no xfails. The earlier result with `14 skipped / 6 xfailed`
is superseded by that run.

The static marker audit still reports many skip *call sites* because it scans
optional-environment and runtime-precondition branches. Those are not realized
Python-reference parity skips in the current full parity proof; they are
classified by the closed taxonomy in §G.1 and fail if an unclassified skip,
xfail, or tolerance appears.

### G.4 Running the gate

```bash
# static debt gate (no engine run) — exits 1 on any untracked xfail/skip/tolerance
python -m tests.integration.parity._marker_audit --check

# as pytest (live-tree gate + negative self-tests)
pytest tests/integration/parity/test_marker_audit.py tests/integration/parity/test_compatibility_ledger.py -q
```

**This gate makes the debt visible and enforceable; it does not bless it.** The
live strict-xfail count is **0**. Future RC-C / RC-D debt must first enter the
ledgered builders / authority file, then the gate guarantees it cannot grow
silently.
