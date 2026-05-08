# `RFModelLeavesRegressor` — Specification

**Status**: SPEC (no implementation yet) — `DECISION_PENDING_CODEX_REVIEW` (D-C-008).
**Owner**: Agent C / MLOps Spine, per `bench/PLAN_REPRISE_2026-05.md` §8 task C5.
**Scope**: prototype lives in `bench/AOM_v0/rf_model_leaves/`; preset membership is `exhaustive_research` only.
**Audience**: Codex (review the design); Agent A (sanity on relation to ASLS-AOM and Multi-Branch-MKL); Agent C (implementation).

---

## 1. Motivation

The benchmark synthesis (`bench/benchmark_synthesis.md` §"Oracle by model class") shows that no single model dominates across all 57 datasets. The TabPFN-opt oracle wins 45/59, AOM-PLS 49/59, AOM-Ridge 45/53. Their disagreements localise to subsets of samples — `Beer_OriginalExtract`, `LMA_spxyG_block2deg`, `Brix_*`, `LUCAS_*` are systematically harder for one family but easier for another. The Adaptive Super Learner (multiview Phase-11) addresses this by stacking on dataset level, but it is `exploratory` (38/61 partial) and structurally pays the full cost of every base learner.

`RFModelLeavesRegressor` proposes a complementary diagnostic: a Random Forest whose leaves contain *fast spectroscopic models* (PLS / Ridge / AOM-PLS-compact) instead of constant predictions. The tree learns a *routing* over samples on the basis of low-dimensional projections (PCA scores, AOM-PLS scores, or metadata); each leaf then fits a small expert on the full spectra of the samples it contains. At prediction time, a sample is routed through every tree to its leaves and the per-leaf expert predictions are bagged.

Three reasons this is worth specifying:

1. **Diagnostic value**. The leaf-assignment matrix exposes which datasets / samples are "different" in the routing sense. This is informative even if final RMSEP is competitive only with TabPFN-opt, not better.
2. **Selector role**. The tree's leaf assignments can be reused as a meta-selector input for AOMRidge-Blender or `AdaptiveSuperLearner` — the routing is a sample-level meta-feature that those stacks currently lack.
3. **Failure-mode characterisation**. The plan §10.1 W5 waypoint asks for a residual workstream classifying failures into baseline/scatter, small-n variance, y-extreme sigmoid, domain shift, nonlinear residual. RF-routed leaf experts naturally surface which subset is responsible for each failure mode.

This is **explicitly not** a bid for `best_current` membership. The plan §8 C5 places this in `exhaustive_research` and requires a diagnostic / selector framing. Promotion to `strong_practical` would require a separate Codex decision and is out of scope for this SPEC.

---

## 2. Definition

`RFModelLeavesRegressor` is an sklearn-compatible regressor with the following signature (proposed):

```python
RFModelLeavesRegressor(
    n_estimators=100,
    routing_features="pca",          # "pca" | "aom_pls_scores" | "metadata" | callable
    n_routing_components=8,
    leaf_model="pls",                # "pls" | "ridge" | "aom_pls_compact" | "fixed_mean"
    min_samples_per_leaf=20,
    leaf_model_fallback="ridge",     # used when leaf size < min_for(leaf_model)
    leaf_model_params=None,          # forwarded to the leaf estimator
    bootstrap=True,
    max_features="sqrt",             # forwarded to underlying decision tree
    random_state=None,
    n_jobs=-1,
)
```

It exposes the standard `fit(X, y)`, `predict(X)`, `score(X, y)` API. `predict` returns a 1D float array. Optional `predict_with_routing(X)` returns the per-sample matrix of leaf identifiers per tree, for diagnostic plots.

---

## 3. Training algorithm

Given training samples `(X, y)` of shape `(n, p)` and `(n,)`:

1. **Routing-feature construction**. Build `X_route ∈ R^(n × d_route)` from `X` (and optional metadata). Default: `PCA(n_components=n_routing_components).fit_transform(StandardScaler(X))`. Alternatives: AOM-PLS scores from a pre-fitted ASLS-AOM-compact-cv5, or a user-supplied callable `routing_features(X) → R^(n × d_route)`.
2. **Tree induction**. For each tree `t = 1 … n_estimators`:
   1. If `bootstrap=True`, draw a bootstrap sample `(X_t, X_route_t, y_t)` of size `n` with replacement.
   2. Fit a `DecisionTreeRegressor(max_features=max_features, min_samples_leaf=min_samples_per_leaf)` on `(X_route_t, y_t)`. The tree splits on the routing features only; raw spectra are never seen by the tree.
   3. Walk the fitted tree to collect the **leaf membership** of each training sample in `(X_t, y_t)`.
3. **Leaf expert fitting**. For each tree `t` and each leaf `ℓ`:
   1. Collect the (within-bag) sample indices assigned to `ℓ`. Call this set `I_{t,ℓ}`.
   2. If `|I_{t,ℓ}| ≥ min_for(leaf_model)`, fit `leaf_model` on `(X[I_{t,ℓ}], y[I_{t,ℓ}])`. Hyperparameters come from `leaf_model_params`. AOM-PLS-compact runs its own internal CV on `I_{t,ℓ}` via `ASLS-AOM-compact-cv5-numpy`'s default config.
   3. Otherwise fall back to `leaf_model_fallback`. If even that fails (extremely small leaf), the leaf stores the leaf-mean of `y[I_{t,ℓ}]`.
4. **Storage**. Per tree, store the fitted decision tree, the leaf experts indexed by leaf-id, and the routing feature transformer. The whole forest is picklable; total memory is `O(n_estimators × avg_leaves × leaf_model_size)`.

`min_for(leaf_model)` defaults: PLS=15 (need ≥3 components × ≥5 samples), Ridge=8, AOM-PLS-compact=30, fixed_mean=1.

---

## 4. Prediction protocol

For test samples `X_test ∈ R^(m × p)`:

1. Apply the routing transformer fitted at training time to obtain `X_route_test ∈ R^(m × d_route)`.
2. For each tree `t`:
   1. Route every test sample through the tree → leaf id.
   2. Look up the leaf expert and call `expert.predict(X_test[i])` for each sample i.
   3. Accumulate per-sample predictions into a column of an `(m × n_estimators)` matrix.
3. Reduce across trees by mean (default) or median (set via constructor flag, mirrors sklearn `RandomForestRegressor.aggregate`).

Diagnostic mode (`predict_with_routing`) additionally returns the `(m × n_estimators)` matrix of leaf ids and a `(m,)` array of "routing entropy" — the empirical entropy of the leaf-id distribution per sample, which surfaces samples whose routing is ambiguous (and therefore whose ensemble variance is high).

---

## 5. Nested-CV gating

The plan §10.1 / §10.2 require nested CV before any selector enters `strong_practical` or `best_current`. `RFModelLeavesRegressor` is constructed to be nested-safe by design:

- All leaf experts are fitted on training samples only. The bootstrap step inside each tree is internal to training, never the outer CV split.
- Routing-feature transformers are fitted on training samples only.
- `predict` never reuses training-set predictions. Out-of-bag predictions are NOT used for selection in this design (no internal early stopping or model-selection step that would peek at left-out training samples).
- Per-leaf AOM-PLS-compact runs its own inner CV inside `I_{t,ℓ}`, which is strictly a subset of the outer training set; this preserves the nested boundary.

**Codex must validate**: that the per-leaf AOM-PLS inner CV does not unintentionally leak through the bootstrap bag. The current proposal performs the bootstrap once per tree, runs AOM-PLS-CV on the bag, and then queries that AOM-PLS at predict time. This is identical to the standard RF + nested expert pattern; we simply have to confirm the subsampling semantics.

---

## 6. Integration hooks for the registry

When the prototype lands, the registry entry will look like:

```yaml
- canonical_name: RFModelLeaves-AOM-pca8-pls-compact
  aliases:
    - rf-leaves-aom-pca8
    - rf-model-leaves-pls
  model_class: RFModelLeavesRegressor
  module: bench.AOM_v0.rf_model_leaves.estimator
  config_template: bench/scenarios/configs/rf_model_leaves_aom_pca8.yaml
  task_types: [regression]
  input_constraints: {min_n: 60}
  supports_predefined_test_split: true
  inner_cv_nested: true
  runtime_tier: slow
  maturity: exploratory
  not_runnable_in_production: false
  notes: |
    Diagnostic / selector. PCA-routed Random Forest with AOM-PLS-compact
    leaf experts. Membership: `exhaustive_research` only.
```

Preset membership: `exhaustive_research` only. The exporter penalty `exploratory_in_non_research_preset` will catch any accidental promotion.

---

## 7. Runtime tier expectations

Reference shape: `n=400`, `p=2000`, `n_estimators=100`, `n_routing_components=8`, `min_samples_per_leaf=20`, `leaf_model=aom_pls_compact`.

| Phase | Estimated cost | Notes |
|---|---|---|
| Routing PCA fit | O(n × p × d_route) | seconds |
| Tree induction × 100 | O(n_estimators × n × d_route × log n) | seconds |
| Leaf experts × ~100 leaves × 100 trees | dominant | each leaf ≈ 5–25 samples; AOM-PLS-compact ≈ 0.05–0.5 s; leaf experts in parallel ⇒ ~30 s on RTX 4090 box CPU |
| Predict × 100 samples | O(m × n_estimators × leaf_model_predict) | seconds |

Tier: **`slow`** (≈ 30 s … 2 min per dataset for `n ≤ 400`). For `n_train > 3000` or `p > 5000` the cost climbs beyond the `slow` tier; the spec gates that case with `input_constraints.max_n` / `max_features`.

---

## 8. Success criteria

`RFModelLeavesRegressor` will be considered useful — and its diagnostic outputs worth carrying in the dashboard — if any of the following hold on the `audit20_transfer_core` cohort:

1. **Per-sample diagnostic value**. Routing entropy correlates with absolute residual on at least 60% of datasets (Pearson |r| > 0.3, p<0.05). This justifies surfacing the routing matrix in the dashboard.
2. **Subset characterisation**. The leaf assignments partition every dataset into clusters whose mean residuals differ by > 1 σ between clusters on at least 50% of datasets. This justifies using the routing as a subset-finder.
3. **Selector signal**. Adding the per-sample leaf id (one-hot encoded) as an extra feature to AOMRidge-Blender's meta layer improves Blender median rel-RMSEP by ≥ 1% across `audit20_transfer_core`. This justifies the "selector" half of the plan §8 C5 framing.

Failure mode: the RF-routed leaves are no different from standard k-means clusters of the routing features, in which case the spec collapses to "do k-means clustering and fit per-cluster experts" — a much simpler design. If the prototype reaches this conclusion, it becomes a reference-only artifact and is documented as a negative result in `bench/AOM_v0/rf_model_leaves/EVALUATION.md` per the same convention as `bench/fck_pls/docs/FCK_EVALUATION.md`.

The criteria are explicitly NOT "RFModelLeavesRegressor beats AOM-Ridge / TabPFN-opt on RMSEP". The plan §8 C5 places this entry in `exhaustive_research` precisely because RMSEP improvement is not expected.

---

## 9. Open questions for Codex

- **OQ1 — Routing features**. Default is PCA with `n_routing_components=8`. AOM-PLS scores from a pre-fitted `ASLS-AOM-compact-cv5-numpy` are likely higher signal but introduce a dependency the registry must wire (the AOM-PLS pre-fit becomes a new step in the harness train path). Should the SPEC default to PCA (simpler, no dep) or AOM-PLS (likely better signal)?
- **OQ2 — Leaf model**. Default is AOM-PLS-compact. PLS is faster but loses the operator-bank advantage. Ridge is even faster but cannot capture spectral structure. Does Codex want one default, or a `_chain_` of three configs (`pls`, `ridge`, `aom_pls_compact`) under `_or_` so the dashboard can compare the three?
- **OQ3 — Bootstrap inside vs outside the harness**. The default bootstrap is inside the estimator (sklearn `RandomForestRegressor` style). Plan §10.2 production-tier asks for Friedman-Nemenyi over multi-seed runs; if the harness already provides 5 outer seeds, do we keep the per-tree bootstrap (variance reduction within seed) or disable it (`bootstrap=False`) and rely on outer seeds? Codex pick.
- **OQ4 — Nested-CV runtime**. AOM-PLS-compact runs internal 5-fold CV on every leaf. With 100 trees × 50 leaves × 5 folds = 25 000 PLS fits per dataset. Acceptable for `exhaustive_research` but the `slow` tier rating may be optimistic on `n_train > 1000`. Should the SPEC tighten `input_constraints` to `max_n: 1500`?
- **OQ5 — Membership**. Plan §8 C5 says `exhaustive_research`. If the prototype fails the success criteria above, it becomes a negative-result artifact. Should it stay in the registry under `not_runnable_in_production: true`, or be deleted entirely from the YAML?

---

## 10. Acceptance for the SPEC document itself

- Status moves from `DECISION_PENDING_CODEX_REVIEW (D-C-008)` to `DECISION_VALIDATED` once Codex answers OQ1…OQ5.
- After validation, this SPEC is the authoritative source for the prototype implementation in `bench/AOM_v0/rf_model_leaves/estimator.py`. Implementation will be a separate pull request, not in this active-wait cycle.

## 11. References

- `bench/PLAN_REPRISE_2026-05.md` §8 task C5 — original spec request.
- `bench/PLAN_REPRISE_2026-05.md` §10 — validation tiers and stat tests.
- `bench/benchmark_synthesis.md` §"Why AOM-PLS was hidden in the first ranking" — the disagreement structure between strategy families that motivates per-leaf experts.
- `bench/AOM_v0/multiview/docs/SUMMARY.md` (Phase 11) — Adaptive Super Learner as a related stacking design.
- `bench/AOM_v0/Ridge/docs/HEADLINE_SPXY3_NESTED_AUDIT.md` — nested-CV pattern AOMRidge-Blender uses, applied here to leaf experts.
- `bench/Subset_analysis/RETHOUGHT_SUBSETS.md` — subset cohorts (`fast12_transfer_core`, `audit20_transfer_core`) that the prototype will run on.
