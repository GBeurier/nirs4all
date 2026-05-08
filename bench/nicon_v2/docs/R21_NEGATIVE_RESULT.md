# r21_curated_oof_multiseed — Negative-Result Memo

**Date**: 2026-05-06 — **LOCKED** after Codex round-4 APPROVE-WITH-CONDITIONS (D-B-012, D-B-013).
**Owner**: Agent B
**Run**: `bench/nicon_v2/benchmark_runs/r21_curated_oof_multiseed/`
**Variant**: `V2L-Residual-AOMPLS-shrinkage`
**Cohort**: curated 39-dataset cohort (same as r20)
**Seeds**: 0 / 1 / 2 / 3 / 4 (195 fits, all OK, no errors)
**Aggregator**: `bench/nicon_v2/benchmarks/aggregate_r21.py`

---

## TL;DR

**Production gate FAIL.** V2L-Residual-AOMPLS-shrinkage does not beat
AOM-Ridge on the cohort (median +7.5 % rmsep, 14.9 % wins). Per plan §7,
P3 is a **negative-result memo, not a paper submission**.

Two relaxed positive findings, both allowed by Codex round-4 with
explicit caveats:

1. **First NN-residual to beat paper CNN** on the curated cohort:
   median **−9.8 %** rmsep, **71.2 % wins** (121/170 paired
   observations). Wins miss the 75 % science gate by 4 pp, so the
   science gate also fails — but the median improvement is large and
   reproducible across 5 seeds.
2. **Do-no-harm gate PASS**: catastrophic rate 1.0 % (well under the
   5 % budget). The shrinkage CV protocol is the do-no-harm enforcer
   and it works — `s = 0` (teacher-only fallback) was selected on 34
   of 195 (17 %) (dataset, seed) pairs.

**Known caveat**: the held-out shrinkage selector is **unstable on
17 / 39 datasets** (44 %), with per-dataset `s*` IQR > 0.3 across the
5 seeds. Codex round-4 approved a hybrid Option A / B for r22+ (true
CV-5 only on the unstable subset) but did not authorise the 24 h GPU
budget for full CV-5 now.

---

## Stop gate verdicts (final)

| Gate | Threshold | Observed | Verdict |
|---|---|---|---|
| Production | median Δ% ≤ −2 % AND ≥ 50 % wins vs `aom_ridge_curated_best` | median **+7.5 %**, wins **14.9 %** (29 / 195) | **FAIL** |
| Science | median Δ% ≤ −5 % AND ≥ 75 % wins vs paper CNN | median **−9.8 %** (passes ≥2× over), wins **71.2 %** (121 / 170; 4 pp under) | FAIL on wins |
| Do-no-harm | ≤ 5 % catastrophic per (dataset, seed) | catastrophic **1.0 %** | **PASS** |

---

## Cohort positioning of V2L-Residual-AOMPLS-shrinkage

(Median Δ% rmsep = `(rmsep / ref_rmse_x − 1)` per (dataset, seed); `wins`
counts paired observations with Δ% < 0; `n` is the number of paired
observations with the reference available.)

| Reference | median Δ% | wins | n |
|---|---:|---:|---:|
| paper CNN (paper Nicon) | **−9.8 %** | 121 / 170 (71.2 %) | 170 |
| paper CatBoost | −2.3 % | 109 / 195 (55.9 %) | 195 |
| paper TabPFN-raw | −0.6 % | 102 / 195 (52.3 %) | 195 |
| paper PLS | +0.9 % | 89 / 195 (45.6 %) | 195 |
| paper Ridge | +4.1 % | 60 / 195 (30.8 %) | 195 |
| paper TabPFN-opt | +9.7 % | 57 / 195 (29.2 %) | 195 |
| **aom_ridge_curated_best** | **+7.5 %** | **29 / 195 (14.9 %)** | 195 |

**Pattern**: V2L-Residual-AOMPLS-shrinkage **beats paper CNN
consistently** (the headline result for which the residual NN was
designed), **ties paper TabPFN-raw and CatBoost**, and **trails AOM-Ridge
and TabPFN-opt** on the cohort. This is the same shape as FCK
(`bench/fck_pls/docs/FCK_EVALUATION.md`): competitive vs paper baselines,
behind AOM-Ridge.

---

## Shrinkage `s*` stability — Option-A reopen flag

Per Codex round-2 condition on D-B-002c-revised: *"if any per-dataset
IQR(`s*`) > 0.3, reopen Option A (true CV-5 shrinkage) before locking
the shrinkage design for r22+."*

Final aggregator: **17 / 39 datasets violate the IQR threshold (44 %).**
Cohort-wide `s*` histogram (across 195 selections):

| `s*` | count | share |
|---:|---:|---:|
| 0.00 | 34 | 17.4 % |
| 0.25 | 14 | 7.2 % |
| 0.50 | 15 | 7.7 % |
| 0.75 | 17 | 8.7 % |
| 1.00 | **115** | **59.0 %** |

The selector is **bimodal**: 59 % of (dataset, seed) pairs trust the
residual NN fully (`s = 1`), 17 % fall back to teacher-only (`s = 0`).
The instability is concentrated on small-n / wide-p datasets — the same
pattern AOM-Ridge struggles with in the master CSV.

Datasets with `s*` IQR ≥ 0.75 (top instability):

- `Biscuit_Sucrose_40_RandomSplit` — IQR **1.0** (s ranges 0.0 → 1.0)
- `DIESEL_bp50_246_hla-b` — IQR **1.0**
- `Rd25_CBtestSite` — IQR **1.0**
- `Quartz_spxy70` — IQR 0.75
- `Fv_Fm_grp70_30` — IQR 0.75
- `WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` — IQR 0.75
- `All_manure_K2O_SPXY_strat_Manure_type` — IQR 0.75
- `V25_spxyG` — IQR 0.75

Full table in
`bench/nicon_v2/benchmark_runs/r21_curated_oof_multiseed/per_dataset_s_star.csv`.

---

## Implications for r22+ (hybrid Option A / B)

**Codex round-4 verdict on D-B-013**: APPROVE the hybrid as an
**exploratory / adaptive diagnostic for r22+**, NOT as a confirmatory
or submission-grade design. Full Option A (24 h GPU on r22) is not
authorised at this stage because the production gate has already
failed.

**Hybrid design** (deferred until paper deadline or a future round
opens):

- 17 unstable datasets → true CV-5 inner shrinkage (Option A,
  ~ 5× train cost). Roughly 17 × 5 seeds × 5 inner folds × 6 min
  ≈ 13 h on the 4090.
- 22 stable datasets → keep held-out (Option B). 22 × 5 × 6 min
  ≈ 11 h.
- Total ≈ 24 h, but the heavy datasets are mostly in the unstable
  subset; in practice this is closer to ~ 15 h.

If the user opens a paper-deadline window, B will queue the hybrid
under D-B-014 (new request). For now, no r22 launch.

---

## Production claims policy (Codex round-4 condition)

- **No paper submission for P3.** P3 becomes a negative-result memo
  per plan §7. No claims of beating AOM-Ridge.
- **Relaxed descriptive claim allowed**: *"V2L-Residual-AOMPLS-shrinkage
  is the first NN-residual variant to beat paper CNN on the curated
  39-dataset cohort by a median −9.8 % with do-no-harm fallback. The
  science gate is missed by 4 percentage points on the wins criterion
  (71.2 % vs the 75 % threshold)."* Use this exact framing or stronger
  language only if accompanied by both numerical caveats.
- **Shrinkage instability disclosure**: any external presentation of
  this result must mention the 17 / 39 IQR > 0.3 finding.

---

## Registry card proposed to Agent C (`exhaustive_research`)

Per Codex round-4 condition on D-B-012 (`protocol_maturity = exploratory`):

```yaml
- canonical_name: V2L-Residual-AOMPLS-shrinkage-r21
  aliases: [v2l_residual_aompls_shrinkage]
  module: bench.nicon_v2.benchmarks.run_baseline_benchmark   # via PHASE_V2_R21_MULTISEED
  estimator: nicon_v2a (V2L-Residual-AOMPLS variant)
  teacher: aompls_extended (5-fold OOF on train)
  shrinkage:
    grid: [0.0, 0.25, 0.5, 0.75, 1.0]
    selector: held-out val partition (deterministic from (seed, val_fraction=0.2))
    catastrophic_threshold: 0.50
  task_types: [regression]
  input_constraints: {min_n: 30, gpu_required: true}
  supports_predefined_test_split: true
  inner_cv_nested: true
  runtime_tier: gpu_long
  maturity: exploratory                              # production gate FAIL on r21 cohort
  evidence: bench/nicon_v2/benchmark_runs/r21_curated_oof_multiseed/results.csv (39 ds × 5 seeds)
  caveats:
    - "Production gate FAIL: median +7.5 % vs aom_ridge_curated_best (14.9 % wins)."
    - "Science gate FAIL on wins: median −9.8 % vs paper CNN (71.2 % wins; need ≥ 75 %)."
    - "Shrinkage s* unstable on 17 / 39 datasets (IQR > 0.3 across seeds)."
    - "Do-no-harm gate PASS: catastrophic rate 1.0 %."
  preset: exhaustive_research
```

Agent C: ingest `r21_curated_oof_multiseed` rows into the master CSV
with `protocol_maturity = exploratory` and the above registry entry.

---

## Codex review checkpoints

- D-B-012 (r21 verdict + negative-result framing) — **APPROVED** in
  Codex round 4. Conditions applied above (relaxed claim wording,
  exploratory tag, instability disclosure).
- D-B-013 (Option-A reopen / hybrid for r22+) — **APPROVED** in Codex
  round 4. Deferred until paper deadline or a future round.
- This memo is now **LOCKED**. Any further r21-related work needs a
  new D-B-XXX request in `bench/SYNC.md`.

---

## Next

1. ✅ Codex round-4 APPROVE recorded; this memo is LOCKED.
2. Hand the registry card above to Agent C via `bench/SYNC.md`.
3. Switch focus to FCKResidualRegressor cohort run (the only remaining
   active item on B's queue).
4. r22 hybrid stays deferred until a paper window opens.
