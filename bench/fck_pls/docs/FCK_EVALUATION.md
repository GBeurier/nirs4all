# FCK Evaluation — GO / NO-GO

**Date**: 2026-05-05 — **LOCKED** after Codex round-3 APPROVE (D-B-011).
**Owner**: Agent B
**Headline cohort per `FCK_PLAN_2026-05.md` §3.3**: full-57 (not run; **audit20 NO-GO**).
**Cohorts evaluated**: `fast12_transfer_core` (12 ds, complete), `audit20_transfer_core` (20 ds, complete), `full-57` (not run — audit20 NO-GO).
**Variants under review**: `FCK-PLS`, `FCK-Ridge`, `FCK-AOMPLS`, `ASLS-FCK-PLS` (fast12-only evidence), `Concat-SNV-FCK-AOMPLS`, `FCKResidual-AOMPLS` (staged, not run on cohort).

---

## TL;DR

**Status**: **audit20 NO-GO; full-57 not run.**

Locking language per Codex round-3 condition: full-57 is the headline
cohort per `FCK_PLAN_2026-05.md` §3.3, but the audit20 evidence is
sufficient to reject FCK-AOMPLS — it fails the strict
audit20→full-57 gate on all three direct criteria by 1.4× to 2.5× the
threshold. **No production claim is made from 20 datasets / 15
AOM-reference rows.** Full-57 stays available as a 70-minute CPU job
if a future round wants the robustness check.

| Cohort | FCK-AOMPLS verdict | Detail |
|---|---|---|
| fast12 (12 ds) | PASS smoke gate | median +14.2 % vs AOM-Ridge, +23.65 % vs PLS-baseline |
| audit20 (20 ds) | **FAIL strict gate** | median +12.6 % (need ≤ +5 %), q90 +57 % (need ≤ +25 %), worst +103 % (need ≤ +75 %) |
| full-57 (57 ds) | not run — audit20 verdict locks NO-GO | |

**Recommendation**: include FCK-AOMPLS, FCK-PLS, Concat-SNV-FCK-AOMPLS,
and ASLS-FCK-PLS in `exhaustive_research` only (ensemble diversity);
exclude from `best_current` and lower preset tiers. Drop FCK-Ridge
permanently (D-B-010, already locked).

---

## Evidence

### fast12_transfer_core (12 datasets × 6 pipelines = 72 rows, all OK)

Source: `bench/fck_pls/runs/smoke_fast12/results.csv` (executed 2026-05-05).
Summariser: `bench/fck_pls/summarize_smoke_fast12.py` (revised gate).

#### Per-pipeline absolute rmsep (median across 12 datasets)

| Pipeline | median | q75 | q90 | worst |
|---|---:|---:|---:|---:|
| **FCK-AOMPLS** | **1.32** | 3.41 | 4.55 | 76.15 |
| FCK-PLS | 1.54 | 3.75 | 4.11 | 63.35 |
| ASLS-FCK-PLS | 1.71 | 3.94 | 4.09 | 73.31 |
| PLS-baseline | 1.73 | 3.83 | 4.59 | 56.21 |
| Concat-SNV-FCK-AOMPLS | 1.80 | 4.08 | 4.57 | 67.32 |
| FCK-Ridge | 2.33 | 4.31 | 13.64 | 50.44 |

#### Δ% rmsep vs `aom_ridge_curated_best` (n = 8 datasets with reference)

| Pipeline | median | q90 | worst | wins/8 | gate |
|---|---:|---:|---:|---:|---|
| **FCK-AOMPLS** | **+14.2 %** | +55.3 % | +72.7 % | 1 / 8 | **PASS** |
| Concat-SNV-FCK-AOMPLS | +21.5 % | +91.9 % | +159.5 % | 0 / 8 | FAIL (improvement) |
| FCK-PLS | +32.2 % | +90.0 % | +106.9 % | 0 / 8 | FAIL (median) |
| ASLS-FCK-PLS | +29.7 % | +87.3 % | +139.6 % | 0 / 8 | FAIL (median) |
| PLS-baseline | +30.5 % | +209.8 % | +226.3 % | 1 / 8 | reference |
| **FCK-Ridge** | +157.3 % | +585.2 % | +675.1 % | 1 / 8 | **FAIL** (drop) |

#### Δ% rmsep vs other paper baselines (FCK-AOMPLS only)

| Reference | median Δ% | wins (FCK-AOMPLS) |
|---|---:|---:|
| paper PLS | +13.2 % | 4/12 |
| paper Ridge | +12.9 % | 4/12 |
| paper TabPFN-raw | +9.7 % | 5/12 |
| paper TabPFN-opt | +28.6 % | 1/12 |
| paper CNN | +3.7 % | 5/10 |
| **paper CatBoost** | **−14.3 %** | **7/12** |

FCK-AOMPLS beats paper CatBoost on the cohort and ties paper CNN. It is
behind AOM-Ridge and TabPFN-opt — expected for a feature-engineering
augmentation that adds 16× wavelength-axis convolutions on top of
AOM-PLS.

### audit20_transfer_core (20 datasets × 4 pipelines = 80 rows, all OK)

Source: `bench/fck_pls/runs/smoke_audit20/results.csv` (executed
2026-05-05; resumed once after a 30-min budget exhaust).
Pipelines: PLS-baseline, FCK-PLS, FCK-AOMPLS, Concat-SNV-FCK-AOMPLS.

#### Δ% rmsep vs `aom_ridge_curated_best` (n = 15 datasets with reference)

| Pipeline | median | q90 | worst | wins/15 | strict gate |
|---|---:|---:|---:|---:|---|
| **FCK-AOMPLS** | **+12.6 %** | +57.1 % | +102.7 % | 2 / 15 | **FAIL** all 3 |
| Concat-SNV-FCK-AOMPLS | +13.8 % | +98.5 % | +159.5 % | 1 / 15 | FAIL |
| FCK-PLS | +19.4 % | +114.4 % | +110 727 %* | 3 / 15 | FAIL |
| PLS-baseline | +20.6 % | +133.1 % | +6 308 %* | 1 / 15 | (reference) |

\* PLS-baseline and FCK-PLS each have one catastrophic outlier dataset
(`Quartz_spxy70` — PLS over-fits the n=24 train split). The catastrophic
behaviour mirrors the FCK-Ridge issue but is an inherent limitation of
PLS at very small n with large p.

#### Δ% rmsep vs other paper baselines (FCK-AOMPLS only)

| Reference | median Δ% | wins (FCK-AOMPLS) |
|---|---:|---:|
| paper PLS | +5.1 % | 7 / 20 |
| paper Ridge | +14.9 % | 5 / 20 |
| paper TabPFN-raw | +1.4 % | 10 / 20 |
| paper TabPFN-opt | +14.4 % | 4 / 20 |
| paper CNN | **−5.5 %** | **9 / 17** |
| **paper CatBoost** | **−1.1 %** | **10 / 20** |

**Interpretation**: FCK-AOMPLS is **competitive with TabPFN-raw**
(median +1.4 %, 10/20 wins) and **beats paper CNN** (median −5.5 %, 9/17
wins). It **ties paper CatBoost** (median −1.1 %, 10/20 wins). It is
**clearly behind AOM-Ridge** (median +12.6 %, q90 +57 %).

The cohort-level pattern: FCK-AOMPLS wins on small-n datasets where
AOM-Ridge over-fits; loses on medium-n / chemometrics where AOM-Ridge's
operator bank already captures most of the spectral signal.

### full-57 — not run; audit20 NO-GO

Per `FCK_PLAN_2026-05.md` §3.3 full-57 is the headline cohort; the
audit20 NO-GO does not equate to a production-tier claim. FCK-AOMPLS
fails the strict §3.3 gate by a wide margin: median +12.6 % (≈ 2.5×
the threshold), q90 +57 % (≈ 2.3× the threshold), worst +103 %
(≈ 1.4× the threshold). The bootstrap-CI excluding +10 % criterion was
not computed; **moot** because the hard median criterion already fails
at +12.6 % vs the +5 % limit (Codex round-3 condition).

Running full-57 would need to flip all three direct criteria —
i.e. the additional 37 datasets would need median Δ% ≤ −5 % vs AOM-Ridge
while the audit20 subset shows +12.6 %. That is implausible given the
fast12 / audit20 consistency.

Status: **full-57 not run; audit20 NO-GO**. A future Codex round can
open the 70-minute CPU job if the robustness check is wanted; nothing
in this memo claims full-57 numbers.

### FCKResidualRegressor

Implementation: `bench/fck_pls/fck_residual.py` (PR-5). 9 unit tests,
ruff/mypy clean. Same shrinkage-CV protocol as r21
(`bench/nicon_v2/docs/B_PLAN_2026-05.md` §2.2 — held-out calibration on
val_fraction=0.2, grid `{0, 0.25, 0.5, 0.75, 1.0}`, catastrophic threshold
+50 %). Smoke run on full-57 pending.

---

## Stop gate verdicts (consolidated)

| Gate | fast12 | audit20 | full-57 |
|---|---|---|---|
| Median Δ% ≤ +25 % vs AOM-Ridge | PASS (FCK-AOMPLS, +14.2 %) | PASS (FCK-AOMPLS, +12.6 %) | not run |
| **Median Δ% ≤ +5 % vs AOM-Ridge** | n/a (smoke gate) | **FAIL (+12.6 %, 2.5× over)** | not run |
| **q90 ≤ +25 % vs AOM-Ridge** | n/a (smoke gate) | **FAIL (+57.1 %, 2.3× over)** | not run |
| **Worst Δ% ≤ +75 % vs AOM-Ridge** | n/a smoke (relax to +200 %, PASS at +72.7 %) | **FAIL (+102.7 %, 1.4× over)** | not run |
| Median improvement ≥ +5 % vs PLS-baseline | PASS (+23.65 %) | PASS (+5.28 %) | not run |

---

## Final recommendation

- **best_current preset**: **EXCLUDE** all FCK variants. FCK-AOMPLS
  fails the strict audit20→full-57 gate on all three criteria; promoting
  it would dilute the preset's quality.
- **strong_practical preset**: **EXCLUDE** all FCK variants. FCK adds
  16× feature blow-up on top of AOM-PLS, costing runtime without a clear
  win vs the simpler AOM-Ridge / AOM-PLS.
- **exhaustive_research preset**: **INCLUDE** FCK-AOMPLS,
  Concat-SNV-FCK-AOMPLS, FCK-PLS. Justification:
  - cohort-level wins vs CatBoost (median −1.1 %, 10/20 on audit20) and
    paper CNN (median −5.5 %, 9/17 on audit20);
  - uncorrelated failure mode vs AOM-Ridge (FCK wins where AOM-Ridge
    over-fits, which improves ensemble coverage);
  - cheap (CPU-only) and reproducible.

  **ASLS-FCK-PLS — INCLUDE with fast12-only evidence caveat.**
  Per Codex round-3 condition, ASLS-FCK-PLS only ran on fast12 (n=8
  AOM-reference rows): +29.7 % median / +87.3 % q90 / +139.6 % worst,
  0/8 wins. Treat its registry card as fast12-only evidence; do not
  cite audit20 numbers for it.

  **Concat-SNV asymmetry** (Codex round-3 flag): Concat-SNV-FCK-AOMPLS
  delivers +15.92 % absolute median-rmsep improvement vs PLS-baseline
  (the strongest of the four), but is *worse* than FCK-AOMPLS on strict
  AOM-Ridge deltas (q90 +98.5 % vs +57.1 %; worst +159.5 % vs +102.7 %).
  Pick `FCK-AOMPLS` for ensemble diversity if only one slot is
  available; `Concat-SNV-FCK-AOMPLS` is the second pick because it
  trades worst-case stability for cohort-level lift over PLS-baseline.
- **fast_reliable preset**: **EXCLUDE** — runtime cost too high for the
  marginal win.
- **Learnable kernels** (`fckpls_torch.py` V1 / V2): **KEEP FROZEN**.
  Static FCK has not demonstrated dominance over AOM-Ridge; the case for
  learnable extension is even weaker. Reopen only if a future audit20
  rerun (with a different teacher, e.g. AOM-Ridge instead of AOM-PLS as
  the AOMPLS step) shows median Δ% ≤ −2 % vs AOM-Ridge.
- **FCK-Ridge**: **ARCHIVE** — dropped permanently from the slate per
  D-B-010 (locked in Codex round 1).
- **FCKResidualRegressor** (`bench/fck_pls/fck_residual.py`):
  **STAGE FOR EXHAUSTIVE_RESEARCH**. Implementation is ready and tested;
  not run on a full cohort yet. Schedule as a follow-up only if the r21
  multiseed result for V2L-Residual-AOMPLS-shrinkage shows that the
  shrinkage CV protocol works in practice (which would validate the
  same protocol applied to FCKResidualRegressor).

---

## Codex review checkpoints

- D-B-009 (fast12 promotion verdict) — **APPROVED** via D-B-009-fix in
  Codex round 2.
- D-B-010 (FCK-Ridge drop) — **APPROVED** in Codex round 1
  (fast12 evidence: +157.3 % median / +585.2 % q90 / +675.1 % worst
  vs AOM-Ridge, 1/8 wins).
- **D-B-011 (audit20 NO-GO + full-57 skip)** — **APPROVED** in Codex
  round 3. Conditions applied:
  1. Bootstrap-CI excluding +10 % marked moot (median +12.6 % already
     fails the +5 % hard threshold).
  2. Full-57 language scoped to "not run; audit20 NO-GO".
  3. Concat-SNV asymmetry recorded.
  4. ASLS-FCK-PLS marked fast12-only evidence.
  5. Registry proposal scoped to `exhaustive_research` only;
     FCK-Ridge excluded.

This memo is now **LOCKED**. Any further FCK promotion work needs a
new D-B-XXX request in `bench/SYNC.md`.

---

## Next

1. ✅ Codex round-3 APPROVE recorded; this memo is LOCKED.
2. Propose `exhaustive_research` registry cards to Agent C for:
   - `FCK-AOMPLS` (audit20 evidence, n=15 AOM-reference rows).
   - `Concat-SNV-FCK-AOMPLS` (audit20 evidence, n=15).
   - `FCK-PLS` (audit20 evidence, n=15; Quartz catastrophic flag).
   - `ASLS-FCK-PLS` (**fast12-only evidence**, n=8 AOM-reference rows).
   - **EXCLUDE FCK-Ridge** (fast12 catastrophic outliers).
3. FCKResidualRegressor stays staged. Run on a cohort only after r21
   multiseed validates the shrinkage-CV protocol.

---

## Addendum (2026-05-08): D-B-016 + D-B-017b results

Three follow-up decisions extended the FCK × AOM matrix:

### D-B-016 — FCKResidualRegressor on full-57

A linear residual head (`FCKStatic + Ridge`) on top of an AOMPLS-compact
teacher was tested as the analogue of r21's V2L-Residual-AOMPLS-shrinkage.
Results on full-57 (n = 39 with aom_ridge reference):

| Pipeline | median Δ% vs aom_ridge | median Δ% vs paper CNN | wins vs CNN |
|---|---:|---:|---:|
| FCKResidual-AOMPLS | +11.0 % | **−3.2 %** | 28 / 48 (58 %) |
| r21 V2L-Residual (CNN head) | +7.5 % | −9.8 % | 121 / 170 (71 %) |

Verdict (Codex round-7 LOCKED): NO-GO; FCKResidual is **strictly
weaker** than V2L-Residual at every cohort tier (6-7 pp). The linear
residual captures less of the non-linear residual signal than the
V2L-CNN head. Registry stays at `exhaustive_research`.

### D-B-017 / D-B-017b — FCK kernels in the AOM-Ridge bank

D-B-017 was opened with the default-hyperparam `AOMRidgePLS` and gave
catastrophic results (median +28 % / +25 % vs aom_ridge_curated_best).
Codex round-7 ruled the comparison unfair (default `ridge_alpha=1.0` vs
the curated reference's tuned alpha) and opened D-B-017b for the
CV-tuned re-run.

D-B-017b results on full-57 (n = 39 with aom_ridge reference):

| Bank | median Δ% | q90 Δ% | worst Δ% | wins / 39 |
|---|---:|---:|---:|---:|
| AOMRidgePLSCV-compact | +11.4 % | +56.6 % | +7 451 % * | 3 / 39 |
| **AOMRidgePLSCV-compact-with-fck** | **+13.4 %** | **+45.9 %** | **+3 173 %** * | **4 / 39** |
| Δ from FCK | +2.0 pp **worse** | **−10.7 pp better** | **−4 278 pp better** | +1 win |

\* Worst-case dominated by `Quartz_spxy70` numerical artifact (reference
~3 e-9 RMSEP explodes the ratio). Real signal is q90.

Verdict (Codex round-8 LOCKED): **stability trade-off** — FCK in the
AOM-Ridge bank trades a small median regression for substantially
better tail behaviour. Mechanistic explanation: AOM-Ridge applies all
operators in parallel as superblocks, so adding 8 FCK blocks makes the
Ridge penalty more conservative — reducing tail risk at the cost of
median fit. Registered as `AOMRidgePLSCV-compact-with-fck` for
`exhaustive_research` only.

### Final FCK × AOM 2 × 2 (full-57, fair comparison)

| | AOM-PLS (default) | AOM-Ridge (CV-tuned) |
|---|---:|---:|
| compact (9 ops) | median Δ% +9.8 % | +11.4 % |
| compact-with-fck (17 ops) | +8.7 % | +13.4 % |
| FCK Δ on median | **−1.1 pp better** | +2.0 pp worse |
| FCK Δ on q90 | +4.7 pp worse | **−10.7 pp better** |

**FCK behaves differently in AOM-PLS vs AOM-Ridge**:
- In AOM-PLS (per-component selector picks one operator per H-th
  component), FCK is selected on 30 % of datasets and produces a small
  median improvement.
- In AOM-Ridge (all ops applied as parallel superblocks), FCK is
  always-present and the Ridge penalty must reabsorb the additional
  blocks — this regularises the tails but slightly hurts median fit.

### Updated registry roster (all `exhaustive_research`)

| Card | Strength |
|---|---|
| AOMPLS-compact-with-fck-full57 (D-B-014/015) | median signal, no-cost augmentation |
| **AOMRidgePLSCV-compact-with-fck (D-B-017b)** | **stability trade-off (q90 / worst)** |
| FCKResidualRegressor (D-B-016) | weaker than V2L-Residual; redundant for any production use |
| FCK-AOMPLS / Concat-SNV-FCK-AOMPLS / FCK-PLS / ASLS-FCK-PLS (D-B-009) | preprocessing-style FCK (audit20 evidence) |

**Excluded**: FCK-Ridge (D-B-010, fast12 catastrophic) and the
default-hyperparam AOMRidgePLS (superseded by D-B-017b).
