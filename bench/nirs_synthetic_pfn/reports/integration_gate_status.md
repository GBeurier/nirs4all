# Integration Gate Status

Date: 2026-04-29

Decision: `NO-GO for nirs4all integration`

No `nirs4all/` files modified.

## Scope

This report assesses Phase E readiness only. It does not propose or perform
production integration. The assessment is based on the master roadmap,
scientific validation protocol, integration gate, and the Phase A-D reports
under `bench/nirs_synthetic_pfn/reports/`.

## Status Table

| phase | structural status | scientific status | gate effect for E |
|---|---|---|---|
| A1 Prior Canonicalization | Partial pass: canonical repair validates 918/1000 samples and documents invalid raw priors. | Not a realism gate. | Supports bench continuation only. |
| A2 Prior-to-Dataset Adapter | Pass: 10/10 curated presets build finite datasets with metadata and target contracts. | Smoke-level only; mode-specific optical physics remains a risk. | Supports bench continuation only. |
| A3 Real-Fit Adapter | Contract pass: fitted-only regeneration produces finite spectra on the exact wavelength grid. | Failed: fitted-only scorecard fails derivative statistics, peak density, and baseline curvature. | Blocks E. |
| B1 Prior Predictive Checks | Pass: 10/10 A2 smoke presets pass hard checks; downstream training marked allowed for B1. | Smoke guardrails only; explicitly carries `A3_failed_documented`. | Does not lift A3/B2. |
| B2 Real/Synthetic Scorecards | Runnable route over local AOM/TabPFN-like cohorts; missing/load failures documented. | Partial diagnostic improvement only: 71 `uncalibrated_raw` compared rows, 71 adversarial AUC smoke failures, 71 stretch failures, 6 blocked evidence gaps, 0 missing AUC rows; BEER/DIESEL/CORN remain named failures and COLZA/TABLET/WOOD rows are blocked on wavelength-grid evidence gaps. | Blocks E. |
| B3 Adversarial AUC Audit | Standalone bench-only audit consumes the B2 CSV without regenerating data. | `NO-GO`: `uncalibrated_raw` authoritative gate fails on 71/71 smoke failures and 6 blocked evidence gaps; SNV/calibrated diagnostics cannot override `uncalibrated_raw`. | Blocks synthetic usefulness claims. |
| B4 Transfer Validation | Gate-first script reads B2/B3 evidence before any synthetic build, real-only fit, TSTR route, or RTSR diagnostic. | `BLOCKED_BY_REALISM_GATE`: blocking reasons `adversarial_auc_raw_gate_NO-GO` and `B2_raw_realism_gate_failed`; 0 synthetic samples generated, 0 fitted models, 0 real-only baseline fits, 0 TSTR/RTSR routes. | Blocks synthetic usefulness claims. |
| B5 Minimal Ablation Attribution | CSV-first, report-only attribution over existing B2/B3/B4 rows; no training, no synthetic generation, no spectra loaded. | `BLOCKED_REPORT_ONLY` in B2/B3 `NO-GO` context with B4 blocked; 35 attribution rows surface failure groupings only, no causal or counterfactual claim. | Does not lift A3/B2/B3/B4. |
| C1-C3 Encoder/View Contracts | Pass structurally: canonical latents, spectral views, and same-latent view factory are validated bench-side. The Phase C precheck reports `BLOCKED_BY_UPSTREAM_REALISM_GATE` with `train_allowed=false`, `tabpfn_allowed=false`, `checkpoint_allowed=false`. | No realism, transfer, encoder training, or downstream gain claim; risk gates remain negative. | Does not lift A3/B2/B3/B4/B5. |
| D1-D3 Task Contracts | Pass structurally: prior task, context/query sampler, and multi-target task contracts are validated bench-side. The Phase D precheck reports `BLOCKED_BY_UPSTREAM_REALISM_GATE` with `task_sampling_allowed=false`, `icl_baseline_allowed=false`, `tabpfn_allowed=false`, `pfn_training_allowed=false`, `benchmark_allowed=false`, and 0 task episodes generated. | No training, transfer, PFN feasibility, or real-data value claim; risk gates remain negative. | Does not lift A3/B2/B3/B4/B5/C. |

## Reasons

1. A3 fitted adapter scientific scorecard failed. The fitted-only adapter
   passes basic dataset contracts but fails the scientific similarity gate on
   `derivative_statistics`, `peak_density`, and `baseline_curvature`. The A3
   report explicitly concludes that the blocker is fitter/generator
   information loss, not a local bench adapter contract bug, and advises not to
   loosen thresholds or use oracle/source provenance for the fitted-only gate.

2. B2 realism remains failed as an integration gate. The current B2 report
   writes 71 `uncalibrated_raw` compared rows with 71 adversarial AUC smoke
   failures, 71 stretch failures, 0 missing AUC rows, and 6 blocked evidence
   gaps. BEER (`Beer_OriginalExtract_60_KS`,
   `Beer_OriginalExtract_60_YbaseSplit`), DIESEL (`bp50_246_b-a`,
   `bp50_246_hla-b`, `bp50_246_hlb-a`), and CORN
   (`Corn_Oil_80_ZhengChenPelegYbaseSplit`,
   `Corn_Starch_80_ZhengChenPelegYbaseSplit`) rows remain named failures, and
   the 6 blocked `uncalibrated_raw` rows cover COLZA (`C_woOutlier`, `N_wOutlier`,
   `N_woOutlier`), TABLET (`Escitalopramt_310_Zhao`), and WOOD
   (`WOOD_Density_402_Olale`, `WOOD_N_402_Olale`) on
   `wavelength_grid_unknown` and `wavelength_grid_overlap` evidence gaps. No
   realism-pass or downstream transfer claim is available under the
   provisional `uncalibrated_raw` smoke thresholds.

3. B3 adversarial AUC audit is `NO-GO`. The standalone bench-only audit
   consumes the B2 CSV without regenerating data and confirms the
   `uncalibrated_raw` authoritative gate fails on 71 smoke failures and 6
   blocked evidence gaps. SNV and calibrated diagnostic evidence cannot
   override `uncalibrated_raw`.

4. B4 transfer validation is `BLOCKED_BY_REALISM_GATE`. The gate-first script
   reads B2 and B3 before any synthetic build and records blocking reasons
   `adversarial_auc_raw_gate_NO-GO` and `B2_raw_realism_gate_failed`. It
   generated 0 synthetic samples, fitted 0 models, ran 0 real-only baselines,
   and produced 0 TSTR/RTSR routes. No supervised TSTR usefulness claim is
   available.

5. B5 minimal ablation attribution is `BLOCKED_REPORT_ONLY`. The 35-row
   attribution CSV is a read-only summary over B2/B3/B4 rows; no training, no
   synthetic generation, and no spectra loading were performed. It does not
   estimate counterfactual effects and cannot lift any upstream gate.

6. Phase C and Phase D prechecks are `BLOCKED_BY_UPSTREAM_REALISM_GATE`. The
   Phase C encoder/TabPFN precheck reports
   `B3_NO-GO;B4_BLOCKED_BY_REALISM_GATE;B5_BLOCKED_REPORT_ONLY;integration_gate_NO-GO`
   with `train_allowed=false`, `tabpfn_allowed=false`, and
   `checkpoint_allowed=false`. The Phase D NIRS-ICL precheck reports the same
   blocking reasons plus `C_BLOCKED_BY_UPSTREAM_REALISM_GATE` with
   `task_sampling_allowed=false`, `icl_baseline_allowed=false`,
   `tabpfn_allowed=false`, `pfn_training_allowed=false`,
   `benchmark_allowed=false`, and 0 task episodes generated. C1-C3 and D1-D3
   structural contracts (canonical latents, spectral views, same-latent view
   factory, prior task, context/query sampler, multi-target task) remain
   validated bench-side only.

## Integration Gate Assessment

Phase E must not start because the integration gate requires bench evidence
for stable production value. The current state satisfies several bench
reproducibility and contract-stability needs, but fails the scientific-value
condition:

- no demonstrated synthetic prior improvement on real-data downstream metrics;
- no encoder or synthetic multi-view result improving few-shot or transfer
  performance;
- no production bug or minimal production API justified independently of model
  gains;
- no defensible supervised TSTR route because target/domain matching is not
  established.

Per `05_INTEGRATION_GATE.md`, work stays in bench when real-data validation is
absent, gains are not reproducible, metadata or realism contracts are
incomplete, or production API would expose research assumptions as stable
behavior. Those stop conditions apply here.

## Minimum Next Actions Before E Can Reopen

1. Resolve or explicitly redesign A3 so fitted-only regeneration passes the
   scientific similarity scorecard without oracle/source-provenance shortcuts.
2. Improve B2 realism until `uncalibrated_raw` adversarial AUC passes the
   documented smoke threshold (currently 71/71 failing) and stretch threshold
   (currently 71/71 failing) on a representative real cohort subset, and
   resolve the 6 blocked evidence gaps (COLZA, TABLET, WOOD) by reconstructing the
   wavelength grid evidence rather than loosening thresholds. Clear named
   BEER, DIESEL, and CORN failures.
3. Re-run the B3 standalone adversarial AUC audit only after B2 raw evidence
   is regenerated; confirm the raw authoritative gate flips before touching
   B4.
4. Reopen B4 transfer validation only after B2 and B3 pass; establish a
   matched real/synthetic target route, then run supervised TSTR/RTSR
   validation against real-only baselines with confidence intervals or split
   statistics.
5. Regenerate B5 attribution after B2/B3/B4 pass so it reflects unblocked
   evidence rather than the current `BLOCKED_REPORT_ONLY` snapshot.
6. Phase C and Phase D prechecks must be re-run by a human after upstream
   realism, transfer, attribution, and integration gate artifacts are
   regenerated with passing evidence; the prechecks themselves never start
   training, sampling, or benchmarking.
7. Write a new integration memo only after the scientific gates above pass,
   naming the minimal API, proposed production files, tests, docs, backward
   compatibility impact, and risks.

## Validation

- `git status --short nirs4all`: inspected before this report; no output.
- `git diff --name-only -- nirs4all`: inspected before this report; no output.
- Full tests skipped: report-only gate assessment; no code was changed.

## Decision

`NO-GO for nirs4all integration`.

Phase E remains closed until A3, B2, and B3 scientific blockers are cleared and
a minimal production API is justified by real-data evidence.
