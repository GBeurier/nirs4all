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
| B2 Real/Synthetic Scorecards | Runnable route over local AOM/TabPFN-like cohorts; missing/load failures documented. | Failed: 60/60 compared rows fail adversarial AUC and 60/60 fail PCA overlap. | Blocks E. |
| B3 Transfer Validation | Real-only baselines and synthetic PCA diagnostic route are runnable. | Supervised TSTR is blocked by target/domain mismatch and B2 realism failure; diagnostic only. | Blocks synthetic usefulness claims. |
| C1-C3 Encoder/View Contracts | Pass structurally: canonical latents, spectral views, and same-latent view factory are validated bench-side. | No realism, transfer, encoder training, or downstream gain claim; risk gates remain negative. | Does not lift A3/B2/B3. |
| D1-D3 Task Contracts | Pass structurally: prior task, context/query sampler, and multi-target task contracts are validated bench-side. | No training, transfer, PFN feasibility, or real-data value claim; risk gates remain negative. | Does not lift A3/B2/B3. |

## Reasons

1. A3 fitted adapter scientific scorecard failed. The fitted-only adapter
   passes basic dataset contracts but fails the scientific similarity gate on
   `derivative_statistics`, `peak_density`, and `baseline_curvature`. The A3
   report explicitly concludes that the blocker is fitter/generator
   information loss, not a local bench adapter contract bug, and advises not to
   loosen thresholds or use oracle/source provenance for the fitted-only gate.

2. B2 realism/adversarial/PCA overlap failed. The B2 scorecard route is
   runnable, but realism smoke success is not established: all 60 compared
   rows fail adversarial AUC and all 60 fail PCA overlap. This means the
   current synthetic spectra are trivially separable from real cohorts under
   the provisional smoke thresholds.

3. B3 TSTR supervised remains blocked and diagnostic only. B3 produced
   real-only baselines and an RTSR-style synthetic PCA diagnostic, but
   supervised TSTR is blocked because A2 synthetic targets are latent preset
   targets, not calibrated to selected real analyte labels, and B2 realism
   failure blocks usefulness claims.

4. C/D structural contracts pass but do not lift realism gates. C1-C3 and
   D1-D3 provide validated bench-side containers, views, samplers, and
   multi-target task contracts. Their own reports explicitly preserve
   `A3_failed_documented = True`, `B2_realism_failed = True`, and
   `claims.realism = False`, `claims.transfer = False`.

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
2. Improve B2 realism until adversarial AUC and PCA overlap pass documented
   thresholds on a representative real cohort subset, with remaining blocked
   rows classified.
3. Establish a matched real/synthetic target route for B3, then run repeated
   supervised TSTR/RTSR validation against real-only baselines with confidence
   intervals or split statistics.
4. If C/D are intended to justify integration, add downstream evidence that
   their encoder/task surfaces improve or remain competitive on real
   benchmarks; otherwise keep them bench-only.
5. Write a new integration memo only after the scientific gates above pass,
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
