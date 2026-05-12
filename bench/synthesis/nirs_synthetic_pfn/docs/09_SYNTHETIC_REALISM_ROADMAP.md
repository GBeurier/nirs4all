# Synthetic Realism Roadmap

Date: 2026-05-01

Scope: post-R9m consolidation for `bench/nirs_synthetic_pfn`, DIESEL
`uncalibrated_raw` realism only. This document does not promote R9m, does not
create R9n, does not change thresholds, metrics, gates, or production APIs, and
does not authorize any `nirs4all/` integration.

Multidataset correction: this roadmap is now historical for the DIESEL case
study only. It must not be read as the global synthetic realism roadmap. The
active cross-dataset plan is
`14_MULTIDATASET_REALISM_REPLAN.md`, which restarts the work from a
representative panel under `bench/tabpfn_paper/data`.

P2-07 update: the current mechanistic stop review is recorded in
`11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md`. It supersedes the
earlier "Immediate next action" sequence below where data-support prerequisites
are concerned: resume only after a wider real DIESEL/fuel cohort or row-bound
geometry metadata is available and exp28 confirms it. The Palier 2 benchmark
plan, stop criteria, and ordered tickets below are retained as historical
context, not as an active queue or promotion path.

## Doctrine

The immediate line remains mechanistic and uncalibrated. R3d
`r3d_diesel_matrix_v1` is the accepted DIESEL baseline. R9e-R9m are diagnostic
evidence only, not gates and not baseline replacements.

Work order:

1. exhaust explicit non-calibrated mechanisms first;
2. only after a data-supported mechanistic block, not the current missing-data
   stop, consider a mechanistic generator plus statistical noise capture layer;
3. only after that second stage is blocked or insufficient, consider ML/DL
   hybrid generation.

For Palier 2, "mechanistic" means predeclared physical or optical assumptions
that do not use labels, targets, splits, downstream metrics, adversarial scores,
PCA loadings, covariance estimates, quantiles, marginal matching, learned
residuals, or row-specific real statistics to tune the synthetic profile.

All proposals stay in bench. Production integration remains governed by
`05_INTEGRATION_GATE.md` and is out of scope here.

## Palier 1 Consolidation

Palier 1 tested whether the residual DIESEL mismatch could be closed by
support-local, R3d-inherited, uncalibrated mechanisms while preserving explicit
fallbacks and auditability. It did not establish a new accepted baseline.

Accepted baseline:

- R3d remains the DIESEL baseline because later variants are diagnostic
  tradeoffs, not clean replacements.

Useful but non-promoted evidence:

- R9e: support-only clean multiplicative attenuation after output clip lowers
  scalar level and morphology gap versus R3d without guard clipping. It is a
  useful pathlength/reference attenuation clue, but it remains behind R4b/R4c
  and is not a baseline.
- R9j: residual damping-only is the strongest single isolated component found
  in R9i-R9k. It lowers morphology gap versus R3d/R9i/R9e/R9f, but it does not
  reproduce R4b/R4c.
- R9l: residual damping plus clean attenuation improves over R3d, R9e, and R9j
  on the repeated-seed cohort, but remains diagnostic-only and still has
  aggregate-vs-paired ambiguity against R4c.

Bounded or negative evidence:

- R9f: moving attenuation before offset and applying it only to the
  continuum/path component is nearly R3d-like and does not beat R9e.
- R9h: support CH centers/drop-1720 alone is effectively null versus R3d.
- R9i: CH width/gain alone is non-null but too small to explain R4b/R4c.
- R9k: continuum-hump-only is effectively null to slightly worse versus R3d.
- R9b/R9c/R9d, from the earlier R9 sequence, bound support intercept, additive
  support shape, and mean-neutral redistribution. They do not establish a
  robust replacement path.

The key scientific result is therefore not "R9m won"; it is that Palier 1
isolated the available support-level levers and found no clean uncalibrated
variant that supersedes R3d.

## R9m NO-GO

R9m combined exactly:

- R9i width/gain: `ch_overtone_width_nm = 36.0`,
  `ch_overtone_gain_range = (0.092, 0.155)`;
- R9j residual damping windows and strength;
- R9e clean support-only attenuation `(0.970, 0.985)` on `750-1550 nm`.

R9m was contract-valid and lower than R3d on the diagnostic cohort, but it is
NO-GO for consolidation because:

- it does not clearly improve over R9l: morphology gap is only slightly lower
  than R9l (`1.516589` vs `1.519558`);
- it introduces a small derivative-under regression versus R9l
  (`-0.063143` vs `-0.062831`);
- the incremental width/gain contribution is tiny and already known from R9i
  to be incomplete;
- it remains behind R4b on aggregate morphology gap and does not produce a
  clean enough tradeoff to replace R3d;
- all evidence is limited to bench-only `uncalibrated_raw` diagnostic audits,
  not a B2/B3/B4/B5 gate or transfer validation.

Decision: R9m is final Palier 1 diagnostic evidence only. There is no R9n,
no retuning of R9m constants, no promotion, no gate, no integration.

## Remaining Mechanistic Hypotheses

The next work should not add another support-only tweak. The remaining plausible
mechanistic gap is at the row-level rendering model: how optical path,
reference attenuation, hydrocarbon overtone damping, continuum behavior, and
readout geometry interact before the final aligned output.

Allowed Palier 2 mechanistic hypotheses:

- row-level pathlength distribution: replace a scalar support attenuation with
  a physical distribution of effective pathlength/reference intensity before
  final support alignment;
- coupled damping and optical depth: express residual damping as a consequence
  of pathlength, absorbance saturation, detector/reference scale, or sample
  presentation rather than as a fixed support correction;
- readout geometry branch: audit whether reflectance/transmittance-style
  assumptions are being represented too similarly for DIESEL-like liquids;
- continuum construction: revisit whether the R3d continuum and micro-path
  terms are coupled in the wrong order or with the wrong smoothness before
  hydrocarbon features are added;
- detector/reference floor: test physically bounded blank/reference intensity
  effects that lower level without negative intercepts, guard clipping, or
  residual copying;
- wavelength support semantics: confirm that the audit cohort being fully
  inside `750-1550 nm` is not hiding off-support behavior that would matter for
  wider instruments.

P2-03 result: `reports/exp26_diesel_support_offsupport_discriminability_audit.md`
confirms the current real-aligned DIESEL cohort has zero off-support wavelength
bins and cannot distinguish R9e support-only attenuation from P2a full-row
attenuation by support/off-support behavior. On the existing generated prior
grid before real alignment, 38 bins are off-support; R9e leaves them unchanged
while P2a attenuates them, so the two routes are directly distinguishable for
wider instruments. This is report-only diagnostic evidence, not a gate,
promotion, profile, or builder behavior change.

Forbidden as immediate next work:

- PCA, covariance, noise capture, marginal or quantile matching;
- learned residuals, ML, or DL generation;
- tuning from labels, targets, train/test splits, adversarial AUC, transfer
  scores, or morphology metrics;
- copying R4b/R4c wholesale as a promoted baseline;
- retuning R9e/R9f/R9j/R9m amplitudes after seeing audit metrics;
- changing thresholds, metrics, gates, or integration status.

## Palier 2 Benchmark Plan

This section is frozen historical planning after P2-07. It does not authorize a
new profile, mechanism, gate, promotion, retune, or immediate follow-up audit.
Before any item below can be reactivated, exp28 must confirm one of the data
prerequisites in `11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md` and a new
mechanistic audit design must be reviewed.

Palier 2 was intended as a sequence of experiments, not a single bigger profile.
Each step was expected to produce a report under `reports/`, a reproducible
command, seed policy, contract checks, and explicit decision: continue, stop, or
escalate.

### P2.0 Decision Freeze

Goal: freeze the Palier 1 conclusion.

Requirements:

- record R3d as accepted baseline;
- record R9m as final Palier 1 NO-GO;
- list all R9e-R9m mechanisms and forbidden retunes;
- keep the R9m audit artifacts as evidence, not as a successor baseline.

Exit: no code change required unless a report reference is missing.

### P2.1 Mechanistic Failure Map

Goal: localize the residual mismatch at rendering-stage level rather than
support-shape level.

Benchmark:

- compare R3d, R9e, R9j, R9l, R9m, R4b, and R4c on identical DIESEL rows;
- decompose level, derivative, correlation, support/off-support weights, guard
  clipping, and dominant gap by render stage where metadata allows;
- add no new generator mechanism yet.

Historical continue criterion: a falsifiable stage-level hypothesis is
identified.

Stop criterion: the failure map only restates "mean_shift remains" without
pointing to a render-stage mechanism.

### P2.2 Pathlength/Reference Generative Branch

Goal: test a predeclared physical pathlength/reference branch that generalizes
R9e without retuning R9e amplitude.

Benchmark:

- apply the branch before final output alignment, with row-level sampled optical
  factors and deterministic metadata;
- keep non-DIESEL rows and non-compliant DIESEL rows byte-identical to R3d;
- compare against R3d, R9e, R9j, R9l, R9m, R4b, and R4c.

Historical continue criterion: lower morphology gap than R3d and R9l without
derivative regression, no guard clipping, and no hidden calibration flags.

Stop criterion: effect is R3d-like, R9e-like only, or improves level only by
breaking derivative/shape behavior.

### P2.3 Coupled Damping/Optical-Depth Mechanism

Goal: replace fixed residual damping with a mechanism tied to optical depth,
sample presentation, or detector/reference physics.

Benchmark:

- constants and distributions must be declared before audit;
- no direct morphology-metric optimization;
- include component ablations proving which coupling matters.

Historical continue criterion: improves over R3d and R9l on aggregate and
paired morphology while keeping derivative no worse than R9l and mean-curve
correlation not regressive versus R3d.

Stop criterion: the mechanism is just R9j/R9l under another name or depends on
metric-driven parameter choice.

### P2.4 Measurement/Readout Branch Audit

Goal: test whether DIESEL realism is blocked by missing or oversimplified
measurement-mode/readout behavior.

Benchmark:

- audit reflectance/transmittance-style assumptions in the bench adapter path;
- keep the comparison space `uncalibrated_raw`;
- report whether the branch explains R4b/R4c-like behavior mechanistically.

Historical continue criterion: the readout branch explains the level/derivative
tradeoff without support-only correction artifacts.

Stop criterion: no measurable effect or effect duplicates prior support
attenuation/damping.

### P2.5 Stop Review

Goal: decide whether non-calibrated mechanisms are genuinely exhausted.

Required evidence before escalation:

- at least two independent Palier 2 reports show mechanistic hypotheses were
  tested and rejected or bounded;
- each failed mechanism has a classified failure mode;
- no remaining simple physical hypothesis is listed by A0/A5 as untested;
- R3d remains better or more defensible than every candidate under unchanged
  metrics and gates.

Only if these conditions are met may the project open the next layer:
mechanistic generation plus statistical capture of noise/residual structure.

## Later Layers, Blocked

Layer 2: mechanistic plus statistical noise capture.

Blocked by P2-07. The current stop is a missing-data-support stop, not approval
to model residual/noise covariance, PCA-like structure, or domain-specific
stochastic artifacts. A later statistical capture plan may be discussed only
after a renewed, data-supported mechanistic audit is itself documented as
blocked or insufficient.

Layer 3: ML/DL hybrid.

Blocked by P2-07 and later than any approved statistical capture layer. Any
learned generator would require leakage controls, train/validation splits,
independent real holdouts, and adversarial checks. It is a later research
program, not a continuation of R9m.

## Stop And Promotion Criteria

These criteria are historical guardrails for a future data-supported mechanism,
not active promotion authorization. P2-07 permits no promotion while exp28 still
reports missing wider real support and missing row-bound geometry metadata.

A future bench-only promotion candidate would require all of:

- mechanism is predeclared and physically interpretable;
- non-DIESEL and non-compliant DIESEL fallbacks are byte-identical to R3d;
- audit flags show no calibration, real-stat capture, PCA/covariance/noise
  capture, ML/DL, labels, targets, splits, thresholds, or metric mutation;
- repeated-seed DIESEL audit has zero blocked rows;
- morphology gap improves versus R3d and the strongest relevant diagnostic
  comparator without derivative-under regression;
- aggregate medians and paired deltas agree or the disagreement is explicitly
  explained;
- result survives a separate tester verification.

Stop or demote if any of:

- improvement depends on retuning constants after seeing metrics;
- the dominant gap merely moves from mean shift to derivative under;
- guard clipping becomes a mechanism;
- the proposal only copies R4b/R4c without explaining the physics;
- the improvement is smaller than the audit noise or only visible in one row;
- the profile requires new metrics, changed gates, or production integration to
  look successful.

## Anti-Leakage And Calibration Guards

For any future data-supported Palier 2 report after P2-07 prerequisites are met,
require explicit flags or equivalent report text:

- `calibration_source = none`;
- `real_stat_source = none`;
- `threshold_source = none`;
- no PCA, covariance, noise capture, quantile, marginal, or residual matching;
- no labels, targets, splits, downstream performance, adversarial scores, or
  gate metrics used for parameter selection;
- no replay of real rows and no row-specific fitting;
- seeds, profile constants, and route markers documented;
- full `git status` summary and command recorded;
- unsupported or blocked rows reported rather than silently skipped.

For future statistical or ML/DL layers, add hard train/validation/test
separation before any modeling of residuals or noise. That guard is intentionally
not activated by the current P2-07 stop because immediate work is blocked
pending data support.

## Ordered Tickets

This queue is superseded by P2-07. It must not be executed until the stop-review
data prerequisites are met and a new mechanistic audit design is approved.

1. P2-00: register this consolidation as the post-R9m decision memo. No code.
2. P2-01: write a Palier 2 experiment spec for a render-stage failure map using
   existing R3d/R9e/R9j/R9l/R9m/R4b/R4c reports and no new mechanism.
3. P2-02: implement only report-side stage diagnostics if current metadata is
   sufficient; otherwise document the missing metadata fields before coding.
4. P2-03: propose one pathlength/reference generative branch with predeclared
   constants and route/fallback metadata. Lead approval required before code.
5. P2-04: audit the pathlength/reference branch against unchanged metrics and
   unchanged comparison space.
6. P2-05: propose one coupled damping/optical-depth mechanism only if P2-04
   does not close the gap cleanly.
7. P2-06: audit measurement/readout branch assumptions if P2-04/P2-05 fail or
   point to geometry as the blocker.
8. P2-07: run a stop review. Decide whether non-calibrated mechanisms are
   exhausted.
9. P2-08: only after a later data-supported stop review, draft a separate plan
   for mechanistic plus statistical noise capture.
10. P2-09: only after that later statistical layer fails, draft ML/DL hybrid
    options.

Immediate next action after P2-07: no new mechanistic, statistical, ML, or DL
work. Ingest wider real DIESEL/fuel support or row-bound geometry metadata, then
re-run exp28. Do not create R9n. Do not retune R9m.
