# P2-04 / P2b Coupled Optical-Depth Damping Decision

Date: 2026-05-01

## Decision

Status: `no_code_design_first`

Do not implement a `p2b_*` generator profile in this pass. The next step is a
pre-code experimental design review for P2b, not a builder change.

Rationale:

- R3d remains the accepted DIESEL baseline.
- P2a already tested the simplest row-level pathlength/reference factor and is
  technically accepted as diagnostic-only, but it is indistinguishable from R9e
  on the current real-aligned DIESEL cohort because off-support count is zero.
- R9j is the best isolated residual damping signal. R9l already combines R9j
  damping with R9e attenuation. R9m adds R9i width/gain on top and is NO-GO.
- A trivial P2b implementation that maps the P2a attenuation draw to the R9j
  damping strength would mostly test correlation between two already-tested
  marginal levers. With the same `[0.970, 0.985]` factor range and R9j
  `[0.05, 0.15]` damping range, the likely effect is a small R9l covariance
  ablation, not a clearly new optical-depth mechanism.
- The builder metadata currently exposes sampled synthetic path/reference
  factors and residual damping stages, but not an independent physical
  optical-depth observable that can justify a bounded coupling law without
  choosing the law from morphology results.

Therefore the guardrail applies: no mechanism is coded until the coupling law
is specified well enough to distinguish a new hypothesis from R9j/R9l/R9m/P2a.

## What P2b Must Test

P2b should test this hypothesis, and no broader one:

> A single pre-output optical-depth latent variable changes both the effective
> reference/pathlength scale and the residual hydrocarbon damping strength
> before final audit alignment. The damping is a consequence of optical depth,
> not an independent R9j correction added beside an attenuation factor.

This is new only if all of the following are true:

- the optical-depth latent is sampled once per synthetic row from a declared
  prior, not inferred from real rows or audit metrics;
- damping strength is deterministically derived from that same latent;
- an ablation with identical marginal attenuation and damping distributions but
  broken coupling is included, so the report can separate "coupling matters"
  from "R9l-like marginals matter";
- non-DIESEL and non-compliant DIESEL rows remain byte-identical to R3d;
- all outputs are report-only: no gate, promotion, threshold change, metric
  change, R9n, or `nirs4all/` integration.

## Experimental Design Before Code

Proposed future audit name:

- `exp27_diesel_coupled_optical_depth_damping_audit`

Required compared profiles:

- `r3d_diesel_matrix_v1`
- `r9e_diesel_pathlength_reference_attenuation_v1`
- `r9j_diesel_residual_damping_isolation_v1`
- `r9l_diesel_residual_damping_clean_attenuation_v1`
- `r9m_diesel_width_gain_damping_clean_attenuation_v1`
- `p2a_diesel_row_pathlength_reference_v1`
- proposed `p2b_*` coupled candidate, only after design approval
- proposed `p2b_*_shuffled_ablation`, or equivalent report-side ablation, only
  if it can keep byte-identical R3d fallback semantics

Predeclared candidate constants, if approved later:

- optical/reference factor range: inherit P2a/R9e `[0.970, 0.985]`;
- damping windows: inherit R9j `[(1180, 46, 0.60), (1425, 54, 0.70)]`;
- damping strength bounds: inherit R9j `[0.05, 0.15]`;
- no CH center/width/gain retune, no continuum hump, no offset change, no
  support intercept/shape/redistribution, no readout transform, no extra clip.

The unresolved design point is the coupling law. It must be declared before any
audit run. The currently obvious linear mapping from factor to damping strength
is not yet accepted because it is physically under-justified and too close to
R9l with correlated draws.

## Required Report Checks

The future exp27 report must include:

- anti-leakage flags: calibration, real-stat capture, PCA, noise capture, ML,
  DL, labels, targets, splits, thresholds, metrics, and source oracle all false;
- provenance fields for the route marker, latent source, constant source, and
  coupling law source;
- paired deltas versus R3d, R9l, P2a, and the shuffled/independent ablation;
- explicit test of whether the coupled candidate differs from R9l by more than
  the covariance structure of existing marginal levers;
- support/off-support counts, with the P2-03 limitation repeated when the
  current real-aligned cohort has zero off-support bins;
- zero builder behavior change outside the explicit P2b route.

## Stop Criteria

Stop before code if any of these remain unresolved:

- the coupling law cannot be justified without referencing morphology outcomes;
- the mechanism only changes the correlation between R9e/P2a attenuation and
  R9j damping while preserving the same independent marginal effects;
- the design cannot include a same-marginals broken-coupling ablation;
- implementation requires new metrics, thresholds, labels, splits, real-row
  statistics, PCA/noise capture, ML/DL, or downstream feedback.

## Next

Lead review should approve or reject a specific coupling law before any
`p2b_*` profile is added. Until then, P2-04 remains design-only and R3d remains
the accepted baseline.
