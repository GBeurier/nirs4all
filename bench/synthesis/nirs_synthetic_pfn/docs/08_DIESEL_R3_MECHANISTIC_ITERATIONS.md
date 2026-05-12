# DIESEL R3 / R4 Mechanistic Iterations

Date: 2026-04-30
Scope: bench-only notes for `uncalibrated_raw` DIESEL realism work.

## Position

`r3d_diesel_matrix_v1` is the current accepted DIESEL diagnostic baseline in
the sentinel morphology scope.

`r3e_diesel_matrix_v1`, `r3f_diesel_matrix_v1`, and `r3g_diesel_matrix_v1`
remain diagnostic variants. They are useful evidence for the shape/offset
tradeoff, but they are not promoted over R3d.

`r4a_diesel_basis_v1` is a new bench-only diagnostic profile that explicitly
inherits `r3d_diesel_matrix_v1` for every non-DIESEL row (NOT R3e/R3f/R3g).
It addresses the R4 diagnostic finding that the real DIESEL basis is supported
on `750-1550 nm` while the 900-1050 nm region is currently synthetic-flat,
the 1100-1250 nm CH overtones are too strong, and the 1350-1500 nm region is
over-structured/inverted. R4a keeps the R3d micro-path continuum and detector
offset, drops the 1720 nm CH center, widens the remaining CH overtones at a
lower gain, damps the residual inside the 1100-1500 nm region without
inversion, and adds a short-continuum hydrocarbon scatter hump centered at
975 nm restricted to the 750-1550 nm support. R4a is diagnostic-only and is
flagged `needs-review`; it is not a promotion over R3d.

`r4b_diesel_derivative_restore_v1` is an additional bench-only diagnostic
profile that explicitly inherits `r3d_diesel_matrix_v1` for every non-DIESEL
row (NOT R4a). It addresses the R4a diagnostic finding that the mean-shift
score improved (`0.1783`) but `derivative_under` dominated (`9/9`) with
`derivative log10 = -0.2405`, i.e. the CH width/gain were too smoothed and the
damping too strong/wide for the first derivative to survive. R4b keeps the R3d
micro-path continuum and detector offset and the same support-only CH centers
as R4a (drops 1720 nm), but uses a narrower CH overtone width
(`38 nm` vs R4a `46 nm`) at a slightly higher gain (`0.085-0.145` vs R4a
`0.055-0.105`), and weaker / narrower residual damping windows
(`(1180, 52, 0.75)` and `(1425, 62, 0.85)` with strength `0.10-0.22` vs R4a
`(1180, 70, 1.0)` / `(1425, 85, 1.0)` at `0.30-0.50`). The short-continuum
hydrocarbon hump is kept on the 750-1550 nm support but is narrower
(`75 nm` vs R4a `90 nm`) and lower-amplitude (`0.00010-0.00035` vs R4a
`0.00025-0.00065`). R4b is diagnostic-only and is flagged `needs-review`;
it is not a promotion over R3d, and is not a fixed envelope (R4b does not
inherit R3g).

Expected R4b audit criteria (single-seed mini-audit, not a B2/B3/B4/B5 gate):

- DIESEL `np.diff(synthetic, axis=1).std()` strictly greater than the same
  metric under R4a on the same seed and dataset (derivative restored);
- DIESEL spectra differ from both R3d and R4a on the same seed;
- non-DIESEL rows byte-for-byte equal to R3d (`np.array_equal` on `X` and `y`
  and equal `r2c_mechanistic_remediation` audit dicts);
- unmarked or non-compliant DIESEL rows fall back to R3d byte-for-byte.

`r4c_diesel_balanced_derivative_v1` is the next bench-only diagnostic profile
that explicitly inherits `r3d_diesel_matrix_v1` for every non-DIESEL row (NOT
R4a/R4b). It addresses the R4b diagnostic finding that the gap mean improved
(`1.4525`) and the normalized mean shift moved to `0.3170` with no rows
dominated by `derivative_under`, but the `derivative log10 = -0.0906` regressed
versus the accepted R3d baseline (`-0.0440`), i.e. R4b's CH overtones were
still slightly too smoothed and its 1100-1500 nm damping still slightly too
strong for the first-derivative energy to match R3d. R4c keeps the R3d
micro-path continuum and detector offset and the same support-only CH centers
as R4a/R4b (drops 1720 nm), but uses a narrower CH overtone width
(`36 nm` vs R4b `38 nm`) at a slightly higher gain
(`0.092-0.155` vs R4b `0.085-0.145`), and weaker / narrower residual damping
windows (`(1180, 46, 0.60)` and `(1425, 54, 0.70)` with strength `0.05-0.15`
vs R4b `(1180, 52, 0.75)` / `(1425, 62, 0.85)` at `0.10-0.22`). The
short-continuum hydrocarbon hump is kept on the 750-1550 nm support but is
narrower (`72 nm` vs R4b `75 nm`) and slightly lower-amplitude
(`0.00010-0.00032` vs R4b `0.00010-0.00035`) so the 900-1050 nm level is still
lifted without flattening the derivative. R4c is diagnostic-only; it is not a
promotion over R3d, is not a fixed envelope (R4c does not inherit R3g), and
does not authorize any `nirs4all/` integration.

Expected R4c audit criteria (single-seed mini-audit, not a B2/B3/B4/B5 gate):

- DIESEL `np.diff(synthetic, axis=1).std()` strictly greater than the same
  metric under R4b on the same seed and dataset (derivative pushed closer to
  R3d than R4b);
- DIESEL spectra differ from both R3d and R4b on the same seed;
- non-DIESEL rows byte-for-byte equal to R3d (`np.array_equal` on `X` and `y`
  and equal `r2c_mechanistic_remediation` audit dicts), including unmarked or
  non-compliant DIESEL source metadata which must fall back to R3d
  byte-for-byte;
- the audit emits `bench_only_r4c_diesel_balanced_derivative_remediation` as
  the audit scope and `none` for every calibration / real-stat / threshold
  source, with no row-specific real statistic captured;
- the rendered markdown emits a `## R4c DIESEL Provenance` block that states
  diagnostic-only, no gate, no promotion, and no `nirs4all/` integration.

None of these profiles passes B2/B3, reopens B4/C/D, or authorizes any
`nirs4all/` integration.

## Evidence Summary

Repeated-seed standard sentinel audit, seeds `20260430`, `20260431`,
`20260432`, `n_synthetic_samples=64`, `max_real_samples=64`.

| profile | status | DIESEL gap mean/min/max | normalized mean shift | derivative log10 | mean_curve_corr | decision |
|---|---|---:|---:|---:|---:|---|
| R3c | diagnostic predecessor | `1.7372/1.6876/1.7738` | not aggregated | `-0.0141/-0.0337/-0.0011` | `0.0682/0.0628/0.0711` | superseded by R3d |
| R3d | accepted baseline | `1.5623/1.5254/1.6140` | `0.3755/0.3504/0.4004` | `-0.0440/-0.0555/-0.0304` | `0.0358/0.0305/0.0419` | keep |
| R3e | diagnostic only | `1.4241/1.3834/1.4479` | `0.2865/0.2710/0.3021` | `-0.0957/-0.1125/-0.0718` | `0.0248/0.0229/0.0267` | not promoted |
| R3f | diagnostic only | `1.5576/1.5115/1.6142` | `0.3657/0.3447/0.4001` | `-0.0469/-0.0636/-0.0138` | `0.0332/0.0282/0.0392` | not promoted |
| R3g | mini-audit diagnostic only | `1.4339/1.3998/1.4597` | `0.2913/0.2759/0.3068` | `-0.0989/-0.1127/-0.0751` | `0.0253/0.0226/0.0298` | not promoted |
| R4a | diagnostic only | `1.5670/1.5390/1.5857` | `0.1783/0.1677/0.1957` | `-0.2405/-0.2526/-0.2241` | `0.0527/0.0486/0.0568` | not promoted |
| R4b | needs-review diagnostic only | `1.4525/1.4053/1.4934` | `0.3170/0.2973/0.3397` | `-0.0906/-0.1040/-0.0755` | `0.0387/0.0328/0.0442` | not promoted |
| R4c | needs-review diagnostic only | `1.5042/1.4680/1.5668` | `0.3451/0.3230/0.3892` | `-0.0713/-0.0937/-0.0339` | `0.0362/0.0305/0.0404` | not promoted |

## Interpretation

The `mean_shift` label is the normalized scalar global mean offset:

`global_mean_delta / real_global_std`

It is not an amplitude, slope, band-shape, or correlation metric. The audit
score also includes `1 - mean_curve_corr`, but `_dominant_gap` only labels
mean-curve correlation directly when it is negative as `mean_curve_inversion`.

R3e and R3g show that lowering the blank-referenced continuum path and detector
offset improves the scalar gap while under-transferring derivative energy and
reducing mean-curve correlation. R3f restores derivative contrast but gives up
nearly all of the R3e/R3g gain and still stays below R3d on mean-curve
correlation.

R4a shows that support-restricted CH centers and a short-continuum hump can
lower scalar mean shift, but the combination of broad bands, low gain, and
wide damping suppresses first-derivative structure too aggressively
(`derivative_under: 9/9`). R4b restores derivative energy relative to R4a and
improves the repeated-seed gap versus both R3d and R4a, but it remains
derivative-regressive versus R3d (`-0.0906` vs `-0.0440`) and all rows are
still dominated by `mean_shift`. R4b therefore stays diagnostic-only and
`needs-review`.

R4c is the next mechanistic step that keeps part of the R4b gap and normalized
mean-shift improvement versus R3d while tightening the mean first-derivative
balance beyond R4b (narrower CH width `36 nm`, higher gain `0.092-0.155`,
weaker / narrower damping `(1180, 46, 0.60)` / `(1425, 54, 0.70)` at strength
`0.05-0.15`, narrower lower-amplitude hump `72 nm` / `0.00010-0.00032`).
The repeated-seed audit (seeds `20260430`, `20260431`, `20260432`) confirms
the intended trade-off: derivative log10 mean moves from R4b `-0.0906` to
`-0.0713`, i.e. closer to the R3d baseline `-0.0440` (paired delta vs R4b
`+0.0193`, vs R3d `-0.0273`). Gap mean is `1.5042` (paired delta vs R3d
`-0.0581`, vs R4b `+0.0517`), normalized mean shift is `0.3451` (paired
delta vs R3d `-0.0304`, vs R4b `+0.0280`), all 9/9 rows are dominated by
`mean_shift` (`derivative_under = 0/9`), and `mean_curve_corr` is
mean-non-regressive vs R3d (paired delta `+0.0005`). The derivative gain vs
R4b is not row-uniform: the paired derivative delta range is
`-0.0182` to `+0.0437`, with one repeated-seed row still below R4b. R4c
remains derivative-regressive vs R3d, so it stays diagnostic-only and
`needs-review`. It is not promoted over R3d and does not authorize any
`nirs4all/` integration.

## Current Scientific Blocker

The remaining DIESEL issue is not just offset. The repeated audits show a weak
mean spectral shape match. Continuing to lower the continuum or detector offset
is not a credible next mechanism.

The next mechanistic investigation must explain the DIESEL mean-curve mismatch
before adding another profile. Acceptable next work:

- inspect wavelength support and optical readout assumptions for the DIESEL
  rows;
- compare whether absorbance, transmittance, or blank-referenced intensity is
  the physically plausible raw space;
- test fixed hydrocarbon band hypotheses from general NIR spectroscopy only;
- report whether the blocker is missing mode/readout physics, missing
  hydrocarbon component shape, or non-mechanistic instrument/noise structure.

Forbidden next work in Palier 1:

- real-stat matching, PCA/covariance/noise capture, quantile/marginal
  calibration, or learned residuals;
- using labels, targets, splits, downstream performance, adversarial scores, or
  threshold changes to tune a profile.

## R6a Audit Outcome

`r6a_diesel_centered_hydrocarbon_shape_v1` was a bench-only diagnostic profile
intended to test whether a centered hydrocarbon shape adjustment, layered on
the R3d continuum and detector offset, could close the residual mean-curve
mismatch flagged in the R3/R4 series. The repeated-seed audit (seeds
`20260430`, `20260431`, `20260432`, `n_synthetic_samples=64`,
`max_real_samples=64`, DIESEL only) returned NO-GO: `mean_curve_corr` only
barely moved versus the R3d baseline and the R4c diagnostic, all 9/9 rows
remained dominated by `mean_shift`, and the audit reported a small negative
absorbance residual after the centered correction. R6a is therefore
diagnostic-only, not promoted over R3d, and does not authorize any further
retune. R3d remains the accepted DIESEL baseline.

## R7a Audit Outcome

`r7a_diesel_support_centered_residual_transfer_v1` was a bench-only diagnostic
profile that attempted, on top of the R3d continuum and detector offset, a
support-centered raw synthetic residual transfer in the DIESEL hydrocarbon
support, with a final nonnegative absorbance clip. The repeated-seed audit
covered seeds `20260430`, `20260431`, `20260432`, profiles
`r3d_diesel_matrix_v1` / `r4a_diesel_basis_v1` / `r4c_diesel_balanced_derivative_v1`
/ `r6a_diesel_centered_hydrocarbon_shape_v1` /
`r7a_diesel_support_centered_residual_transfer_v1`, DIESEL only, with
`n_synthetic_samples=64` and `max_real_samples=64`.

R7a metrics on this audit:

- `mean_curve_corr` mean `0.0354`, versus R3d `0.0358` and R4c `0.0362`
  (no mean-curve gain);
- `normalized_mean_shift` mean `0.4262`, versus R4c `0.3451` (mean-shift
  regression);
- `derivative log10` mean `-0.0051`, which is within the derivative
  threshold and does not by itself block the profile;
- gap mean `1.6555`, versus R3d `1.5623`;
- paired gap improvement vs R3d `0/9` rows;
- `final_clip_fraction` mean `69.95%`, max `70.68%`;
- no negative final absorbance after the nonnegative clip.

R7a is NO-GO and diagnostic-only. The blocker is mechanistic, not numerical:
the raw synthetic residual transferred into the DIESEL support was too large
relative to the micro-path absorbance base, so the final nonnegative clip
truncated about `70%` of the cells in the support and destroyed the mean
hydrocarbon shape. Retuning R7a constants after this audit is not authorized;
the profile must be replaced by a new mechanistic hypothesis, not iterated.
R7a is not promoted over R3d and does not authorize any `nirs4all/`
integration.

## R8a Audit Outcome

`r8a_diesel_mean_preserving_micro_path_modulation_v1` was a bench-only
diagnostic profile that replaced the R7a additive support-centered residual
transfer with a mean-preserving multiplicative modulation on top of the R4a
base. The synthetic-only `feature_residual` is masked to the 750-1550 nm
support, row-centered on the support, divided row-wise by a robust internal
scale (`p95_abs` with epsilon `1e-9`), and clipped to a dimensionless
`[-1, +1]` shape; the modulation factor is `exp(strength * shape)` with
`strength` drawn from a fixed `[0.10, 0.30]` range, applied after the R4a
non-negative base clip, then renormalized multiplicatively so the support row
mean is exactly preserved. Outside the 750-1550 nm support the readout is
identically the R4a base. The repeated-seed audit covered seeds `20260430`,
`20260431`, `20260432`, profiles `r3d_diesel_matrix_v1` /
`r4a_diesel_basis_v1` / `r4c_diesel_balanced_derivative_v1` /
`r6a_diesel_centered_hydrocarbon_shape_v1` /
`r7a_diesel_support_centered_residual_transfer_v1` /
`r8a_diesel_mean_preserving_micro_path_modulation_v1`, DIESEL only, with
`n_synthetic_samples=64` and `max_real_samples=64`.

R8a metrics on this audit (mean over 9 rows = 3 datasets x 3 seeds):

- `mean_curve_corr` mean `0.0474`, versus R3d `0.0358` (paired delta
  `+0.0117`, every paired row improves; minimum paired delta `+0.0073`);
- `normalized_mean_shift` mean `0.1783`, versus R3d `0.3755` and R4c
  `0.3451` (matches the R4a global level by construction since the
  modulation is mean-preserving on the support);
- `derivative log10` mean `-0.2261`, versus R3d `-0.0440` (regression,
  inherits the R4a flatter-derivative deficit);
- gap mean `1.5270`, versus R3d `1.5623`;
- paired gap improvement vs R3d on `7/9` rows (worst paired delta
  `+0.0068`, best paired delta `-0.1086`);
- `guard_clip_fraction` mean `0.0000`, max `0.0000` on every paired row,
  versus R7a `final_clip_fraction` mean `69.95%` and max `70.68%`
  (the R7a clip blocker is removed by construction);
- `support_mean_abs_delta_max <= 8e-18` on every paired row (mean
  preservation at machine precision after the multiplicative
  renormalization);
- no negative final absorbance (`mod_min_after_guard = 0.0000` on every
  paired row);
- dominant morphology gap mode `derivative_under` on `9/9` rows
  (R4a inheritance).

R8a clears `8/10` of the R7a-style criteria, including the R7a clip blocker,
but fails the two derivative criteria (`log10 derivative` mean and
`derivative_under + derivative_over <= 2/9`): the dominant morphology gap
mode flips to `derivative_under` on every paired row, inherited from R4a's
flatter-derivative base. R8a is reported diagnostic-only and is not promoted
over R3d. R8a is not a B2/B3/B4/B5 gate, not a calibration, not a real-stat
capture, not a thresholds-modified evaluator, and does not authorize any
`nirs4all/` integration. Non-DIESEL rows fall back byte-identical to R3d,
and audit-generated DIESEL rows carry a compliant R3d fallback marker so
non-compliant R8a routes also fall back byte-identical to R3d.

## R8b Audit Outcome

`r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1` was the selected
R4c-base plus R8a-modulation hybrid. It is bench-only, Palier 1 uncalibrated,
diagnostic-only, and not a promotion over R3d. The audit scope is
`bench_only_r8b_diesel_r4c_base_mean_preserving_micro_path_modulation`.

Design rationale (kept for the record): R8b inherits R3d for every non-DIESEL
row (NOT R4a/R4b/R4c/R5a/R5b/R5c/R6a/R7a/R8a). It changes only explicitly
DIESEL-marked petrochem fuel rows that carry `_r8b_diesel_micro_path_route`.
Audit-generated explicit DIESEL rows also carry a compliant
`_r3d_diesel_readout_route` marker, so a missing or non-compliant R8b route
falls back byte-identical to R3d. The R8b DIESEL base is the R4c
balanced-derivative absorbance pipeline byte-for-byte. After the base
non-negative clip, R8b applies the same fixed support-mean-preserving
multiplicative modulation contract as R8a on the 750-1550 nm support:
synthetic internal residual only, row-centered on the support, normalized by
`p95_abs` with epsilon `1e-9`, clipped to `[-1, 1]`, applied as
`exp(strength * shape)` with strength `[0.10, 0.30]`, then renormalized to
preserve the R4c support row mean.

The repeated-seed audit covered seeds `20260430`, `20260431`, `20260432`,
profiles `r3d_diesel_matrix_v1` / `r4a_diesel_basis_v1` /
`r4c_diesel_balanced_derivative_v1` /
`r6a_diesel_centered_hydrocarbon_shape_v1` /
`r7a_diesel_support_centered_residual_transfer_v1` /
`r8a_diesel_mean_preserving_micro_path_modulation_v1` /
`r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1`, DIESEL only,
with `n_synthetic_samples=64` and `max_real_samples=64` (n=9 rows per profile
= 3 datasets x 3 seeds).

R8b metrics on this audit (mean over 9 rows):

- `log10_derivative_std_p50_ratio` mean `-0.056`, versus R8a `-0.226`
  (paired R8b - R8a delta `+0.170`, median `+0.166`, every paired row
  improves; range `+0.143` / `+0.202`). R8b conserves the first-derivative
  magnitude noticeably better than R8a on every (seed, dataset) pair, as
  intended by the R4c base;
- `morphology_gap_score` mean `1.5183`, versus R3d `1.5623` (paired delta
  `-0.044`, every paired row improves) and versus R4c `1.5042` (paired delta
  `+0.014`, R8b is consistently slightly worse than R4c on the gap);
- `mean_curve_corr` mean `0.0327`, versus R3d `0.0358` (paired delta
  `-0.0030`) and R4c `0.0362` (paired delta `-0.0035`); R8b does not beat
  prior bests on mean-curve correlation;
- `normalized_mean_shift_abs` mean `0.3451`, identical to R4c (paired
  R8b - R4c delta `0.0` on every row); the modulation is mean-preserving on
  the support and off-support pixels are unchanged versus R4c, so R8b
  inherits the R4c mean shift exactly and does not close the residual
  mean-shift component;
- dominant morphology gap mode `mean_shift` on `9/9` rows (R4c inheritance);
  R4a and R8a remain the only profiles that flip to `derivative_under` on
  this cohort.

R8b Palier 1 invariants (uniform across seeds):

- `guard_clip_fraction` exactly `0.0` on every paired row; the multiplicative
  shape keeps the support strictly non-negative by construction and the
  R7a clip blocker stays absent;
- `support_mean_abs_delta_max <= 1.04e-17` on every seed, well below the
  test tolerance `1e-9`, i.e. mean preservation on the support at machine
  precision after multiplicative renormalization;
- `calibration_source = real_stat_source = threshold_source = "none"` on
  every row; every audit flag (`oracle`, `label_inputs_used`,
  `target_inputs_used`, `split_inputs_used`, `source_oracle_used`,
  `learned`, `real_stat_capture`, `thresholds_modified`,
  `metrics_modified`, `imputed`, `replays_real_rows`) is `False`;
- non-DIESEL rows fall back byte-identical to R3d, and audit-generated
  DIESEL rows that lack a compliant R8b route also fall back byte-identical
  to R3d.

R8b is therefore reported diagnostic-only and is not promoted over R3d. R8b
confirms the R4c-base hypothesis (the R4c balanced-derivative base preserves
much more of the first-derivative magnitude through the same R8a-style
mean-preserving multiplicative modulation than the R4a base), but on this
DIESEL cohort R8b does not outperform R4c on the morphology gap or
`mean_curve_corr`, and it does not improve on R3d on `mean_curve_corr`. R8b
does not authorize any `nirs4all/` integration, is not a B2/B3/B4/B5 gate, is
not a calibration, is not a real-stat capture, and does not modify any
threshold or metric.

## R9a Audit Outcome

`r9a_diesel_mean_shift_localization_audit` is a bench-only, diagnostic-only
mean-shift localization snapshot in the `uncalibrated_raw` comparison space.
It is not a promotion, not a gate, not a calibration, not a real-stat
capture, not a thresholds-modified evaluator, and does not authorize any
`nirs4all/` integration. The audit decomposes the scalar
`synthetic_mean - real_mean` over the fixed 750-1550 nm DIESEL support into
a `support_weighted_delta` and an `off_support_weighted_delta` that sum to
the global mean delta to floating-point tolerance.

The audit covered 3 DIESEL datasets (`DIESEL_bp50_246_b-a`,
`DIESEL_bp50_246_hla-b`, `DIESEL_bp50_246_hlb-a`), seeds `20260430`,
`20260431`, `20260432`, and 7 profiles (`r3d_diesel_matrix_v1`,
`r4a_diesel_basis_v1`, `r4c_diesel_balanced_derivative_v1`,
`r5a_diesel_absorbance_readout_v1`, `r5b_diesel_transmittance_readout_v1`,
`r5c_diesel_blank_referenced_intensity_v1`,
`r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1`), giving
`3 x 3 x 7 = 63` compared rows and `0` blocked rows, with
`n_synthetic_samples=64` and `max_real_samples=64`.

Localization findings:

- `off_support_weight` median and max are exactly `0.0` on every compared
  row. The aligned real wavelength grid `900-1550 nm` is fully inside the
  fixed support window `750-1550 nm`, so the off-support diagnostic is
  structurally null on this cohort and cannot, by construction, drive any
  next mechanistic step.
- `decomposition residual` max is `0.0` across all 63 rows, i.e. the
  identity `global_mean_delta = support_weighted_delta + off_support_weighted_delta`
  holds at machine precision in this audit.
- Every audit flag is `false`: `oracle`, `label_inputs_used`,
  `target_inputs_used`, `split_inputs_used`, `source_oracle_used`,
  `learned`, `real_stat_capture`, `thresholds_modified`, `metrics_modified`,
  `imputed`, `replays_real_rows`. `calibration_source`, `real_stat_source`,
  and `threshold_source` are all `none`. The audit scope is
  `bench_only_r9a_diesel_mean_shift_localization_audit`.

Profile-level mean-shift decomposition (median over 9 rows per profile,
all profiles share `support_weight = 1.0` and `off_support_weight = 0.0`):

- `r4c_diesel_balanced_derivative_v1`,
  `r5a_diesel_absorbance_readout_v1`, and
  `r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1` share the
  same median `global_mean_delta = 0.004896`. R5a is the absorbance-readout
  encoding of the same R4c-base support mean, and R8b inherits the R4c
  support mean exactly because its mean-preserving multiplicative
  modulation acts only on the support. R8b vs R4c/R5a differences on this
  cohort therefore belong to support shape/derivative diagnostics, not to
  the mean-shift decomposition.
- `r5b_diesel_transmittance_readout_v1` is scale-incompatible with the
  uncalibrated absorbance comparison space: median
  `global_mean_delta = 0.979090` and median morphology gap score
  `70.094514`, i.e. R5b is a readout outlier in this lane and not a
  candidate for closing the residual mean-shift component.
- `r5c_diesel_blank_referenced_intensity_v1` aggravates the mean-shift
  versus R3d / R4c (median `global_mean_delta = 0.014492` versus R3d
  `0.005419` and R4c `0.004896`).
- `r4a_diesel_basis_v1` is the only profile that reduces the median
  `global_mean_delta` to `0.002536`, but the R4a improvement on the
  scalar mean is bought back by the dominant morphology gap mode flipping
  to `derivative_under = 9/9` rows on this cohort, consistent with the
  R4a/R8a derivative-flatter inheritance documented above.

R9a thus localizes the residual DIESEL `mean_shift` component entirely
inside the 750-1550 nm support and confirms that off-support contributions
cannot be the mechanistic lever in this comparison space. R9a is purely
diagnostic and does not modify R3d's accepted-baseline status.

## R9b Audit Outcome

`r9b_diesel_support_intercept_v1` was a bench-only, diagnostic-only support-only
intercept profile. It is NOT a promotion over R3d, NOT a B2/B3/B4/B5 gate, NOT
a calibration, NOT a real-stat capture, NOT a thresholds-modified evaluator,
and does NOT authorize any `nirs4all/` integration. R9b inherits the full R4c
balanced-derivative absorbance base byte-for-byte and adds a single
pre-declared mechanistic absorbance constant (generic detector reference /
blank-cell baseline support-level absorbance prior) on the 750-1550 nm DIESEL
support after the R4c non-negative output clip; outside the support the
readout is byte-identical to the R4c base. The constant is not fitted to any
mean-shift residual and not derived from any real-stat, PCA, quantile, ML/DL,
label, target, split, AUC, morphology gap, threshold, or downstream feedback.

The exp11 audit (seeds `20260501`, `20260502`, `20260503`, `n_synthetic_samples=64`,
`max_real_samples=64`, sentinel tokens `DIESEL`, profiles
`r3d_diesel_matrix_v1` / `r4a_diesel_basis_v1` /
`r4c_diesel_balanced_derivative_v1` /
`r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1` /
`r9b_diesel_support_intercept_v1`) returned `45` compared rows and `0` blocked
rows over `3` seeds x `3` DIESEL sentinels x `5` profiles. `off_support_weight`
is structurally `0.0` and `support_weight` is `1.0` on every compared row in
this cohort, because the aligned real grid stays inside the fixed
750-1550 nm support window.

Per-profile medians on this cohort:

| profile | median global mean delta == support mean delta | median morphology gap | dominant gap distribution |
|---|---:|---:|---|
| `r3d_diesel_matrix_v1` | `0.005680` | `1.5982` | `mean_shift = 9` |
| `r4a_diesel_basis_v1` | `0.002657` | `1.5323` | `derivative_under = 7`, `mean_shift = 2` |
| `r4c_diesel_balanced_derivative_v1` | `0.005126` | `1.5111` | `mean_shift = 9` |
| `r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1` | `0.005126` | `1.5210` | `mean_shift = 9` |
| `r9b_diesel_support_intercept_v1` | `0.007123` | `1.6487` | `mean_shift = 9` |

Paired R9b vs R4c (per `(seed, dataset)`, `9/9` rows): R9b worsens both
`global_mean_delta` and `support_mean_delta` by exactly `+0.001997` on every
pair (the deterministic support intercept addition, off-support pixels
unchanged from the R4c base by construction), and worsens
`morphology_gap_score` by approximately `+0.138` on every pair. R9b therefore
moves R4c uniformly in the wrong direction on both the support mean shift and
the aggregate morphology gap on this cohort.

The corrected exp11 CSV
(`bench/nirs_synthetic_pfn/reports/r9b_diesel_support_mechanism_audit.csv`)
now carries 18 paired-delta columns (R9b minus R4c, R4a, and R3d on the
matching morphology and decomposition fields). These columns are audit /
reporting only; they are not new metrics and they do not modify any gate or
threshold.

R9b conclusion: rejected as a remediation candidate. The support-only
intercept form (a single positive mechanistic constant added on the 750-1550 nm
support on top of the R4c base) is exhausted on this cohort and is not
pursued. R9b is diagnostic-only, is not promoted over R3d / R4c / R8b, and
does not authorize any `nirs4all/` integration.

## R9c Audit Outcome

`r9c_diesel_selective_ch_bandwidth_damping_v1` was a bench-only,
diagnostic-only support-shape mechanism layered on top of the R3d absorbance
base. It is NOT a promotion over R3d, NOT a B2/B3/B4/B5 gate, NOT a calibration,
NOT a real-stat capture, NOT a thresholds-modified evaluator, and does NOT
authorize any `nirs4all/` integration. R9c inherits the full R3d absorbance
base / pipeline byte-for-byte for non-DIESEL rows and for non-compliant DIESEL
rows; on compliant DIESEL rows it adds, after the R3d non-negative output clip,
a positive-area support-shape additive profile that is support-local
(750-1550 nm), wavelength-dependent, and built from pre-declared general
liquid-hydrocarbon NIR constants. The mechanism uses no calibration, no
real-stat capture, no PCA, no covariance, no noise capture, no ML or DL, no
labels, targets, splits, AUC, morphology gap, threshold, or downstream
feedback. Off-support pixels are byte-identical to the R3d base.

The audit (`bench/nirs_synthetic_pfn/reports/r9c_diesel_support_shape_mechanism_audit.md`,
verified in `bench/nirs_synthetic_pfn/reports/r9c_tester_verification.md`)
returned `63` compared rows and `0` blocked rows. R9c is contract-valid: every
audit flag is `false`, `calibration_source = real_stat_source = threshold_source = "none"`,
non-DIESEL rows fall back byte-identical to R3d, and non-compliant DIESEL
rows fall back byte-identical to R3d. The mechanism is, by construction,
mean-shift-additive on the support, in the same positive sign as R9b.

R9c metrics on this cohort (medians):

- median `global_mean_delta` `0.066477`, versus R3d `0.005677`, R4c `0.005122`,
  and R9b `0.007119` (R9c is roughly an order of magnitude worse than every
  prior accepted or audited profile on the scalar mean shift);
- median `morphology_gap_score` `7.030112`, versus R3d `1.597554`, R4c
  `1.509282`, and R9b `1.646547` (R9c is roughly four times worse than the
  R3d / R4c / R9b cluster on the aggregate morphology gap);
- dominant morphology gap mode `mean_shift` on `9/9` paired rows;
- the residual is structurally additive on the 750-1550 nm support and
  inherits R9b's positive-area sign, only with a wavelength-dependent profile
  instead of a scalar.

R9c conclusion: contract-valid but scientifically rejected. A positive-area
support-shape additive profile on top of the R3d base, even with a
pre-declared wavelength-dependent shape rather than a scalar, moves R3d
uniformly in the wrong direction on both `global_mean_delta` and
`morphology_gap_score` (with R4c, R9b, and R8b cited only as comparison
profiles) and keeps the dominant gap mode locked at `mean_shift`. R9c is
diagnostic-only, is not promoted over R3d / R4c / R8b / R9b, and does not
authorize any `nirs4all/` integration. Retuning R9c constants is not
authorized; the positive-area support-shape additive form is exhausted on
this cohort and must be replaced by a different mechanistic hypothesis, not
iterated.

## R9d Audit Outcome

`r9d_diesel_energy_normalized_support_redistribution_v1` was a bench-only,
diagnostic-only mean-neutral, energy-normalized multiplicative support
redistribution layered on the R3d absorbance base. It is NOT a promotion over
R3d, NOT a B2/B3/B4/B5 gate, NOT a calibration, NOT a real-stat capture, NOT a
thresholds-modified evaluator, and does NOT authorize any `nirs4all/`
integration. R9d inherits the full R3d absorbance base / pipeline byte-for-byte
for non-DIESEL rows and for non-compliant DIESEL rows; on compliant DIESEL rows
it applies, after the R3d non-negative output clip, an `exp(strength * shape)`
multiplicative factor on the 750-1550 nm support followed by a per-row
multiplicative renormalization that preserves the pre-redistribution support
mean. The shape is a fixed mean-neutral, max-abs-normalized basis built from a
sum of Gaussian CH overtone bands at 1150 / 1210 / 1390 / 1460 nm with per-band
widths `(40, 40, 44, 48)` nm (mean-subtracted on the support and clipped to
`[-1, 1]`); per-row `strength` is bounded in `[0.035, 0.095]`. The
redistribution constants are PRE-DECLARED MECHANISTIC CONSTANTS from a general
liquid-hydrocarbon NIR energy redistribution prior; they were not derived from
any R9a/R9b/R9c residual, real spectra, marginal statistic, PCA loading,
quantile, ML/DL output, label, target, split, AUC, morphology gap score,
threshold, calibration, or downstream feedback. R4c is referenced only as a
comparison profile, not as the R9d base.

The audit
(`bench/nirs_synthetic_pfn/reports/r9d_diesel_energy_normalized_support_redistribution_audit.md`,
verified in `bench/nirs_synthetic_pfn/reports/r9d_tester_verification.md`)
returned `72` compared rows and `0` blocked rows over `3` seeds
(`20260501`, `20260502`, `20260503`) x `3` DIESEL sentinels x `8` profiles
(`r3d_diesel_matrix_v1`, `r4a_diesel_basis_v1`, `r4b_diesel_derivative_restore_v1`,
`r4c_diesel_balanced_derivative_v1`,
`r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1`,
`r9b_diesel_support_intercept_v1`,
`r9c_diesel_selective_ch_bandwidth_damping_v1`,
`r9d_diesel_energy_normalized_support_redistribution_v1`), with
`n_synthetic_samples=64` and `max_real_samples=64`. R9d is contract-valid:
every audit flag is `false` (`oracle`, `label_inputs_used`,
`target_inputs_used`, `split_inputs_used`, `source_oracle_used`, `learned`,
`real_stat_capture`, `thresholds_modified`, `metrics_modified`, `imputed`,
`replays_real_rows`, `calibration`, `uses_pca`, `captures_noise`, `uses_ml`,
`uses_dl`, `adds_offset`),
`support_redistribution_only=true`,
`support_redistribution_mean_neutral=true`,
`support_redistribution_energy_normalized=true`,
`constants_source=predeclared_general_liquid_hydrocarbon_nir_energy_redistribution_prior`,
audit scope is
`bench_only_r9d_diesel_energy_normalized_support_redistribution_audit`,
non-DIESEL rows fall back byte-identical to R3d, and non-compliant DIESEL
rows fall back byte-identical to R3d.

R9d metrics on this cohort (medians over 9 paired rows = 3 datasets x
3 seeds):

- median `global_mean_delta` `0.005679`, identical to R3d `0.005679`. By
  construction the aligned real grid is fully inside the 750-1550 nm support
  (`support_weight = 1.0`, `off_support_weight = 0.0` on every compared row),
  off-support pixels are byte-identical to R3d, and the per-row support mean
  is preserved within numerical tolerance, so R9d cannot move the scalar
  mean-shift in this audit geometry. The paired R9d - R3d delta on
  `global_mean_delta` is `0.0` on every `(seed, dataset)` pair. R9d therefore
  remains worse than R4c `0.005106`, R4b `0.004630`, and R8b `0.005106` on
  this scalar median, while only improving over the already-rejected positive
  support-additive R9b `0.007103` and R9c `0.066479` profiles;
- median `morphology_gap_score` `1.602604`, versus R3d `1.597481`,
  R4c `1.507567`, R4b `1.451708`, and R8b `1.519762` (R9d worsens the
  aggregate morphology gap versus R3d, R4c, R4b, and R8b on this cohort);
- median `mean_curve_corr` `0.032231`, versus R3d `0.034466` and
  R4c `0.036304` (R9d does not gain on mean-curve correlation);
- median `log10_derivative_std_p50_ratio` `-0.031972`, versus R3d
  `-0.032212`, i.e. derivative magnitude is essentially unchanged from the
  R3d base (paired delta is small and not the rejection driver);
- dominant morphology gap mode `mean_shift` on `9/9` paired rows.

R9d conclusion: contract-valid but scientifically rejected. A mean-neutral,
energy-normalized multiplicative support redistribution layered on the R3d
base, with pre-declared liquid-hydrocarbon CH-overtone shape constants and a
bounded per-row strength, cannot move `global_mean_delta` in this audit
geometry (full-support coverage + mean preservation + off-support byte
identity), worsens the median `morphology_gap_score` versus R3d / R4c / R4b /
R8b, does not gain on `mean_curve_corr`, and keeps the dominant gap mode
locked at `mean_shift` on every paired row. R9d is diagnostic-only, is not
promoted over R3d / R4c / R4b / R8b / R9b / R9c, and does not authorize any
`nirs4all/` integration. Retuning R9d constants is not authorized; the
mean-neutral / energy-normalized support redistribution form is exhausted on
this cohort and must be replaced by a different mechanistic hypothesis that
does not assume the support shape alone is the dominant residual.

## R9e0 Audit Outcome

`r9e0_diesel_signed_support_actuator_audit` was a bench-only,
diagnostic-only, probe-only exp14 audit of post-hoc signed support actuators
applied after the R3d base render and wavelength-grid alignment. R9e0 defines
no registered builder profile, promotes no probe, changes no gate or threshold,
and authorizes no `nirs4all/` integration. R3d remains the accepted DIESEL
baseline. The probes exist only as descriptive GO/NO-GO evidence relative to
R3d in the `uncalibrated_raw` comparison space.

The exp14 audit
(`bench/nirs_synthetic_pfn/reports/r9e0_diesel_signed_support_actuator_audit.md`,
verified in `bench/nirs_synthetic_pfn/reports/r9e0_tester_verification.md`)
returned `117` compared rows and `0` blocked rows over `3` seeds
(`20260501`, `20260502`, `20260503`) x `3` DIESEL sentinels x `13` profile /
probe rows. The R9e0 probe subset contains `45` rows: `5` probes x `9`
rows per probe. The full bench test suite reported `1220 passed` in the
tester verification. The exp14 audit and verification also confirm
probe-only metadata (`profile_registered=false`, `probe_only=true`), R3d base
fallback, no calibration, no real-stat capture, no PCA/covariance/noise
capture, no ML/DL, no labels/targets/splits/downstream feedback, and no
threshold or metric mutation.

R9e0 medians on this cohort:

| profile / probe | median global mean delta | median morphology gap | median guard clip fraction |
|---|---:|---:|---:|
| `r3d_diesel_matrix_v1` baseline | `0.005672` | `1.598182` | NA |
| `r9e0_negative_blank_intercept_0p0010` | `0.005374` | `1.571736` | `0.712711` |
| `r9e0_negative_blank_intercept_0p0020` | `0.005100` | `1.545629` | `0.732650` |
| `r9e0_multiplicative_attenuation_0p985` | `0.005538` | `1.582328` | `0.000000` |
| `r9e0_multiplicative_attenuation_0p970` | `0.005405` | `1.566373` | `0.000000` |
| `r9e0_negative_intercept_0p0010_plus_r9d_shape_0p035` | `0.005374` | `1.574303` | `0.712711` |

Strict interpretation: every R9e0 probe gives descriptive GO-evidence versus
R3d on the two reported medians (`global_mean_delta` and
`morphology_gap_score`), but no probe is promoted and no probe replaces R3d.
The negative intercept probes are scientifically risky because the
non-negative guard clips roughly `71-73%` of support cells; therefore they
must not become direct profile templates. The multiplicative attenuation
probes are the cleanest R9e0 signal because they reduce level and gap without
any guard clipping, but this is still not a promotion. The next mechanistic
step must investigate a physically grounded scale / pathlength / attenuation /
continuum mechanism that lowers the DIESEL level without clip and without
falling into `derivative_under`; it must not be a calibration, a R9d retune,
or a profile selected from exp14 probes.

## R9e Audit Outcome

`r9e_diesel_pathlength_reference_attenuation_v1` is the registered bench-only,
diagnostic-only pathlength / blank-reference attenuation profile derived from
the clean R9e0 multiplicative-attenuation signal, but implemented as a
predeclared mechanistic component rather than a promoted probe. R9e is NOT a
promotion over R3d, NOT a B2/B3/B4/B5 gate, NOT a calibration, NOT a real-stat
capture, NOT a thresholds-modified evaluator, and does NOT authorize any
`nirs4all/` integration. R3d remains the accepted DIESEL baseline.

Mechanism contract: R9e inherits the R3d absorbance base and fallback. On
compliant DIESEL rows only, it applies a positive row-wise uniform
multiplicative attenuation factor in `[0.970, 0.985]` after the R3d output
clip, restricted to the fixed `750-1550 nm` support. Off-support cells are
byte-identical to R3d. R9e adds no offset, applies no additional guard clip,
performs no support-mean renormalization, uses no R9d shape, and does not use
calibration, real-stat capture, PCA/covariance/noise capture, ML/DL, labels,
targets, splits, downstream feedback, threshold mutation, or metric mutation.

The exp15 audit
(`bench/nirs_synthetic_pfn/reports/r9e_diesel_pathlength_reference_attenuation_audit.md`,
verified in `bench/nirs_synthetic_pfn/reports/r9e_tester_verification.md`)
returned `54` compared rows and `0` blocked rows over `3` seeds
(`20260501`, `20260502`, `20260503`) x `3` DIESEL sentinels x `6` profiles
(`r3d_diesel_matrix_v1`, `r4b_diesel_derivative_restore_v1`,
`r4c_diesel_balanced_derivative_v1`,
`r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1`,
`r9d_diesel_energy_normalized_support_redistribution_v1`,
`r9e_diesel_pathlength_reference_attenuation_v1`), with
`n_synthetic_samples=64` and `max_real_samples=64`.

R9e metrics on this cohort (medians over 9 paired rows = 3 datasets x
3 seeds):

- median `global_mean_delta` `0.005471` and median `support_mean_delta`
  `0.005471`, versus R3d `0.005676` and R9d `0.005676`;
- median `morphology_gap_score` `1.571865`, versus R3d `1.596743` and R9d
  `1.602339`;
- median `log10_derivative_std_p50_ratio` `-0.040459`, close to R3d
  `-0.032350` and better preserved than R4c `-0.059055` / R4b `-0.094243`
  on this cohort;
- median `mean_curve_corr` `0.035650`, essentially tied with R3d `0.035653`
  and above R9d `0.033092`, but below R4b `0.040178` and R4c `0.037393`;
- median `guard_clip_fraction` `0.000000`; no extra guard clip is applied;
- dominant morphology gap mode remains `mean_shift` on `9/9` paired rows.

R9e is a clean mechanistic component: support-only positive multiplicative
attenuation lowers the scalar level and aggregate gap versus R3d and R9d
without introducing off-support changes, offsets, guard clipping,
renormalization, or the R9d support shape. It does not beat the R4b / R4c
family on the main gap / level medians: R4b has median `global_mean_delta`
`0.004658` and gap `1.455727`, while R4c has median `global_mean_delta`
`0.005104` and gap `1.511024`. Therefore R9e improves R3d/R9d but is not a
new accepted baseline.

Verification status from the independent tester report: targeted R9e builder
tests passed (`10 passed, 406 deselected`), exp15 tests passed (`4 passed`),
exp14 + exp15 tests passed (`14 passed`), the full
`bench/nirs_synthetic_pfn/tests` suite passed (`1234 passed, 4 warnings`),
`ruff check .` passed, and scoped mypy on the four R9e Python files passed.
Root `mypy .` is out-of-scope as a signal in this workspace because it fails
while traversing `.venv/Sphinx`
(`.venv/lib/python3.13/site-packages/sphinx/util/typing.py`).

R9e conclusion: diagnostic-only, non-gate, non-promoted. It is useful
evidence that a clean attenuation / pathlength / reference-level mechanism can
lower the residual DIESEL level without clipping, but it does not displace
R3d. R3d remains the accepted DIESEL baseline, and no integration follows.

## R9f Audit Outcome

`r9f_diesel_pre_offset_pathlength_reference_attenuation_v1` is the registered
bench-only, diagnostic-only pre-offset pathlength / blank-reference attenuation
profile. It tests whether the clean R9e attenuation signal should act earlier
in the R3d render, only on the continuum/path component before additive
baseline and output clipping. R9f is NOT a promotion over R3d, NOT a
B2/B3/B4/B5 gate, NOT a calibration, NOT a real-stat capture, NOT a
thresholds-modified evaluator, and does NOT authorize any `nirs4all/`
integration. R3d remains the accepted DIESEL baseline.

Mechanism contract: R9f inherits the R3d absorbance base and fallback. On
compliant DIESEL rows only, it applies a positive row-wise uniform attenuation
factor in `[0.970, 0.985]` on the fixed `750-1550 nm` support, before additive
baseline and output clip, and only on the continuum/path component
(`continuum * path_factors[:, None] * path_profile`). Feature residuals,
additive baseline / offsets, readout transform, R9d shape, support-mean
renormalization, negative intercept, and extra clipping are unchanged or not
used. Calibration, real-stat capture, PCA/covariance/noise capture, ML/DL,
labels, targets, splits, downstream feedback, threshold mutation, and metric
mutation remain absent.

The exp16 audit
(`bench/nirs_synthetic_pfn/reports/r9f_diesel_pre_offset_pathlength_reference_attenuation_audit.md`,
verified in `bench/nirs_synthetic_pfn/reports/r9f_tester_verification.md`)
returned `63` compared rows and `0` blocked rows over `3` seeds
(`20260501`, `20260502`, `20260503`) x `3` DIESEL sentinels x `7` profiles
(`r3d_diesel_matrix_v1`, `r4b_diesel_derivative_restore_v1`,
`r4c_diesel_balanced_derivative_v1`,
`r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1`,
`r9d_diesel_energy_normalized_support_redistribution_v1`,
`r9e_diesel_pathlength_reference_attenuation_v1`,
`r9f_diesel_pre_offset_pathlength_reference_attenuation_v1`), with
`n_synthetic_samples=64` and `max_real_samples=64`. Each profile contributes
`9` compared rows.

R9f metrics on this cohort (medians over 9 paired rows = 3 datasets x
3 seeds):

- median `global_mean_delta` `0.00565240766844` and median
  `support_mean_delta` `0.00565240766844`;
- median `off_support_mean_delta` `0` in this audit geometry;
- median `morphology_gap_score` `1.59542861178`;
- median `log10_derivative_std_p50_ratio` `-0.0323313599657`;
- median `mean_curve_corr` `0.0342291346304`;
- median `guard_clip_fraction` `0`;
- dominant morphology gap mode remains `mean_shift` on `9/9` paired rows.

R9f is contract-valid and clean: it preserves the diagnostic-only Palier 1
contract, introduces no clipping, and confines the attenuation to the intended
pre-offset continuum/path component. Scientifically, it does not beat R9e and
does not beat the R4b / R4c family. It is nearly R3d-like on both scalar level
and aggregate morphology gap (`0.00565240766844` / `1.59542861178` for R9f
versus R9f-cohort R3d `0.005684` / `1.598164`), while R9e is lower on the
same R9f audit cohort (`0.005479` / `1.573312`) and R4b/R4c remain better on
the gap. The tested stage, pre-offset attenuation of the continuum/path
component alone, is therefore not the principal lever.

Verification status from the independent tester report: targeted R9f builder
tests passed (`11 passed, 416 deselected`), exp16 tests passed (`4 passed`),
exp15 + exp16 tests passed (`8 passed`), the full
`bench/nirs_synthetic_pfn/tests` suite passed (`1249 passed, 4 warnings`),
`ruff check .` passed, and scoped mypy on the four R9f Python files passed.
Root `mypy .` is out-of-scope as a signal in this workspace because it fails
while traversing `.venv/Sphinx`
(`.venv/lib/python3.13/site-packages/sphinx/util/typing.py`).

R9f conclusion: diagnostic-only, non-gate, non-promoted. It provides
negative diagnostic evidence for pre-offset continuum/path-only attenuation,
does not authorize retuning the amplitude, does not authorize Palier 2, does
not authorize any integration, and does not displace R3d. R3d remains the
accepted DIESEL baseline.

## R9g R4 Component Attribution Audit Outcome

`r9g_diesel_r4_component_attribution_audit` is a diagnostic-only comparative
audit, not a new profile and not a retuning pass. It re-ran the same
repeated-seed DIESEL cohort as R9e/R9f with profiles
`r3d_diesel_matrix_v1`, `r4a_diesel_basis_v1`,
`r4b_diesel_derivative_restore_v1`,
`r4c_diesel_balanced_derivative_v1`,
`r9e_diesel_pathlength_reference_attenuation_v1`, and
`r9f_diesel_pre_offset_pathlength_reference_attenuation_v1` on seeds
`20260501`, `20260502`, `20260503`, with `n_synthetic_samples=64`,
`max_real_samples=64`, DIESEL-only sentinel rows, and comparison space
`uncalibrated_raw`.

The R9g medians over 9 paired rows are:

- R3d: `global_mean_delta = support_mean_delta = 0.005670`,
  `morphology_gap_score = 1.598551`, derivative `-0.032190`,
  `mean_curve_corr = 0.034744`;
- R4a: `0.002659`, `1.532352`, derivative `-0.223922`,
  `mean_curve_corr = 0.047905`, with dominant gap
  `derivative_under=7, mean_shift=2`;
- R4b: `0.004654`, `1.454853`, derivative `-0.094652`,
  `mean_curve_corr = 0.039911`;
- R4c: `0.005118`, `1.507308`, derivative `-0.059950`,
  `mean_curve_corr = 0.036717`;
- R9e: `0.005465`, `1.572196`, derivative `-0.040299`,
  `mean_curve_corr = 0.034727`, guard clip `0`;
- R9f: `0.005639`, `1.595322`, derivative `-0.032568`,
  `mean_curve_corr = 0.034270`, guard clip `0`.

Paired attribution deltas show that R4b beats R9e/R9f on both scalar support
mean and morphology gap (median gap deltas `-0.105024` vs R9e and
`-0.124968` vs R9f), while R4c also beats R9e/R9f on those same fields
(`-0.054364` vs R9e and `-0.073823` vs R9f). R4b/R4c do not win by
continuum/path attenuation: their derivative is lower than R9e/R9f, while
their mean-curve correlation is modestly higher. Relative to R4a, R4b and R4c
recover first-derivative energy strongly (median derivative deltas `+0.137223`
and `+0.156545`) and remove the R4a dominant `derivative_under` failure mode,
while keeping more of the R4-family gap reduction than clean attenuation.

R9g attribution conclusion: the signal comes primarily from a coupled R4
hydrocarbon-basis package. The support CH centers / drop of the 1720 nm center
and the 975 nm continuum hump appear to set the support level; R4b/R4c
width/gain derivative restoration and weaker/narrower damping prevent the
R4a derivative collapse. The damping windows and 975 nm hump are not strictly
isolable from the CH-center/drop-1720 change without creating new builder
variants. This phase intentionally did not modify `builder_adapter.py`, so the
formal conclusion is coupling non-isolable without a new variant. R4b/R4c
remain explanatory diagnostic evidence only; they are not promoted, do not
open a gate, do not authorize retuning R9e/R9f attenuation amplitude, and do
not authorize any `nirs4all/` integration. R3d remains the accepted DIESEL
baseline.

## R9h Support CH-Center Isolation Audit Outcome

`r9h_diesel_support_ch_center_drop1720_isolation_v1` is the follow-up
bench-only diagnostic isolation profile created to split the R9g coupled R4
component conclusion. R9h is diagnostic-only, non-gate, and non-promoted. It
does not authorize any `nirs4all/` integration, threshold change, metric
change, calibration, real-stat capture, PCA/covariance/noise capture, ML/DL,
labels, targets, splits, downstream feedback, or attenuation retuning. R3d
remains the accepted DIESEL baseline.

Mechanism contract: R9h inherits the full R3d pipeline and changes only one
R4-family component on compliant explicit DIESEL petrochem rows:
`ch_overtone_centers_nm=(1150, 1210, 1390, 1460)`, i.e. the support CH centers
with the 1720 nm center dropped. It intentionally adds no 975 nm continuum
hump, no damping windows, no R4 width/gain changes, no attenuation, no support
intercept, no support shape, no redistribution, no readout transform, and no
extra clip. Width/gain remain R3d-like, not R4b/R4c-like.

The exp18 repeated-seed audit used seeds `20260501`, `20260502`, and
`20260503`, `n_synthetic_samples=64`, `max_real_samples=64`, DIESEL-only
sentinel rows, and comparison space `uncalibrated_raw`. It compared 63 rows,
blocked 0 rows, and produced 9 rows per audited profile.

R9h medians over its 9 paired rows are:

- `global_mean_delta = support_mean_delta = 0.005676745221`;
- `morphology_gap_score = 1.59918713004`;
- `log10_derivative_std_p50_ratio = -0.032328094721`;
- `mean_curve_corr = 0.035081087950`;
- dominant gap mode remains `mean_shift` on `9/9` rows.

Paired R9h - R3d median deltas are effectively zero:

- global/support mean delta: `-1.49e-11`;
- morphology gap: `-6.70e-10`;
- derivative: `-1.76e-10`;
- mean-curve correlation: `-7.30e-11`.

The result is null. Support CH centers / drop 1720 alone does not explain the
R4b/R4c gap reduction and does not materially move R3d on the current cohort.
Against R4b and R4c, R9h is worse on the morphology gap, so the R4b/R4c
evidence cannot be attributed to the center/drop1720 component in isolation.
The next isolation should target width/gain derivative restoration or the
damping / continuum part of the R4 package, not another scalar retune or
support-center retune.

Verification status from the R9h tester report: targeted R9h builder tests
passed (`9 passed, 427 deselected`), exp18 tests passed (`3 passed`), exp17 +
exp18 tests passed (`7 passed`), the full `bench/nirs_synthetic_pfn/tests`
suite passed (`1265 passed, 4 warnings`), `ruff check .` passed, and scoped
mypy on the R9h source/test paths passed. Root `mypy .` is out-of-scope as a
signal in this workspace because it is blocked while traversing
`.venv/Sphinx` (`.venv/lib/python3.13/site-packages/sphinx/util/typing.py`).

R9h conclusion: diagnostic-only, non-gate, non-promoted, no integration. R3d
remains the accepted DIESEL baseline.

## Doctrine Reaffirmation

Palier 1 stays uncalibrated mechanistic. No real-stat matching, no PCA,
covariance, or noise capture, no learned residuals, no ML or DL, and no use of
labels, targets, splits, or downstream feedback to tune a profile. A documented
mechanistic blockage in this file is not by itself authorization to open
statistical calibration; any Palier 2 transition requires a separate explicit
gate. ML and DL remain closed unless that statistical avenue is later opened and
exhausted. R6a, R7a, R8a, and R8b are documented mechanistic outcomes on the
centered-shape, residual-transfer, R4a-base mean-preserving multiplicative
modulation, and R4c-base mean-preserving multiplicative modulation
directions; none of them opens the statistical or learned tier, and R8b in
particular is diagnostic-only despite removing the R7a clip blocker and
restoring the first-derivative magnitude that R8a destroyed. R9a is a
diagnostic-only mean-shift localization snapshot with no calibration, no
real-stat capture, no PCA, no ML/DL, no labels/targets/splits, and no
threshold or metric mutation; it does not by itself open any Palier 2 lane.
R9c is a documented mechanistic outcome on the positive-area support-shape
additive direction layered on the R3d base; it is contract-valid but
scientifically rejected on this cohort, remains diagnostic-only, and does
not by itself authorize any Palier 2 transition. R9d is now audited and
rejected on the mean-neutral, energy-normalized multiplicative support
redistribution direction layered on the R3d base; it is contract-valid but
scientifically rejected on this cohort, remains diagnostic-only, and does
not by itself authorize any Palier 2 transition. The mean-neutral /
energy-normalized support redistribution lever is therefore exhausted on
this cohort. R9e0 is a post-hoc R3d probe-only exp14 diagnostic audit; it
does not register a builder profile, promotes no probe, opens no gate, and
does not by itself authorize any Palier 2 transition. No Palier 2
authorization follows from R9c, R9d, or R9e0. Retuning R9d constants is not
authorized, and using R9e0 probes as direct profile templates is not
authorized. R9e is the audited clean attenuation implementation of that
direction, remains diagnostic-only / non-gate / non-promoted, and does not
authorize any integration or Palier 2 transition. R9f is the audited
pre-offset continuum/path-only attenuation implementation; it is
contract-valid and clean but nearly R3d-like on gap/global, does not beat
R9e or R4b/R4c, remains diagnostic-only / non-gate / non-promoted, and does
not authorize retuning, integration, or any Palier 2 transition. R9h is the
audited support CH-center/drop1720 isolation on the R3d base; it is
contract-valid but null versus R3d and does not explain R4b/R4c, remains
diagnostic-only / non-gate / non-promoted, and does not authorize retuning,
integration, or any Palier 2 transition. R9i is the audited CH width/gain
isolation on the R3d base; it is tiny non-null versus R3d/R9h, but insufficient
and does not explain the R4b/R4c gap advantage, remains diagnostic-only /
non-gate / non-promoted, and does not authorize retuning, integration, or any
Palier 2 transition. R9j is the audited residual-damping-only isolation on the
R3d base; it is the first strong isolated R4-family component versus R3d/R9i,
but still does not explain the full R4b/R4c gap advantage, remains
diagnostic-only / non-gate / non-promoted, and does not authorize retuning,
integration, a combined damping/continuum copy, or any Palier 2 transition.
R9k is the audited continuum-hump-only isolation on the R3d base; it is
neutral to slightly worse versus R3d, does not explain R4b/R4c, remains
diagnostic-only / non-gate / non-promoted, and does not authorize retuning,
integration, a combined successor, or any Palier 2 transition. PCA,
statistical / real-stat capture, noise capture, ML, DL, and calibration remain
closed; labels, targets, splits, and downstream feedback remain unread;
threshold and metric mutations remain forbidden; no `nirs4all/` integration
is authorized.

## Next Direction

R8b, R9a, R9b, R9c, R9d, R9e0, R9e, R9f, R9h, R9i, R9j, and R9k are audited
and remain diagnostic-only. None of them is a promotion over R3d. R9e0 does
not open Palier 2 and does not authorize selecting an exp14 probe as a builder
profile;
R9e is the audited attenuation profile, but it remains non-gate and
non-promoted. R9f is the audited pre-offset continuum/path attenuation profile,
but it remains non-gate and non-promoted. R9h is the audited support
CH-center/drop1720 isolation profile, but it is null versus R3d and remains
non-gate and non-promoted. R9i is the audited CH width/gain isolation profile,
but it is insufficient versus the R4b/R4c gap advantage and remains non-gate
and non-promoted. R9j is the audited residual-damping-only isolation profile;
it is a strong isolated component versus R3d/R9i, but still insufficient versus
the full R4b/R4c gap advantage and remains non-gate and non-promoted. R9k is
the audited continuum-hump-only isolation profile; it is neutral to slightly
worse than R3d and does not explain the R4b/R4c gap advantage, so it remains
non-gate and non-promoted. Together they close or constrain several sub-axes:

- R8b exhausts the layering of further support-mean-preserving multiplicative
  shape modulations on top of R4c: any such variant inherits the R4c support
  row mean by construction and cannot, by itself, close the residual
  `mean_shift`-dominated component. Retuning R8a or R8b constants after the
  R8b audit is not authorized.
- R9a structurally rules out an off-support route on the current DIESEL
  grids: `off_support_weight` is `0.0` median and max across the R9a 63
  compared rows and across the R9b 45 compared rows, because the aligned
  real grid stays inside the fixed 750-1550 nm support. There is no
  off-support lever to pull on this cohort.
- R9a also rules out an R5b transmittance route (scale-incompatible /
  readout outlier in the `uncalibrated_raw` comparison space) and an R5c
  blank-referenced intensity route (aggravates the median mean-shift versus
  R3d and R4c). Switching readout in this lane is not the next mechanism.
- R9b exhausts the simplest fixed support-level intercept form in the
  positive-additive sign on top of R4c: paired R9b vs R4c worsens both
  `global_mean_delta` / `support_mean_delta` by exactly `+0.001997` on every
  `(seed, dataset)` pair and worsens the morphology gap by approximately
  `+0.138` on every pair, with the dominant gap mode locked at `mean_shift`
  on `9/9` rows. A scalar offset added on the support, in this sign and this
  form, is not the lever.
- R9c extends that exhaustion to the positive-area support-shape additive
  form: a wavelength-dependent positive-area additive profile on the
  750-1550 nm support on top of the R3d base, built from pre-declared
  general liquid-hydrocarbon NIR constants, is contract-valid but moves R3d
  roughly an order of magnitude in the wrong direction on
  `global_mean_delta` (median `0.066477` versus R3d `0.005677`, with R4c
  `0.005122` and R9b `0.007119` as comparison profiles) and roughly fourfold
  worse on `morphology_gap_score` (median `7.030112` versus R3d `1.597554`,
  with R4c `1.509282` and R9b `1.646547` as comparison profiles), with the
  dominant gap mode still locked at `mean_shift`. Any positive-area additive
  profile on the support, scalar or shape, is therefore exhausted on this
  cohort.
- R9d closes the symmetric mean-neutral / energy-normalized direction on
  top of the R3d base. A wavelength-dependent multiplicative redistribution
  `exp(strength * shape)` on the 750-1550 nm support, with a fixed
  mean-neutral max-abs-normalized CH-overtone shape and a bounded per-row
  strength, followed by a per-row support-mean-preserving renormalization,
  cannot move `global_mean_delta` in this audit geometry (median `0.005679`,
  identical to R3d, paired delta `0.0` on every `(seed, dataset)` pair
  because `support_weight = 1.0`, off-support is byte-identical to R3d, and
  the support mean is preserved by construction) and remains worse than R4c
  `0.005106`, R4b `0.004630`, and R8b `0.005106`; worsens the median
  `morphology_gap_score` (median `1.602604` versus R3d `1.597481`,
  R4c `1.507567`, R4b `1.451708`, and R8b `1.519762`); does not gain on
  `mean_curve_corr` (median `0.032231` versus R3d `0.034466` and R4c
  `0.036304`); and keeps the dominant gap mode locked at `mean_shift` on
  `9/9` paired rows. The mean-neutral / energy-normalized support
  redistribution form, scalar or shape, layered on R3d, is therefore
  exhausted on this cohort. Retuning R9d constants is not authorized.
- R9e0 constrains the signed support-actuator direction. Negative support
  intercepts and the negative-intercept-plus-R9d-shape combo reduce the
  median `global_mean_delta` and `morphology_gap_score` versus R3d, but they
  do so with high guard clipping (`0.712711` to `0.732650` medians), so they
  are scientifically risky and cannot be promoted or copied into a direct
  profile. Pure multiplicative attenuation also reduces the two medians
  versus R3d and keeps guard clipping at `0.000000`; that makes attenuation
  the cleanest R9e0 signal, but still only as descriptive evidence for a
  future mechanism.
- R9e implements that clean attenuation direction as an actual bench-only
  profile on the R3d base: support-only positive multiplicative factor
  `[0.970, 0.985]`, off-support byte-identical to R3d, no offset, no guard
  clip, no renormalization, and no R9d shape. It improves R3d/R9d on the
  median scalar level and gap (`0.005471` / `1.571865` for R9e versus
  `0.005676` / `1.596743` for R3d and `0.005676` / `1.602339` for R9d),
  but it remains behind R4b/R4c on the gap/global tradeoff and keeps the
  dominant gap mode at `mean_shift`. R9e is therefore evidence, not a
  baseline replacement.
- R9f moves the same `[0.970, 0.985]` pathlength / reference attenuation
  before additive baseline and output clip, and limits it to the
  continuum/path component on the 750-1550 nm support. Feature residuals,
  additive baseline / offsets, readout transform, R9d shape, support-mean
  renormalization, negative intercept, and extra clipping stay unchanged or
  absent. It is contract-valid and clean, but its medians are nearly R3d-like
  (`global_mean_delta = support_mean_delta = 0.00565240766844`,
  `morphology_gap_score = 1.59542861178`, derivative
  `-0.0323313599657`, `mean_curve_corr = 0.0342291346304`, guard clip `0`)
  and it does not beat R9e or R4b/R4c. Pre-offset continuum/path-only
  attenuation is therefore evidence against that stage as the main lever, not
  a baseline replacement.
- R9h isolates only the R4-family support CH centers / drop 1720 decision on
  top of the complete R3d pipeline:
  `ch_overtone_centers_nm=(1150, 1210, 1390, 1460)`. It carries no 975 nm
  hump, no damping, no R4 width/gain, no attenuation, no support intercept,
  no support shape, no redistribution, no readout transform, and no extra
  clip. Its medians are numerically indistinguishable from R3d
  (`global_mean_delta = support_mean_delta = 0.005676745221`,
  `morphology_gap_score = 1.59918713004`, derivative `-0.032328094721`,
  `mean_curve_corr = 0.035081087950`), with paired R9h - R3d median deltas
  near zero (`-1.49e-11`, `-6.70e-10`, `-1.76e-10`, `-7.30e-11` on
  global/support, morphology gap, derivative, and correlation respectively).
  Support CH centers/drop1720 alone is therefore exhausted as an explanation
  for the R4b/R4c improvement.
- R9i isolates only the R4c CH width/gain decision on top of the complete R3d
  pipeline: `ch_overtone_width_nm = 36.0` and
  `ch_overtone_gain_range = (0.092, 0.155)`. It keeps the R3d CH centers,
  including `1720`, and carries no damping, no 975 nm continuum hump, no
  attenuation, no support intercept, no support shape, no redistribution, no
  readout transform, and no extra clip. Its medians are only tiny non-null
  versus R3d (`global_mean_delta = support_mean_delta = 0.005673977762`,
  `morphology_gap_score = 1.597661319467`, derivative `-0.031691908358`,
  `mean_curve_corr = 0.034749730996`; paired R9i - R3d medians
  `-0.000019827477`, `-0.000019827477`, `-0.002510451761`,
  `-0.000352744725`, `+0.000001640062`). R9i remains worse than R4b and R4c
  on morphology gap (paired gap deltas `+0.125233019344` and
  `+0.074106130408`) and worse than R9e on the same field (paired gap delta
  `+0.021494864972`). CH width/gain alone is therefore insufficient as an
  explanation for the R4b/R4c improvement.
- R9j isolates only the R4c residual damping decision on top of the complete
  R3d pipeline: `damping_windows_nm = ((1180.0, 46.0, 0.60), (1425.0, 54.0, 0.70))`
  and `damping_strength_range = (0.05, 0.15)`. It keeps the R3d CH centers,
  including `1720.0`, and keeps R3d width/gain. It carries no 975 nm continuum
  hump, no attenuation, no support intercept, no support shape, no
  redistribution, no readout transform, and no extra clip. Its medians are a
  meaningful non-null improvement versus R3d/R9i (`global_mean_delta =
  support_mean_delta = 0.005203`, `morphology_gap_score = 1.543487`,
  derivative `-0.052600`, `mean_curve_corr = 0.036826`; paired R9j - R3d
  medians `-0.000445`, `-0.000445`, `-0.053797`, `-0.023068`, `+0.002198`).
  R9j remains worse than R4b and R4c on morphology gap (paired gap deltas
  `+0.073230` and `+0.020151`) while improving over R9e on the same field
  (paired gap delta `-0.029373`). Residual damping-only is therefore the first
  strong isolated component and explains a meaningful part, but not all, of
  the R4-family advantage.
- R9k isolates only the R4c 975 nm continuum hump on top of the complete R3d
  pipeline: center `975.0`, width `72.0`, amplitude range
  `(0.00010, 0.00032)`, and support `(750.0, 1550.0)`. It keeps the R3d CH
  centers, including `1720.0`, and keeps R3d width/gain. It carries no
  damping, no attenuation, no support intercept, no support shape, no
  redistribution, no readout transform, and no extra clip. Its medians are
  effectively R3d-like and slightly worse on the aggregate gap
  (`global_mean_delta = support_mean_delta = 0.005670`,
  `morphology_gap_score = 1.597067`, derivative `-0.031546`,
  `mean_curve_corr = 0.035300`; paired R9k - R3d medians `+0.000001`,
  `+0.000001`, `+0.000099`, `+0.000003`, `-0.000005`). R9k remains worse
  than R9j, R4b, and R4c on morphology gap (paired gap deltas `+0.053816`,
  `+0.127906`, and `+0.076716` respectively). Continuum-hump-only is
  therefore exhausted as an explanation for the R4b/R4c improvement.

R8b + R9b + R9c + R9d + R9e + R9f + R9h + R9i + R9j + R9k together close or bound the
support-only mean-shift, support-only support-shape, clean attenuation, and
support-center/drop1720 levers currently available at Palier 1 on this cohort:
support-mean-preserving multiplicative modulation (R8a/R8b), positive-area
scalar intercept (R9b), positive-area wavelength-dependent additive shape
(R9c), mean-neutral / energy-normalized multiplicative redistribution (R9d),
clean support-only multiplicative attenuation after output clip (R9e) or
before offset / output clip on only the continuum/path component (R9f), and
isolated support CH centers/drop1720 (R9h), CH width/gain (R9i), residual
damping (R9j), and continuum hump (R9k) on the R3d base all fail to establish
a new accepted baseline over R3d. R9e0 added the descriptive
constraint that lowering the level can help the two medians if clipping is
avoided; R9e confirms that as a clean component, R9f shows that moving that
attenuation earlier onto the continuum/path component alone is not the main
lever, R9h shows that support centers/drop1720 alone is a null mechanism, and
R9i shows that CH width/gain alone is too small to explain R4b/R4c. R9j shows
that residual damping-only is a meaningful isolated component but still not
the full R4b/R4c explanation. R9k shows that continuum-hump-only is neutral to
slightly worse than R3d and does not explain the R4b/R4c gap advantage.
The next mechanistic step must therefore remain uncalibrated and mechanistic,
must not retune the attenuation amplitude or support centers, and must not
pass Palier 2. The individual isolations are complete enough to justify a
strictly controlled combination hypothesis, likely damping + width/gain or
damping + R9e, but only after Lead approval. No integration is authorized, and
R4b/R4c are not accepted as the baseline.

Acceptable direction at Palier 1 (no constants decided here, no constants to
be cited as facts):

- propose mechanistic hypotheses for attenuation / pathlength / continuum
  scale behaviour that go beyond support-only shape redistribution and
  beyond support-only additive lifts, e.g. revisit micro-path / CH-overtone /
  readout assumptions of the R3d / R4 family at the row level (path-length
  distribution, CH-overtone gain / damping balance, micro-path continuum
  behaviour) rather than adding a fixed support-shape correction on top of an
  unchanged base, while keeping the dominant morphology gap mode out of
  `derivative_under` and avoiding non-negative guard clipping;
- after Lead approval only, define one strictly controlled combination
  hypothesis from completed isolations, likely damping + width/gain or damping
  + R9e. R9h shows the support CH-center/drop1720 decision alone is null versus
  R3d, R9i shows the CH width/gain decision alone is only tiny non-null, R9j
  shows damping-only is meaningful but incomplete, and R9k shows
  continuum-hump-only is neutral to slightly worse versus R3d;
- any successor profile must inherit R3d byte-for-byte for every non-DIESEL
  row and provide a byte-identical R3d fallback for non-compliant DIESEL
  rows;
- audit geometry note: the aligned real grid is fully inside the
  `750-1550 nm` support on this cohort (`support_weight = 1.0`,
  `off_support_weight = 0.0`), so any pure support-only mean-preserving
  mechanism is structurally null on `global_mean_delta` here. Successors
  must not rely on closing the scalar mean shift through a support-only
  mean-preserving redistribution.
- no successor should simply retune the R9e/R9f attenuation amplitude. R9f
  specifically tested the pre-offset continuum/path-only stage and found that
  this stage is not the principal lever on the current cohort.

Forbidden in any such successor at Palier 1, unchanged from above:

- real-stat matching, PCA, covariance, noise capture, quantile or marginal
  calibration, learned residuals, ML, or DL;
- using labels, targets, splits, downstream performance, adversarial scores,
  or threshold or metric mutations to tune the profile;
- any `nirs4all/` integration;
- retuning R9d constants or proposing another mean-neutral /
  energy-normalized support redistribution variant on this cohort;
- promoting an R9e0 probe directly, using high-clip negative intercept probes
  as direct profile templates, recasting R9e0 attenuation as calibration, or
  promoting R9e from diagnostic evidence to accepted baseline;
- retuning the R9e/R9f attenuation amplitude, promoting R9f, or using R9f as
  a gate or integration trigger.
- retuning the R9h support CH centers/drop1720 mechanism, promoting R9h, or
  using R9h as a gate or integration trigger.
- retuning the R9i CH width/gain mechanism, promoting R9i, using R9i as a
  gate or integration trigger, or copying R4 damping and continuum-hump changes
  together as a combined successor.
- retuning the R9j residual damping mechanism, promoting R9j, using R9j as a
  gate or integration trigger, or copying R9j damping with the continuum hump
  as a direct combined successor before individual isolations are complete.
- retuning the R9k continuum-hump mechanism, promoting R9k, using R9k as a
  gate or integration trigger, or using R9k as authorization for integration
  or a combined successor without Lead approval.

## R9i CH Width/Gain Isolation Audit Outcome

`r9i_diesel_ch_width_gain_isolation_v1` is the Palier 1 strict isolation of
the R4c CH width/gain component after R9h showed that support CH centers /
drop1720 alone are null. R9i is bench-only, diagnostic-only, non-gate, and
non-promoted. It is not a calibration, not a real-stat / PCA / covariance /
noise capture, not ML/DL, uses no labels, targets, splits, or downstream
feedback, mutates no thresholds or metrics, and authorizes no `nirs4all/`
integration. R3d remains the accepted DIESEL baseline.

Mechanism contract: R9i inherits the full R3d pipeline and fallback. On
compliant explicit DIESEL `petrochem_fuels` rows carrying
`_r9i_diesel_ch_width_gain_route`, it changes only:

- `ch_overtone_width_nm = 36.0`;
- `ch_overtone_gain_range = (0.092, 0.155)`.

R9i does not change the R3d CH centers, so the 1720 nm band remains present.
It adds no damping window, no 975 nm continuum hump, no R9e/R9f attenuation,
no support intercept / shape / redistribution, no readout transform, and no
extra clip. Its seed source is R3d so the R3d draw order is preserved and only
the width/gain transformation changes. Non-DIESEL rows and non-compliant
DIESEL rows fall back byte-identical to R3d.

Exp19 used seeds `20260501`, `20260502`, `20260503`, `n_synthetic_samples=64`,
`max_real_samples=64`, sentinel token `DIESEL`, comparison space
`uncalibrated_raw`, and profiles `r3d_diesel_matrix_v1`,
`r4a_diesel_basis_v1`, `r4b_diesel_derivative_restore_v1`,
`r4c_diesel_balanced_derivative_v1`,
`r9e_diesel_pathlength_reference_attenuation_v1`,
`r9f_diesel_pre_offset_pathlength_reference_attenuation_v1`,
`r9h_diesel_support_ch_center_drop1720_isolation_v1`, and
`r9i_diesel_ch_width_gain_isolation_v1`. It returned `72` compared rows,
`0` blocked rows, and `8` profiles x `9` rows. The generated audit outputs
were written to `/tmp/r9i_diesel_ch_width_gain_isolation_audit.md` and
`/tmp/r9i_diesel_ch_width_gain_isolation_audit.csv`.

Median results on the compared cohort:

- R3d: `global_mean_delta = support_mean_delta = 0.005693805239`,
  `morphology_gap_score = 1.600106837591`, derivative `-0.031396545440`,
  `mean_curve_corr = 0.034748090934`;
- R9h: identical to R3d at the displayed precision;
- R9i: `global_mean_delta = support_mean_delta = 0.005673977762`,
  `morphology_gap_score = 1.597661319467`, derivative `-0.031691908358`,
  `mean_curve_corr = 0.034749730996`;
- R4b/R4c remain better on morphology gap
  (`1.455114020584` / `1.509393136690`);
- R9e remains lower on level/gap than R9i
  (`0.005485855193` / `1.574988291060`);
- R9f is nearly tied with R9i (`0.005662711423` / `1.597367827333`).

Paired median R9i deltas:

| reference | global | support | morphology gap | derivative | mean_curve_corr |
|---|---:|---:|---:|---:|---:|
| R3d | `-0.000019827477` | `-0.000019827477` | `-0.002510451761` | `-0.000352744725` | `+0.000001640062` |
| R9h | `-0.000019827462` | `-0.000019827462` | `-0.002510451119` | `-0.000352744063` | `+0.000001640204` |
| R4b | `+0.000978956901` | `+0.000978956901` | `+0.125233019344` | `+0.057158925703` | `-0.005764369682` |
| R4c | `+0.000501594617` | `+0.000501594617` | `+0.074106130408` | `+0.025546815405` | `-0.001831037436` |
| R9e | `+0.000181591199` | `+0.000181591199` | `+0.021494864972` | `+0.009666209932` | `+0.000005556684` |
| R9f | `+0.000011080103` | `+0.000011080103` | `+0.000293492133` | `+0.000202625590` | `+0.000501787272` |

Conclusion: width/gain-only isolation produces a tiny non-null improvement
over R3d/R9h, but it does not recover the R4b/R4c morphology gap advantage and
does not beat the clean R9e attenuation result. R9i is therefore reported as
diagnostic-only evidence and is not promoted. R9j subsequently isolated the
damping-only component, so the remaining Palier 1 single-component candidate is
continuum-hump-only isolation, not a combined damping/continuum copy.

Verification status from the R9i tester report: targeted R9i builder tests
passed (`9` passed), exp19 tests passed (`3` passed), combined exp18 + exp19
tests passed (`6` passed), the full bench test suite passed (`1277` passed,
`4` sklearn warnings), `ruff check .` passed, and scoped mypy on the R9i
source/test paths passed. Root `mypy .` is out-of-scope as a pass/fail signal
for this audit because it is blocked before project analysis by
`.venv/Sphinx`.

Any follow-up audit will be reported diagnostic-only by default and is not a
promotion over R3d.

## R9j Residual Damping-Only Isolation Audit Outcome

`r9j_diesel_residual_damping_isolation_v1` is the Palier 1 strict isolation
of the R4c residual damping component. R9j is bench-only, diagnostic-only,
non-gate, and non-promoted. It is not a calibration, not a real-stat / PCA /
covariance / noise capture, not ML/DL, uses no labels, targets, splits, or
downstream feedback, mutates no thresholds or metrics, and authorizes no
`nirs4all/` integration. R3d remains the accepted DIESEL baseline.

Mechanism contract: R9j inherits the full R3d pipeline and fallback. On
compliant explicit DIESEL `petrochem_fuels` rows carrying
`_r9j_diesel_residual_damping_route`, it changes only:

- `damping_windows_nm = ((1180.0, 46.0, 0.60), (1425.0, 54.0, 0.70))`;
- `damping_strength_range = (0.05, 0.15)`.

R9j keeps the R3d CH centers, including `1720.0`, and keeps the R3d width
`34.0` and gain range `(0.11, 0.18)`. It adds no 975 nm continuum hump, no
R9e/R9f attenuation, no support intercept / shape / redistribution, no readout
transform, and no extra clip. Non-DIESEL rows and non-compliant DIESEL rows
fall back byte-identical to R3d.

Exp20 used seeds `20260501`, `20260502`, `20260503`,
`n_synthetic_samples=64`, `max_real_samples=64`, sentinel token `DIESEL`,
comparison space `uncalibrated_raw`, and profiles `r3d_diesel_matrix_v1`,
`r4a_diesel_basis_v1`, `r4b_diesel_derivative_restore_v1`,
`r4c_diesel_balanced_derivative_v1`,
`r9e_diesel_pathlength_reference_attenuation_v1`,
`r9f_diesel_pre_offset_pathlength_reference_attenuation_v1`,
`r9h_diesel_support_ch_center_drop1720_isolation_v1`,
`r9i_diesel_ch_width_gain_isolation_v1`, and
`r9j_diesel_residual_damping_isolation_v1`. It returned `81` compared rows,
`0` blocked rows, and `9` profiles x `9` rows. Exp20 defaults point at
`bench/nirs_synthetic_pfn/reports/`; the tester run explicitly passed
`--report /tmp/r9j_diesel_residual_damping_isolation_audit.md` and
`--csv /tmp/r9j_diesel_residual_damping_isolation_audit.csv`, so the generated
outputs were kept under `/tmp` and no additional repository report files are
required for this R9j file budget.

Median results on the compared cohort:

- R3d: `global_mean_delta = support_mean_delta = 0.005675846187`,
  `morphology_gap_score = 1.597283669841`, derivative `-0.031201602053`,
  `mean_curve_corr = 0.034724390004`;
- R9i: `global_mean_delta = support_mean_delta = 0.005656018710`,
  `morphology_gap_score = 1.594589158894`, derivative `-0.031496964971`,
  `mean_curve_corr = 0.034721717142`;
- R9j: `global_mean_delta = support_mean_delta = 0.005202978895`,
  `morphology_gap_score = 1.543486847073`, derivative `-0.052600025650`,
  `mean_curve_corr = 0.036825594934`;
- R4b/R4c remain better on morphology gap
  (`1.456261781427` / `1.509377175897`);
- R9j has a lower median morphology gap than the clean R9e/R9f attenuation
  profiles (`1.570698646934` / `1.593994333776`) on this diagnostic cohort,
  while still not matching R4b/R4c.

Paired median R9j deltas:

| reference | global | support | morphology gap | derivative | mean_curve_corr |
|---|---:|---:|---:|---:|---:|
| R3d | `-0.000445117720` | `-0.000445117720` | `-0.053796822768` | `-0.023068152599` | `+0.002197700425` |
| R9i | `-0.000423880920` | `-0.000423880920` | `-0.051102311822` | `-0.022652494100` | `+0.002196874952` |
| R4b | `+0.000547675680` | `+0.000547675680` | `+0.073229755553` | `+0.033771547067` | `-0.003524263996` |
| R4c | `+0.000077713697` | `+0.000077713697` | `+0.020151323027` | `+0.005097237121` | `+0.000310822010` |
| R9e | `-0.000237548641` | `-0.000237548641` | `-0.029373339645` | `-0.014126236225` | `+0.002216939084` |
| R9f | `-0.000412420481` | `-0.000412420481` | `-0.050649592423` | `-0.022331098127` | `+0.002691477483` |

Conclusion: damping-only isolation is a materially non-null Palier 1 signal
versus R3d/R9i/R9e/R9f and is the first strong isolated component in the R4
family, but it still does not recover the R4b/R4c morphology gap advantage.
It explains a meaningful part of the R4 advantage, not all of it. R9j is
diagnostic evidence only and is not promoted. The result does not authorize a
combined damping + continuum copy, any retuning, or any `nirs4all/`
integration.

Verification status from the R9j tester report: targeted R9j builder tests
passed (`9` passed), exp20 tests passed (`3` passed), combined exp19 + exp20
tests passed (`6` passed), the full bench test suite passed (`1289` passed,
`4` sklearn warnings), `ruff check .` passed, and scoped mypy on the R9j
source/test paths passed. Root `mypy .` is out-of-scope as a pass/fail signal
for this audit because it is blocked before project analysis by
`.venv/Sphinx`.

The next Palier 1 direction after R9j was continuum-hump-only isolation, now
reported as R9k below. A direct combined damping/continuum copy remains
unauthorized.

## R9k Continuum-Hump-Only Isolation Audit Outcome

`r9k_diesel_continuum_hump_isolation_v1` is the Palier 1 strict isolation
of the R4c 975 nm continuum hump component after R9j. R9k is bench-only,
diagnostic-only, non-gate, and non-promoted. It is not a calibration, not a
real-stat / PCA / covariance / noise capture, not ML/DL, uses no labels,
targets, splits, or downstream feedback, mutates no thresholds or metrics,
and authorizes no `nirs4all/` integration. R3d remains the accepted DIESEL
baseline.

Mechanism contract: R9k inherits the full R3d pipeline and fallback. On
compliant explicit DIESEL `petrochem_fuels` rows carrying
`_r9k_diesel_continuum_hump_route`, it changes only:

- `continuum_hump_center_nm = 975.0`;
- `continuum_hump_width_nm = 72.0`;
- `continuum_hump_amplitude_range = (0.00010, 0.00032)`;
- `continuum_hump_support_nm = (750.0, 1550.0)`.

R9k keeps the R3d CH centers, including `1720.0`, and keeps the R3d width
`34.0` and gain range `(0.11, 0.18)`. It adds no damping windows, no damping
strength, no R9e/R9f attenuation, no support intercept / shape /
redistribution, no readout transform, and no extra clip. Non-DIESEL rows and
non-compliant DIESEL rows fall back byte-identical to R3d. R9k is explicitly
not a damping + continuum combination.

Exp21 used seeds `20260501`, `20260502`, `20260503`,
`n_synthetic_samples=64`, `max_real_samples=64`, sentinel token `DIESEL`,
comparison space `uncalibrated_raw`, and profiles `r3d_diesel_matrix_v1`,
`r4a_diesel_basis_v1`, `r4b_diesel_derivative_restore_v1`,
`r4c_diesel_balanced_derivative_v1`,
`r9e_diesel_pathlength_reference_attenuation_v1`,
`r9f_diesel_pre_offset_pathlength_reference_attenuation_v1`,
`r9h_diesel_support_ch_center_drop1720_isolation_v1`,
`r9i_diesel_ch_width_gain_isolation_v1`,
`r9j_diesel_residual_damping_isolation_v1`, and
`r9k_diesel_continuum_hump_isolation_v1`. It returned `90` compared rows,
`0` blocked rows, and `10` profiles x `9` rows. The tester run explicitly
passed `--report /tmp/r9k_diesel_continuum_hump_isolation_audit.md` and
`--csv /tmp/r9k_diesel_continuum_hump_isolation_audit.csv`, so the generated
outputs were kept under `/tmp`.

Median results on the compared cohort:

- R3d: `global_mean_delta = support_mean_delta = 0.005669445953`,
  `morphology_gap_score = 1.596964517101`, derivative `-0.031535917770`,
  `mean_curve_corr = 0.035305875951`;
- R9i: `global_mean_delta = support_mean_delta = 0.005649618476`,
  `morphology_gap_score = 1.594442836534`, derivative `-0.031831280688`,
  `mean_curve_corr = 0.035306287380`;
- R9j: `global_mean_delta = support_mean_delta = 0.005202149524`,
  `morphology_gap_score = 1.543251604085`, derivative `-0.052934341368`,
  `mean_curve_corr = 0.037519400968`;
- R9k: `global_mean_delta = support_mean_delta = 0.005670157208`,
  `morphology_gap_score = 1.597067200256`, derivative `-0.031545748019`,
  `mean_curve_corr = 0.035300075246`;
- R4b/R4c remain better on morphology gap
  (`1.455932288513` / `1.510276068009`);
- R9k is also worse than the clean R9e/R9f attenuation profiles on median
  morphology gap (`1.570414812383` / `1.593694357469`).

Paired median R9k deltas:

| reference | global | support | morphology gap | derivative | mean_curve_corr |
|---|---:|---:|---:|---:|---:|
| R3d | `+0.000000621813` | `+0.000000621813` | `+0.000099479597` | `+0.000003257912` | `-0.000005358862` |
| R9i | `+0.000020538732` | `+0.000020538732` | `+0.002621999326` | `+0.000357455352` | `-0.000004371295` |
| R9j | `+0.000445721647` | `+0.000445721647` | `+0.053815596171` | `+0.023065593973` | `-0.002208955044` |
| R4b | `+0.000997338770` | `+0.000997338770` | `+0.127906426812` | `+0.057586993124` | `-0.005811998114` |
| R4c | `+0.000523435344` | `+0.000523435344` | `+0.076716328710` | `+0.026133505481` | `-0.001845454534` |
| R9e | `+0.000204186096` | `+0.000204186096` | `+0.024131651857` | `+0.009915418122` | `+0.000010739178` |
| R9f | `+0.000030932550` | `+0.000030932550` | `+0.002840847489` | `+0.000510957593` | `+0.000508275669` |

Conclusion: continuum-hump-only isolation is effectively null versus R3d and
does not explain the R4b/R4c morphology gap advantage. R9j damping-only
remains the meaningful isolated Palier 1 component, but it is still incomplete
and non-promoted. R9k is diagnostic evidence only and is not promoted. The
result does not authorize a combined damping + continuum copy, any retuning,
or any `nirs4all/` integration.

Verification status from the R9k tester report: targeted R9k builder tests
passed (`9` passed), exp21 tests passed (`3` passed), combined exp20 + exp21
tests passed (`6` passed), the full bench test suite passed (`1301` passed,
`4` sklearn warnings), `ruff check .` passed, and scoped mypy on the R9k
source/test paths passed. Root `mypy .` is out-of-scope as a pass/fail signal
for this audit because it is blocked before project analysis by
`.venv/Sphinx`.

The next Palier 1 direction is not another single-component isolation. The
individual isolations are complete enough to justify a strictly controlled
combination hypothesis, likely damping + width/gain or damping + R9e, but only
after Lead approval. No integration is authorized.

## R9l Residual-Damping + Clean-Attenuation Controlled Combination Outcome

`r9l_diesel_residual_damping_clean_attenuation_v1` is the approved Palier 1
controlled-combination diagnostic after R9k. It combines exactly the R9j
residual damping constants with the R9e clean support-only attenuation stage
on the complete R3d base. It is bench-only, diagnostic-only, non-gate, and
non-promoted. R3d remains the accepted DIESEL baseline.

Mechanism contract: R9l inherits the full R3d pipeline and byte-identical R3d
fallback. It activates only for compliant explicit DIESEL `petrochem_fuels`
rows carrying `_r9l_diesel_residual_damping_clean_attenuation_route`. It keeps
the R3d CH centers including `1720.0`, R3d width `34.0`, and R3d gain
`(0.11, 0.18)`. It adds:

- R9j exact damping: `damping_windows_nm = ((1180.0, 46.0, 0.60), (1425.0, 54.0, 0.70))`
  and `damping_strength_range = (0.05, 0.15)`;
- R9e exact clean attenuation: support-only factor range `(0.970, 0.985)`
  on `(750.0, 1550.0)`, after the R3d output clip.

It adds no continuum hump, no R9f pre-offset attenuation, no support intercept,
no support shape, no redistribution, no readout transform, and no extra guard
clip. The seed source profile remains `r3d_diesel_matrix_v1`; R9l uses the
existing R9j damping stream and the R9e attenuation stage so only the declared
transforms change. Metadata explicitly marks no calibration, no real stats,
no PCA, no covariance/noise capture, no ML/DL, no labels, no targets, no
splits, no threshold/metric mutation, and no integration.

Exp22 used seeds `20260501`, `20260502`, `20260503`,
`n_synthetic_samples=64`, `max_real_samples=64`, sentinel token `DIESEL`,
comparison space `uncalibrated_raw`, and profiles `r3d_diesel_matrix_v1`,
`r4a_diesel_basis_v1`, `r4b_diesel_derivative_restore_v1`,
`r4c_diesel_balanced_derivative_v1`,
`r9e_diesel_pathlength_reference_attenuation_v1`,
`r9f_diesel_pre_offset_pathlength_reference_attenuation_v1`,
`r9h_diesel_support_ch_center_drop1720_isolation_v1`,
`r9i_diesel_ch_width_gain_isolation_v1`,
`r9j_diesel_residual_damping_isolation_v1`,
`r9k_diesel_continuum_hump_isolation_v1`, and
`r9l_diesel_residual_damping_clean_attenuation_v1`. It returned `99` compared
rows, `0` blocked rows, and wrote `/tmp/r9l_diesel_residual_damping_clean_attenuation_audit.md`
plus `/tmp/r9l_diesel_residual_damping_clean_attenuation_audit.csv`.

Median results on the compared cohort:

- R3d: morphology gap `1.599563`;
- R9e: morphology gap `1.572969`;
- R9j: morphology gap `1.544989`;
- R9l: `global_mean_delta = support_mean_delta = 0.005014`,
  `morphology_gap_score = 1.521286`, derivative `-0.061540`,
  `mean_curve_corr = 0.037006`;
- R4b/R4c remain better on aggregate median morphology gap
  (`1.455520` / `1.509091`), while the paired median R9l-minus-R4c gap delta
  is slightly negative and therefore must not be conflated with the aggregate
  ordering.

Paired median R9l deltas:

| reference | global | support | morphology gap | derivative | mean_curve_corr |
|---|---:|---:|---:|---:|---:|
| R3d | `-0.000633` | `-0.000633` | `-0.077056` | `-0.032626` | `+0.002192` |
| R9j | `-0.000194` | `-0.000194` | `-0.022530` | `-0.009619` | `-0.000005` |
| R9e | `-0.000435` | `-0.000435` | `-0.053019` | `-0.022985` | `+0.002211` |
| R4b | `+0.000352` | `+0.000352` | `+0.050686` | `+0.025134` | `-0.003523` |
| R4c | `-0.000120` | `-0.000120` | `-0.004964` | `-0.003920` | `+0.000293` |

Conclusion: R9l has a lower aggregate median morphology gap than R3d, R9e, and
R9j on this cohort, so damping + clean attenuation is useful controlled Palier 1
evidence. It is not a promotion signal. It remains behind R4b/R4c on aggregate
median morphology gap, while showing only a small paired median advantage versus
R4c because row-level distributions cross. R9l is diagnostic evidence only: not
promoted, no gate, no retuning, and no `nirs4all/` integration.

## R9m Width/Gain + Residual-Damping + Clean-Attenuation Final Controlled Combination Outcome

`r9m_diesel_width_gain_damping_clean_attenuation_v1` is the final Palier 1
controlled-combination diagnostic after R9l. It combines exactly the R9i
CH width/gain constants, the R9j residual damping constants, and the R9e clean
support-only attenuation stage on the complete R3d base. It is bench-only,
mechanistic, `uncalibrated_raw`, diagnostic-only, non-gate, and non-promoted.
R3d remains the accepted DIESEL baseline.

Mechanism contract: R9m inherits the full R3d pipeline and byte-identical R3d
fallback. It activates only for compliant explicit DIESEL `petrochem_fuels`
rows carrying `_r9m_diesel_width_gain_damping_clean_attenuation_route`. It
keeps the R3d CH centers including `1720.0`. It changes only:

- R9i exact width/gain: `ch_overtone_width_nm = 36.0` and
  `ch_overtone_gain_range = (0.092, 0.155)`;
- R9j exact damping: `damping_windows_nm = ((1180.0, 46.0, 0.60), (1425.0, 54.0, 0.70))`
  and `damping_strength_range = (0.05, 0.15)`;
- R9e exact clean attenuation: support-only factor range `(0.970, 0.985)`
  on `(750.0, 1550.0)`, after the R3d output clip.

It adds no continuum hump, no support intercept, no support shape, no
redistribution, no readout transform, no R9f pre-offset attenuation, no extra
guard clip, no center/drop-1720 change, no calibration, no real-stat capture,
no PCA/covariance/noise capture, no ML/DL, no labels/targets/splits, no
threshold/metric mutation, and no `nirs4all/` integration.

Exp23 used seeds `20260501`, `20260502`, `20260503`,
`n_synthetic_samples=64`, `max_real_samples=64`,
`max_sentinel_datasets=8`, sentinel token `DIESEL`, comparison space
`uncalibrated_raw`, and profiles `r3d_diesel_matrix_v1`,
`r4b_diesel_derivative_restore_v1`,
`r4c_diesel_balanced_derivative_v1`,
`r9e_diesel_pathlength_reference_attenuation_v1`,
`r9i_diesel_ch_width_gain_isolation_v1`,
`r9j_diesel_residual_damping_isolation_v1`,
`r9l_diesel_residual_damping_clean_attenuation_v1`, and
`r9m_diesel_width_gain_damping_clean_attenuation_v1`. It returned `72`
compared rows, `0` blocked rows, and wrote
`/tmp/r9m_diesel_width_gain_damping_clean_attenuation_audit.md` plus
`/tmp/r9m_diesel_width_gain_damping_clean_attenuation_audit.csv`.

Median results on the compared cohort:

- R3d: morphology gap `1.597786`, derivative `-0.031737`;
- R4b: morphology gap `1.455638`, derivative `-0.093962`;
- R4c: morphology gap `1.511325`, derivative `-0.058086`;
- R9e: morphology gap `1.571210`, derivative `-0.039846`;
- R9i: morphology gap `1.595093`, derivative `-0.032033`;
- R9j: morphology gap `1.544025`, derivative `-0.053136`;
- R9l: morphology gap `1.519558`, derivative `-0.062831`;
- R9m: `global_mean_delta = support_mean_delta = 0.004983`,
  `morphology_gap_score = 1.516589`, derivative `-0.063143`,
  `mean_curve_corr = 0.036845`.

Paired median R9m deltas:

| reference | global | support | morphology gap | derivative | mean_curve_corr |
|---|---:|---:|---:|---:|---:|
| R3d | `-0.000654` | `-0.000654` | `-0.079356` | `-0.032996` | `+0.002190` |
| R9i | `-0.000632` | `-0.000632` | `-0.077022` | `-0.032573` | `+0.002189` |
| R9j | `-0.000214` | `-0.000214` | `-0.024843` | `-0.010008` | `-0.000001` |
| R9l | `-0.000019` | `-0.000019` | `-0.002538` | `-0.000370` | `+0.000004` |
| R9e | `-0.000455` | `-0.000455` | `-0.055289` | `-0.023298` | `+0.002209` |
| R4b | `+0.000332` | `+0.000332` | `+0.048356` | `+0.024764` | `-0.003504` |
| R4c | `-0.000140` | `-0.000140` | `-0.007494` | `-0.004289` | `+0.000294` |

Aggregate-vs-paired nuance: R9m is slightly better than R9l on aggregate
median morphology gap (`1.516589` vs `1.519558`) and also slightly better in
paired median morphology gap (`-0.002538`). However, R9m has a small
derivative-under regression versus R9l (`-0.063143` vs `-0.062831`; paired
median derivative delta `-0.000370`). Under the R9m decision rule this is a
NO-GO for consolidation: it does not clearly improve aggregate median over R9l
without derivative-under regression.

Conclusion: R9m is diagnostic-only evidence that adding the tiny R9i
width/gain lever to R9l is non-null but not clean enough to consolidate. R9m
remains a final Palier 1 bench-only variant only: not promoted, no gate, no
retuning, no integration, and R3d remains the accepted DIESEL baseline.
