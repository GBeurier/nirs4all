# R2a Sentinel Mechanistic Profile Ablation

Date: 2026-04-29
Scope: bench-only diagnostic lane. No production package changes. No
production integration. No model training authorization.

## Disclaimer

R2a is a **non-gate, report-only** lane that runs alongside the existing
B2/B3/B4/B5 audits without modifying them. Nothing in this lane changes any
gate decision, threshold, or metric definition. Frozen audits remain
authoritative:

- B2 `reports/real_synthetic_scorecards.md` — `uncalibrated_raw` 71/71 smoke
  AUC failures, 71/71 stretch failures, 6 blocked rows, AUC median 1.0.
- B3 `reports/adversarial_auc.md` — `NO-GO`.
- B4 `BLOCKED_BY_REALISM_GATE`.
- B5 `BLOCKED_REPORT_ONLY`.

R2a does **not** unblock any of these. Any future re-evaluation of the gates
must come from the B2/B3 paths and not from this report.

## Objective

Run a fixed mechanistic ablation over a small set of sentinel real datasets to
measure how strongly each individual mechanism in the bench-only sentinel set
shifts the spectra-only realism diagnostics in the `uncalibrated_raw` lane.
The lane is **not** trying to claim realism — it is trying to expose
mechanistic levers without using calibration, capture, or learning.

## Profiles

Defined in `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/adapters/builder_adapter.py`
as `R2A_MECHANISTIC_PROFILES`. Each profile is fully determined by its name and
the dataset/preset seed; none consults real spectra, labels, splits, targets,
or AUC.

| profile | mechanism | parameters |
|---|---|---|
| `r2a_baseline` | identity control | none |
| `r2a_pathlength_drift` | per-sample multiplicative pathlength factor | uniform in [0.85, 1.15] |
| `r2a_baseline_curvature` | per-sample additive linear+quadratic baseline drift | linear in [-0.02, 0.02], quad in [-0.01, 0.01] |
| `r2a_emsc_like_scatter` | per-sample multiplicative scale + additive offset | scale in [0.9, 1.1], offset in [-0.01, 0.01] |
| `r2a_instrumental_broadening` | row-wise Gaussian convolution at fixed FWHM | FWHM = 8 nm |
| `r2a_structured_noise` | smoothed correlated noise added to each row | amp 0.005, sigma 2 bins |

These transforms are bench approximations for documentation, not changes to the
production synthesis generator.

## Audit Contract

The builder always adds `metadata['r2a_mechanistic_profile']`. With
`mechanistic_profile=None` this is a disabled audit record only; it does not
change spectra, targets, seeds, thresholds, metrics, or any gate behavior.
When a profile is requested, the audit records:

- `enabled` — true only when a profile is requested.
- `profile`, `seed`, `input_seed`, `transform_params` — full provenance.
- `oracle=false`, `label_inputs_used=false`, `target_inputs_used=false`,
  `split_inputs_used=false`, `source_oracle_used=false`,
  `learned=false`, `real_stat_capture=false`,
  `thresholds_modified=false`, `metrics_modified=false`,
  `imputed=false`, `replays_real_rows=false`.

If any future R2a transform requires a flag flipped to `true`, that flag must
be made explicit in the audit dictionary and the experiment must declare the
escalation in its report.

`exp08_mechanistic_sentinel_ablation.py` persists the non-oracle audit into
every `AblationRow` and CSV row as explicit columns:

- `audit_oracle`, `audit_label_inputs_used`, `audit_target_inputs_used`,
  `audit_split_inputs_used`, `audit_source_oracle_used`, `audit_learned`,
  `audit_real_stat_capture`, `audit_thresholds_modified`,
  `audit_metrics_modified`, `audit_imputed`, `audit_replays_real_rows`.
- `profile_input_seed`, `profile_scope`, and compact stable JSON
  `profile_transform_params`.

Blocked rows that fail before synthetic profile application still carry the
same non-oracle audit flags with empty `{}` profile parameters and null profile
seed/input seed.

## Inputs and Outputs

- Adapter: `bench/nirs_synthetic_pfn/src/nirsyntheticpfn/adapters/builder_adapter.py`
  — adds optional `mechanistic_profile` kwarg to `build_synthetic_dataset_run`;
  default path unchanged.
- Experiment: `bench/nirs_synthetic_pfn/experiments/exp08_mechanistic_sentinel_ablation.py`.
- Reports: `bench/nirs_synthetic_pfn/reports/r2a_mechanistic_sentinel_ablation.md`
  and `r2a_mechanistic_sentinel_ablation.csv`.

## Command

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src \
  python bench/nirs_synthetic_pfn/experiments/exp08_mechanistic_sentinel_ablation.py \
    --profiles r2a_baseline,r2a_pathlength_drift,r2a_baseline_curvature,r2a_emsc_like_scatter,r2a_instrumental_broadening,r2a_structured_noise \
    --n-synthetic-samples 64 \
    --max-real-samples 64 \
    --max-sentinel-datasets 8 \
    --sentinel-tokens BEER,DIESEL,CORN,MILK,LUCAS,PHOSPHORUS,MANURE,SOIL,BERRY,PEACH,PLUMS,FRUIT \
    --seed 20260430
```

Pass `--max-sentinel-datasets 0` (or any non-positive value) to score every
runnable cohort row. Pass a narrower comma-separated `--profiles` list to drop
profiles.

## Sentinel Selection (Priority-Based)

Selection is sentinel-first **and primary-first** whenever
`--max-sentinel-datasets > 0`:

1. `discover_local_real_datasets` enumerates every runnable AOM/TabPFN cohort row.
2. Each row whose `source`, `task`, `database_name`, or `dataset` contains any
   token from `--sentinel-tokens` (case-insensitive substring match) is kept.
3. Each kept row is assigned the **highest priority** of its matching tokens
   (lowest priority index wins when several tokens hit the same row).
4. Rows are sorted by `(priority, original_cohort_index)` so the priority
   groups stay in order and within each group the original cohort order is
   preserved.
5. The sorted list is truncated to `--max-sentinel-datasets`.

### Default Token Priority Groups

| priority index | group              | default tokens                                  |
|---------------:|--------------------|-------------------------------------------------|
|              0 | primary            | `BEER`, `DIESEL`, `CORN`                        |
|              1 | secondary milk     | `MILK`                                          |
|              2 | secondary soil     | `LUCAS`, `PHOSPHORUS`, `MANURE`, `SOIL`         |
|              3 | secondary fruit    | `BERRY`, `PEACH`, `PLUMS`, `FRUIT`              |
|              4 | custom user tokens | any `--sentinel-tokens` outside the groups above |

The grouping constants live in
`exp08_mechanistic_sentinel_ablation.SENTINEL_PRIORITY_GROUPS`; the flat list
remains available as `DEFAULT_SENTINEL_TOKENS`. Custom user tokens are placed
in a single implicit final group; their relative order follows the cohort
order, not the order they appeared on the CLI.

### Default Cap

`DEFAULT_MAX_SENTINEL_DATASETS = 8`. The current AOM benchmarks ship 2 BEER +
3 DIESEL + 2 CORN = 7 named primary rows; a default of `8` therefore covers
every named primary plus exactly one secondary slot for diagnostic comparison.
This default is independent of any B2/B3/B4/B5 gate and is purely a bench-only
ergonomic choice; downstream gate paths are unaffected.

### Cap Bypass

When `--max-sentinel-datasets <= 0`, the token filter and the priority sort
are both intentionally bypassed and every runnable cohort row is scored in the
order returned by `discover_local_real_datasets`.

### Result Surface

The result dictionary and Markdown report expose:

- `real_runnable_count` — total runnable cohort rows discovered.
- `real_sentinel_candidate_count` — rows surviving the token filter (equals
  `real_runnable_count` when the cap is `<= 0`).
- `real_selected_count` — rows actually scored after the cap is applied.
- `sentinel_tokens` — the token list used for the run (echoed in the JSON
  summary block of the Markdown report).

R2a's selection mechanics are bench-only and never feed back into the B2, B3,
B4, or B5 gates; the disclaimer at the top of this document still applies.

## R2h BERRY Readout Provenance

`r2h_sentinel_matrix_v1` is a later R2 remediation profile, not part of the
R2a ablation matrix. It is documented here because it uses the same
sentinel-first, non-gate audit contract.

For BERRY-routed `beverage_juice` rows, R2h adds
`cloudy_berry_percent_transmittance_readout`. The constants are fixed
mechanistic optical priors for cloudy berry juice in apparent
percent-transmittance/intensity units:

- `absorbance_path_factor_range=[4.25, 4.75]`
- `detector_baseline_percent=30.0`
- `detector_dynamic_percent=20.0`
- `turbidity_offset_percent_range=[-20.0, 20.0]`
- `output_clip_percent=[0.0, 100.0]`

The mechanism is a readout transform for raw instrument datasets whose axis is
percent/intensity-like: generated absorbance is clipped to non-negative values,
scaled by a fixed prior path-factor draw, mapped through `10^-A`, shifted onto
a detector percent/intensity axis, offset by a broadband turbidity term, and
clipped to `[0, 100]`.

This is not a calibration or statistical capture. The constants are not derived
from BERRY spectra, labels, targets, splits, PCA/covariance/marginal/quantile
summaries, adversarial AUC, morphology gap scores, thresholds, or downstream
performance. The builder audit records this as
`constant_status=fixed_mechanistic_prior`,
`readout_space=apparent_percent_transmittance_intensity`,
`calibration_source=none`, `real_stat_source=none`, and
`threshold_source=none`.

Validity is limited to cloudy berry juice percent/intensity readout
hypotheses. The current R2h morphology snapshot still shows amplitude risk on
BERRY rows, so R2h is not a scientific gate and does not reopen B2/B3/B4/B5.
Repeated seeds, robust/adversarial checks, and real-vs-real context remain
mandatory before any candidate can be promoted.

## What R2a Does NOT Do

- It does not call any B2/B3 gate; it does not write to any gate report path.
- It does not fit any calibration to real spectra and applies no marginal,
  covariance, or quantile mapping.
- It does not capture noise, baseline, or scatter statistics from real data.
- It does not train, evaluate, or invoke any ML/DL component.
- It does not change provisional thresholds or metric definitions.
- It does not claim realism for any preset, profile, dataset, or cohort.

## Interpretation Guidance

R2a aggregates per profile (mean and median adversarial AUC, smoke-failure
count) are diagnostic only. A profile that lowers AUC in this lane is a
candidate for follow-up mechanistic work through the standard
generator-development path. R2a aggregates cannot be treated as evidence that a
profile has improved the synthetic prior — that claim requires re-running the
B2 audit on a regenerated, gate-eligible synthetic prior.
