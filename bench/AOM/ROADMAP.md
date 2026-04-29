# AOM-PLS Publication Roadmap

**Scope.** Operational roadmap from today (2026-04-21) to a submittable
manuscript. This is the day-level plan wired to actual files, scripts and
datasets in the repository. It is the executable counterpart of
`PUBLICATION_PLAN.md` (the argument) and `PUBLICATION_BACKLOG.md` (the
workstream specs).

**Reference substrate.**

- Data: [bench/tabpfn_paper/data/regression/](nirs4all/bench/tabpfn_paper/data/regression/)
  (238 dataset splits across 29 families) and
  [bench/tabpfn_paper/data/classification/](nirs4all/bench/tabpfn_paper/data/classification/)
  (16 splits across 11 families).
- Selection: [bench/tabpfn_paper/data/DatabaseDetail.xlsx](nirs4all/bench/tabpfn_paper/data/DatabaseDetail.xlsx)
  names the subset. `scan_datasets.py` already honours this list.
- Reference scores: [bench/tabpfn_paper/master_results.csv](nirs4all/bench/tabpfn_paper/master_results.csv)
  (336 rows, 61 distinct dataset splits, 6 models: CNN, CatBoost, PLS,
  Ridge, TabPFN-Raw, TabPFN-opt). *This CSV is the oracle.* Every run in
  this roadmap produces rows with the same schema (see §0.3) and a
  ΔRMSE column against the reference.
- The paper behind the scores:
  [bench/tabpfn_paper/Robin_s_article-1.pdf](nirs4all/bench/tabpfn_paper/Robin_s_article-1.pdf).
  Citation anchor; read once for protocol details.

**Existing scripts to adapt, not duplicate.**

- [run_reg_pls.py](nirs4all/bench/tabpfn_paper/run_reg_pls.py) — reference
  runner for PLS, Ridge, CatBoost, TabPFN, NICON. Already hooked into
  nirs4all pipelines and `DatasetConfigs`. Template for every new runner.
- [run_reg_aom.py](nirs4all/bench/tabpfn_paper/run_reg_aom.py) — AOM-PLS
  runner, but the model-step is wired to the *old* Optuna-piggy-back API
  (`operator_index`, fixed `n_components=(1, 27)`). This is the baseline
  runner we *replace* at M2 once W3 lands.
- [scan_datasets.py](nirs4all/bench/tabpfn_paper/scan_datasets.py) —
  dataset ID cards with statistics. Already does its job; reused in M0
  for the benchmark-cohort selection.
- Workspace lives at [bench/tabpfn_paper/AOM_workspace/](nirs4all/bench/tabpfn_paper/AOM_workspace/).
  Stays as-is for reproducibility of past runs; new workstreams create
  fresh workspaces under [bench/AOM/roadmap_runs/W*](nirs4all/bench/AOM/).

**Workstream mapping.** Each milestone below executes one or more
workstreams from `PUBLICATION_BACKLOG.md`. The legend `(W1)` ties a
milestone to its workstream.

---

## 0. Ground-truth layer (M0) — *weeks 1–2*

Purpose: make sure every workstream after this is measured against the
same oracle and uses the same benchmark harness. No algorithmic change
yet.

### M0.1 — Benchmark-cohort freeze (W7)

Output: `bench/AOM/roadmap_runs/cohort.csv` listing every dataset split
we run on, with its reference RMSE from master_results.csv.

- Run [scan_datasets.py](nirs4all/bench/tabpfn_paper/scan_datasets.py) on
  the XLSX-selected subset. Confirm the 61 dataset splits in
  master_results.csv all exist on disk and parse cleanly.
- Emit `cohort.csv` with columns:
  `dataset_family, dataset, n_train, n_test, p, has_nan, signal_type_detected,
   ref_rmse_pls, ref_rmse_tabpfn_opt, ref_rmse_tabpfn_raw, ref_rmse_cnn,
   ref_rmse_catboost, ref_rmse_ridge`.
- Drop from the cohort the splits flagged in `run_reg_aom.py` as NAN or
  CRASH (`ALPINE_C_424_KS`, `ALPINE_C_424_RobustnessAlps`,
  `ALPINE_N_552_KS`, `FUSARIUM/Tleaf_grp70_30`, and the two LUCAS
  datasets that are "SKIPPED TOO BIG"). Record the reason column.
- Final expected cohort size: ~55 regression splits + classification
  subset to be defined in M0.3.

### M0.2 — Benchmark harness (W7)

Output: `bench/AOM/roadmap_runs/harness/run_benchmark.py`. *One* script
everyone calls. Nobody writes a one-off runner after this milestone.

- API:
  ```bash
  python run_benchmark.py \
      --cohort cohort.csv \
      --pipeline pipelines/W1_press.py \
      --workspace bench/AOM/roadmap_runs/W1_press/ \
      --model-name "AOM-PLS-PRESS" \
      --seeds 0,1,2,3,4,5,6,7,8,9 \
      --n-jobs -1
  ```
- Per (dataset × seed), runs `nirs4all.run(...)` and appends a row to a
  unified results CSV with the master_results.csv schema, plus the
  extra columns: `seed, wall_clock_s, selected_operator, k_selected,
  n_orth, delta_rmse_vs_pls, delta_rmse_vs_tabpfn_opt, notes`.
- Honours `verbose=0` by default; `--verbose` flips to 1. Uses
  `CacheConfig(memory_warning_threshold_mb=32768)` like the existing
  AOM runner.
- Resumable: appends rows, skips already-completed (dataset, seed, model)
  triples. Critical for the 55×10-seed runs.

### M0.3 — Statistics library (W7)

Output: `bench/AOM/roadmap_runs/stats/` with three tested helpers.

- `nadeau_bengio_ttest(rmse_a, rmse_b, n_train, n_test, folds=5)` —
  corrected resampled paired t-test (Nadeau & Bengio, JMLR 2003).
- `friedman_nemenyi(rmse_matrix, model_names) -> (p_value, cd_svg)` —
  Friedman rank test + Nemenyi post-hoc + critical-difference diagram
  (Demšar, JMLR 2006).
- `bootstrap_ci(rmse_vec, n_boot=10000, alpha=0.05)` — per-dataset
  RMSE confidence intervals.
- Unit tests in `bench/AOM/roadmap_runs/stats/test_*.py` against the
  worked examples from those two papers. CI must be green before M1.

### M0.4 — Classification cohort decision

The paper is primarily regression, but `AOMPLSClassifier` exists and
would reviewer-proof the submission against "does this only work for
regression?" Output: a short decision memo (`bench/AOM/roadmap_runs/
classification_decision.md`) answering:

- Is there sufficient reference signal in the 16 classification splits
  to include them? (master_results.csv has no classification rows —
  need to check [metrics.xlsx](nirs4all/bench/tabpfn_paper/metrics.xlsx)
  and the PDF.)
- If yes, include 8–12 classification splits as a secondary experiment.
  If no, cite classification as future work and scope the paper to
  regression.

**Exit gate for M0.** Benchmark cohort frozen, harness runs one
PLS-baseline row end-to-end on the full cohort reproducing the
master_results.csv PLS RMSE to within 2%. Stats helpers pass unit tests.

---

## 1. Mechanical unification (M1, W5 part 1) — *week 3*

Purpose: unify the AOM-PLS and POP-PLS code paths *before* any decision
is measured on either policy alone. Pure refactor. No experiments. Every
subsequent milestone runs on both `selection="global"` (AOM) and
`selection="per_component"` (POP). This is what keeps M2–M5's decisions
policy-aware rather than AOM-biased.

**Rationale.** If M2 picks `selection_criterion="hybrid"` based on
global-policy results, and POP silently prefers PRESS, we ship the wrong
default for POP users. Same risk at M3 (`max_components` default), M4
(bank contents) and M5 (presets). Putting the unification first means
every downstream benchmark has a 2-row-per-dataset layout
(`selection="global"`, `selection="per_component"`), and policy-specific
divergences show up where they exist.

### M1.1 — Unified backend

New private module: `nirs4all/operators/models/sklearn/_adjoint_pls_core.py`.

- `_adjoint_pls_fit_numpy(X, Y, bank, max_components, n_components,
  selection, selection_criterion, n_orth)`. Dispatches on `selection`:
  - `"global"`: the current AOM-PLS flow.
  - `"per_component"`: the current POP-PLS flow.
- `selection_criterion` ships with one value only at this milestone:
  `"holdout_20"` (current behaviour, pre-W1). M2 adds the alternatives.
- `"auto"` is **not** implemented at M1. That's the late-stage W5
  experiment (M6).

### M1.2 — `AOMPLSRegressor` rewire

File: [aom_pls.py](../../nirs4all/operators/models/sklearn/aom_pls.py).

- Keep the class name, keep the public imports.
- `fit()` delegates to `_adjoint_pls_fit_numpy`.
- New parameter `selection: str = "global"`. Default stays "global" so
  the post-M1 library behaves identically for existing users.
- New parameter `selection_criterion: str = "holdout_20"`. Also
  identical to current behaviour at M1.
- Old parameters `gate` and `operator_index` are kept at M1 for
  backwards path equivalence; they are removed at M2 once we have
  evidence to justify it (sparsemax removal) and at M3 once
  `n_components` is an int (Optuna hook replacement).

### M1.3 — `POPPLSRegressor` rewire

File: [pop_pls.py](../../nirs4all/operators/models/sklearn/pop_pls.py).

Reduce to:

```python
class POPPLSRegressor(AOMPLSRegressor):
    def __init__(self, **kw):
        kw.setdefault("selection", "per_component")
        kw.setdefault("operator_bank", "compact")
        super().__init__(**kw)
```

Plus whatever constructor signature preservation the existing users
need.

### M1.4 — Parity tests

New files:
- `tests/unit/operators/models/test_aom_pls_parity.py`
- `tests/unit/operators/models/test_pop_pls_parity.py`

On three synthetic datasets with fixed seeds, confirm bit-equivalence
between:

- `AOMPLSRegressor(selection="global")` post-M1 and the pre-M1
  `AOMPLSRegressor` snapshot (tagged `aom-pls-m0`).
- `AOMPLSRegressor(selection="per_component")` post-M1 and the pre-M1
  `POPPLSRegressor` snapshot.

Tolerance: 1e-10. These tests block every subsequent merge.

### M1.5 — Harness update

`bench/AOM/roadmap_runs/harness/run_benchmark.py` adds a
`--selection` flag that forwards to the estimator. The result CSV schema
gains a `selection` column so every row is tagged with the policy it
was run under. Every M2–M5 benchmark will be invoked at least twice —
once per policy — using the same seed set.

**Exit gate for M1.** Parity tests green. Library post-M1 produces
identical results to library pre-M1 for both default AOM and default
POP configurations. A pilot benchmark run with 3 datasets × 2 seeds ×
2 policies completes cleanly and populates the `selection` column in
the results CSV.

---

## 2. Selection criterion (M2, W1) — *weeks 4–6*

Purpose: decide what replaces the 20% holdout. Quantify the holdout's
cost on the cohort. Ship the new default.

### M2.1 — Five configurations × two policies, 10 seeds each

Output: `bench/AOM/roadmap_runs/W1_selection_criterion/results.csv`.

**Two policies.** Every configuration below runs under both
`selection="global"` and `selection="per_component"`. This is the whole
point of putting M1 before M2: the criterion default might differ by
policy, and we need the evidence to pick either a shared default or a
policy-conditioned one.

Configurations (pipelines live in
`bench/AOM/roadmap_runs/pipelines/`):

1. `W1_holdout20.py` — current deployed behaviour. Baseline.
2. `W1_press.py` — PRESS for both operator and `k`. Port the PRESS path
   from `_poppls_press_pass` in
   [pop_pls.py](../../nirs4all/operators/models/sklearn/pop_pls.py).
3. `W1_cv5.py` — 5-fold CV for both operator and `k`. Uses sklearn's
   `KFold` per candidate. Expected to be 5× slower.
4. `W1_hybrid.py` — **primary candidate**. PRESS for operator
   selection, 5-fold CV for `k` selection on the winning operator.
   Two-pass structure: (a) PRESS scan over all ~60 operators at
   a fixed-`k` pilot (say `k = 10`), pick argmin; (b) 5-fold CV over
   `k ∈ [1, max_components]` using the winning operator.
5. `W1_cv5x2.py` — Dietterich's 5×2 CV. Ground-truth reference. Run
   only on a 20-dataset subset for wall-clock reasons.

Total row count: ~55 datasets × 10 seeds × 5 criteria × 2 policies ≈
5500 rows (+ the 20-subset × 2-policy × 10-seed cv5x2 extra ≈ 400 rows).

### M2.2 — Measurement and analysis

Output: `bench/AOM/roadmap_runs/W1_selection_criterion/report.md`.

- Per-dataset RMSE (mean over seeds), selected operator (mode over
  seeds), `k_selected` distribution — **tabulated per policy**.
- Wall-clock comparison.
- **Operator stability index** per dataset per (config, policy):
  fraction of 10 seeds agreeing on the winning operator. Holdout-20
  is expected to score < 0.7 here; that's the main argument for
  removal.
- Friedman–Nemenyi rank across the four fast configurations,
  **separate CD diagrams per policy** and a third diagram across the
  union. If the two per-policy diagrams agree on the winner, the paper
  ships one shared default. If they disagree, the paper documents two
  policy-conditioned defaults.
- Nadeau–Bengio paired t-test against the holdout-20 baseline for
  every other configuration, per policy. Datasets with `p < 0.05` are
  tabulated.

### M2.3 — Decision memo

Output: `bench/AOM/roadmap_runs/W1_selection_criterion/decision.md`.
Structure:

1. Shared default (if the two policies agree).
2. Per-policy default (if they disagree), with the divergence
   quantified and the reason explained.
3. Acceptance-criteria table from backlog W1 filled in, per policy.

Expected outcome (pre-registered hypothesis): hybrid wins or ties pure
PRESS for both policies. If hybrid wins by < 1% mean RMSE or < 0.1 in
Friedman rank on either policy, ship pure PRESS on simplicity grounds.

**Exit gate for M2.** Winning `selection_criterion` decided in writing
for both policies. Decision memo committed. The M3/M4/M5 runs will use
the per-policy winner going forward (policy-conditioned defaults are
cheap at the harness level — just a `--selection-criterion` flag
defaulting by policy).

---

## 3. `max_components` and `n_components` API (M3, W3) — *weeks 7–9*

Purpose: fix the three inconsistent code paths, add the user-facing
forced/auto distinction, pick the `max_components` default from the
cohort evidence. **All measurements run on both policies.**

### M3.1 — Constructor refactor

File to edit:
[aom_pls.py](../../nirs4all/operators/models/sklearn/aom_pls.py).

- Add `max_components: int = 25` and change `n_components: int | str = "auto"`.
- In `fit`, `n_comp` (extraction ceiling) now comes from `max_components`,
  capped to `min(n_samples − 1, n_features)`.
- Collapse the remaining branches in `_adjoint_pls_fit_numpy` (the
  unified backend from M1) so that for both `selection="global"` and
  `selection="per_component"`:
  1. Extract up to `n_comp` components per operator (per component in
     the per-component policy).
  2. Score with `selection_criterion` (M2 winner per policy).
  3. If `n_components == "auto"`: pick `k*` by argmin of score.
  4. If `n_components` is an integer: no prefix search; trim or pad;
     warn if `> max_components`.
- Remove `operator_index` (the Optuna-piggy-back parameter). Users who
  want to force a single operator now pass a singleton
  `operator_bank=[op]` plus an integer `n_components`.
- Remove the `_select_prefix` method. Prefix selection is unified
  inside the main flow.
- Update `AOMPLSRegressor.__repr__` and docstring.

### M3.2 — Cohort measurement to pick `max_components` default

Output: `bench/AOM/roadmap_runs/W3_n_components/histogram.png` and
`decision.md`.

- Run `run_benchmark.py` with `max_components ∈ {15, 20, 25, 30, 40}` and
  `n_components="auto"` using each policy's M2 winner as selection
  criterion. Both `selection="global"` and `selection="per_component"`.
- **Two histograms** of auto-selected `k*` — one per policy. These
  distributions are expected to differ (per-component tends to pick
  smaller `k*` because each component is locally optimised).
- Pick `max_components` default: the smallest value that covers ≥ 99%
  of datasets under **both** policies without truncation. If one
  policy needs materially more headroom, ship policy-specific defaults.
- Measure wall-clock cost per `max_components` increment.

### M3.3 — Forced vs auto measurement

Output: same folder, `forced_vs_auto.csv`.

Run three configurations × two policies on the full cohort:

1. `n_components="auto"`.
2. `n_components=k_median` where `k_median` is the per-policy median
   auto-selected `k` from M3.2.
3. `n_components=10` (arbitrary domain-expert value; common in
   chemometrics defaults).

Report ΔRMSE (mean, median, worst-case) of forced policies against auto,
split by `selection`. This is the "does `n_components` need tuning?"
answer for the paper, and it may have a policy-dependent story.

### M3.4 — External Optuna sanity check

Output: `bench/AOM/roadmap_runs/W3_n_components/optuna_sanity.csv`.

- On 10 datasets from the cohort × 2 policies, run Optuna with
  `n_components ∈ {1..max_components}` as a categorical integer over 50
  trials. Compare the Optuna-best RMSE vs `n_components="auto"`.
- Expected: < 1% mean RMSE gain at 10× wall-clock. If larger, the
  paper needs to acknowledge that external tuning is worthwhile for
  niche datasets (and potentially one policy more than the other).

**Exit gate for M3.** `AOMPLSRegressor` exposes `max_components` and
`n_components="auto"`. `operator_index` is removed. Docstring updated
with the "when to let AOM-PLS choose, when to force it" paragraph.
Regression tests confirm `n_components="auto"` under each policy's M2
criterion matches or beats the M2 per-policy winner.

---

## 4. Bank extensions (M4, W4) — *weeks 10–12*

Purpose: OSC (supervised linear), EPO (supervised linear), Whittaker
baseline, wavelet/FFT promotion. Signal-type sub-experiment.
**Every bank comparison runs under both policies**, because OSC may be
picked by `selection="global"` on component 1 but rotated away from by
`selection="per_component"` after c1 — the per-operator selection
frequency is policy-dependent and informs default-bank membership.

### M4.1 — `LinearOperator` ABC extension

File: [aom_pls.py](../../nirs4all/operators/models/sklearn/aom_pls.py)
(class `LinearOperator`, line 52).

- Add `fit(self, X, y=None)` hook with default `return self` (no-op).
- Supervised operators override it.
- Document the "linear at apply, fitted at setup" contract in the
  class docstring.

### M4.2 — New operators

New module:
`nirs4all/operators/models/sklearn/aom_pls_operators.py` (or extend
`aom_pls.py` if size stays under 1500 lines).

- `OSCOperator(n_orth)` — port from
  [orthogonalization.py](../../nirs4all/operators/transforms/orthogonalization.py).
  Forward: `x (I − P_o P_o^T)`. Adjoint: same (symmetric projection).
  `fit(X, y)` stores `P_o`.
- `EPOOperator(reference_block)` — same structure, projection from an
  external block. Only added to the bank when a reference block is
  provided at construction.
- `WhittakerBaselineOperator(lam)` — banded solve. Pre-compute the
  banded factorisation in `initialize(p)`. Test at
  `lam ∈ {1e4, 1e6, 1e8}`.
- Unit tests in
  `tests/unit/operators/models/test_aom_operators.py`:
  adjoint identity `<A x, y> = <x, A^T y>` within 1e-8 tolerance for
  each new operator.

### M4.3 — Bank-extension experiments (both policies)

Output: `bench/AOM/roadmap_runs/W4_bank/results.csv`.

Four banks × two policies, on the cohort, using each policy's M2
`selection_criterion` and M3 `max_components`:

1. `{default}` — post-M3 baseline. Control.
2. `{default + OSC×3}` where OSC×3 = `OSCOperator(1), OSCOperator(2), OSCOperator(3)`.
3. `{default + OSC×3 + Whittaker×3}` with `lam ∈ {1e4, 1e6, 1e8}`.
4. `{default + OSC×3 + Whittaker×3 + wavelet + FFT}` — promote the
   extended-bank operators that are currently not defaults.

Analysis: Friedman–Nemenyi across the four banks **per policy**;
per-dataset Nadeau–Bengio vs the default per policy. Operator-selection
frequency per (policy, operator) — the OSC-rotation hypothesis above
gets tested here. Init-cost wall-clock comparison for the
`WaveletProjectionOperator` on the largest `p` in the cohort (LUCAS
or SOIL_ESDAC_19969 likely).

### M4.4 — Signal-type sub-experiment

Output: `bench/AOM/roadmap_runs/W4_bank/signal_type.csv` and a
paragraph in the W4 report.

- For every cohort dataset, attempt to convert the signal using
  `SignalType` detection and `convert_to_absorbance()`. Keep only
  datasets where both reflectance and absorbance representations are
  well-defined (positive, within Beer-Lambert range).
- Run AOM-PLS on both representations × both policies. Report:
  - Whether the winning operator changes (boolean), per policy.
  - ΔRMSE between representations, per policy.
- Target: "signal-type choice changes the winning operator on X% of
  datasets and shifts RMSE by Y% on half of those" — a single paper-
  ready sentence (or two, one per policy if they diverge).

### M4.5 — SNV/MSC non-inclusion confirmation

Output: one short section in the W4 report with the
branch+merge stacking baseline numbers:

```python
pipeline = [
    {"branch": [
        [SNV(), AOMPLSRegressor(...)],
        [AOMPLSRegressor(...)],
        [MSC(), AOMPLSRegressor(...)],
    ]},
    {"merge": "predictions"},
    Ridge(),
]
```

Measured on the cohort. The resulting table is the "SNV and MSC are
handled outside the bank; here is the stacking pattern that does it
cleanly" evidence for the paper's discussion.

**Exit gate for M4.** New-bank defaults decided (OSC almost certainly
in; Whittaker and wavelet/FFT decided empirically). W4 report committed
with signal-type sub-experiment numbers.

---

## 5. Operator-bank API (M5, W6) — *weeks 13–14*

Purpose: user-facing API for selecting and combining bank presets.
Lives in the library, used by the paper's examples.

### M5.1 — `OperatorBankSpec` dataclass

New module: `nirs4all/operators/models/sklearn/aom_pls_bank.py`.

```python
@dataclass
class OperatorBankSpec:
    base: str = "default"              # "default", "compact", "sg_only",
                                        # "derivatives_only", "extended"
    include: list[str] = field(default_factory=list)   # family names
    exclude: list[str] = field(default_factory=list)
    custom_operators: list[LinearOperator] = field(default_factory=list)

    def resolve(self, p: int) -> list[LinearOperator]: ...
```

- Family names: `sg_smoothing`, `sg_d1`, `sg_d2`, `detrend`,
  `finite_difference`, `norris_williams`, `composed`, `wavelet`, `fft`,
  `osc`, `epo`, `whittaker`.
- `resolve(p)` initialises all operators for `p`-dimensional input and
  returns the concrete list.

### M5.2 — Five named presets

In `aom_pls_bank.py`, five functions:

- `default_operator_bank()` — keeps existing behaviour, possibly
  revised by M4 bank-extension outcome.
- `compact_operator_bank()` — POP-PLS's 9-operator bank (see
  `pop_pls_operator_bank` in
  [pop_pls.py](../../nirs4all/operators/models/sklearn/pop_pls.py)).
- `sg_only_operator_bank()`.
- `derivatives_only_operator_bank()`.
- `extended_operator_bank()` — existing function, updated.

### M5.3 — Constructor integration

File: [aom_pls.py](../../nirs4all/operators/models/sklearn/aom_pls.py).

- Accept `operator_bank: str | list[LinearOperator] | OperatorBankSpec | None`.
  - `None` or `"default"` → `default_operator_bank()`.
  - `str` → named preset.
  - `list` → explicit override (current behaviour).
  - `OperatorBankSpec` → `.resolve(p)` at fit time.

### M5.4 — Preset benchmark (both policies)

Output: `bench/AOM/roadmap_runs/W6_bank_api/preset_ranks.csv`.

- Run the five presets on the cohort under M2/M3/M4 defaults, for both
  `selection="global"` and `selection="per_component"`.
- Report mean Friedman rank per preset **per policy** and per
  dataset-size quartile. This is the empirical "which preset to pick
  when" guide in the paper, with a policy conditioning if one is
  warranted.

### M5.5 — `bank_summary()` introspection

- Add `AOMPLSRegressor.bank_summary() -> pd.DataFrame` returning
  `family, name, params_json, selected (bool), gamma_weight` columns
  post-fit. Used in the paper's `get_preprocessing_report()` figure.

**Exit gate for M5.** Preset API documented in the `AOMPLSRegressor`
docstring. One example in `examples/user/` demonstrates
`operator_bank="compact"` + `selection="per_component"`. Backward compat
preserved: existing `operator_bank=list(...)` callers still work.

---

## 6. `"auto"` policy + sparsemax decision (M6, W5 part 2) — *weeks 15–16*

Purpose: answer the one remaining W5 question that needs M2–M5 defaults
to be settled — whether a data-driven policy selector (`"auto"`) beats
the two fixed policies, and whether the experimental `sparsemax` gate
still earns its place in the shipped code. The mechanical unification
shipped in M1; this milestone is pure experimentation and cleanup.

### M6.1 — `"auto"` implementation

Extend `_adjoint_pls_core.py` (from M1) so that `selection="auto"`:

1. For each policy in `{"global", "per_component"}`, runs the full
   scan under `selection_criterion` (M2 default for that policy).
2. Picks the policy whose best candidate wins the criterion.
3. Refits with that policy on the full data.

Cost: 1× the more expensive per-policy cost (PRESS/CV is computed once
per `(policy, b, k)` candidate; the per-component policy dominates).

### M6.2 — `"auto"` policy benchmark

Output: `bench/AOM/roadmap_runs/W5_unification/auto_vs_fixed.csv`.

- Run three configurations on the cohort, using each policy's defaults
  from M2–M5:
  1. `selection="global"`.
  2. `selection="per_component"`.
  3. `selection="auto"`.
- Friedman–Nemenyi ranks. Acceptance criterion for shipping `"auto"`:
  ties both individual policies in mean rank and shows no statistically
  significant regression on any dataset.
- **Can `"auto"` be predicted without running it?** Measure whether a
  cheap diagnostic (e.g., "does the winning operator change between
  components 1 and 2 under `selection="per_component"`?") predicts
  which policy wins. If yes, this becomes an alternative to the
  full-scan auto and is worth a one-paragraph recommendation in the
  paper.
- If `"auto"` does not clear the bar: remove it from the public API,
  document the attempt in the paper's "tried but did not generalise"
  note.

### M6.3 — Sparsemax gate decision

The `gate="sparsemax"` code path has not shown an advantage on the
benchmark suite so far (see `PUBLICATION_PLAN.md` §2.8 and §4.3). Now
that M2's selection criterion is settled, re-measure once:

- Run `selection="global"` with `gate="hard"` (the default) and
  `gate="sparsemax"` under the M2 `selection_criterion` on the full
  cohort. Same for `selection="per_component"`.
- Acceptance criterion for keeping sparsemax: a statistically
  significant win on ≥ 10% of the cohort under at least one policy,
  without regressions elsewhere.
- Expected outcome: removal. If removal holds, delete the sparsemax
  code path from `_adjoint_pls_core.py` and the hand-written
  `_sparsemax` function from
  [aom_pls.py](../../nirs4all/operators/models/sklearn/aom_pls.py) at
  this milestone — not carried into the paper.

### M6.4 — Docstring and examples

- Rewrite the `AOMPLSRegressor` docstring to reflect the final shipped
  API:
  - One-sentence summary.
  - Parameter table led by `max_components`, `n_components`,
    `selection`, `selection_criterion`, `operator_bank`.
  - "What `selection` changes" paragraph.
  - "What `selection_criterion` changes" paragraph.
  - Minimal examples for the three most common configurations.
- Add three example scripts in `examples/user/models/`:
  - `aom_pls_default.py` — out-of-the-box.
  - `aom_pls_per_component.py` — POP-style (`selection="per_component"`,
    `operator_bank="compact"`).
  - `aom_pls_custom_bank.py` — preset + include/exclude.

**Exit gate for M6.** `"auto"` shipped iff it cleared the bar (or
removed with documented evidence). Sparsemax removed (or kept with
documented evidence). Docstring rewritten to the final API.

---

## 7. Large-scale benchmark (M7, W7 final pass) — *weeks 17–18*

Purpose: produce the paper's headline experiments. Everything below
runs on the M6 `AOMPLSRegressor` with the M2/M3/M4/M5 defaults (plus
`"auto"` if M6 kept it).

### M7.1 — Final comparison matrix

Output: `bench/AOM/roadmap_runs/M7_final/results.csv` and
`cd_diagrams/`.

Models in the comparison:
- **Primary contribution:** `AOMPLSRegressor(selection="global")`,
  `AOMPLSRegressor(selection="per_component")`, and if shipped,
  `AOMPLSRegressor(selection="auto")`.
- **Prior exploration (not shipped):** baseline AOM-PLS with
  holdout-20 (snapshot tagged `aom-pls-m0`, pre-M2 behaviour). This is
  the paper's "before/after" story for the selection-criterion switch.
- **Chemometrics baselines:** PLS, PLS + grid-searched SG/SNV/MSC.
  Reuse the pipeline from
  [run_reg_pls.py](../tabpfn_paper/run_reg_pls.py).
- **From prior nirs4all work:** FCK-PLS (run once for the table).
  POP-PLS is already covered by `selection="per_component"` post-M1;
  a pre-M1 POP snapshot run is optional depending on what the paper
  wants to say about the unification.
- **Reference models from master_results.csv:** CNN, CatBoost, Ridge,
  TabPFN-Raw, TabPFN-opt. Reference values already in the CSV; no
  re-run needed.

10 seeds per dataset. Total new runs: ~55 datasets × 10 seeds × (3
primary AOM variants + 1 pre-M2 snapshot + 1 FCK-PLS) ≈ 2750, plus the
PLS / PLS-grid baselines if not already in the cohort measurements.
At 5–30 s per run on the typical cohort dataset, this is ~5–30 hours of
wall-clock with `n_jobs=-1` on a 16-core box.

### M7.2 — Statistical analysis

Output: `bench/AOM/roadmap_runs/M7_final/report.md`.

- Friedman–Nemenyi across all models; CD diagram as the paper's Figure X.
- Per-pair Nadeau–Bengio t-tests: every combination of
  (AOM-PLS-variants) × (baselines). Table in supplementary.
- Per-dataset rank plot.
- The "before/after" comparison: pre-M2 AOM-PLS (holdout-20) vs
  post-M6 AOM-PLS (M2 criterion + M3 API + M4 bank + M5 presets). This
  is the second headline finding.

### M7.3 — Signal-type and stacking tables

- Signal-type sub-experiment (M4.4 results formatted for the paper).
- SNV/MSC-stacking control (M4.5).

### M7.4 — Per-dataset diagnostic reports

- For five representative datasets (covering small-n,
  derivative-dominated, scatter-dominated, and the two where AOM loses
  to CNN or TabPFN-opt in M7.1), produce
  `AOMPLSRegressor.bank_summary()` tables and the selected-operator
  history. These become the paper's qualitative-interpretability
  figures.

**Exit gate for M7.** All paper figures and tables are artefacts in
the M7 folder. Any result in the paper should be traceable to a
specific row in `M7_final/results.csv`.

---

## 8. Paper drafting (M8, W8) — *weeks 19–24*

Not a research milestone. Planning listed for calendar honesty.

### M8.1 — Outline and structure

Use `PUBLICATION_PLAN.md` §1 as the structural spine. Confirm target
journal — *Chemometrics and Intelligent Laboratory Systems* or
*Journal of Chemometrics* are the natural homes; *Analytical Chemistry*
if scope widens. Journal choice dictates length limits and figure
count.

### M8.2 — Section drafts (parallel)

- Introduction and motivation.
- Related work: POP-PLS (cite as sibling method in the family — already
  collapsed into `AOMPLSRegressor(selection="per_component")` since
  M1), FCK-PLS (cite as prior exploration), MoE-PLS / DARTS-PLS /
  Zero-Shot Router as prior exploration benchmarks.
- Method: the adjoint trick, identity dominance, the two selection
  policies, the selection criterion (formalise the M2 winner as the
  default), the operator-bank composition, the unification.
- Experiments: cohort description, protocol, main results from M7,
  ablations (M2 selection criterion, M3 `max_components`, M4 OSC,
  M6 `"auto"` if shipped).
- Discussion: when AOM-PLS helps (supervised + linear-at-apply
  operators, moderate n), when it doesn't (non-linear scatter not
  corrected upstream, n < 30), the three caveats from plan §5.5.
- Conclusion.

### M8.3 — Supplementary material

- Per-dataset results table (all 55 × N models × 2 policies).
- Reproducibility README pointing at
  `bench/AOM/roadmap_runs/` with the exact commands to replay every
  experiment.
- The cohort CSV (`cohort.csv`) with any redistributable metadata.

### M8.4 — Internal and external review

- Internal review: the project's author reads, then one second pair of
  eyes from the chemometrics community. Iterate once.
- Pre-submission: post an arXiv draft to gather early feedback (≤ 2
  weeks of waiting time built in).

**Exit gate for M8.** Manuscript submitted.

---

## Calendar summary

| Weeks | Milestone | Workstreams | Output |
|---|---|---|---|
| 1–2 | M0: ground-truth layer | W7 (partial) | cohort.csv, harness, stats lib |
| 3 | M1: mechanical unification | W5 part 1 | `AOMPLSRegressor(selection=...)`, POP as subclass, parity tests |
| 4–6 | M2: selection criterion (both policies) | W1, W2 | decision memo, winning `selection_criterion` per policy |
| 7–9 | M3: `max_components` + `n_components` (both policies) | W3 | API split, `max_components` default(s) |
| 10–12 | M4: bank extensions (both policies) | W4 | OSC / EPO / Whittaker, signal-type result |
| 13–14 | M5: bank API | W6 | preset API, `bank_summary` |
| 15–16 | M6: `"auto"` + sparsemax decision | W5 part 2 | ship-or-remove verdicts on `"auto"` and sparsemax |
| 17–18 | M7: final benchmark | W7 (final pass) | paper figures and tables |
| 19–24 | M8: drafting and submission | W8 | manuscript |

Total: ~6 months from 2026-04-21 to submission ~2026-10-20. The
expensive unknowns are M2 (the hybrid criterion may need more tuning
than planned; 2×-policy doubles the cost), M4 (Whittaker/EPO benchmarks
may require data not in the cohort), and M6 (`"auto"` may not clear the
bar — fast-fail path built in).

**Why M1 goes first.** Putting the mechanical unification before M2
ensures every downstream measurement (criterion, `max_components`,
bank contents, bank API) is made under both `selection="global"` and
`selection="per_component"`. The alternative — pushing unification to
M5 as an earlier draft of this roadmap had it — would have baked
AOM-biased defaults into the library and forced us to re-run M2–M4
under per-component at M5 to validate. That would have cost more
calendar, not less.

---

## Cross-cutting conventions

**Pipeline-file naming.** Every pipeline lives as a standalone `.py`
file in `bench/AOM/roadmap_runs/pipelines/`. Name format:
`{milestone}_{config}.py`, e.g. `W1_hybrid.py`. The harness imports
the pipeline's `build_pipeline(cfg)` function and calls it per
dataset.

**Result CSV schema.** All results CSVs share the columns:

```
dataset_family, dataset, task, model, seed,
n_train, n_test, p, signal_type,
rmsecv, rmse_train, rmse_test, mae_test, r2_test,
wall_clock_s, peak_mem_mb,
selected_operator, k_selected, n_orth,
bank_name, selection, selection_criterion, max_components,
delta_rmse_vs_ref_pls, delta_rmse_vs_ref_tabpfn_opt,
notes
```

The `selection` column is critical: every row is tagged with the
policy it was run under (`"global"` or `"per_component"`). All
per-policy analyses pivot on this column.

The first six columns plus `rmsecv`, `rmse_test`, `mae_test`, `r2_test`
are the master_results.csv superset. Reference deltas come from
joining against master_results.csv on `dataset`.

**Seed discipline.** Benchmark uses 10 seeds (0–9). Every workstream
runs the same seed set to make Nadeau–Bengio valid across comparisons.

**Dual-policy discipline.** Every benchmark from M2 onwards invokes
the harness once per `selection` value. The harness is resumable and
indexed on `(dataset, seed, model, selection)`, so policy cohorts can
be filled in any order.

**Workspace layout.** Each milestone writes to a dedicated workspace:

```
bench/AOM/roadmap_runs/
├── cohort.csv
├── harness/
│   ├── run_benchmark.py
│   └── stats/
├── pipelines/
│   ├── W1_holdout20.py
│   ├── W1_press.py
│   ├── W1_cv5.py
│   ├── W1_hybrid.py
│   ├── ...
├── M1_unification/            # parity tests + pilot run
│   └── parity_report.md
├── W1_selection_criterion/    # M2 outputs
│   ├── workspace/             # nirs4all workspace (SQLite + arrays)
│   ├── results.csv            # rows for both policies
│   ├── report.md
│   └── decision.md
├── W3_n_components/           # M3 outputs
├── W4_bank/                   # M4 outputs
├── W6_bank_api/               # M5 outputs
├── W5_unification/            # M6 outputs (auto + sparsemax)
└── M7_final/
```

**Code-change gates.** Every code change to the library needs:
- `ruff check .` green.
- `mypy .` green (or an explicit targeted ignore).
- `pytest tests/unit/operators/models/` green.
- The parity tests (M1.4) green for all AOM-PLS / POP-PLS changes.

**Reproducibility tag.** At every milestone exit, tag the repo:
`aom-pls-m0`, `aom-pls-m1`, ..., `aom-pls-m7`, `aom-pls-submitted`.
The paper's reproducibility section cites `aom-pls-submitted`.
`aom-pls-m0` is especially important: it is the pre-unification
snapshot the parity tests in M1.4 compare against, and the "before"
half of the before/after comparison in M7.2.

---

## What this roadmap does *not* do

- It does not expand the cohort beyond the 61 master_results.csv
  splits. If a reviewer asks for IDRC/Cargill Corn or other public NIRS
  corpora, that becomes a post-revision addition.
- It does not add a DARTS-style search to the shipped code. DARTS
  remains in `bench/AOM/darts_pls.py` as a negative-result reference
  for the paper.
- It does not integrate MoE-PLS into `AOMPLSRegressor`. MoE belongs
  to the stacking / branch+merge pattern at the *library* level, not
  inside the estimator.
- It does not pursue `Gamma` / sparsemax-based soft gating unless M6
  experiments surface a win. The default expected outcome is to remove
  that code path in M6.
