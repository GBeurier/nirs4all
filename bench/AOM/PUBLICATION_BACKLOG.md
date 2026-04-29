# AOM-PLS Publication Backlog

**Relation to `PUBLICATION_PLAN.md`.** The plan is the *argument* for the
paper. This backlog is the *ordered list of experiments and code changes*
that must land before the paper can be submitted. Each workstream is
self-contained enough to be run and reported independently; dependencies
are called out explicitly.

**Benchmark substrate.** All empirical workstreams run on the author's
60-dataset NIRS corpus with reference RMSE/R² scores. The headline figure
per workstream is always `ΔRMSE` against the reference *and* against the
current deployed baseline, per dataset, with bootstrap CIs. Per-dataset
results go into a supplementary table; paper body reports
Friedman-Nemenyi ranks and 5×2 CV Nadeau–Bengio p-values.

**Ordering rationale.** Workstreams are ordered so that each one's output
is either standalone or feeds cleanly into the next. Selection-policy
questions (W1, W2, W3) come before scope questions (W4, W5, W6) because
the latter will have to be re-evaluated under whichever selection policy
wins.

---

## Decisions already locked in (set by the author, 2026-04-21)

These are inputs to the workstreams below, not open questions.

- **The unified estimator keeps the `AOMPLSRegressor` name.** The joint
  AOM/POP algorithm is exposed under the existing class; the new behaviour
  is controlled by `selection` and `selection_criterion` parameters (see
  W3/W5). `AdjointPLSRegressor` from earlier drafts is dropped as a name.
- **The 20% internal holdout is removed** — but *after* the W1 head-to-head
  that quantifies what it costs on the 60-dataset corpus. The measurement
  exists so the paper can defend the removal, not so the decision is
  reconsidered.
- **Component budget is separated from selection.** A new
  `max_components` parameter (default 25 or 30, decided by W3) bounds the
  NIPALS extraction. `n_components` becomes the *selection output* by
  default (`"auto"`), or an *enforced integer* when the user overrides.
- **The operator-bank preset + include/exclude API (W6) is in scope.**
- **Signal-type influence on operator choice is a sub-experiment of W4.**

## Workstream index

| # | Workstream | Blocks what? | Rough effort |
|---|---|---|---|
| W1 | Selection criterion head-to-head (PRESS / CV / holdout / hybrid) | W3, W4, W5, W6 | Medium |
| W2 | Holdout removal — confirmatory measurement | — | Small |
| W3 | `max_components` + `n_components` API split and auto-selection study | W4, W5 | Medium |
| W4 | OSC, EPO and the "linear at apply time" extended bank | W6 | Medium |
| W5 | AOM/POP unification under `AOMPLSRegressor(selection=...)` | W6 | Large |
| W6 | Custom operator banks — preset + include/exclude API | — | Small–Medium |
| W7 | Statistical machinery and reproducibility package | — | Small |
| W8 | Paper drafting | — | Large |

---

## W1. Selection criterion head-to-head — PRESS, CV, hybrid, and the holdout it replaces

**Question.** Among PRESS, k-fold CV, 5×2 CV, and a hybrid, *which
selection criterion* should become the default `selection_criterion` for
AOM-PLS? The 20% holdout is included in the comparison as the baseline
being replaced — it is in the experiment to quantify what we leave behind,
not to win.

**Context.** The deployed code picks the winning operator and the winning
prefix by minimising RMSE on a 20%-held-out subsample (when no external
`X_val` is supplied). The seed is fixed (`RandomState(42)`) so the run is
reproducible, but the *choice* of 20% and the fixed permutation are
arbitrary from the data's point of view. Three alternatives exist:

- **PRESS** — `Σ_i ((y_i − ŷ_i) / (1 − h_ii))²`, an analytical LOO
  estimate that needs only one fit. POP-PLS already implements this
  (`pop_pls.py`, `auto_select=True`). Deterministic in the data, no seed.
  Caveat: needs a well-conditioned hat matrix (Tikhonov or PLS-score-space
  version for `p ≫ n`).
- **k-fold CV** — standard. Robust but `k` fits per candidate. Cheap on
  small `k` (k=3, k=5); expensive on large operator banks.
- **5×2 CV** — Dietterich's protocol. Pairs well with Nadeau–Bengio
  corrected t-tests downstream.
- **Hybrid: PRESS for operator, CV for k.** Operator selection has ~60
  candidates where stability matters → PRESS. Prefix `k` has ~25
  candidates where selection noise matters less and CV is affordable →
  5-fold CV. This is the candidate most likely to win on both
  stability and wall-clock.

**Hypotheses.**

- H1.a — PRESS is *at least as good* as the 20% holdout on mean RMSE
  across the 60-dataset corpus, with significant wins on small-n
  (`n < 80`) datasets where the 12-sample holdout is noisiest.
- H1.b — The hybrid (PRESS for operator, 5-fold CV for `k`) dominates
  pure PRESS on datasets with strong prefix sensitivity (those with
  chemically noisy spectra where `k` is hard to pin down), and ties
  pure PRESS elsewhere. Wall-clock stays within 1.5× of pure PRESS
  because CV runs on only 1 operator, not 60.
- H1.c — Pure 5×2 CV is the ground-truth reference but is 5–10× slower
  than any of the other three. It does not become the default; it
  is a sanity check that the default's rank is within noise of
  ground truth.
- H1.d — All three non-holdout criteria produce *more stable operator
  choices across random seeds* than the 20% holdout (trivially for
  PRESS, empirically for CV at k≥5).

**Deliverables.**

- `bench/AOM/W1_selection_criterion/` benchmark script running the 60
  datasets through five configurations:
  1. `selection_criterion="holdout_20"` — baseline being replaced.
  2. `selection_criterion="press"` — full PRESS for both operator and `k`.
  3. `selection_criterion="cv5"` — 5-fold CV for both.
  4. `selection_criterion="hybrid"` — PRESS for operator, 5-fold CV for
     `k`. *Primary candidate for the new default.*
  5. `selection_criterion="cv5x2"` — ground-truth reference, run on a
     subset of 20 datasets for wall-clock reasons.
- Per-dataset RMSE, selected operator, selected `k`, wall-clock, plus
  cross-seed operator-stability index for each configuration.
- Aggregate Friedman–Nemenyi ranks across the four fast configurations.
- Decision memo naming the winning default.

**Acceptance criteria for the new default.**

- Mean Friedman rank at least as good as the 20% holdout.
- No dataset shows a statistically significant regression
  (Nadeau–Bengio p < 0.05) beyond a pre-specified 3% RMSE margin.
- Wall-clock within 2× of the 20% holdout on datasets with `n < 500`.
- Operator-stability index (fraction of 10 seeds agreeing on the winning
  operator) ≥ 0.9. Holdout-20 is expected to fall below this threshold;
  this is the failure mode we are documenting.

**Risks.**

- PRESS on small-n datasets with `p ≫ n` is numerically fragile. The
  POP-PLS implementation uses the PLS score-space hat matrix which
  sidesteps this; port it verbatim.
- Hybrid has two knobs (operator criterion, `k` criterion) which could
  be over-tuned to the corpus. Mitigation: fix the hybrid recipe
  *before* running the benchmark and report it as a pre-registered
  configuration.
- If the hybrid wins by a hair over pure PRESS, Occam's razor says
  ship PRESS. Decision threshold: hybrid must beat pure PRESS by
  ≥ 1% mean RMSE *or* ≥ 0.1 in Friedman rank *or* on ≥ 5% of the
  corpus with significance. Otherwise PRESS wins on simplicity.

**Dependencies.** None. First workstream.

**Feeds into.** W2 (confirms the holdout's documented weakness before
removing it), W3 (prefix selection uses the winning criterion), W5
(the unified estimator exposes whichever criterion wins as the default
`selection_criterion`).

---

## W2. Holdout removal — confirmatory measurement

**Status.** The 20% holdout is being removed from AOM-PLS. This
workstream exists to produce the evidence cited in the paper when we
remove it. Outcome is pre-committed; measurement is not.

**Questions the paper needs answered.**

- By how much is the 20% holdout *worse* than the winning criterion
  (from W1) on the 60-dataset corpus? Answer in mean rank, median ΔRMSE,
  number of datasets regressed.
- What is the *seed sensitivity* of the 20% holdout? Report the RMSE
  range across 10 seeds per dataset. This is the number chemometricians
  will cite as the reason to prefer deterministic selection.
- Were any datasets *actually helped* by the accidentally-lucky
  `RandomState(42)` permutation? This is a fair-play check: rerun with
  10 different seeds and see whether any cases change the winning
  operator.

**Deliverables.**

- A short report (one page) with:
  - Friedman–Nemenyi plot of the 20% holdout versus the W1 winner.
  - Table of datasets where the holdout gives statistically different
    RMSE from the W1 winner (in either direction).
  - Plot of RMSE spread across 10 seeds per dataset for the holdout
    path.
- Code deletion PR that removes the internal-holdout code path from
  `_aompls_fit_numpy` once the report is published. External `X_val`
  support stays (users who pass validation data still benefit from
  external selection).

**Acceptance criteria.**

- Report published alongside the W1 memo.
- Removal PR merged. `aom_pls.py` no longer contains a `RandomState(42)`
  internal split.

**Dependencies.** W1 (winning criterion is the comparison baseline).

---

## W3. `max_components` + `n_components` API split and auto-selection study

**Decisions locked.** The constructor will expose two distinct
parameters:

```python
AOMPLSRegressor(
    max_components=25,              # default TBD: 25 or 30 — decided in W3
    n_components="auto",            # "auto" or an integer
    selection="global",             # "global" (AOM) or "per_component" (POP) — see W5
    selection_criterion="press",    # or "cv5", "hybrid" — winner from W1
    operator_bank="default",        # see W6
    ...
)
```

Semantics:

- `max_components` — the hard ceiling on NIPALS extraction. Always a
  cap. Also capped by `min(n_samples − 1, n_features)` internally
  (unchanged behaviour from today).
- `n_components="auto"` — the algorithm picks the prefix `k*` via the
  `selection_criterion` over `k ∈ [1, max_components]`. This is the
  default and the common path.
- `n_components=<int>` — **forced**. No prefix search. NIPALS extracts
  exactly that many components (capped by `max_components` if the user
  passes an inconsistent pair; warn if so).
- `selection` — already documented in W5. Governs whether one operator
  is picked for all components (`"global"`, AOM behaviour) or a
  different operator per component (`"per_component"`, POP behaviour).
- `selection_criterion` — how the algorithm scores candidates. Chosen
  by W1.

**The study's two real questions.**

1. What is the right default for `max_components`?
   - Too small → silently caps the algorithm at a suboptimum on complex
     spectra.
   - Too large → NIPALS cost and PRESS hat-matrix cost grow linearly.
   - The answer depends on the 60-dataset corpus empirics: what fraction
     of datasets select `k* ≥ 20`? If negligible, `max_components=20`
     is defensible. Current default (15) is plausibly too low.
2. Does `n_components="auto"` need an escape hatch for the user?
   - In the Optuna use case, users tune `n_components` externally. That
     becomes a literal integer — no special hook needed. This is
     simpler than the current `operator_index` Optuna piggy-back.
   - In the "I always use k=10 because my domain expert says so" use
     case: same, pass `n_components=10`.

**Hypotheses.**

- H3.a — On the 60-dataset corpus, the empirical distribution of
  auto-selected `k*` is right-skewed with median around 6–10 and a
  long tail to 20+. A `max_components` of 25 covers ≥ 99% of datasets;
  30 buys essentially nothing extra. Propose **25** as the default.
- H3.b — Forced `n_components=<int>` (no search) loses ~3–8% mean RMSE
  against `"auto"` on the corpus, because no single integer fits all
  datasets. This justifies `"auto"` as the default.
- H3.c — External Optuna tuning of `n_components` (treating it as a
  categorical integer) gains < 1% RMSE over `"auto"` at 10× wall-clock.
  Not recommended as default; worth mentioning as an option for
  benchmarking rigour.

**Deliverables.**

- Constructor refactor: split `n_components` into `max_components` +
  `n_components: int | "auto"`. Update `_aompls_fit_numpy` so the three
  code paths (hard gate, Optuna hook, sparsemax) collapse into one
  canonical flow:
  1. Run NIPALS up to `max_components` for each operator.
  2. If `n_components == "auto"`: score all `(operator, k)` pairs with
     `selection_criterion` and pick the argmin.
  3. If `n_components == <int>`: skip prefix search, keep exactly that
     many.
  4. Trim and store.
- Remove the `operator_index` Optuna-piggy-back parameter (replaced by
  a combination of `operator_bank=[single_operator]` + `n_components=<int>`).
- Update the docstring to distinguish auto vs forced cleanly. One
  paragraph explaining: "`max_components` bounds work; `n_components`
  decides whether the algorithm picks or the user picks."
- Histogram of auto-selected `k*` across the 60-dataset corpus.
  Decision memo on `max_components` default (25 vs 30).
- Per-dataset ΔRMSE of three policies: `"auto"`, forced-median (the
  median of auto-selected `k*` applied uniformly), forced-domain
  (e.g., `n=10`).
- Deprecation note for `operator_index` (removed, not backward-compat
  shimmed — consistent with the "no deprecated code" project policy).

**Acceptance criteria.**

- One code path in `_aompls_fit_numpy` for all gates and both
  `n_components` modes.
- `max_components` default supported by the empirical distribution
  from the corpus.
- Documentation addition: a short "when to let AOM-PLS choose, when to
  force it" section with the corpus-backed recommendation.

**Dependencies.** W1 (the `selection_criterion` default must be set
before this workstream tests `"auto"`).

---

## W4. Extending the bank — OSC, EPO and the "linear at apply time" candidates

**Key distinction, settled up front.** SNV, MSC and OSC are *not*
interchangeable for bank membership. Their linearity differs:

| Operator | Linear at apply time? | Data dependency at apply time | Bank-eligible? |
|---|---|---|---|
| **SNV** | **No** | Divides by per-sample `std(x)`, which is non-linear in `x` | No |
| **MSC** | **No** | Divides by per-sample OLS slope `a_i` fitted against the reference spectrum | No |
| **OSC (DOSC)** | **Yes** | Fixed `P_o` stored at fit time; `x → x (I − P_o P_o^T)` at apply time | **Yes**, with a supervised-operator caveat |
| **EPO** | **Yes** | Fixed external-block projection stored at fit time | **Yes**, same caveat |
| **Wavelet approximation projection** | Yes | Fixed wavelet filter bank, no data dependency | Already in extended bank |
| **FFT bandpass** | Yes | Fixed frequency mask | Already in extended bank |
| **Whittaker / ArPLS baseline** | Yes (once `λ` is fixed) | Banded-matrix inverse applied identically to every sample | **Probably yes** — candidate to test |
| **Area / max / min normalisation** | No | Divides by per-sample statistic | No |
| **Kubelka–Munk, log(1/R)** | No | Per-sample non-linear transform | No (signal-type conversion, belongs upstream) |

SNV and MSC remain *non-linear*. They cannot be included in the bank as
strict linear operators. The pseudo-linear SNV adjoint was tried (§2.2 of
the plan) and rejected for exactly this reason. **The paper's
recommendation for SNV/MSC is upstream placement or branch+merge
stacking, not bank inclusion.** This workstream concerns the *linear-at-
apply* candidates: OSC, EPO, Whittaker, and confirming the promotion of
wavelet/FFT operators from the extended bank to the default.

**The "supervised-operator caveat."** OSC and EPO store their projection
matrices during `fit(X, y)` using information from `y` (for OSC) or from
an external block (for EPO). At apply time they are linear, but they
cannot be initialised independently of the data the way SG or detrend
can. Concretely: the bank cannot pre-build them with just `p` — they
need `(X, y)` at fit time. Two consequences:

1. **API change.** `LinearOperator.initialize(p)` becomes insufficient.
   We need an optional `fit(X, y)` step for supervised operators. The
   `PseudoLinearSNVOperator` prototype already did something similar
   (`bench/AOM/pseudo_linear_aom.py:26`). Standardise that into the
   `LinearOperator` ABC.
2. **Guarantee restated.** `AOM-PLS ≥ PLS` now says: on the *selection
   criterion*, AOM-PLS dominates PLS. If the criterion is PRESS and the
   bank contains supervised operators, PRESS is computed on the same
   data OSC was fit on, which is consistent (both use the same
   training set); no information leak. If the criterion is a holdout
   or CV, OSC must be re-fitted inside each fold to avoid the leak.
   This is standard CV hygiene; the W1 implementation must honour it.

**Signal-type sub-experiment.** NIRS data come in reflectance (R),
absorbance (A = log(1/R)), transmittance, and Kubelka–Munk forms. The
transform from raw R to A is non-linear and always applied upstream —
nirs4all already supports this via `SignalType` detection and
`convert_to_absorbance()`. But the *optimal operator selected by AOM-PLS
may differ between R and A* for the same dataset: on reflectance,
derivatives emphasise scattering; on absorbance, derivatives align with
chemical bands. The paper should measure this.

**Hypotheses.**

- H4.a — OSC as a linear operator in the bank (with `n_orth ∈ {1, 2, 3}`
  giving three bank entries) wins statistically on ~15–25% of the
  60-dataset corpus — those with strong systematic scatter, temperature
  drift, or batch effects — without regressing on the rest.
- H4.b — EPO helps on the subset of corpus datasets where a donor/target
  instrument pair exists or where replicate blocks are available. On
  the rest it is not applicable, so it's a conditional operator: bank
  entry is added at construction time only when an EPO reference block
  is provided.
- H4.c — Whittaker baseline removal (at two or three `λ` presets) adds
  marginal signal over `DetrendProjectionOperator(degree={1,2})` on
  spectra with curved non-polynomial baselines; neutral elsewhere.
  Decision is pure ΔRMSE on the corpus.
- H4.d — Promoting wavelet approximation and FFT bandpass from the
  extended bank to the default bank improves coverage on ~5–10% of
  the corpus (high-noise or strong-baseline spectra) and does no harm
  elsewhere. The only reason not to promote them is initialisation
  cost: `WaveletProjectionOperator` builds a `p × p` projection matrix
  at init, which is `O(p²)` per operator. W4 must benchmark init cost
  on the largest `p` in the corpus.
- H4.e — Running AOM-PLS on absorbance rather than reflectance changes
  the winning operator on ≥ 20% of corpus datasets where both domains
  are available, and shifts RMSE by ≥ 2% on half of those. This is the
  "your signal-type choice matters, here's by how much" result.

**Deliverables.**

- **Bank additions:**
  - `OSCOperator(n_components)` as a supervised `LinearOperator`.
    Forward: `x (I − P_o P_o^T)`. Adjoint: same (projection is
    symmetric). `fit(X, y)` stores `P_o`.
  - `EPOOperator(reference_block)` — same structure, projection from
    an external block.
  - `WhittakerBaselineOperator(lam)` — banded solve, linear at apply.
    Test at `lam ∈ {1e4, 1e6, 1e8}`.
  - Extend `LinearOperator` ABC with an optional
    `fit(X, y=None) -> self` hook. Document the supervised vs
    unsupervised contract.
- **Non-inclusion confirmation.** A short section in the paper
  explicitly stating *why SNV and MSC do not enter the bank* with the
  §2.2 rejection as evidence. No code change.
- **Experiments:**
  1. Bank `{default}` vs `{default + OSC×3}` on all 60 datasets.
  2. Bank `{default + OSC}` vs `{default + OSC + wavelet + FFT}` —
     promotion test.
  3. Bank `{default + OSC + Whittaker}` vs `{default + OSC}` —
     Whittaker value test.
  4. Signal-type sub-experiment: for the subset of corpus datasets
     where R and A are both derivable, run the chosen W4 bank on
     both and report operator-choice divergence and ΔRMSE.
  5. Reference baseline: branch+merge stacking
     `[SNV → AOM-PLS, AOM-PLS, MSC → AOM-PLS] → Ridge`. This is the
     "what if the user handles SNV/MSC outside" control.
- **Decision memo** on the new default bank content.
- **Paper section:** "The linear-at-apply principle and why SNV/MSC
  are handled via stacking."

**Acceptance criteria.**

- Each proposed bank extension (OSC, Whittaker, wavelet-promotion,
  FFT-promotion) has a pre-registered Friedman-Nemenyi evidence for
  inclusion or rejection.
- Signal-type sub-experiment produces a one-paragraph finding with a
  number — this goes straight into the paper's discussion.
- `LinearOperator` ABC documents the supervised contract cleanly.

**Dependencies.** W1 (selection criterion — supervised operators need
leakage-free evaluation), W3 (`max_components` must be set before
wavelet promotion is benchmarked, since wavelet's init cost compounds
with extraction depth).

---

## W5. AOM/POP unification — `AOMPLSRegressor(selection=...)`

**Decision locked.** The joint algorithm **keeps the `AOMPLSRegressor`
name** in the public API. The algorithmic behaviour is controlled by two
constructor parameters, which are the user-facing knobs for what the
model actually does:

- `selection` — *how many operators the model uses*.
  - `"global"` — one operator for all components (classical AOM behaviour).
  - `"per_component"` — possibly different operator per component
    (classical POP behaviour).
  - `"auto"` — data-driven pick between the two using
    `selection_criterion`, if W5 demonstrates it works. Otherwise
    dropped.
- `selection_criterion` — *how the model scores candidates*. Winner of W1
  (`"press"`, `"cv5"`, `"hybrid"`, etc.). Shared between operator
  selection and, if `n_components="auto"`, prefix selection.

**Documentation mandate.** The AOMPLSRegressor docstring must open with
a clear statement of what `selection` and `selection_criterion` change
in the algorithm's behaviour, because renaming at the parameter level
(rather than the class level) makes the parameters the primary
documentation surface. The docstring paragraph is itself a W5
deliverable.

**POPPLSRegressor.** Remains in the library as a convenience wrapper
that sets `selection="per_component"` (and POP's compact bank as a
default preset). It stays for discoverability; it is not a separate
model. The paper presents `AOMPLSRegressor` with its `selection` knob
as the canonical artefact.

**Context.** AOM-PLS and POP-PLS share ~80% of infrastructure (linear
operator bank, adjoint trick, NIPALS deflation, OPLS pre-filter). The
difference is the granularity of the operator commitment. Merging them
is primarily code hygiene; the scientific contribution is studying when
one policy is preferred.

**Sub-questions.**

- On the 60-dataset corpus, what fraction of datasets benefit from
  `selection="per_component"` over `"global"`, and by how much?
  (Prior informal runs suggest ~30% but this needs a clean measurement
  under the W1 criterion.)
- Can `"auto"` pick the right policy from the training data alone, or
  does the selection noise it introduces wipe out the gains?
- Is there a simple predictor — e.g. "the winning operator for
  components 1 and 2 differs" — that predicts `"per_component"` wins
  without actually running both policies?

**Hypotheses.**

- H5.a — `"per_component"` wins over `"global"` on ~25–35% of datasets,
  loses on ~20%, is equivalent on the rest. The win/loss split is
  driven by whether the dominant operator changes across components.
- H5.b — A PRESS-based `"auto"` selector picks the right policy
  often enough that it ties or beats both individual policies in
  mean rank on Friedman-Nemenyi.
- H5.c — The unified code path does not cost appreciably more
  wall-clock than either policy alone: PRESS scores are computed once
  per `(b, k)` candidate and `"global"` is a trivial subset of the
  `"per_component"` candidate set.

**Deliverables.**

- Refactor `_aompls_fit_numpy` and the POP-PLS equivalent into a single
  `_adjoint_pls_fit_numpy` dispatching on `selection`. Keep the public
  `AOMPLSRegressor` class; rewire its `fit()` to call the unified
  backend.
- Reduce `POPPLSRegressor` to a thin wrapper class: forwards to
  `AOMPLSRegressor(selection="per_component", operator_bank="compact")`
  at construction.
- Implement `selection="auto"`: for each candidate policy, score the
  whole operator bank under `selection_criterion`; pick the policy
  whose best candidate wins. This is a one-line outer loop around the
  existing per-policy scan.
- Parity tests: `AOMPLSRegressor(selection="global")` and the current
  AOM-PLS produce identical predictions on fixed-seed datasets. Same
  for `"per_component"` vs current POP-PLS.
- Paper-quality docstring for `AOMPLSRegressor` that leads with
  `selection` and `selection_criterion` and explains what they actually
  change. (This is the "enforce the documentation" deliverable.)
- Benchmark on the 60-dataset corpus: `{global, per_component, auto}`
  × `{whatever selection_criterion wins W1}`. Report wins, losses,
  ties with Nadeau–Bengio significance.

**Acceptance criteria.**

- Parity tests pass on fixed seeds for both `"global"` and
  `"per_component"`.
- One code path (`_adjoint_pls_fit_numpy`) replaces the two.
- `AOMPLSRegressor` docstring clearly distinguishes the four knobs
  (`max_components`, `n_components`, `selection`, `selection_criterion`)
  and their interactions.
- If `"auto"` does not statistically beat `"per_component"` + bank
  defaults, drop it from the shipped API and mention it only as a
  tried-and-rejected experiment in the paper.

**Risks.**

- Scope creep. If `"auto"` becomes a research project, it could delay
  the paper. Mitigation: ship `"global"` and `"per_component"` first
  (trivial refactor); measure `"auto"` on the corpus; only ship it if
  it clears the acceptance bar.
- Parameter interaction confusion. Four knobs
  (`max_components`, `n_components`, `selection`, `selection_criterion`)
  is at the edge of what a clean API can support. Mitigation:
  aggressive docstring discipline and one example per common
  configuration in `examples/user/`.

**Dependencies.** W1 (selection criterion default), W3 (unified
`n_components` semantics), W4 (unified bank semantics across both
policies).

---

## W6. Custom operator banks — public API for user-selected preprocessings

**Question.** Can users specify *which* preprocessings AOM-PLS / POP-PLS /
the unified estimator manages, as a first-class parameter of the
estimator, without having to subclass or monkey-patch?

**Context.** Today the bank can be overridden via `operator_bank=list(...)`
in the constructor (`aom_pls.py:1314`). This works but requires the user
to import the internal `LinearOperator` classes from
`nirs4all.operators.models.sklearn.aom_pls`. There is no recipe-oriented
API like "use SG derivatives only" or "use the POP-PLS compact bank" or
"use AOM default minus composed operators." Users who want to experiment
with custom banks currently reach for the `enhanced_aom` monkey-patch
pattern (`bench/AOM/enhanced_aom.py`), which is not something the paper
can recommend.

**Requirements.**

1. **Named presets.** At least:
   - `"default"` — current deployed bank (possibly revised by W4).
   - `"compact"` — the POP-PLS 9-operator bank
     (`pop_pls.py:68, pop_pls_operator_bank()`).
   - `"sg_only"` — Savitzky–Golay family only.
   - `"derivatives_only"` — SG, finite difference, Norris–Williams.
   - `"extended"` — default + supervised operators added by W4 (OSC,
     EPO if applicable) + promoted wavelet/FFT operators.
2. **Family-level additive/subtractive selection.**
   ```python
   AOMPLSRegressor(
       selection="global",
       selection_criterion="press",
       operator_bank=OperatorBankSpec(
           base="default",
           include=["norris_williams", "osc"],
           exclude=["composed"],
       ),
   )
   ```
   where `include` and `exclude` name families: `sg_smoothing`, `sg_d1`,
   `sg_d2`, `detrend`, `finite_difference`, `norris_williams`,
   `composed`, `wavelet`, `fft`, `osc`, `epo`, `whittaker`.
3. **Explicit operator override** (already exists — passing a
   `list[LinearOperator]` directly — keep it as the escape hatch).
4. **Introspection.** `estimator.bank_summary()` returns a DataFrame of
   the bank as actually used, with family, parameters, and (after fit)
   selection frequency / gamma weights. This is a documentation aid;
   chemometrics readers will ask for it.

**Hypotheses.**

- H6.a — The compact bank wins on small-n datasets (less selection
  noise), the default bank wins on medium-n, and the extended bank
  wins on rare cases with high-frequency noise (wavelet) or strong
  baseline (FFT). The default-by-corpus-size recommendation is worth a
  paragraph in the paper.
- H6.b — No single bank is uniformly best. This makes the
  family-level API a publication-relevant artefact: the paper should
  show the sensitivity curve and recommend `"default"` as a safe
  starting point.

**Deliverables.**

- `OperatorBankSpec` dataclass with `base`, `include`, `exclude`, and
  `custom_operators` fields. Resolves to a concrete list of
  `LinearOperator` instances at fit time.
- Preset catalogue (5 named banks minimum).
- `bank_summary()` method.
- Docstring additions explaining when to pick which preset.
- Benchmark on the 60-dataset corpus: compare the 5 presets as the
  default operator_bank. Report per-preset rank distributions.

**Acceptance criteria.**

- Public API documented, tested, and used by at least one example in
  `examples/user/`.
- Benchmark produces a defensible recommendation for each dataset-size
  regime.

**Dependencies.** W5 (the unified estimator is the natural home for
this API, so ideally W6 lands on top of W5). If W5 is delayed, W6 can
ship on `AOMPLSRegressor` directly and be lifted later.

---

## W7. Statistical machinery and reproducibility package

**Scope.** Cross-cutting infrastructure that every other workstream
will emit results into. Best done once up front so all workstreams
share the same machinery.

**Deliverables.**

- `bench/AOM/stats/` helpers:
  - `nadeau_bengio_ttest(rmse_a, rmse_b, n_train, n_test)` — corrected
    resampled t-test (JMLR 2003).
  - `friedman_nemenyi(rmse_matrix, model_names)` — Friedman test with
    Nemenyi post-hoc, plus a CD-diagram plotter (Demšar, JMLR 2006).
  - `bootstrap_ci(rmse_vec, n_boot=10000, alpha=0.05)` — per-dataset
    CIs on RMSE.
- Standard benchmark driver: reads the 60-dataset corpus, runs a
  configurable model list, writes per-dataset RMSE + wall-clock to a
  single CSV that all workstreams append to. No one reimplements the
  run loop.
- Reference-score tracker: for each of the 60 datasets, the corpus
  already has a reference RMSE. Every benchmark row reports
  `ΔRMSE = RMSE_model − RMSE_reference`. Aggregate: "W1 changed the
  mean `ΔRMSE` from X to Y" becomes a single number the paper can
  cite.
- CD diagrams (critical-difference) as SVGs, one per workstream.

**Acceptance criteria.**

- Each workstream's benchmark emits rows into the same CSV schema.
- Stats helpers are covered by unit tests (easy: compare against
  published worked examples).
- CD diagrams are reproducible from the CSV alone.

**Dependencies.** None. Start this in parallel with W1.

---

## W8. Paper drafting

Not a research workstream — listed for completeness so the publication
timeline closes.

**Deliverables.**

- Draft outline (can steal structure from `PUBLICATION_PLAN.md` §1).
- Sections: introduction, related work (AOM prior exploration + POP-PLS
  + FCK-PLS + MoE + DARTS), method (adjoint trick formalisation,
  identity-dominance proposition, selection policies), experiments
  (60-dataset corpus, statistical tests, ablations from W1–W6),
  discussion (caveats from §5.5 of plan), conclusion.
- Figures: method diagram, CD diagrams from W7, example of
  `get_preprocessing_report()` output on one dataset.
- Supplementary material: per-dataset results, benchmark scripts,
  reproducibility README.

**Dependencies.** W1–W6 complete. W7 provides all figures.

---

## Cross-workstream risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Hybrid (PRESS for operator, CV for k) wins W1 by a marginal amount — harder to justify than pure PRESS | Medium — affects paper narrative | Pre-register a significance threshold for hybrid > pure PRESS; if not met, ship pure PRESS on simplicity grounds |
| 60-dataset reference scores use a slightly different benchmark protocol than this paper (different splitters, different preprocessors in the reference pipeline) | Medium — comparisons are apples-to-oranges | Run a sanity pass first: re-run the reference pipeline on all 60 datasets under the *paper's* protocol, produce a "reference-protocol-aligned" column that every workstream compares against |
| W4 confirms OSC helps — then identity-dominance needs restating under "consistent evaluation" constraint | Low — guarantee becomes conditional but still holds | Paper clearly states the two-tier contract: strict linearity → unconditional `AOM-PLS ≥ PLS` on the criterion; supervised linearity → same guarantee conditional on leakage-free evaluation (PRESS satisfies this; CV must refit supervised ops per fold) |
| `selection="auto"` (W5) does not clear the acceptance bar | Low — fallback is clean | Ship `"global"` and `"per_component"` only; mention `"auto"` in the paper's future work |
| Custom bank API (W6) breaks backward compat with existing YAML pipelines | Low | Keep `operator_bank=list(...)` as a supported form; `OperatorBankSpec` is additive. `operator_bank="default"` is the new string-preset default |
| `max_components` default of 25 cuts off ~1% of corpus datasets that need `k > 25` | Very low | W3 histogram dictates the default; ship `max_components=30` if the tail is meaningful |

---

## Timeline sketch (not a commitment)

Month 1: W7 (stats/infra) + W1 (selection criterion head-to-head) +
reference-protocol alignment pass on the 60-dataset corpus.

Month 2: W2 (holdout removal confirmatory report + PR) + W3
(`max_components` + `n_components` API split and auto-selection study).

Month 3: W4 (OSC / EPO / Whittaker bank extensions, signal-type
sub-experiment) + W6 (custom banks, preset + include/exclude API).

Month 4: W5 (`selection`/`selection_criterion` unification in
`AOMPLSRegressor`, parity tests, documentation).

Month 5–6: W8 (paper drafting, revisions, supplementary material).

This is a ~6-month track to submission assuming no major surprises. W1
is the one whose outcome could meaningfully compress the schedule (if
pure PRESS wins outright, hybrid variants collapse to a paragraph and
W5's `"auto"` becomes trivial).
