# Custom CV splitter for AOM-PLS — Design Brief

Date: 2026-04-29 (revised after codex review)

## Revisions from codex review

1. **`get_n_splits(X, y, groups=None)` is preferred over `.n_splits`** — it
   is the sklearn `BaseCrossValidator` contract that nirs4all also
   honours. We probe `get_n_splits` first and fall back to
   `.n_splits` only if missing.
2. **Two CV paths, not one**. The fixed-sequence / POP scorer flows
   through `aompls.scorers.cv_score_regression`, not
   `_cv_score_per_prefix`. Both must accept `cv_splitter`.
3. **`repeats > 1` semantics revised**. Cloning with a new
   `random_state` is only valid for splitters whose split logic
   actually depends on it (KFold yes; SPXYFold no — its
   `random_state` only seeds PCA). For V1 we forbid `repeats > 1`
   together with a custom splitter and raise a clear error. Users who
   want repeated SPXY-style CV should pass a factory wrapper or a
   `RepeatedKFold`-style wrapper externally.
4. **Variant labels encode the splitter spec**. Resume key is
   `(database, dataset, model, seed)`, so the variant label must carry
   every splitter parameter that affects results
   (`SPXY-AOM-compact-cv5-numpy` already does for the headline case).
   The `cv_splitter` column in diagnostics records `type_name(n_splits=k,
   y_metric=...)`.
5. **No silent fallback**. If the splitter raises on a fold, propagate
   the exception (rows are written with `status="failed"`); do not
   substitute KFold mid-run.
6. **Groups are explicit out-of-scope for V1**. The estimator and
   benchmark `fit(X, y)` paths do not plumb groups through. Document
   this and add a TODO; the splitter contract still includes the
   `groups=None` parameter so the future plumbing is forward-compatible.

## Problem

The AOM-PLS selection criterion currently uses a random `KFold(n_splits=cv,
shuffle=True, random_state=seed)`. For NIRS calibration this is suboptimal
because:

1. Random folds may have correlated samples (group leakage when there are
   repetitions of the same physical sample).
2. Random folds do not respect the chemistry / scatter distribution of
   the dataset, so the selector can pick an operator that won the random
   draw rather than the genuinely best operator.
3. nirs4all already ships better splitters (`KennardStoneSplitter`,
   `SPXYFold`, `SPXYGFold`, `KMeansSplitter`, `SystematicCircularSplitter`,
   `KBinsStratifiedSplitter`) that exploit `(X, y)` structure or group
   information.

We want to expose a knob so the user can pass a domain-aware splitter
to AOM-PLS, and keep the random KFold path as the backward-compatible
default.

## API contract

A "splitter" is anything sklearn-compatible:

- `.split(X, y, groups=None)` returns an iterator of
  `(train_idx, val_idx)` numpy index arrays
- `.n_splits` attribute (int) — number of folds the splitter will yield

That covers `sklearn.model_selection.{KFold, GroupKFold, ...}` and
`nirs4all.operators.splitters.{SPXYFold, SPXYGFold, KennardStoneSplitter,
KMeansSplitter, ...}`.

Optional:
- `.random_state` attribute — used by `repeats > 1` to seed independent
  draws.

## API surface

### `aompls.scorers.CriterionConfig`

```python
@dataclass
class CriterionConfig:
    kind: str = "cv"
    cv: int = 5                      # KFold n_splits (used iff cv_splitter is None)
    cv_splitter: Any = None          # NEW: optional sklearn-compat splitter
    repeats: int = 1
    one_se_rule: bool = False
    random_state: int = 0
    ...
```

### `aompls.estimators.AOMPLSRegressor`

```python
AOMPLSRegressor(
    ...
    cv: int = 5,
    cv_splitter=None,                # NEW: optional sklearn-compat splitter
    ...
)
```

Default `cv_splitter=None` keeps current `KFold(cv, shuffle, seed)` path
intact. Passing a splitter instance overrides `cv` (the splitter's own
`n_splits` is used).

### `aompls.selection._cv_score_per_prefix`

```python
def _cv_score_per_prefix(
    Xc, yc, fit_predict_per_prefix,
    n_splits, random_state, n_components,
    repeats=1,
    cv_splitter=None,               # NEW
):
    if cv_splitter is not None:
        splitter = cv_splitter
        n_splits = splitter.n_splits
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    ...
```

For `repeats > 1`:
- If `cv_splitter` has a writable `random_state`, clone the splitter
  `repeats` times with seeds `[random_state, random_state+1, ...]`.
- Else: warn once that `repeats` is ignored for this deterministic
  splitter, fall back to running it once.

### Benchmark variant config

A new `cv_splitter_factory` field, a callable `(seed: int) -> splitter`,
keeps the benchmark's resume mechanism working (variant labels are
strings, splitter instances are not JSON-serialisable):

```python
{
    "label": "SPXY-AOM-compact-cv5-numpy",
    "cv_splitter_factory": lambda seed: SPXYFold(n_splits=5, random_state=seed),
    ...
}
```

`_build_estimator` calls the factory with the run seed and passes the
resulting splitter to `AOMPLSRegressor(cv_splitter=...)`.

## Edge cases

1. **Splitter without `n_splits`**. Reject with `ValueError("splitter must
   expose n_splits")`. All sklearn / nirs4all splitters do.

2. **Splitter raises on small `n`**. Wrap with try/except in
   `_cv_score_per_prefix`; if all folds fail, fall back to KFold (the
   AOM selection cannot run otherwise). Log the fallback.

3. **`one_se_rule` interaction**. Unchanged: the rule uses per-prefix
   mean scores, which are computed identically regardless of splitter.

4. **Bit-parity**. Default `cv_splitter=None` preserves the KFold path
   bit-for-bit. Existing tests must still pass.

5. **JSON / CSV serialisation**. We store the splitter's type name and
   `n_splits` as a diagnostics dict entry
   (`"cv_splitter": "SPXYFold(n_splits=5)"`) so the result CSV remains
   parseable.

6. **Groups (rep_to_sources)**. Out of scope for this iteration. The
   `splitter.split(X, y, groups=None)` call signature is forwarded; if
   the user provides a group-aware splitter we'd need to plumb groups
   through. TODO marker only.

## Variants to add

For the SPXYFold benchmark, three new variants on the top-3 banks, all
with ASLS preprocessing (the winning recipe from the previous run):

- `SPXY-AOM-compact-cv5-numpy`     (5-fold SPXYFold + ASLS + compact bank)
- `SPXY-AOM-response-dedup-cv3-numpy` (3-fold + ASLS + response-dedup)
- `SPXY-AOM-family-pruned-cv3-numpy`  (3-fold + ASLS + family-pruned)

The hypothesis the user wants to test: if the per-dataset variance of
the random-CV selection is the bottleneck (as the HPO failure suggests),
a chemistry-aware splitter should give a more stable selection criterion
without needing per-dataset HPO.

## Caveat: criterion-score comparability across splitters (added in review)

SPXYFold's farthest-from-centroid initialisation and max-min assignment
produce folds whose held-out points are interpolation-like rather than
extrapolation-like. Empirically, this lowers fold-RMSE relative to a
random KFold split on the same data, so **the inner-CV criterion score
is not comparable across splitters**.

Implications:

- The **selector's argmin** (which operator is picked) can differ
  between SPXYFold and KFold even on the same data — that is the
  intended effect.
- The **test-set RMSEP** comparison (variant vs PLS-standard on the
  fixed train/test split defined by the cohort) is unaffected: the
  test set is independent of the inner CV.
- Result tables that mix SPXY-CV and KFold-CV variants must compare
  them on test RMSEP, not on the internal CV score, and should flag
  the splitter explicitly.

## Risks

1. **Computation**: SPXYFold has a `O(n^2 * p)` distance computation. For
   LUCAS_SOC_Cropland (n=6111, p=4200) the distance matrix alone is ~150
   MB and ~25 s to compute. Acceptable on the cohort.

2. **Tied folds on duplicates**: SPXY assignment can give degenerate
   folds if many samples are exact X duplicates. nirs4all's
   implementation handles this; we just propagate failures up.

3. **Comparability with master cohort**: TabPFN-paper Ridge / TabPFN-opt
   used random CV inside their HPO. Switching the AOM CV to SPXY
   changes one factor at a time, which is the right ablation. Just
   needs to be flagged in the writeup.

## Test plan

1. Unit test: `_cv_score_per_prefix` with `cv_splitter=KFold(...)`
   matches the integer-`cv` path bit-for-bit.
2. Unit test: `_cv_score_per_prefix` with `cv_splitter=SPXYFold(...)`
   produces deterministic output with fixed `random_state`.
3. Smoke benchmark: run on Beer (n=40) with both KFold-CV5 and
   SPXYFold-CV5; SPXY should give a different selected operator on at
   least one fold, but the test-set RMSE should be comparable.
4. Full 57-cohort: 3 SPXY variants × 57 datasets ≈ 171 runs; budget
   30-60 minutes (compact bank is fast, SPXY adds ~2-5 s/fit).
