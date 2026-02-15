# AOM-PLS: Adaptive Operator-Mixture Multi-Block PLS (Torch Alternative) - Updated Design

## 1) Goal and scope

Implement a PLS-family estimator for NIR spectra that learns preprocessing blocks in one training run over a **linear operator bank**:

- preserve PLS interpretability (components, loadings, scores)
- avoid materializing all preprocessed views `X_b`
- expose per-component block weights `gamma[k, b]`
- provide a pragmatic path: CPU-first core, Torch backend second

Target data:

- `X in R^(n x p)` (samples x wavelengths)
- `y in R^n` (scalar regression), extendable to `Y in R^(n x q)`

Scope of v1:

- linear operator bank: SG + detrend + identity
- sparse block gating (sparsemax first)
- SIMPLS-style predictive components
- scatter correction handled outside the estimator (pipeline-level)

---

## 2) Review claims adopted (with justification)

The following review claims are incorporated because they are technically correct and implementation-relevant.

| Review claim | Decision | Justification |
| --- | --- | --- |
| SG adjoint can be wrong at boundaries for non-linear padding semantics | Adopted | Exact adjoint guarantees require explicit linear padding semantics. v1 defines bank operators with fixed linear padding (`same` + zero padding) and tests `<Ax,y> == <x,A^T y>`. |
| Raw block score `||g_b||^2` is scale-biased across operators | Adopted | Derivative operators can dominate by norm only. v1 uses normalized scoring to make gating comparable across blocks. |
| Identity/no-preprocessing block is required | Adopted | Guarantees reduction to standard PLS and gives a hard sanity baseline. |
| SIMPLS + sparse gating has no known convergence theorem | Adopted | We frame AOM-PLS as a heuristic PLS variant and validate empirically with acceptance gates. |
| Deflation behavior differs from classical MB-PLS block-wise deflation | Adopted | Document now states this explicitly: global predictive deflation in original space, not per-block MB-PLS deflation. |
| Scatter transforms are not linear operators in this framework | Adopted | v1 moves scatter selection upstream to existing pipeline branching/search. |
| Start with sparsemax before entmax | Adopted | Sparsemax is dependency-light and enough for v1 sparsity behavior. |
| Skip mixed precision in v1 | Adopted | Numerical stability is more important than speed at NIRS scale; v1 uses float64 (NumPy) and float32 (Torch backend) only. |
| Benchmark before GPU-heavy investment | Adopted | Phase gates added: Torch backend only after CPU core proves value. |
| OPLS ordering was ambiguous | Adopted | v1 defines OPLS as an optional pre-filter before predictive AOM components. |

Claims not adopted as mandatory in v1:

- "Only implement as MBPLS extension, not a new estimator": deferred. A standalone estimator is clearer for API and experiments.
- "PRESS-only model selection": deferred. Validation-prefix selection is simpler first; PRESS can be added later.
- "GPU is never useful for NIRS": rejected as absolute. Kept as conditional backend based on measured scaling.

---

## 3) Model definition

### 3.1 Operator-defined blocks

Each block is a linear operator on wavelengths:

- `X_b = X A_b`
- `A_b` from operator bank `B`

In v1:

- `IdentityOperator`
- `SavitzkyGolayOperator(window, polyorder, deriv, delta, mode="same_zero")`
- `DetrendProjectionOperator(degree in {1, 2})`
- compositions `A_b = D_b @ SG_b` and identity-only variants

### 3.2 Adjoint trick

For any linear `A_b`:

- `X_b^T u = (X A_b)^T u = A_b^T (X^T u)`

Per component, compute `c = X^T u` once, then apply all `A_b^T` cheaply.

### 3.3 What "one-shot" means now

One-shot applies to **linear operator selection and latent extraction**.
Scatter correction (`SNV`, `MSC`, `EMSC`) is outside v1 estimator and handled by pipeline composition/search.

---

## 4) Mathematical formulation

### 4.1 Predictive direction

Let centered/scaled `X` and residual target `y_(k-1)`.

- `c_k = X^T y_(k-1)`
- `g_(b,k) = A_b^T c_k`

### 4.2 Normalized block scoring (adopted)

To avoid operator-scale bias:

- precompute `nu_b = ||A_b||_F^2` (or empirical response norm)
- score:
  - `s_(b,k) = ||g_(b,k)||_2^2 / (nu_b + eps)`

Optional alternative (future): cosine-based score.

### 4.3 Sparse gating

- `gamma_k = sparsemax(log(s_k + eps) / tau)`
- `gamma_k` is simplex-constrained and sparse

### 4.4 Effective loading

- normalized block directions:
  - `w_hat_(b,k) = g_(b,k) / (||g_(b,k)||_2 + eps)`
- effective loading in original space:
  - `w_k = sum_b gamma_(b,k) * A_b * w_hat_(b,k)`
- component score:
  - `t_k = X w_k`

### 4.5 Deflation semantics (clarified)

v1 uses SIMPLS-style/global predictive deflation in original space.
This is **not** classical MB-PLS per-block deflation. Block weights remain interpretable diagnostics, but deflation is global.

### 4.6 OPLS ordering (clarified)

If enabled:

1. extract/remove `n_orth` orthogonal components from `X` (OPLS-style pre-filter)
2. run predictive AOM component extraction on filtered `X`

---

## 5) v1 algorithm

Inputs:

- `X` and `y` (already scatter-corrected by upstream pipeline if needed)
- operator bank `{A_b}_{b=1..B}` including identity
- `k_max`, `tau`, `n_orth`, `standardize`

Fit loop:

1. center/scale `X`, `y`
2. optional OPLS pre-filter (`n_orth > 0`)
3. for `k in 1..k_max`:
   - `c_k = X^T y_(k-1)`
   - `g_(b,k) = A_b^T c_k` for all blocks
   - `s_(b,k) = ||g_(b,k)||^2 / (nu_b + eps)`
   - `gamma_k = sparsemax(log(s_k + eps) / tau)`
   - build `w_k`, compute `t_k = X w_k`
   - orthogonalize/deflate with SIMPLS-style step
   - store prefix regression coefficients
4. evaluate prefixes `k=1..k_max` on validation criterion
5. choose `k*` and persist artifacts

Artifacts:

- `gamma_[k,b]`
- `w_[k,p]`, `t_[n,k]` (optional persistence controls)
- selected `k*` and optional `n_orth*`
- block metadata (`block_name`, operator params)

---

## 6) API proposal

`AOMPLSRegressor` (sklearn-compatible)

Core params:

- `n_components=25`
- `operator_bank` (explicit list or generated config)
- `gate="sparsemax"` (`"entmax"` optional later)
- `tau=1.0`
- `n_orth=0`
- `standardize=True`
- `selection="validation"` (future: `"press"`)
- `random_state=None`
- `backend="numpy"` (future: `"torch"`)

Core methods:

- `fit(X, y, X_val=None, y_val=None)`
- `predict(X)`
- `transform(X)` (scores)
- `get_block_weights()`
- `get_preprocessing_report()`

---

## 7) Implementation roadmap (repository-mapped)

This roadmap is ready for implementation in the current codebase.

### Phase 0: Scaffolding (must complete first)

Files:

- `nirs4all/operators/models/sklearn/aom_pls.py` (new)
- `nirs4all/operators/models/sklearn/__init__.py`
- `nirs4all/operators/models/__init__.py`

Tasks:

- add estimator skeleton (`AOMPLSRegressor`) with sklearn API
- add operator interfaces and identity operator
- export symbols in both `__init__.py` files

Definition of done:

- estimator instantiates, clones, `get_params`/`set_params` pass

### Phase 1: CPU core (value proof phase)

Files:

- `nirs4all/operators/models/sklearn/aom_pls.py`
- `tests/unit/operators/models/test_aom_pls.py` (new, preferred)
- `tests/unit/operators/models/test_sklearn_pls.py` (optional minimal import smoke)

Tasks:

- implement SG operator with explicit linear padding semantics
- implement detrend projection operator (degree 1/2)
- implement adjoint pair tests for each operator and compositions
- implement sparsemax gate
- implement normalized block scoring
- implement SIMPLS-style extraction loop with prefix selection
- implement report artifacts (`gamma_`, block rankings)

Definition of done:

- all new unit tests pass
- identity-only bank matches `SIMPLS` predictions within tolerance
- normalized scoring invariance test passes under operator rescaling

### Phase 2: Integration and usability

Files:

- `nirs4all/operators/models/sklearn/aom_pls.py`
- `nirs4all/operators/models/sklearn/__init__.py`
- `nirs4all/operators/models/__init__.py`
- `docs/` usage docs (new page or update existing model docs)

Tasks:

- add model metadata (`_webapp_meta`)
- document recommended pipeline composition for scatter handling:
  - branch/search over `StandardNormalVariate`, `MultiplicativeScatterCorrection`, `ExtendedMultiplicativeScatterCorrection`, `None`
  - then `AOMPLSRegressor`
- add example snippets for branch and cartesian workflows

Definition of done:

- model appears in imports and docs with reproducible example

### Phase 3: Benchmark gate (go/no-go for Torch backend)

Files:

- `bench/aom-pls/benchmark_aom_pls.py` (new)
- optional dataset configs under `bench/aom-pls/`

Must-run comparisons:

1. `AOMPLSRegressor` (identity+SG+detrend bank)
2. best cartesian preprocessing + `SIMPLS`/`PLSRegression`
3. `MBPLS` with explicit selected blocks
4. `FCKPLS` baseline

Gate criteria:

- quality: >= 3% RMSECV improvement over best cartesian baseline on at least 2 datasets, or equal quality with materially better runtime/interpretability
- stability: block rankings stable across folds (measured by rank correlation)
- runtime: CPU v1 no worse than 2x one best-fit PLS and clearly better than full grid search wall-clock

If gate fails, stop before Torch and reassess (possibly fold sparse gating into `MBPLS` instead).

### Phase 4: Torch backend (only after Phase 3 pass)

Proposed files:

- `nirs4all/operators/models/pytorch/aom_pls.py` (new backend core)
- `nirs4all/operators/models/sklearn/aom_pls.py` (backend dispatch)
- `tests/unit/operators/models/test_aom_pls.py` (backend parity tests)

Tasks:

- implement `apply_A_batch` / `apply_AT_batch` with `torch.nn.functional.conv1d`
- float32 default, no AMP in first Torch release
- parity tests versus NumPy backend on fixed seeds/data

Definition of done:

- prediction parity within tolerance against NumPy backend
- benchmark shows benefit for larger `(n, p, B, K)` regimes

---

## 8) Test plan (minimum set)

Unit tests:

- operator adjoint identity:
  - `|<A x, y> - <x, A^T y>| < tol`
- identity-only bank recovers SIMPLS behavior
- sparsemax properties:
  - simplex sum to 1
  - zeros appear for weak blocks
- normalized scoring reduces scale bias under synthetic rescaling
- sklearn compatibility (`clone`, `cross_val_score`)

Integration tests:

- pipeline with upstream scatter branch + `AOMPLSRegressor`
- serialization/deserialization with stored bank metadata and `gamma_`

Regression tests:

- deterministic output under fixed `random_state`
- stable selection of `k*` on fixed synthetic benchmark

---

## 9) Risks and mitigations

- Risk: no formal convergence guarantee.
  - Mitigation: explicit heuristic framing + empirical acceptance gates.
- Risk: operator implementation drift from transform semantics.
  - Mitigation: compare SG/detrend outputs to existing transform references and unit-test tolerances.
- Risk: limited gains over existing search pipelines.
  - Mitigation: enforce benchmark gate before Torch work.
- Risk: interpretability instability fold-to-fold.
  - Mitigation: add stability metrics and report confidence intervals.

---

## 10) Version plan

### v1.0 (implementation target)

- standalone `AOMPLSRegressor` with NumPy backend
- operator bank: identity + SG + detrend
- sparsemax gating with normalized scoring
- optional OPLS pre-filter (`n_orth`)
- validation-prefix model selection (`k*`)
- full tests and benchmark gate report

### v1.1

- entmax option
- optional region-mask operators (interval selection)
- improved reporting and visualization

### v2.0 (conditional)

- Torch backend (if Phase 3 gate passed)
- optional PRESS-based selection
- deeper architectural unification path with filter-bank estimators (`FCKPLS` family)

