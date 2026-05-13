# AOMPLS Operator Exploration

## Purpose

This document records the design direction for exploring much larger
preprocessing spaces in `bench/AOM_v0`.

The current AOM/POP framework uses a curated bank of strict linear spectral
operators. That is scientifically clean, but a bank of 10 to 60 operators may
be too small to explore the useful preprocessing space. The objective here is
to grow the candidate space aggressively while keeping the final model cheap,
interpretable, and compatible with the existing operator-adaptive PLS engines.

The central idea is:

```text
Generate many candidate operators.
Score them cheaply in covariance space.
Keep a small diverse active bank.
Run exact AOM/POP selection only on that active bank.
```

The large exploration bank is a search space, not necessarily the final bank
passed to CV/PRESS.

## Core Constraint

A native AOM/POP bank operator must be strict linear at application time:

```text
X_b = X A_b^T
```

with fixed `A_b in R^{p x p}` for the fold/model fit.

Then:

```text
X_b^T Y = A_b X^T Y
```

This identity is the reason large operator exploration is feasible. We can
apply candidates to:

```text
S = X^T Y
```

instead of materializing:

```text
X A_b^T
```

For PLS1, `S` is only a vector of length `p`. For PLS2, `S` is `p x q`, and
`q` is usually small.

## Higher-Order Compositions

Compositions preserve linearity. If `A` and `W` are strict linear operators:

```text
X1 = X A^T
X2 = X1 W^T
X2 = X A^T W^T
X2 = X (W A)^T
```

So:

```text
C = W A
X2 = X C^T
```

The covariance identity also composes:

```text
S = X^T Y
S2 = C S = W A S
```

The adjoint used by NIPALS reverses the chain:

```text
C^T v = A^T W^T v
```

Therefore chains such as:

```text
SavGol -> Whittaker
Detrend -> SavGol derivative
Whittaker -> SavGol derivative
Shift -> derivative
Low-pass -> derivative
```

are mathematically valid if every stage has fixed parameters.

### Nonlinear Exclusions

The following are not strict linear operators and should stay outside the
native bank unless represented by a deliberately separate approximation:

```text
SNV
MSC
sample-wise normalization
adaptive baseline correction fitted per sample
data-dependent clipping or thresholding
wavelet denoising with coefficient thresholding
```

They can still be explored as upstream preprocessing forks, but not as native
operators in the covariance-space bank.

## Computational Credibility

Dense matrix multiplication should be avoided for production exploration.

Do not do this for large banks:

```text
C = W @ A
Xb = X @ C.T
```

Keep chains lazy:

```python
def apply_cov(S):
    out = S
    for op in chain:
        out = op.apply_cov(out)
    return out

def adjoint_vec(v):
    out = v
    for op in reversed(chain):
        out = op.adjoint_vec(out)
    return out
```

The cost of a chain is the sum of its stages on `p x q`, not the cost of a
dense `p x p` product. For PLS1, this is usually very cheap.

Example:

```text
SG.apply_cov(S)        -> O(p q) convolution
Whittaker.apply_cov(S) -> O(p q) banded solve
FFT.apply_cov(S)       -> O(p log p q)
Detrend.apply_cov(S)   -> O(p d q), if implemented as low-rank projection
```

For `SG -> Whittaker` in PLS1, this is basically one convolution plus one
banded solve on a vector.

## Candidate Families

The exploration bank should be built from a relatively small set of primitive
families whose parameters cover a large but structured space.

### Dense Savitzky-Golay Scale Space

Instead of a few SG filters, define a large grid:

```text
deriv = 0, 1, 2, maybe 3
window_length = 5, 7, 9, ..., max_window
polyorder = deriv + 1, ..., 5
```

Rules:

```text
window_length odd
polyorder < window_length
deriv <= polyorder
window_length <= p // 3 or another p-aware cap
```

This creates many candidates, but many are nearly duplicate. It should be
paired with similarity pruning.

### Whittaker Smoothers

Whittaker is valuable because it provides a smoothness continuum not identical
to SG:

```text
lambda = 1e-2, 1e-1, ..., 1e8
```

It is especially relevant for broad baseline and smooth chemical variation.
With the existing banded solver it is cheap on `S`.

### Gaussian Derivative Filters

Derivative-of-Gaussian filters are a good complement to SG derivatives:

```text
sigma = log-spaced or half-octave grid
order = 0, 1, 2, 3
```

They naturally form a scale-space and are convolutional. They also make
higher-order composition easier to reason about because smoothing scale and
derivative order are explicit.

### Finite and Fractional Derivatives

Integer finite differences are already present. Fractional derivatives could
cover the continuum between identity, first derivative, and second derivative:

```text
alpha = 0.0, 0.1, 0.2, ..., 2.5
```

This can be represented as convolutional kernels. This family may be more
efficient than adding many SG variants because it spans a different axis of
spectral emphasis.

### FFT or DCT Masks

Frequency-domain masks can generate many cheap candidates:

```text
low-pass
high-pass
band-pass
octave bands
overlapping bands
soft tapers
```

Implementation should compute the transform once per response vector/matrix
and apply many masks if possible. This is useful as a broad exploratory family,
but final selected operators should still be described clearly.

### Baseline Projection Operators

These are strict linear if the basis is fixed:

```text
remove polynomial degree 0..5
remove low-frequency DCT basis
remove spline basis with fixed knots
remove known nuisance subspace
```

For polynomial detrend:

```text
P_d2 P_d1 = P_max(d1, d2)
```

approximately or exactly depending on construction, so repeated detrend
operators should be canonicalized rather than stacked.

### Fixed Spectral Shifts

Small wavelength shifts can be represented by fixed interpolation matrices:

```text
shift = -3, -2, -1, -0.5, 0.5, 1, 2, 3 points
```

This is useful for instrument drift or wavelength misalignment. It remains
linear if interpolation weights are fixed.

## Higher-Order Grammar

Do not generate all possible chains. If there are `m` primitive operators:

```text
degree 1: m
degree 2: m^2
degree 3: m^3
```

This explodes immediately.

Use a grammar that allows plausible compositions and rejects redundant ones.

Credible chains:

```text
detrend -> smooth -> derivative
detrend -> whittaker -> derivative
smooth -> derivative -> light_smooth
shift -> derivative
lowpass -> derivative
derivative -> bandpass
baseline_projection -> derivative
```

Usually redundant chains:

```text
smooth -> smooth -> smooth
whittaker -> whittaker -> whittaker
detrend_d1 -> detrend_d2
derivative2 -> derivative2
highpass -> highpass
lowpass -> lowpass
```

Canonicalization rules should simplify chains before scoring:

```text
convolution -> convolution     => fuse kernels
FFT mask -> FFT mask           => multiply masks
detrend degree a -> degree b   => keep max/equivalent projection
identity anywhere              => remove
duplicate symmetric smoother   => keep one or merge parameters
```

Whittaker generally should stay as its own lazy stage, but compositions such as
`convolution -> Whittaker` are still cheap.

## Similarity: Global vs Dataset-Dependent

There are two useful notions of similarity.

### Intrinsic Similarity

This depends only on the transformations:

```text
sim(A, B) = <A, B>_F / (||A||_F ||B||_F)
```

For convolutional filters, compare frequency responses:

```text
sim(A, B) = corr(abs(FFT(k_A)), abs(FFT(k_B)))
```

Intrinsic pruning is useful offline. It removes obvious duplicates from the
global candidate bank.

A practical offline alternative is to compare operator actions on a synthetic
probe basis:

```text
Dirac impulses
DCT/sinusoidal multi-frequency signals
low-degree polynomials
Gaussian peaks with multiple widths
synthetic NIRS-like spectra
white and correlated noise
```

For each operator:

```text
R_A = A Z
```

where `Z` is the probe matrix. Compare flattened `R_A`.

This captures boundary effects, Whittaker behavior, shifts, and projections
better than a purely analytic comparison.

### Dataset-Dependent Similarity

AOM/POP does not use an abstract operator in isolation. It sees:

```text
A S
```

with:

```text
S = X^T Y
```

or a residual/projected version of `S`.

Two operators that are different globally can be equivalent on a dataset if
`S` has energy only where their responses are similar. Conversely, two similar
operators can differ on a chemically important narrow band.

Therefore use two stages:

```text
1. Offline/global pruning by intrinsic similarity.
2. In-fold active-bank pruning by similarity of A S.
```

Suggested pattern:

```python
global_bank = prune_by_operator_similarity(candidate_bank, threshold=0.995)

S = X_train.T @ y_train
active_bank = prune_by_response_similarity(
    global_bank,
    S,
    threshold=0.98,
    top_m=30,
)
```

The global bank should be broad and non-redundant. The active bank should be
specific to the fold and target.

## Controlling Cost With Beam Search

The main risk is not the cost of one higher-order chain. The main risk is the
number of chains.

Use beam search:

```text
primitive_ops: 100 to 300
max_degree: 2 or 3
beam_width: 32 or 64
final top_m: 20 to 40
```

Instead of scoring all `m^d` chains:

```text
degree 1: score all primitives, keep B diverse states
degree 2: expand B x m, keep B diverse states
degree 3: expand B x m, keep B diverse states
```

For `m=200`, `B=32`, `degree=3`:

```text
full enumeration: 8,000,000 chains
beam search: about 12,800 expansions
```

Each expansion applies one primitive operator to an existing response:

```text
new_response = op.apply_cov(previous_response)
```

Do not recompute the entire chain from scratch.

### Candidate State

A search state should store:

```python
@dataclass
class OperatorCandidateState:
    chain: tuple[LinearSpectralOperator, ...]
    response: np.ndarray       # A_chain S
    score: float
    family_signature: str
    gain_estimate: float
```

The final selected chains can be converted into `ComposedOperator` instances
or specialized fused operators.

### Beam Search Sketch

```python
states = [
    OperatorCandidateState(
        chain=(),
        response=S,
        score=base_score(S),
        family_signature="identity",
        gain_estimate=1.0,
    )
]

kept_by_depth = []

for depth in range(1, max_degree + 1):
    candidates = []

    for state in states:
        for op in primitive_ops:
            if not grammar_allows(state.chain, op):
                continue

            chain = canonicalize(state.chain + (op,))
            if chain_is_duplicate(chain):
                continue

            response = op.apply_cov(state.response)
            score = normalized_response_score(response, chain)

            candidates.append(
                OperatorCandidateState(
                    chain=chain,
                    response=response,
                    score=score,
                    family_signature=signature(chain),
                    gain_estimate=estimate_gain(chain),
                )
            )

    states = keep_top_diverse(
        candidates,
        beam_width=beam_width,
        cosine_threshold=0.98,
        per_family_limit=4,
    )
    kept_by_depth.extend(states)

active = keep_top_diverse(
    kept_by_depth,
    beam_width=final_top_m,
    cosine_threshold=0.97,
    per_family_limit=6,
)
```

## Scoring and Normalization

A naive covariance score:

```text
score(A) = -||A S||
```

is fast but can be biased by operator gain. Derivatives, high-pass filters, or
chains with large amplification may win because of scale rather than predictive
content.

Use a normalized score:

```text
score(A) = - ||A S|| / gain(A)
```

Possible `gain(A)` estimates:

```text
||A||_F
spectral norm estimate by power iteration
RMS response on synthetic probes
noise gain from white-noise probes
chain product of primitive gains
```

For convolutional filters, a cheap proxy is:

```text
gain(k) = sqrt(mean(abs(FFT(k))^2))
```

For lazy chains, empirical probe gain is often simplest:

```text
gain(A) = rms(A Z) / rms(Z)
```

with fixed deterministic probes `Z`.

The final CV/PRESS stage still decides. The normalized covariance score is only
for prescreening.

## Diversity Selection

After sorting by score, keep candidates only if they add a new response
direction:

```text
cosine(A_i S, A_j S) < threshold
```

For PLS1:

```python
def response_cosine(a, b, eps=1e-12):
    return abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)
```

For PLS2:

```python
def response_cosine(A, B, eps=1e-12):
    av = A.ravel()
    bv = B.ravel()
    return abs(float(av @ bv)) / (np.linalg.norm(av) * np.linalg.norm(bv) + eps)
```

Use both:

```text
score rank
response diversity
family quotas
max chain degree
```

Example default:

```text
cosine_threshold = 0.98 for beam
cosine_threshold = 0.97 for final active bank
per_family_limit = 4 per depth
final_top_m = 30
```

## Active Bank Generation in AOM/POP

### Global AOM

For global selection:

```text
S = X_train^T Y_train
active_bank = explore_operators(S, primitive_bank, max_degree=3)
run exact global selection on active_bank
```

The selected operator is one chain for the whole model.

### Per-Component POP

For POP, the active bank can be regenerated at each component:

```text
S_a = current residual/projected covariance
active_bank_a = explore_operators(S_a, primitive_bank, max_degree=2 or 3)
select one operator for component a
commit component
```

This is more expensive but more faithful. A cheaper compromise:

```text
Generate a broad active bank once from S_0.
Reuse it for all components.
Optionally refresh every r components.
```

Recommended starting point:

```text
global AOM: max_degree=3, final_top_m=30
POP: max_degree=2, final_top_m=20 per component
```

## Matrix-Free Operator Design

The current `ComposedOperator` already implements the right idea:

```text
transform: apply each stage
apply_cov: apply each stage
adjoint_vec: apply stages in reverse
```

For large generated banks, add optional specialized operators:

```text
ConvolutionKernelOperator
FusedConvolutionOperator
FFTMaskOperator
DCTProjectionOperator
ShiftInterpolationOperator
GeneratedComposedOperator
```

The dense `matrix()` method can exist for tests and small `p`, but generation
and scoring should not call it.

## Proposed AOM_v0 Components

Add these modules when implementing the exploration layer:

```text
aompls/operator_generation.py
    primitive grid builders
    composition grammar
    canonicalization
    signatures

aompls/operator_explorer.py
    covariance-space beam search
    response similarity pruning
    active-bank generation

aompls/operator_similarity.py
    intrinsic probe matrix
    offline pruning
    response cosine functions
    gain estimates
```

`banks.py` should remain for curated named presets. The explorer can be used
to build temporary active banks from a preset or from a larger primitive grid.

Potential API:

```python
active_bank = explore_operator_bank(
    S=S,
    primitive_ops=primitive_bank,
    max_degree=3,
    beam_width=32,
    final_top_m=30,
    cosine_threshold=0.98,
    score_normalization="probe_gain",
    grammar="spectral_v1",
)
```

Estimator-level parameters could be:

```python
AOMPLSRegressor(
    operator_bank="exploratory",
    candidate_bank="spectral_v1",
    candidate_max_degree=3,
    candidate_beam_width=32,
    candidate_top_m=30,
    candidate_similarity_threshold=0.98,
)
```

## Testing Requirements

Every generated operator or chain must pass the standard operator tests:

```text
shape
linearity
adjoint identity
transform vs matrix for small p
covariance identity
```

For generated chains:

```text
ComposedOperator.apply_cov(S) equals matrix product on small p.
ComposedOperator.adjoint_vec(v) equals explicit C^T v on small p.
Fused convolution equals sequential convolution within tolerance.
Canonicalized chain equals original chain within tolerance.
```

For exploration:

```text
beam search is deterministic
identity remains available
final active bank size <= top_m
response diversity threshold is respected
no candidate uses validation/test data
CV folds generate active banks from training fold only
```

The last point is critical. If active banks are generated inside CV, the bank
must be generated using only the training fold. Otherwise operator generation
leaks validation information.

## Recommended Initial Implementation Path

1. Add intrinsic/probe similarity tools.
2. Add a dense SG grid and Whittaker lambda grid as primitives.
3. Add response-space `keep_top_diverse`.
4. Add covariance beam search for degree 1 and 2.
5. Integrate active-bank generation into `criterion="hybrid"` only.
6. Add degree 3 after tests and timing are stable.
7. Add Gaussian derivative and fractional derivative families.
8. Add FFT/DCT masks and shift operators.

This order keeps the system scientifically conservative while giving a clear
path toward thousands of virtual candidates.

## Practical Defaults

Start with:

```text
primitive_ops: 100 to 200
max_degree: 2
beam_width: 32
final_top_m: 20
cosine_threshold: 0.98
per_family_limit: 4
score_normalization: probe_gain
```

Then benchmark:

```text
max_degree: 3
beam_width: 64
final_top_m: 30 or 40
```

Do not start with an unbounded composition grammar. The first useful result is
not "all possible transformations"; it is "many plausible transformations,
screened cheaply and pruned aggressively".

## Summary

Higher-order preprocessing exploration is credible:

```text
Mathematically: yes, strict linearity is preserved by composition.
Computationally: yes, if chains are lazy and scored on S = X^T Y.
Scientifically: yes, if the search space is structured and pruned.
```

The final architecture should separate three concepts:

```text
Candidate bank:
    very large, generated, matrix-free

Active bank:
    small, fold-specific, diverse, covariance-screened

Selection bank:
    the active bank passed to exact CV/PRESS AOM/POP selection
```

This gives AOM_v0 a way to explore hundreds or thousands of preprocessing
operators without turning the model into an expensive brute-force preprocessing
grid.
