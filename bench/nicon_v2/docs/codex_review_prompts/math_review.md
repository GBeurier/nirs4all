# Codex Math Review Prompt — nicon_v2

You are reviewing a math/algorithm specification and its implementation in `bench/nicon_v2/`. The project is a research bench under `bench/` of the `nirs4all` workspace; it does NOT modify `nirs4all` itself.

## Scope

* `bench/nicon_v2/docs/MATH_SPEC.md` — formal definitions of preprocessing layers, model architectures, loss functions, augmentation, ensembles, conformal UQ.
* `bench/nicon_v2/nicon_v2/preprocessing.py`
* `bench/nicon_v2/nicon_v2/augmentation.py`
* `bench/nicon_v2/nicon_v2/uncertainty.py`
* `bench/nicon_v2/nicon_v2/models/*.py`

## What to check

1. **Mathematical correctness.**
   * Preprocessing operators (SNV, MSC, SG-derivative, EMSC) match the spec.
   * Conformal calibration uses the correct `(|C|+1)(1−α)/|C|`-quantile.
   * Deep-ensemble variance decomposition is consistent (E[σ²] + Var[μ]).
   * Loss functions and gradient flow on the y-scaled scale are stable; inverse transform at evaluation is faithful.

2. **Numerical stability.**
   * `log σ̂` clamping range is sensible.
   * Gradient through the SG convolution is correct (kernel non-trainable).
   * Conformal quantile interpolation does not introduce rank reversal.

3. **Bias / variance balance of the augmentations.**
   * Bjerrum amplitudes do not violate the spectrum domain (no negative values for absorbance-only data).
   * C-Mixup label-distance kernel is normalised; sampling preserves dataset-level bias.

4. **Statistical reporting.**
   * Wilcoxon test conforms to the BENCHMARK_PROTOCOL (paired, two-sided, with multiplicity correction across phases).

## Output format

For each finding produce:
* **Severity** (High / Medium / Low / Info).
* **Location** (file:line).
* **Issue** (≤ 80 words).
* **Suggested fix** (≤ 80 words).

Plus a single one-line **verdict** at the end: `LGTM`, `LGTM_WITH_NITS`, `NEEDS_FIXES`, or `BLOCKED`.

## Reference materials

* `source_materials/literature_review/LITERATURE_REVIEW.md` (theme-by-theme review with DOIs).
* `bench/AOM_v0/docs/AOMPLS_MATH_SPEC.md` (analogous tone for sister project).
