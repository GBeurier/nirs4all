# Cover letter

**To:** The Editors, Chemometrics and Intelligent Laboratory Systems

**Re:** Submission of "Operator-Adaptive Partial Least Squares for
Near-Infrared Spectroscopy: Unified AOM/POP Selection, Covariance-Space
SIMPLS, and PLS-DA"

Dear Editors,

We are pleased to submit the enclosed manuscript for consideration at
*Chemometrics and Intelligent Laboratory Systems*. The paper presents a
parity-validated *Operator-Adaptive Partial Least Squares*
(AOM/POP-PLS) framework that formalises the deployed `nirs4all`
AOM-PLS estimator and evaluates controlled research extensions around
it. The framework rests on the central operator identity
`(X A^T)^T Y = A X^T Y` for strict linear spectral operators, which
gives rise to a covariance-space SIMPLS engine that evaluates candidate
operators without ever materialising the transformed matrices. The
framework is extended to PLS-DA via class-balanced latent coding and a
leakage-safe logistic-regression calibrator on the latent scores; the
classification benchmark itself is not presented as a headline result.

## Scope fit

The manuscript fits the journal's scope on three axes:

1. *Chemometric methodology.* It is a clean, traceable extension of
   PLS that respects standard chemometric practice (NIPALS, SIMPLS,
   Savitzky-Golay, Norris-Williams, Whittaker, detrend, finite
   difference). It does not introduce any deep architecture.
2. *Reproducible benchmarking.* The protocol is fully aligned with the
   recently published TabPFN/NIRS benchmark schema (master result CSV)
   and adds AOM/POP rows to that schema with stable column names.
3. *Open implementation.* The reference implementation
   (`bench/AOM_v0/aompls/`) is shipped under the open-source
   `nirs4all` library with production-parity tests, NumPy and PyTorch
   backends, and an Elsevier-CAS-compatible LaTeX manuscript.

## Novelty

The paper's specific contributions are:

- A unified mathematical contract that subsumes standard PLS, AOM
  (global), POP (per-component), soft mixture, and superblock multi-view
  PLS under a single operator-bank formulation.
- A covariance-space SIMPLS engine that follows directly from the
  operator identity above; it is mathematically equivalent to
  materialised SIMPLS in `global` mode and to residualised NIPALS in
  per-component mode, and is asserted as such in the unit tests.
- A class-balanced (`Y_ic = 1/sqrt(pi_c)`) PLS-DA extension with
  leakage-safe calibration via logistic regression on latent scores,
  with a temperature-scaled softmax fallback for degenerate regimes.
- A leakage-safe experimental protocol that joins AOM/POP variants to
  the published TabPFN/NIRS regression benchmark cohort and reports
  per-pair signed deltas against PLS and TabPFN.

## Comparison to TabPFN

We make no claim of universal superiority over TabPFN-Raw or
TabPFN-opt, both of which are reported as reference baselines from the
master CSV. TabPFN-opt remains the strongest single estimator overall
in the joined comparison. Our results instead isolate the chemometric
lessons for AOM-PLS: compact banks and 3-fold CV reduce holdout
selection variance, SNV in front of compact-bank AOM gives the best
AOM-family row (32/57 wins versus PLS, median relative RMSEP 0.984),
and unregularised POP at `K=15` is unstable rather than a recommended
default.

## arXiv preprint disclosure

The manuscript will be deposited to arXiv prior to journal review, in
order to make the open-source release citable for the chemometric
community. The arXiv version will be identical to the submitted version
modulo the journal's preferred typographic class. We disclose this
upfront in keeping with the journal's policy on overlapping
publications.

## Conflict of interest

The authors declare no conflict of interest, and no external funding
specific to this work.

## Suggested reviewers

We respectfully suggest reviewers with expertise in:

- chemometric PLS (NIPALS / SIMPLS / OPLS),
- NIRS preprocessing and operator banks,
- benchmarking small-sample tabular prediction with PLS, ridge,
  CatBoost, and TabPFN.

We confirm that the work has not been published elsewhere and is not
under consideration at another journal. The corresponding author
welcomes any editorial feedback and stands ready to provide additional
material upon request.

Sincerely,

Grégory Beurier and the `nirs4all` contributors

`gregory.beurier@cirad.fr`
