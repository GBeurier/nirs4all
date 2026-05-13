# Context Review

## Existing AOM Material

The existing `bench/AOM` material supports a publication-quality AOM-PLS
project, but it is centered on the deployed NIPALS/adjoint path. The roadmap and
publication plan already identify the main weakness: current global AOM selection
uses a fixed internal 20% holdout when no validation set is supplied. That is
reproducible, but it is not a strong scientific default on small NIRS datasets.

Current deployed concepts:

- `AOMPLSRegressor`: global operator selection; all components use one selected
  preprocessing operator.
- `POPPLSRegressor`: per-component operator selection; each component may use a
  different operator.
- Operator bank: identity, Savitzky-Golay smoothing/derivatives, finite
  differences, detrend projections, Norris-Williams derivatives, compositions,
  optional wavelet/FFT operators.
- Current prototypes: Bandit AOM, DARTS PLS, zero-shot router, MoE PLS, and
  pseudo-linear SNV variants.

Important implementation findings:

- AOM and POP are separate implementations. The new work must unify them under
  one core with `selection="global"` and `selection="per_component"`.
- SIMPLS already exists in `nirs4all/operators/models/sklearn/simpls.py`, but it
  is standard SIMPLS only, not operator-adaptive covariance-space SIMPLS.
- Torch AOM exists, but it is a global-selection NIPALS acceleration, not a full
  numpy/torch parity implementation of all AOM/POP/SIMPLS variants.
- Existing tests cover basic operator adjoints and smoke API behavior, but not
  all equivalences needed for a scientific paper.

## FCK-PLS Review

FCK-PLS provides useful design constraints for AOM_v0:

- It makes spectral filtering learnable through Torch kernels while preserving a
  PLS-style solved head.
- Its most important methodological lesson is leakage control: the PLS head is
  solved on one training subset, while kernel optimization loss is computed on a
  held-out subset. AOM_v0 must preserve this discipline whenever it trains gates,
  calibrators, or probability mappings.
- The FCK-PLS Torch prototype supports both free learnable kernels and
  alpha/sigma-parametric kernels. AOM_v0 should not reimplement FCK-PLS, but its
  benchmark table is a related baseline and its validation split logic should
  influence all trainable or differentiable AOM variants.

## TabPFN Review

`bench/tabpfn_paper/master_results.csv` is the regression benchmark oracle for
AOM_v0. Current local profile:

- 335 rows.
- 61 unique regression dataset splits.
- 29 columns.
- Models present: `TabPFN-Raw`, `TabPFN-opt`, `Catboost`, `PLS`, `Ridge`, `CNN`.
- Model coverage by dataset count: TabPFN-Raw 61, TabPFN-opt 58, Catboost 57,
  PLS 54, Ridge 54, CNN 51.
- Status values: TabPFN-Raw rows are `ok`; other model rows are mostly `partial`.

The AOM_v0 benchmark must write rows compatible with the same schema and append
extra columns for operator-adaptive diagnostics. It must not invent a separate
benchmark table that cannot be joined against `master_results.csv`.

The TabPFN paper draft reports a regression protocol with strict calibration/test
separation, preprocessing/model selection on calibration folds, and final RMSEP
on the untouched test set. It also describes 15 classification datasets evaluated
with balanced accuracy. However, the local `DatabaseDetail.xlsx` only lists
regression rows, so AOM_v0 must build classification cohorts by scanning
`bench/tabpfn_paper/data/classification`.

## Classification Review

Existing `AOMPLSClassifier`, `POPPLSClassifier`, and `PLSDA` wrappers implement
basic PLS-DA by encoding classes and then fitting a regressor. Their probability
handling is weak:

- Binary probabilities are clipped regression scores.
- Multiclass probabilities are plain softmax over regression outputs.
- There is no explicit class-prior correction, class-balanced coding, latent-space
  classifier, or leakage-safe calibration policy.

AOM_v0 must implement a stronger classification design:

- Encode classes into a class-balanced one-hot matrix for PLS2 extraction.
- Select operators/components using classification-aware metrics when
  `task="classification"`.
- Fit a multinomial logistic calibration model on training latent scores for
  `predict_proba`.
- In benchmark mode, probabilities and calibration must be learned inside each
  training fold only.
- Report balanced accuracy, macro-F1, log loss, Brier score, and expected
  calibration error where possible.

## Scientific Positioning

The strongest contribution is not another prototype but a unified family:

`Operator-Adaptive PLS = selection policy x PLS engine x operator bank x task`.

Required variants:

- Standard PLS as identity-only operator-adaptive PLS.
- AOM global hard selection.
- POP per-component hard selection.
- Soft mixture AOM/POP as experimental.
- Superblock/multi-view baseline.
- NIPALS materialized reference and adjoint implementation.
- SIMPLS materialized reference and covariance-space implementation.
- PLS1, PLS2, and PLS-DA.

The paper should present this as a mathematical framework with reproducible
benchmarks, not as a loose collection of implementation tricks.
