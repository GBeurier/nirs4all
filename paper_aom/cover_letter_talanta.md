Dear Editor,

We are pleased to submit our manuscript entitled **"Reframing preprocessing
selection as model-internal calibration in near-infrared spectroscopy: a
large-scale benchmark of operator-adaptive PLS and Ridge models"** for
consideration in *Talanta*.

This manuscript addresses a central bottleneck in analytical NIRS method
development: the dependence of calibration performance on large and poorly
auditable preprocessing searches. We propose operator-adaptive calibration, a
framework that embeds linear preprocessing alternatives inside PLS and Ridge
models while preserving original-wavelength coefficients. Nonlinear or
sample-adaptive corrections such as SNV, MSC and ASLS are handled as
fold-local branches to maintain leakage-safe validation.

The approach is evaluated on more than 50 heterogeneous NIRS calibration
datasets and compared with conventional PLS, Ridge, CatBoost and CNN baselines
under documented search budgets. Compact operator-adaptive PLS combined with
ASLS branch preprocessing achieved a median RMSEP/PLS ratio of 0.960 with
42 wins on 57 datasets, while a deployable AOM-Ridge selector improved over
tuned Ridge by a median 2.22% with 35 wins on 52 datasets. Beyond predictive
accuracy, the proposed framework reduces calibration search complexity,
produces traceable preprocessing decisions, retains interpretable coefficients,
and supports fast refitting for routine calibration updates.

We believe the manuscript fits *Talanta* because it provides a practical and
broadly applicable advance for analytical calibration workflows rather than a
single-application optimization. The accompanying supplementary material
documents derivations, benchmark provenance, deployable versus oracle result
separation, leakage controls, paired statistics and the experiments required
for final journal-level validation.

Sincerely,

Grégory Beurier, Robin Reiter, Camille Noûs, Lauriane Rouan and Denis Cornet
