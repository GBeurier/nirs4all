"""
U08 - Native Finetuning: the portable ``n4m`` optimizer
=======================================================

Tune hyperparameters with the **native optimizer** from ``nirs4all-methods``
(``libn4m``) instead of Optuna — by adding a single key, ``"engine": "n4m"``,
to ``finetune_params``.

Why use it
----------
* **Portable** — the same C-ABI optimizer drives finetuning in Python, R,
  MATLAB/Octave and WebAssembly. At a fixed ``seed`` the trial sequence is
  identical across every binding.
* **Same DSL** — ``model_params`` / ``train_params`` are declared exactly as
  with the Optuna engine (see :ref:`U02_hyperparameter_tuning`); only the engine
  changes.
* **Rich algorithm menu** — samplers ``random, sobol, lhs, ternary, ga, pso,
  cmaes, tpe, gp_ei`` and pruners ``median, successive_halving (asha),
  hyperband, racing``.

Optuna stays the default engine; nothing here removes it. Drop ``"engine"``
(or set it to ``"optuna"``) to go back.

Prerequisites
-------------
* Complete :ref:`U02_hyperparameter_tuning` first.
* An up-to-date ``nirs4all-methods`` install exposing the native optimizer
  (``python -c "from n4m.model_selection.optimizer import Optimizer"``).

Duration: ~1 minute
Difficulty: ★★★☆☆
"""
import argparse

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate

parser = argparse.ArgumentParser(description="U08 Native Finetuning Example")
parser.parse_args()

# Is the native optimizer available?
try:
    from n4m.model_selection.optimizer import Optimizer  # noqa: F401
    HAS_N4M = True
except Exception:
    HAS_N4M = False

print("\n" + "=" * 60)
print("U08 - Native Finetuning (n4m engine)")
print("=" * 60)

if not HAS_N4M:
    print(
        "\nThe native n4m optimizer is not available in this environment.\n"
        "Install/upgrade nirs4all-methods, then re-run:\n"
        "    python -c 'from n4m.model_selection.optimizer import Optimizer'\n"
    )
    raise SystemExit(0)

# =============================================================================
# Section 1: Switch engines with one key
# =============================================================================
print("\n" + "-" * 60)
print("Section 1: TPE on the native engine")
print("-" * 60)
print("""
The ONLY change vs. the Optuna tutorial is  "engine": "n4m".
Everything else — the search-space DSL, approaches, metrics — is the same.
""")

result_tpe = nirs4all.run(
    pipeline=[
        StandardNormalVariate(),
        MinMaxScaler(),
        ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
        {
            "model": PLSRegression(),
            "name": "PLS-n4m-TPE",
            "finetune_params": {
                "engine": "n4m",            # <-- native optimizer (this is the whole change)
                "n_trials": 20,
                "sampler": "tpe",           # tree-structured Parzen estimator
                "approach": "grouped",
                "metric": "rmse",
                "seed": 42,
                "verbose": 1,
                "model_params": {
                    "n_components": ("int", 1, 30),
                },
            },
        },
    ],
    dataset="sample_data/regression",
    name="n4m-TPE",
    verbose=1,
)
print(f"\nBest RMSE (native TPE): {result_tpe.best_score:.4f}")

# =============================================================================
# Section 2: CMA-ES for continuous params + log-scale
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: CMA-ES on a log-scale regularization")
print("-" * 60)
print("""
CMA-ES is a strong choice for smooth, continuous search spaces such as a
regularization strength spanning several orders of magnitude.
""")

result_cmaes = nirs4all.run(
    pipeline=[
        StandardNormalVariate(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "model": Ridge(),
            "name": "Ridge-n4m-CMAES",
            "finetune_params": {
                "engine": "n4m",
                "n_trials": 25,
                "sampler": "cmaes",
                "approach": "single",
                "metric": "r2",             # maximize R2 (direction auto-inferred)
                "seed": 42,
                "verbose": 1,
                "model_params": {
                    "alpha": ("float_log", 1e-4, 1e2),
                    "fit_intercept": [True, False],
                },
            },
        },
    ],
    dataset="sample_data/regression",
    name="n4m-CMAES",
    verbose=1,
)
print(f"\nBest R2 (native CMA-ES): {result_cmaes.best_score:.4f}")

# =============================================================================
# Section 3: Early stopping with a native pruner
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Pruning unpromising trials (median rule)")
print("-" * 60)
print("""
Pruners stop unpromising trials early using the per-fold score stream.
Native pruners: median, successive_halving (ASHA), hyperband, racing.
(`racing` is the fold-safe choice when folds are exchangeable.)
""")

result_pruned = nirs4all.run(
    pipeline=[
        StandardNormalVariate(),
        ShuffleSplit(n_splits=4, random_state=42),
        {
            "model": PLSRegression(),
            "name": "PLS-n4m-Pruned",
            "finetune_params": {
                "engine": "n4m",
                "n_trials": 20,
                "sampler": "tpe",
                "pruner": "median",
                "approach": "grouped",
                "seed": 42,
                "verbose": 1,
                "model_params": {
                    "n_components": ("int", 1, 30),
                },
            },
        },
    ],
    dataset="sample_data/regression",
    name="n4m-Pruned",
    verbose=1,
)
print(f"\nBest RMSE (native + median pruner): {result_pruned.best_score:.4f}")

# =============================================================================
# Section 4: Conditional attributes (object__attribute in the search space)
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Conditional hyperparameters (`when` clause)")
print("-" * 60)
print("""
Some attributes only matter for a particular choice of another attribute — e.g.
an SVR's `gamma` is only relevant for the RBF/poly kernels, and `degree` only for
poly. A `when` clause makes a parameter active *only* when a sibling categorical
takes a given value, so the sampler never wastes trials on irrelevant dimensions
and the model is built with exactly the right attributes.
""")

from sklearn.svm import SVR

result_cond = nirs4all.run(
    pipeline=[
        StandardNormalVariate(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "model": SVR(),
            "name": "SVR-n4m-Conditional",
            "finetune_params": {
                "engine": "n4m",
                "n_trials": 20,
                "sampler": "tpe",           # tree-structured — handles conditional spaces
                "approach": "single",
                "metric": "rmse",
                "seed": 42,
                "verbose": 1,
                "model_params": {
                    "kernel": ["linear", "rbf", "poly"],
                    "C": ("float_log", 1e-1, 1e3),
                    # gamma only when kernel is rbf or poly:
                    "gamma": {"type": "float_log", "min": 1e-4, "max": 1e1,
                              "when": {"kernel": ["rbf", "poly"]}},
                    # degree only when kernel is poly:
                    "degree": {"type": "int", "min": 2, "max": 4,
                               "when": {"kernel": "poly"}},
                },
            },
        },
    ],
    dataset="sample_data/regression",
    name="n4m-Conditional",
    verbose=1,
)
print(f"\nBest RMSE (conditional SVR): {result_cond.best_score:.4f}")

# =============================================================================
# Section 5: Operators in the search space (choose the operator, tune each)
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Operators in the search space")
print("-" * 60)
print("""
Go beyond primitive hyperparameters: put the OPERATORS themselves in the search
space. With a scikit-learn Pipeline as the model, a `{"options": {...}}`
categorical selects which operator fills a step, and `object__attribute`
addressing (with a `when` clause) tunes each operator's own hyperparameters
conditionally — one joint search over structure + hyperparameters.
""")

from sklearn.pipeline import Pipeline

result_ops = nirs4all.run(
    pipeline=[
        StandardNormalVariate(),
        ShuffleSplit(n_splits=3, random_state=42),
        {
            "model": Pipeline([("scale", StandardScaler()), ("est", PLSRegression())]),
            "name": "Pipeline-OperatorSearch",
            "finetune_params": {
                "engine": "n4m",
                "n_trials": 30,
                "sampler": "tpe",
                "approach": "single",
                "metric": "rmse",
                "seed": 42,
                "verbose": 1,
                "model_params": {
                    # choose the scaler operator:
                    "scale": {"type": "categorical",
                              "options": {"standard": StandardScaler(), "minmax": MinMaxScaler()}},
                    # choose the estimator operator:
                    "est": {"type": "categorical",
                            "options": {"pls": PLSRegression(), "ridge": Ridge()}},
                    # tune each estimator's own params, conditionally:
                    "est__n_components": {"type": "int", "min": 1, "max": 20, "when": {"est": "pls"}},
                    "est__alpha": {"type": "float_log", "min": 1e-4, "max": 1e2, "when": {"est": "ridge"}},
                },
            },
        },
    ],
    dataset="sample_data/regression",
    name="OperatorSearch",
    verbose=1,
)
print(f"\nBest RMSE (operator search): {result_ops.best_score:.4f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Native finetuning = the Optuna DSL + one key:

  {
    "model": PLSRegression(),
    "finetune_params": {
      "engine": "n4m",                 # native libn4m optimizer (portable)
      "n_trials": 30,
      "sampler": "tpe",                # random|sobol|lhs|ternary|ga|pso|cmaes|tpe|gp_ei
      "pruner": "median",              # none|median|successive_halving|hyperband|racing
      "approach": "grouped",           # single|grouped|individual
      "metric": "rmse",                # direction auto-inferred
      "seed": 42,                      # same seed -> same trials in every binding
      "model_params": {
        "n_components": ("int", 1, 30),
      },
    }
  }

Same seed + same data -> identical trial sequence in Python, R, MATLAB and WASM.
Drop "engine" (or set "optuna") to use the Optuna engine instead.
""")

if __name__ == "__main__":
    pass
