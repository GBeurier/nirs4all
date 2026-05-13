"""End-to-end demo: nirs4all pipeline driving the AOM_lib AOM-PLS backend.

This script accompanies the AOM-PLS Talanta submission and demonstrates
that the nirs4all pipeline engine can drive the *dedicated*
``aompls.AOMPLSCompact`` C++/Eigen implementation (shipped under
``bench/AOM_lib/python/src``) through the
:class:`AOMPLSAomlibRegressor` wrapper. The legacy pure-Python
``AOMPLSRegressor`` remains available; this example exercises the new
wrapper specifically.

Run it from the repository root with ``aompls`` on the Python path::

    PYTHONPATH=bench/AOM_lib/python/src \\
        python examples/aom_paper/aomlib_nirs4all_regression.py

The script falls back to a synthetic regression dataset when the default
``examples/sample_data/regression`` folder is not reachable (e.g. when
this file is executed from outside the ``examples/`` directory).
"""

from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import ShuffleSplit

import nirs4all
from nirs4all.operators.models.sklearn import AOMPLSAomlibRegressor

# The nirs4all example loader resolves dataset paths relative to the
# current working directory; we therefore probe both layouts.
_HERE = Path(__file__).resolve().parent
_CANDIDATE_DATASETS = [
    _HERE.parent / "sample_data" / "regression",  # examples/sample_data/regression
    Path("sample_data") / "regression",            # when CWD == examples/
]

dataset: object
for candidate in _CANDIDATE_DATASETS:
    if candidate.exists():
        dataset = str(candidate)
        break
else:
    # Synthetic fallback so the example always runs end-to-end.
    dataset = nirs4all.generate.regression(n_samples=200, n_features=200)


# nirs4all pipeline: a simple CV splitter feeding the AOM_lib backend.
pipeline = [
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=0),
    {"model": AOMPLSAomlibRegressor(n_components=10, cv=3, random_state=0)},
]


result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    name="AOMLib-AOMPLS-Demo",
    verbose=1,
)


print("\n=== AOM_lib AOM-PLS via nirs4all ===")
print(f"Best RMSE     : {result.best_rmse:.4f}")
print(f"Best R^2      : {result.best_r2:.4f}")

# The wrapper records the operator selected by the C++ CV loop; we surface
# it here so the paper can reproduce the per-run preprocessing choice.
best = result.top(1)[0] if result.top(1) else {}
selected_op = best.get("selected_operator_sequence", None)
if selected_op is not None:
    print(f"Selected op   : {selected_op}")
else:
    print("Selected op   : (see nirs4all run metadata for per-fold details)")
