"""Public process-boundary coverage for the strict native full-refit lane."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def _native_v3_ready() -> bool:
    """Return whether this interpreter has the explicitly provisioned lane."""

    if not os.environ.get("N4M_LIB_PATH"):
        return False
    try:
        import dag_ml
        import n4m  # noqa: F401
        import nirs4all_core  # noqa: F401
    except ImportError:
        return False
    return all(
        callable(getattr(dag_ml, name, None))
        for name in (
            "execute_methods_portable_full_refit",
            "build_archive_v3_native_refit_payloads",
            "replay_loaded_methods_portable_refit_package_v3",
        )
    )


@pytest.mark.skipif(not _native_v3_ready(), reason="native Methods V3 runtime is not provisioned")
def test_native_full_refit_archive_v3_replays_in_a_fresh_process(tmp_path: Path) -> None:
    """Train → V3 refit → archive → fresh-process PREDICT never touches legacy."""

    import nirs4all

    X = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]]
    )
    y = np.arange(6.0, dtype=float)
    parent = nirs4all.run(
        [KFold(n_splits=3), {"model": PLSRegression(n_components=1)}],
        {"X": X, "y": y, "sample_ids": [f"fit-{index}" for index in range(len(X))]},
        engine="native",
        save_charts=False,
    )
    child = nirs4all.retrain(
        parent,
        {
            "X": X + 0.1,
            "y": y + 0.25,
            "sample_ids": [f"refit-{index}" for index in range(len(X))],
        },
        engine="native",
        verbose=0,
    )
    archive = tmp_path / "full-refit.n4a"
    assert child.export(archive) == archive

    prediction_X = X + 0.2
    prediction_ids = [f"predict-{index}" for index in range(len(X))]
    direct = nirs4all.predict(
        child,
        {"X": prediction_X, "sample_ids": prediction_ids},
        engine="native",
    )
    script = """
import json
import os
import numpy as np
import nirs4all
loaded = nirs4all.load_session(os.environ['N4A_V3_ARCHIVE'], engine='native')
assert loaded.archive_schema_version == 3
values = np.asarray(json.loads(os.environ['N4A_V3_VALUES']), dtype=float)
ids = json.loads(os.environ['N4A_V3_IDS'])
result = nirs4all.predict(data={'X': values, 'sample_ids': ids}, session=loaded, engine='native')
print(json.dumps({'sample_ids': result.metadata['sample_ids'], 'values': result.y_pred.tolist()}))
"""
    environment = dict(os.environ)
    environment.update(
        {
            "N4A_V3_ARCHIVE": str(archive),
            "N4A_V3_VALUES": json.dumps(prediction_X.tolist()),
            "N4A_V3_IDS": json.dumps(prediction_ids),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    observed = json.loads(completed.stdout)
    assert observed["sample_ids"] == prediction_ids
    np.testing.assert_allclose(np.asarray(observed["values"]), direct.y_pred, rtol=0.0, atol=1e-12)


@pytest.mark.skipif(not _native_v3_ready(), reason="native Methods V3 runtime is not provisioned")
def test_native_pls_ridge_stack_full_refit_replays_in_a_fresh_process(tmp_path: Path) -> None:
    """The R2 PLS×2→Ridge path survives archive and process boundaries natively."""

    import nirs4all

    X = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [4.0, 1.0, 0.0],
            [5.0, 0.0, 1.0],
            [6.0, 1.0, 0.0],
            [7.0, 0.0, 1.0],
        ]
    )
    y = np.asarray([0.1, 1.2, 1.9, 3.1, 4.0, 5.2, 5.9, 7.1])
    pipeline = [
        KFold(n_splits=2),
        {
            "branch": [
                [{"model": PLSRegression(n_components=1)}],
                [{"model": PLSRegression(n_components=1)}],
            ]
        },
        {"merge": "predictions"},
        {"model": Ridge(alpha=0.25)},
    ]
    parent = nirs4all.run(
        pipeline,
        {"X": X, "y": y, "sample_ids": [f"stack-fit-{index}" for index in range(len(X))]},
        engine="native",
        save_charts=False,
    )
    child = nirs4all.retrain(
        parent,
        {
            "X": X + np.asarray([0.05, 0.0, 0.0]),
            "y": y + 0.2,
            "sample_ids": [f"stack-refit-{index}" for index in range(len(X))],
        },
        engine="native",
        verbose=0,
    )
    archive = tmp_path / "pls-ridge-stack-full-refit.n4a"
    assert child.export(archive) == archive

    prediction_X = X + np.asarray([0.15, 0.0, 0.0])
    prediction_ids = [f"stack-predict-{index}" for index in range(len(X))]
    direct = nirs4all.predict(
        child,
        {"X": prediction_X, "sample_ids": prediction_ids},
        engine="native",
    )
    script = """
import json
import os
import numpy as np
import nirs4all
loaded = nirs4all.load_session(os.environ['N4A_V3_ARCHIVE'], engine='native')
assert loaded.archive_schema_version == 3
values = np.asarray(json.loads(os.environ['N4A_V3_VALUES']), dtype=float)
ids = json.loads(os.environ['N4A_V3_IDS'])
result = nirs4all.predict(data={'X': values, 'sample_ids': ids}, session=loaded, engine='native')
print(json.dumps({'sample_ids': result.metadata['sample_ids'], 'values': result.y_pred.tolist()}))
"""
    environment = dict(os.environ)
    environment.update(
        {
            "N4A_V3_ARCHIVE": str(archive),
            "N4A_V3_VALUES": json.dumps(prediction_X.tolist()),
            "N4A_V3_IDS": json.dumps(prediction_ids),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    observed = json.loads(completed.stdout)
    assert observed["sample_ids"] == prediction_ids
    np.testing.assert_allclose(np.asarray(observed["values"]), direct.y_pred, rtol=0.0, atol=1e-12)
