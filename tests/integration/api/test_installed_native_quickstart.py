"""Installed public quickstart must work without native-library path overrides."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.methods
@pytest.mark.skipif(
    os.environ.get("NIRS4ALL_REQUIRE_NATIVE_ARCHIVE_V2") != "1",
    reason="set NIRS4ALL_REQUIRE_NATIVE_ARCHIVE_V2=1 with installed native wheels",
)
def test_installed_quickstart_and_fresh_replay_without_native_overrides(tmp_path):
    # -I intentionally ignores source-tree PYTHONPATH: this gate requires a
    # genuinely installed library/runtime stack in the tested interpreter.
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("N4M_") and key not in {"LD_LIBRARY_PATH", "PYTHONPATH", "NIRS4ALL_CORE_LIVE_METHODS_LIBRARY", "N4A_ENGINE"}
    }
    script = """
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import nirs4all

archive = Path(sys.argv[1])
if sys.argv[2] == "train":
    rng = np.random.default_rng(73)
    X = rng.normal(size=(30, 4))
    y = 2 * X[:, 0] - 0.6 * X[:, 1] + 0.03 * rng.normal(size=30)
    with nirs4all.run([KFold(3), {"model": PLSRegression(2)}],
                     {"X": X, "y": y, "sample_ids": [f"train.{i}" for i in range(30)]},
                     verbose=0) as result:
        assert result.execution_engine == "native"
        result.export(archive)
with nirs4all.load_session(archive) as session:
    prediction = session.predict({"X": [[0.2, -0.1, 0.4, 0.0]], "sample_ids": ["new.0"]})
    print(json.dumps({"sample_ids": prediction.metadata["sample_ids"], "y_pred": prediction.y_pred.tolist()}))
"""
    archive = tmp_path / "installed.n4a"
    results = []
    for operation in ("train", "replay"):
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", script, str(archive), operation],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
        assert observed["sample_ids"] == ["new.0"]
        values = np.asarray(observed["y_pred"])
        assert values.shape == (1, 1)
        assert np.isfinite(values).all()
        results.append(values)
    np.testing.assert_allclose(results[0], results[1], atol=1e-12, rtol=1e-12)
