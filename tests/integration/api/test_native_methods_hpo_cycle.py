"""Opt-in real API-001 HPO then DAG-001 detached replay witness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

_REQUIRE_ENV = "NIRS4ALL_REQUIRE_NATIVE_METHODS_HPO"
_LIBRARY_ENV = "NIRS4ALL_CORE_LIVE_METHODS_LIBRARY"


@pytest.mark.methods
@pytest.mark.skipif(
    os.environ.get(_REQUIRE_ENV) != "1",
    reason=f"set {_REQUIRE_ENV}=1 with the selected native wheels to run",
)
def test_native_methods_hpo_exports_a_fresh_process_replayable_winner(
    tmp_path: Path,
) -> None:
    import nirs4all

    library = os.environ.get(_LIBRARY_ENV)
    if not library or not Path(library).is_file():
        pytest.fail(f"{_LIBRARY_ENV} must identify the exact Methods runtime file")
    rng = np.random.default_rng(11)
    features = rng.normal(size=(18, 4))
    targets = 1.5 * features[:, 0] - 0.7 * features[:, 1] + 0.05 * rng.normal(size=18)
    pipeline = [
        KFold(n_splits=3, shuffle=True, random_state=9),
        {
            "model": PLSRegression(n_components=1),
            "finetune_params": {
                "engine": "n4m",
                "n_trials": 3,
                "sampler": "random",
                "pruner": "none",
                "approach": "grouped",
                "seed": 7,
                "metric": "rmse",
                "direction": "minimize",
                "model_params": {"n_components": ["int", 1, 3]},
            },
        },
    ]
    result = nirs4all.run(
        pipeline,
        {
            "X": features,
            "y": targets,
            "sample_ids": [f"train.{index}" for index in range(len(features))],
        },
        engine="native",
        save_charts=False,
        methods_library_path=library,
    )

    assert result.tuning_best_params == {"model.n_components": 3}
    assert result.tuning_best_value == pytest.approx(0.08963750601197541)
    assert result._native_outcome["methods_hpo_resume_state"]["trial_history_len"] == 3
    assert result._native_outcome["diagnostics"]["nirs4all_execution"] == ("methods_controller_owned_hpo_archive_v2")
    archive = result.export(tmp_path / "hpo-winner.n4a")
    result.close()
    assert result.native_execution_is_live is False

    script = """
import json
import sys
import nirs4all

session = nirs4all.load_session(sys.argv[1])
prediction = session.predict(
    {"X": [[0.2, -0.1, 0.4, 0.0], [-0.5, 0.3, 0.1, 0.2]],
     "sample_ids": ["fresh.0", "fresh.1"]},
    engine="native",
    methods_library_path=sys.argv[2],
)
print(json.dumps({"sample_ids": prediction.metadata["sample_ids"],
                  "target_names": prediction.metadata["target_names"],
                  "y_pred": prediction.y_pred.tolist()}, sort_keys=True))
session.close()
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(archive), library],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    replay = json.loads(completed.stdout)
    assert replay["sample_ids"] == ["fresh.0", "fresh.1"]
    assert replay["target_names"] == ["y"]
    np.testing.assert_allclose(
        replay["y_pred"],
        [[0.3797156088884486], [-0.9579359466789124]],
        atol=1.0e-12,
        rtol=1.0e-12,
    )
