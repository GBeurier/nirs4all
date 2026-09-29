"""Qualify the installed four-modality U07 demo across separate processes.

Run from outside the source checkout after installing candidate wheels. This
script does not publish packages or require any real dataset.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples/user/02_data_handling/U07_multimodal.py"


def run(*args: str, cwd: Path, env: dict[str, str]) -> dict:
    """Run one isolated Python process and decode its JSON report."""
    completed = subprocess.run(
        [sys.executable, "-I", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Qualification process failed ({completed.returncode}):\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def main() -> None:
    """Check installed imports, durable tuning and prediction-only replay."""
    with tempfile.TemporaryDirectory(prefix="n4a-installed-u07-") as directory:
        root = Path(directory)
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        env["MPLCONFIGDIR"] = str(root / "matplotlib")
        origins = run(
            "-c",
            'import json, nirs4all, nirs4all_io, dag_ml; print(json.dumps({name: module.__file__ for name, module in (("nirs4all", nirs4all), ("nirs4all_io", nirs4all_io), ("dag_ml", dag_ml))}))',
            cwd=root,
            env=env,
        )
        for name, origin in origins.items():
            if Path(origin).resolve().is_relative_to(REPO):
                raise RuntimeError(f"{name} imported from the source checkout: {origin}")

        output = root / "run"
        stopped = run(str(EXAMPLE), "--output", str(output), "--stop-after", "2", cwd=root, env=env)
        assert stopped["status"] == "cancelled" and stopped["completed_trials"] == 2

        run(str(EXAMPLE), "--output", str(output), "--resume", cwd=root, env=env)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["engine"] == "dag-ml" and len(report["tuning"]["trials"]) == 8
        assert set(report["raw_shapes"]) == {"nir", "image", "series", "metadata"}
        assert report["training_performed_on_reload"] is False
        assert math.isfinite(report["cv_rmse"]) and math.isfinite(report["test_rmse"])

        replay = run(str(EXAMPLE), "--output", str(root / "replay"), "--replay", str(output / "multimodal.n4a"), cwd=root, env=env)
        assert replay["training_performed_on_reload"] is False
        assert len(replay["new_predictions"]) == 12
        assert replay["new_predictions"] == report["new_predictions"]
        print(json.dumps({"status": "INSTALLED_U07_OK", "trial_count": 8, "prediction_count": 12, "fit_on_replay": False, "origins": origins}, sort_keys=True))


if __name__ == "__main__":
    main()
