"""Bounded native parallel HPO over early and learned late typed fusion.

Requires actual sequential-CPU Methods build capabilities and matching DAG-ML.
The deterministic U07 cohort validates software; it is not a scientific dataset.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import nirs4all

_PATH = Path(__file__).with_name("U21_structural_hpo_early_late.py")
_SPEC = importlib.util.spec_from_file_location("parallel_topology_fixture", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_fixture)


def make_tuning(directory: Path, *, workers: int = 2, resume: bool = False) -> dict[str, Any]:
    """Keep all native topology/conditional-axis declarations and bound workers."""
    return {**_fixture.make_tuning(directory, resume=resume), "n_jobs": workers, "sampler": "random", "pruner": None, "study_name": "parallel-typed-topologies", "n_trials": 7}


def main(output_path: str | Path | None = None, *, workers: int = 2) -> Path | None:
    """Run native joined windows, then replay the portable fitted closure."""
    temporary = tempfile.TemporaryDirectory(prefix="n4a-parallel-typed-") if output_path is None else None
    output = Path(temporary.name) if temporary is not None else Path(str(output_path))
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "training-workspace"
    new = _fixture.make_dataset(29, prediction=True)
    try:
        with nirs4all.run(
            _fixture.make_pipeline(),
            _fixture.make_dataset(),
            tuning=make_tuning(output / "study", workers=workers),
            engine="dag-ml",
            workspace_path=workspace,
            refit=True,
            random_state=17,
            cpu_threads=1,
            gpu_devices=[],
            save_charts=False,
            verbose=0,
        ) as result:
            archive = result.export(output / "parallel-winner.n4a")
            expected = nirs4all.predict(archive, new, engine="dag-ml").y_pred
            audit = result.structural_tuning_candidate_audit
            assert len(audit) == 7 and all(candidate["closed"] for candidate in audit)
            print(f"Selected grouped CV RMSE: {result.tuning_best_value:.6f}")
            print(f"Active alphas: {result.tuning_best_params}; closed candidates: {len(audit)}")
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.rmtree(output / "study")
        actual = nirs4all.predict(archive, new, engine="dag-ml")
        np.testing.assert_array_equal(actual.y_pred, expected)
        assert actual.metadata["training_performed"] is False
        print("Complete raw-input winner replayed without training state.")
        return archive if temporary is None else None
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Directory for the retained portable winner archive")
    parser.add_argument("--workers", choices=(2, 3, 4), type=int, default=2, help="Native candidate worker bound")
    arguments = parser.parse_args()
    main(arguments.output, workers=arguments.workers)
