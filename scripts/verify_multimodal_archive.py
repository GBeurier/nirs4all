"""Verify a synthetic demo archive in an installed environment, forbidding fit.

Run with the isolated environment's Python and ``-I``. Only the copied archive,
prediction_dataset.json and report.json are needed. Repeated --deny-root options
can forbid both the original repository tree and training output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--deny-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    roots = [str(root.resolve()) for root in args.deny_root]

    def audit(event: str, values: tuple[Any, ...]) -> None:
        if event in {"open", "os.listdir", "os.scandir"} and values and isinstance(values[0], str | bytes | os.PathLike):
            candidate = os.path.abspath(os.fsdecode(values[0]))
            if any(candidate == root or candidate.startswith(root + os.sep) for root in roots):
                raise AssertionError(f"replay accessed a forbidden training/source directory: {candidate}")

    sys.addaudithook(audit)
    import numpy as np
    from nirs4all_io import MultimodalDataset
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    import nirs4all
    from nirs4all.operators.models import MBPLS, MultimodalClassifier, MultimodalRegressor, TensorPCA
    from nirs4all.pipeline.runner import PipelineRunner

    dataset = MultimodalDataset.from_dict(json.loads(args.dataset.read_text()))
    expected_report = json.loads(args.expected.read_text())
    expected = np.asarray(expected_report["new_predictions"])

    def forbidden(*unused_args: Any, **unused_kwargs: Any) -> Any:
        raise AssertionError("archive replay attempted training")

    with ExitStack() as patches:
        for operator in (MultimodalRegressor, MultimodalClassifier, TensorPCA, PCA, Ridge, LogisticRegression, MBPLS, StandardScaler, OneHotEncoder, ColumnTransformer):
            for method in ("fit", "fit_transform", "partial_fit"):
                if hasattr(operator, method):
                    patches.enter_context(patch.object(operator, method, forbidden))
        patches.enter_context(patch.object(PipelineRunner, "run", forbidden))
        prediction = nirs4all.predict(args.archive, dataset, verbose=0)
    actual = np.asarray(prediction.y_pred)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    maximum_error: float | None = None
    if expected_report.get("task_type") != "classification" and expected.dtype.kind in "biuf":
        actual = np.asarray(actual, dtype=float)
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
        maximum_error = float(np.max(np.abs(actual - expected)))
    else:
        np.testing.assert_array_equal(actual, expected)
    assert prediction.metadata["training_performed"] is False
    assert prediction.metadata["artifact_integrity_verified"] is True
    packages = ("nirs4all", "nirs4all-io", "dag-ml", "dag-ml-data", "nirs4all-methods", "numpy", "scikit-learn")
    print(json.dumps({
        "status": "passed", "predictions": actual.size, "prediction_shape": list(actual.shape),
        "comparison": "numeric_tolerance" if maximum_error is not None else "exact_labels", "max_absolute_error": maximum_error,
        "fit_calls": 0, "artifact_integrity_verified": True, "forbidden_roots": roots,
        "nirs4all_module": nirs4all.__file__, "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
