"""End-to-end native ``.n4a`` export for one captured dag-ml REFIT model."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.operators.transforms.scalers import StandardNormalVariate as SNV
from nirs4all.pipeline.bundle import BundleLoader
from nirs4all.pipeline.dagml.run_backend import run_via_dagml

from ._datasets import dataset_path


def _dagml_cli() -> Path:
    configured = os.environ.get("N4A_DAGML_CLI")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3].parent / "dag-ml" / "target" / "release" / "dag-ml-cli"


@pytest.mark.parity
def test_native_single_model_bundle_export_never_refits_legacy(tmp_path, monkeypatch) -> None:
    cli = _dagml_cli()
    if not cli.exists():
        pytest.skip(f"dag-ml-cli binary unavailable: {cli}")
    pipeline = [SNV(), KFold(n_splits=3, shuffle=True, random_state=42), {"model": PLSRegression(n_components=5)}]
    result = run_via_dagml(
        pipeline,
        dataset_path("regression"),
        workdir=tmp_path / "work",
        dagml_cli=str(cli),
        venv_python=sys.executable,
        results_path=tmp_path / "results",
        random_state=42,
    )
    assert result._dagml_results_dir is not None  # noqa: SLF001
    assert len(result._dagml_refit_artifacts) == 1  # noqa: SLF001

    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setattr(run_module, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy run used")))
    bundle = result.export(tmp_path / "native.n4a")
    assert result._dagml_legacy_result is None  # noqa: SLF001

    dataset = dataset_path("regression")
    from nirs4all.data.config import DatasetConfigs

    source = DatasetConfigs(dataset).get_dataset_at(0)
    test_ids = [int(sample) for sample in source.index_column("sample", {"partition": "test"})]
    x_test = np.asarray(source.x_rows(test_ids, layout="2d"))
    expected_rows = result.predictions.filter_predictions(partition="test", fold_id="final")
    assert len(expected_rows) == 1
    expected = np.asarray(expected_rows[0]["y_pred"], dtype=float).ravel()
    actual = np.asarray(BundleLoader(bundle).predict(x_test), dtype=float).ravel()
    assert np.allclose(actual, expected, atol=1e-6)
