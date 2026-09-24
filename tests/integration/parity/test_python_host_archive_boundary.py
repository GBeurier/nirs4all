"""Python model replay works on both DAG mechanisms without claiming portable Raw."""

import json
import zipfile

import numpy as np
import pytest
from sklearn.linear_model import Ridge

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.bundle.loader import BundleLoader
from nirs4all.pipeline.dagml.rt import RtError

from ._dagml_cli import dagml_cli_path


@pytest.mark.parity
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
def test_sklearn_host_archive_replays_on_python_but_does_not_claim_raw_portability(tmp_path, monkeypatch, mechanism: str) -> None:
    if mechanism == "subprocess":
        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")

    rng = np.random.default_rng(21)
    features = rng.normal(size=(20, 5))
    targets = 2 * features[:, 0] - features[:, 1] + 0.1 * rng.normal(size=20)

    def dataset() -> SpectroDataset:
        result = SpectroDataset("python_host_archive")
        result.add_samples(features[:16], {"partition": "train"}, headers=[str(index) for index in range(5)])
        result.add_samples(features[16:], {"partition": "test"})
        result.add_targets(targets.reshape(-1, 1))
        return result

    expected = Ridge(alpha=1).fit(features[:16], targets[:16]).predict(features[16:])
    legacy = nirs4all.run([Ridge(alpha=1)], dataset(), engine="legacy",
                          workspace_path=tmp_path / "legacy", save_charts=False, verbose=0)
    try:
        archive = legacy.export(tmp_path / "legacy.n4a")
        np.testing.assert_allclose(np.asarray(BundleLoader(archive).predict(features[16:])).ravel(), expected, atol=1e-7)
        with pytest.raises(RtError, match="legacy .n4a model requires conversion"):
            nirs4all.predict(archive, features[16:])
    finally:
        legacy.close()

    native = nirs4all.run([Ridge(alpha=1)], dataset(), engine="dag-ml", allow_fallback=False,
                          workspace_path=tmp_path / "native", save_charts=False, verbose=0)
    try:
        assert native.execution_engine == "dag-ml"
        archive = native.export(tmp_path / "native.n4a")
        np.testing.assert_allclose(np.asarray(nirs4all.predict(archive, features[16:]).y_pred).ravel(), expected, atol=1e-6)
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
            assert manifest["source_type"] == "dagml_native"
            assert "dagml_initial_full_refit_package_ref" not in manifest
            assert len([name for name in bundle.namelist() if name.startswith("artifacts/") and name.endswith(".joblib")]) == 1
    finally:
        native.close()
