"""Public oracle for a sequential classifier followed by a MetaModel."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from nirs4all.operators.models import MetaModel


@pytest.mark.parametrize("mechanism", ["pyo3", "cli"])
@pytest.mark.parametrize("use_proba", [False, True])
def test_sequential_classification_metamodel_uses_native_oof_or_refuses_proba(use_proba, mechanism, monkeypatch):
    import nirs4all

    if mechanism == "cli":
        from tests.integration.parity._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "pyo3" else "0")

    rng = np.random.default_rng(79)
    features = rng.normal(size=(30, 6))
    targets = (features[:, 0] + features[:, 1] > 0).astype(int)
    pipeline = [
        StratifiedKFold(2, shuffle=True, random_state=42),
        LogisticRegression(max_iter=300),
        {"model": MetaModel(model=LogisticRegression(max_iter=300), use_proba=use_proba)},
    ]
    legacy = nirs4all.run(pipeline, (features, targets), engine="legacy", refit=False,
                          save_artifacts=False, save_charts=False, verbose=0)
    assert legacy.cv_best_score == pytest.approx(0.9333333333333333)

    if use_proba:
        with pytest.raises(Exception, match="sequential MetaModel.*probability features"):
            nirs4all.run(pipeline, (features, targets), engine="dag-ml", refit=False,
                         save_artifacts=False, save_charts=False, verbose=0)
    else:
        native = nirs4all.run(pipeline, (features, targets), engine="dag-ml", refit=False,
                             save_artifacts=False, save_charts=False, verbose=0)
        assert native.cv_best_score == pytest.approx(legacy.cv_best_score, abs=1e-8)
        assert native.cv_best["model_name"] == "MetaModel_LogisticRegression"
