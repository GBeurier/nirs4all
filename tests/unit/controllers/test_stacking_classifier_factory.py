"""Runtime classification hints must not become sklearn constructor arguments."""

from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from nirs4all.controllers.models.factory import ModelFactory


def _classifier():
    return StackingClassifier(
        estimators=[("linear", LogisticRegression()), ("tree", RandomForestClassifier(n_estimators=3, max_depth=2, random_state=6))],
        final_estimator=LogisticRegression(), cv=2,
    )


def test_meta_estimator_filters_runtime_hints_and_preserves_nested_parameters():
    model = _classifier()
    rebuilt = ModelFactory.build_single_model(
        {"model_instance": model}, SimpleNamespace(is_classification=True, num_classes=2),
        force_params={"final_estimator__C": .25},
    )
    assert isinstance(rebuilt, StackingClassifier)
    assert rebuilt.get_params()["final_estimator__C"] == .25
    assert [name for name, _ in rebuilt.estimators] == ["linear", "tree"]
    assert "num_classes" not in rebuilt.get_params()


def test_real_stacking_classifier_pipeline(tmp_path):
    import nirs4all

    X = np.random.default_rng(54).normal(size=(48, 4))
    y = (X[:, 0] > 0).astype(int)
    with nirs4all.run([_classifier()], (X, y, {"train": 40}), engine="legacy", workspace_path=tmp_path,
                     verbose=0, save_charts=False, refit=False) as result:
        predictions = result.predictions.filter_predictions(partition="test", load_arrays=True)
        assert predictions and len(predictions[0]["y_pred"]) == 8
        assert set(np.asarray(predictions[0]["y_pred"]).ravel()) <= {0, 1}
