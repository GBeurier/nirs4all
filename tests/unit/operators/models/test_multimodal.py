"""Source-local fitting, tensor boundaries and persistence for multimodal models."""

import io

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier, is_regressor
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor, TensorPCA
from nirs4all.operators.models.sklearn.mbpls import MBPLS


class RecordingTensorPCA(TensorPCA):
    """Observe the encoder boundary without retaining samples."""

    def fit(self, X, y=None):
        self.seen_shape_ = X.shape
        self.target_sum_ = np.sum(y)
        return super().fit(X, y)


@pytest.fixture
def multimodal_data():
    rng = np.random.default_rng(42)
    n_samples = 36
    spectral = rng.normal(size=(n_samples, 8))
    image = rng.normal(size=(n_samples, 3, 4, 2))
    temporal = rng.normal(size=(n_samples, 6, 2))
    tabular = pd.DataFrame({"temperature": rng.normal(size=n_samples), "instrument": np.tile(["a", "b", "c"], 12)})
    y = 2 * spectral[:, 0] + image[:, 0, 0, 0] - temporal[:, 0, 0] + tabular["temperature"].to_numpy()
    return [spectral, image, temporal, tabular], y


def make_regressor(fusion="early"):
    tabular = ColumnTransformer([
        ("numeric", StandardScaler(), ["temperature"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["instrument"]),
    ])
    return MultimodalRegressor(
        transformers={
            "nirs": StandardScaler(),
            "image": RecordingTensorPCA(3),
            "temporal": make_pipeline(RecordingTensorPCA(2), StandardScaler()),
            "tabular": tabular,
        },
        model=Ridge(alpha=0.7) if fusion == "early" else MBPLS(n_components=2),
        fusion=fusion,
        source_weights={"image": 0.5, "temporal": 0.25},
    )


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
def test_fusion_matches_explicit_source_pipelines(multimodal_data, fusion):
    blocks, y = multimodal_data
    model = make_regressor(fusion).fit(blocks, y)
    encoded = []
    for (name, transformer), block in zip(model.transformers.items(), blocks, strict=True):
        fitted = clone(transformer).fit(block, y)
        encoded.append(fitted.transform(block) * model.source_weights_.get(name, 1.0))
    expected_model = clone(model.model).fit(np.concatenate(encoded, axis=1) if fusion == "early" else encoded, y)
    expected = expected_model.predict(np.concatenate(encoded, axis=1) if fusion == "early" else encoded)
    np.testing.assert_allclose(model.predict(blocks), expected)
    assert is_regressor(model)
    assert model.source_names_ == ("nirs", "image", "temporal", "tabular")
    assert model.input_shapes_ == {"nirs": (8,), "image": (3, 4, 2), "temporal": (6, 2), "tabular": (2,)}


def test_fold_local_encoders_receive_raw_rank_and_targets(multimodal_data):
    blocks, y = multimodal_data
    train = [block.iloc[:24] if isinstance(block, pd.DataFrame) else block[:24] for block in blocks]
    model = make_regressor().fit(train, y[:24])
    image = model.transformers_["image"]
    temporal = model.transformers_["temporal"].steps[0][1]
    assert image.seen_shape_ == (24, 3, 4, 2)
    assert temporal.seen_shape_ == (24, 6, 2)
    assert image.target_sum_ == pytest.approx(y[:24].sum())
    assert temporal.target_sum_ == pytest.approx(y[:24].sum())
    np.testing.assert_allclose(image.pca_.mean_, train[1].reshape(24, -1).mean(axis=0))
    np.testing.assert_allclose(model.transformers_["nirs"].mean_, train[0].mean(axis=0))
    assert not hasattr(model.transformers["image"], "pca_")
    assert not hasattr(model.model, "coef_")


def test_clone_and_nested_parameter_updates(multimodal_data):
    blocks, y = multimodal_data
    model = make_regressor().fit(blocks, y)
    fresh = clone(model)
    assert not hasattr(fresh, "model_")
    assert fresh.transformers["image"] is not model.transformers["image"]
    fresh.set_params(
        transformers__image__n_components=2,
        transformers__temporal__recordingtensorpca__n_components=1,
        model__alpha=2.0,
        source_weights__image=0.0,
    )
    params = fresh.get_params(deep=True)
    assert params["transformers__image__n_components"] == 2
    assert params["transformers__temporal__recordingtensorpca__n_components"] == 1
    assert params["model__alpha"] == 2.0
    assert params["source_weights__image"] == 0.0
    fresh.fit(blocks, y)
    assert fresh.output_widths_["image"] == 2
    assert model.output_widths_["image"] == 3
    assert model.source_weights["image"] == 0.5


def test_replacements_apply_before_nested_parameters():
    model = MultimodalRegressor({"image": None}, Ridge())
    model.set_params(transformers__image__n_components=2, transformers__image=TensorPCA(1), model__alpha=3.0, model=Ridge())
    assert model.transformers["image"].n_components == 2
    assert model.model.alpha == 3.0
    model.set_params(source_weights__image=0.5)
    assert model.source_weights == {"image": 0.5}


@pytest.mark.parametrize("params", [
    {"transformers__missing__n_components": 1},
    {"source_weights__missing": 0.5},
    {"source_weights__image__invalid": 0.5},
    {"unknown": True},
    {"transformers__image__invalid": 2},
])
def test_invalid_parameters_are_rejected(params):
    with pytest.raises(ValueError):
        make_regressor().set_params(**params)


@pytest.mark.parametrize("change,match", [
    (lambda blocks: blocks[:-1], "Expected 4 source blocks"),
    (lambda blocks: blocks + [blocks[0]], "Expected 4 source blocks"),
    (lambda blocks: [blocks[0], blocks[1].reshape(36, 4, 3, 2), *blocks[2:]], "expected input shape"),
    (lambda blocks: [blocks[0], blocks[1][:, :, :, 0], *blocks[2:]], "expected input shape"),
    (lambda blocks: [blocks[0][:-1], *blocks[1:]], "inconsistent numbers of samples"),
    (lambda blocks: tuple(blocks), "requires a list"),
])
def test_prediction_refuses_changed_source_contract(multimodal_data, change, match):
    blocks, y = multimodal_data
    model = make_regressor().fit(blocks, y)
    with pytest.raises(ValueError, match=match):
        model.predict(change(blocks))


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
def test_joblib_replay_never_fits(multimodal_data, monkeypatch, fusion):
    blocks, y = multimodal_data
    model = make_regressor(fusion).fit(blocks, y)
    expected = model.predict(blocks)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Loading or predicting must not fit an estimator")

    for estimator in (TensorPCA, PCA, StandardScaler, ColumnTransformer, Ridge, MBPLS):
        monkeypatch.setattr(estimator, "fit", forbidden_fit)
    restored = joblib.load(buffer)
    np.testing.assert_allclose(restored.predict(blocks), expected)
    assert not {"X_", "y_", "training_data_"}.intersection(vars(restored))
    assert not {"X_", "y_", "training_data_"}.intersection(vars(restored.transformers_["image"]))


def test_tensor_pca_uses_sklearn_projection_and_strict_shape(multimodal_data):
    blocks, y = multimodal_data
    image = blocks[1]
    encoder = TensorPCA(3, whiten=True).fit(image, y)
    expected = PCA(3, whiten=True).fit(image.reshape(36, -1), y)
    np.testing.assert_allclose(encoder.transform(image[:5]), expected.transform(image[:5].reshape(5, -1)))
    assert encoder.input_shape_ == (3, 4, 2)
    with pytest.raises(ValueError, match="expected input shape"):
        encoder.transform(image.reshape(36, 24))


def test_unfitted_operators_refuse_prediction(multimodal_data):
    blocks, _ = multimodal_data
    with pytest.raises(NotFittedError):
        make_regressor().predict(blocks)
    with pytest.raises(NotFittedError):
        TensorPCA(2).transform(blocks[1])


def test_passthrough_requires_explicit_tensor_encoder(multimodal_data):
    blocks, y = multimodal_data
    model = MultimodalRegressor({"nirs": "passthrough"}, Ridge()).fit([blocks[0]], y)
    assert model.predict([blocks[0]]).shape == y.shape
    with pytest.raises(ValueError, match="numeric 2-D representation"):
        MultimodalRegressor({"image": None}, Ridge()).fit([blocks[1]], y)


@pytest.mark.parametrize("weights", [{"missing": 1}, {"nirs": -1}, {"nirs": np.inf}, {"nirs": np.nan}])
def test_invalid_source_weights_fail_before_training(multimodal_data, weights):
    blocks, y = multimodal_data
    with pytest.raises(ValueError, match="[Ss]ource weights"):
        MultimodalRegressor({"nirs": None}, Ridge(), source_weights=weights).fit([blocks[0]], y)


def test_sparse_categorical_encoding_is_accepted():
    categories = np.array([["a"], ["b"], ["a"], ["b"]])
    model = MultimodalRegressor({"category": OneHotEncoder(handle_unknown="ignore")}, Ridge()).fit([categories], [1, 2, 1, 2])
    assert model.predict([np.array([["unknown"]])]).shape == (1,)


class BlockLogisticClassifier(ClassifierMixin, BaseEstimator):
    """Real list-input classifier exposing a deliberately reversed class order."""

    def fit(self, X, y):
        assert isinstance(X, list)
        self.model_ = LogisticRegression(max_iter=300).fit(np.concatenate(X, axis=1), y)
        self.classes_ = self.model_.classes_[::-1].copy()
        return self

    def predict(self, X):
        return self.model_.predict(np.concatenate(X, axis=1))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.concatenate(X, axis=1))[:, ::-1]


def make_classifier(fusion="early"):
    regressor = make_regressor(fusion)
    # Plain TensorPCA preserves text labels; the regression recording fixture
    # above intentionally computes a numeric target sum.
    return MultimodalClassifier(
        {**regressor.transformers, "image": TensorPCA(3), "temporal": make_pipeline(TensorPCA(2), StandardScaler())},
        LogisticRegression(max_iter=300) if fusion == "early" else BlockLogisticClassifier(),
        fusion=fusion,
        source_weights=regressor.source_weights,
    )


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
@pytest.mark.parametrize("class_names", [[-8, 4, 37], ["zebra", "apple", "pear"]])
def test_classifier_preserves_labels_and_probability_class_order(multimodal_data, fusion, class_names):
    blocks, targets = multimodal_data
    labels = np.asarray(class_names)[np.digitize(targets, np.quantile(targets, [1 / 3, 2 / 3]))]
    classifier = make_classifier(fusion).fit(blocks, labels)
    encoded = []
    for (name, transformer), block in zip(classifier.transformers.items(), blocks, strict=True):
        fitted = clone(transformer).fit(block, labels)
        encoded.append(fitted.transform(block) * classifier.source_weights_.get(name, 1.0))
    fused = np.concatenate(encoded, axis=1) if fusion == "early" else encoded
    expected = clone(classifier.model).fit(fused, labels)
    np.testing.assert_array_equal(classifier.classes_, expected.classes_)
    np.testing.assert_array_equal(classifier.predict(blocks), expected.predict(fused))
    probabilities = classifier.predict_proba(blocks)
    np.testing.assert_allclose(probabilities, expected.predict_proba(fused))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(labels)))
    np.testing.assert_array_equal(classifier.classes_[probabilities.argmax(axis=1)], classifier.predict(blocks))
    assert probabilities.shape == (36, 3)
    assert classifier.input_shapes_["image"] == (3, 4, 2)
    assert classifier.input_shapes_["temporal"] == (6, 2)
    assert is_classifier(classifier) and not is_regressor(classifier)


def test_classifier_fold_local_fit_does_not_learn_held_out_class_or_statistics(multimodal_data, monkeypatch):
    blocks, _ = multimodal_data
    labels = np.asarray(["train_a", "train_b"] * 12 + ["heldout_only"] * 12)
    train = [block.iloc[:24] if isinstance(block, pd.DataFrame) else block[:24] for block in blocks]
    seen = []
    original_fit = TensorPCA.fit

    def observe_fit(self, X, y=None):
        seen.append((X.shape, y.copy()))
        return original_fit(self, X, y)

    monkeypatch.setattr(TensorPCA, "fit", observe_fit)
    classifier = make_classifier().fit(train, labels[:24])
    assert [shape for shape, _ in seen] == [(24, 3, 4, 2), (24, 6, 2)]
    for _, targets in seen:
        np.testing.assert_array_equal(targets, labels[:24])
    np.testing.assert_array_equal(classifier.classes_, ["train_a", "train_b"])
    np.testing.assert_allclose(classifier.transformers_["image"].pca_.mean_, train[1].reshape(24, -1).mean(axis=0))
    np.testing.assert_allclose(classifier.transformers_["nirs"].mean_, train[0].mean(axis=0))
    assert not hasattr(classifier.transformers["image"], "pca_")
    assert not hasattr(classifier.model, "classes_")
    second_fold = clone(classifier).fit(blocks, labels)
    np.testing.assert_array_equal(second_fold.classes_, ["heldout_only", "train_a", "train_b"])
    assert classifier.transformers_["image"] is not second_fold.transformers_["image"]
    np.testing.assert_array_equal(classifier.classes_, ["train_a", "train_b"])


def test_classifier_clone_nested_parameters_and_public_export(multimodal_data):
    from nirs4all.operators.models import MultimodalClassifier as PublicClassifier

    blocks, y = multimodal_data
    classifier = make_classifier().fit(blocks, np.where(y > 0, "yes", "no"))
    fresh = clone(classifier).set_params(transformers__image__n_components=2, model__C=0.3, source_weights__image=0.0)
    assert PublicClassifier is MultimodalClassifier
    assert not hasattr(fresh, "classes_") and not hasattr(fresh, "model_")
    assert fresh.get_params()["transformers__image__n_components"] == 2
    assert fresh.get_params()["model__C"] == 0.3
    assert fresh.get_params()["source_weights__image"] == 0.0
    assert classifier.get_params()["transformers__image__n_components"] == 3
    fresh.fit(blocks, np.where(y > 0, "yes", "no"))
    assert fresh.predict(blocks).shape == (36,)


def test_classifier_probabilities_are_exposed_only_when_supported(multimodal_data):
    blocks, y = multimodal_data
    classifier = make_classifier().set_params(model=LinearSVC())
    assert not hasattr(classifier, "predict_proba")
    classifier.fit(blocks, np.where(y > 0, 13, -2))
    assert not hasattr(classifier, "predict_proba")
    assert set(classifier.predict(blocks)) <= {-2, 13}
    probabilistic = make_classifier()
    assert hasattr(probabilistic, "predict_proba")
    with pytest.raises(NotFittedError):
        probabilistic.predict_proba(blocks)
    with pytest.raises(NotFittedError):
        probabilistic.predict(blocks)


@pytest.mark.parametrize("target,match", [
    (None, "classification targets"),
    (np.arange(36) + 0.5, "Unknown label type"),
    (np.ones((36, 2)), "1d array"),
    (np.tile([0, 1], 17), "inconsistent numbers of samples"),
    (np.array([0, 1, np.nan] * 12), "NaN"),
])
def test_classifier_rejects_invalid_targets_before_fitting(multimodal_data, monkeypatch, target, match):
    blocks, _ = multimodal_data

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid targets reached an encoder fit")

    monkeypatch.setattr(TensorPCA, "fit", forbidden)
    with pytest.raises(ValueError, match=match):
        make_classifier().fit(blocks, target)


def test_classifier_rejects_regressor_model_and_schema_drift(multimodal_data):
    blocks, y = multimodal_data
    labels = np.where(y > 0, "yes", "no")
    with pytest.raises(ValueError, match="classifier as model"):
        make_classifier().set_params(model=Ridge()).fit(blocks, labels)
    classifier = make_classifier().fit(blocks, labels)
    for predict in (classifier.predict, classifier.predict_proba):
        with pytest.raises(ValueError, match="Expected 4 source blocks"):
            predict(blocks[:-1])
        with pytest.raises(ValueError, match="expected input shape"):
            predict([blocks[0], blocks[1].reshape(36, 4, 3, 2), *blocks[2:]])


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
def test_classifier_joblib_replay_keeps_classes_and_probabilities_without_fit(multimodal_data, monkeypatch, fusion):
    blocks, y = multimodal_data
    classifier = make_classifier(fusion).fit(blocks, np.where(y > 0, "ripe", "unripe"))
    predictions, probabilities = classifier.predict(blocks), classifier.predict_proba(blocks)
    buffer = io.BytesIO()
    joblib.dump(classifier, buffer)
    buffer.seek(0)

    def forbidden(*args, **kwargs):
        pytest.fail("Classifier replay must not fit")

    for estimator in (MultimodalClassifier, TensorPCA, PCA, StandardScaler, ColumnTransformer, LogisticRegression, BlockLogisticClassifier):
        monkeypatch.setattr(estimator, "fit", forbidden)
    restored = joblib.load(buffer)
    np.testing.assert_array_equal(restored.classes_, classifier.classes_)
    np.testing.assert_array_equal(restored.predict(blocks), predictions)
    np.testing.assert_array_equal(restored.predict_proba(blocks), probabilities)


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
def test_multioutput_regression_preserves_targets_predictions_and_replay(multimodal_data, monkeypatch, fusion):
    blocks, y = multimodal_data
    targets = np.column_stack([y, blocks[0][:, 1] - y, blocks[1][:, 1, 0, 0] + 0.3 * y])
    train = [block.iloc[:24] if isinstance(block, pd.DataFrame) else block[:24] for block in blocks]
    regressor = make_regressor(fusion).fit(train, targets[:24])
    assert regressor.transformers_["image"].target_sum_ == pytest.approx(targets[:24].sum())
    encoded = [
        transformer.transform(block) * regressor.source_weights_[name]
        for (name, transformer), block in zip(regressor.transformers_.items(), train, strict=True)
    ]
    expected_model = clone(regressor.model).fit(np.concatenate(encoded, axis=1) if fusion == "early" else encoded, targets[:24])
    expected = expected_model.predict(np.concatenate(encoded, axis=1) if fusion == "early" else encoded)
    np.testing.assert_allclose(regressor.predict(train), expected)
    predictions = regressor.predict(blocks)
    assert predictions.shape == (36, 3)
    assert is_regressor(regressor) and not is_classifier(regressor)
    buffer = io.BytesIO()
    joblib.dump(regressor, buffer)
    buffer.seek(0)

    def forbidden(*args, **kwargs):
        pytest.fail("Multioutput replay must not fit")

    for estimator in (TensorPCA, PCA, Ridge, MBPLS):
        monkeypatch.setattr(estimator, "fit", forbidden)
    np.testing.assert_allclose(joblib.load(buffer).predict(blocks), predictions)


def test_sklearn_tags_preserve_estimator_roles_and_multioutput_capability():
    import sklearn.utils

    if not hasattr(sklearn.utils, "get_tags"):
        assert make_regressor()._get_tags()["multioutput"]
        assert not make_classifier()._get_tags()["multioutput"]
        return

    classifier_tags = sklearn.utils.get_tags(make_classifier())
    regressor_tags = sklearn.utils.get_tags(make_regressor())
    assert classifier_tags.estimator_type == "classifier"
    assert classifier_tags.classifier_tags is not None and classifier_tags.regressor_tags is None
    assert classifier_tags.target_tags.required and not classifier_tags.target_tags.multi_output
    assert regressor_tags.estimator_type == "regressor"
    assert regressor_tags.regressor_tags is not None and regressor_tags.classifier_tags is None
    assert regressor_tags.target_tags.multi_output


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
@pytest.mark.parametrize("hidden_value", [np.nan, np.inf, 1e100])
def test_per_target_chains_fit_only_observed_raw_rows(multimodal_data, fusion, hidden_value):
    blocks, y = multimodal_data
    targets = np.column_stack([y, blocks[0][:, 1] - y, y * 0.3])
    rows = np.arange(len(y))
    observed = np.column_stack([rows % 3 != 0, rows % 2 == 0, rows < 25])
    masked_targets = targets.copy()
    masked_targets[~observed] = hidden_value
    regressor = make_regressor(fusion).set_params(target_policy="per_target").fit(blocks, masked_targets, target_mask=observed)
    assert not hasattr(regressor, "model_") and not hasattr(regressor, "transformers_")
    np.testing.assert_array_equal(regressor.target_counts_, [24, 18, 25])
    assert len(regressor.target_models_) == 3
    expected = []
    for index, model in enumerate(regressor.target_models_):
        selected = observed[:, index]
        subset = [block.iloc[selected] if isinstance(block, pd.DataFrame) else block[selected] for block in blocks]
        oracle = make_regressor(fusion).fit(subset, targets[selected, index])
        expected.append(oracle.predict(blocks).reshape(-1))
        assert model.target_policy == "complete"
        assert model.transformers_["image"].seen_shape_ == (int(selected.sum()), 3, 4, 2)
        assert model.transformers_["image"].target_sum_ == pytest.approx(targets[selected, index].sum())
        np.testing.assert_allclose(model.transformers_["image"].pca_.mean_, blocks[1][selected].reshape(selected.sum(), -1).mean(axis=0))
        np.testing.assert_allclose(model.transformers_["nirs"].mean_, blocks[0][selected].mean(axis=0))
    np.testing.assert_allclose(regressor.predict(blocks), np.column_stack(expected))
    assert regressor.target_models_[0].transformers_["image"] is not regressor.target_models_[1].transformers_["image"]
    assert not hasattr(regressor.transformers["image"], "pca_")
    assert not {"target_mask_", "y_", "X_"}.intersection(vars(regressor))


@pytest.mark.parametrize("column_target", [False, True])
def test_per_target_preserves_single_target_rank_and_clones_configuration(multimodal_data, column_target):
    blocks, y = multimodal_data
    targets = y[:, None] if column_target else y
    observed = np.ones(targets.shape, dtype=bool)
    observed[::3] = False
    model = make_regressor().set_params(target_policy="per_target").fit(blocks, targets, target_mask=observed)
    assert model.predict(blocks).shape == targets.shape
    fresh = clone(model).set_params(transformers__image__n_components=2, model__alpha=1.5, source_weights__image=0.2)
    assert fresh.get_params()["target_policy"] == "per_target"
    assert not hasattr(fresh, "target_models_")
    fresh.fit(blocks, targets, target_mask=observed)
    assert fresh.target_models_[0].output_widths_["image"] == 2
    assert fresh.target_models_[0].model.alpha == 1.5
    assert model.target_models_[0].output_widths_["image"] == 3


def test_complete_policy_remains_joint_and_refitting_switches_fitted_state(multimodal_data):
    blocks, y = multimodal_data
    targets = np.column_stack([y, y * 0.4 + blocks[0][:, 2]])
    model = make_regressor().fit(blocks, targets, target_mask=np.ones(targets.shape, dtype=bool))
    joint_predictions = model.predict(blocks)
    assert hasattr(model, "model_") and not hasattr(model, "target_models_")
    mask = np.ones(targets.shape, dtype=bool)
    mask[::2, 1] = False
    model.set_params(target_policy="per_target").fit(blocks, targets, target_mask=mask)
    assert not hasattr(model, "model_") and not hasattr(model, "transformers_")
    assert len(model.target_models_) == 2
    model.set_params(target_policy="complete").fit(blocks, targets)
    assert hasattr(model, "model_") and not hasattr(model, "target_models_")
    np.testing.assert_allclose(model.predict(blocks), joint_predictions)


@pytest.mark.parametrize("invalid", ["policy", "policy_type", "mask_shape", "mask_dtype", "observed_nan", "observed_inf", "empty_target", "complete_partial", "implicit_missing", "target_rank"])
def test_invalid_target_contract_is_rejected_before_any_encoder_fit(multimodal_data, monkeypatch, invalid):
    blocks, y = multimodal_data
    targets = np.column_stack([y, y * 0.3])
    mask = np.ones(targets.shape, dtype=bool)
    model = make_regressor().set_params(target_policy="per_target")
    match = "target"
    if invalid == "policy":
        model.set_params(target_policy="impute")
    elif invalid == "policy_type":
        model.set_params(target_policy=[])
    elif invalid == "mask_shape":
        mask = mask[:, 0]
    elif invalid == "mask_dtype":
        mask = mask.astype(int)
    elif invalid in {"observed_nan", "observed_inf"}:
        targets[2, 1] = np.nan if invalid == "observed_nan" else np.inf
    elif invalid == "empty_target":
        mask[:, 1] = False
    elif invalid == "complete_partial":
        model.set_params(target_policy="complete")
        mask[0, 1] = False
    elif invalid == "implicit_missing":
        targets[1, 1] = np.nan
        mask = None
    else:
        targets = targets[:, :, None]

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid target contract reached encoder.fit")

    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(TensorPCA, "fit", forbidden)
    with pytest.raises(ValueError, match=match):
        model.fit(blocks, targets, target_mask=mask)


def test_per_target_joblib_replay_never_fits_and_rejects_raw_shape_drift(multimodal_data, monkeypatch):
    blocks, y = multimodal_data
    targets = np.column_stack([y, y * 0.3])
    observed = np.ones(targets.shape, dtype=bool)
    observed[::3, 0] = False
    model = make_regressor().set_params(target_policy="per_target").fit(blocks, targets, target_mask=observed)
    expected = model.predict(blocks)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    def forbidden(*args, **kwargs):
        pytest.fail("Per-target replay must not fit")

    for estimator in (MultimodalRegressor, TensorPCA, PCA, StandardScaler, Ridge):
        monkeypatch.setattr(estimator, "fit", forbidden)
    restored = joblib.load(buffer)
    np.testing.assert_array_equal(restored.predict(blocks), expected)
    np.testing.assert_array_equal(restored.target_counts_, model.target_counts_)
    with pytest.raises(ValueError, match="expected input shape"):
        restored.predict([blocks[0], blocks[1].reshape(36, 4, 3, 2), *blocks[2:]])


def test_per_target_accepts_single_output_models_and_declares_multioutput():
    import sklearn.utils
    from sklearn.svm import SVR

    values = np.arange(24.0).reshape(12, 2)
    targets = np.column_stack([values[:, 0], -values[:, 1]])
    model = MultimodalRegressor({"x": None}, SVR(), target_policy="per_target").fit([values], targets)
    assert model.predict([values]).shape == targets.shape
    if hasattr(sklearn.utils, "get_tags"):
        assert sklearn.utils.get_tags(model).target_tags.multi_output
    else:
        assert model._get_tags()["multioutput"]


def test_classification_does_not_accept_target_masks(multimodal_data):
    blocks, y = multimodal_data
    with pytest.raises(TypeError, match="target_mask"):
        make_classifier().fit(blocks, y > 0, target_mask=np.ones(y.shape, dtype=bool))
