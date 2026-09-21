"""Absent raw modalities never reach fitted encoders or affect predictions."""

import io

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from nirs4all.operators.models.multimodal import MultimodalClassifier, MultimodalRegressor, TensorPCA
from tests.unit.operators.models.test_multimodal import make_regressor
from tests.unit.operators.models.test_multimodal import multimodal_data as multimodal_data


def masks_for(blocks):
    rows = np.arange(len(blocks[0]))
    return {name: (rows + index) % 4 != 0 for index, name in enumerate(make_regressor().transformers)}


def subset(block, mask):
    return block.iloc[mask] if hasattr(block, "iloc") else block[mask]


def poison(blocks, masks, value):
    result = []
    for block, mask in zip(blocks, masks.values(), strict=True):
        altered = block.copy()
        if hasattr(altered, "iloc"):
            altered.iloc[~mask, 0] = value
            altered.iloc[~mask, 1] = "unobserved-category"
        else:
            altered[~mask] = value
        result.append(altered)
    return result


@pytest.mark.parametrize("fusion", ["early", "intermediate"])
def test_partial_sources_match_explicit_present_row_encoders(multimodal_data, fusion):
    blocks, y = multimodal_data
    masks = masks_for(blocks)
    raw = poison(blocks, masks, np.nan)
    model = make_regressor(fusion).set_params(missing_source_policy="zero_with_indicator").fit(raw, y, source_masks=masks)
    encoded = []
    for (name, transformer), block in zip(model.transformers.items(), raw, strict=True):
        present = masks[name]
        fitted = clone(transformer).fit(subset(block, present), y[present])
        values = fitted.transform(subset(block, present))
        fused = np.zeros((len(y), values.shape[1] + 1))
        fused[present, :-1] = values
        fused[:, -1] = present
        encoded.append(fused * model.source_weights_.get(name, 1.0))
    features = np.concatenate(encoded, axis=1) if fusion == "early" else encoded
    reference = clone(model.model).fit(features, y)
    np.testing.assert_allclose(model.predict(raw, source_masks=masks), reference.predict(features))
    np.testing.assert_allclose(model.transformers_["nirs"].mean_, blocks[0][masks["nirs"]].mean(axis=0))
    for name in ("image", "temporal"):
        fitted = model.transformers_[name]
        if name == "temporal":
            fitted = fitted.steps[0][1]
        assert fitted.seen_shape_[0] == int(masks[name].sum())
        assert fitted.target_sum_ == pytest.approx(y[masks[name]].sum())


def test_hidden_source_values_cannot_change_fit_or_prediction(multimodal_data):
    blocks, y = multimodal_data
    masks = masks_for(blocks)
    template = make_regressor().set_params(missing_source_policy="zero_with_indicator")
    raw_nan = poison(blocks, masks, np.nan)
    raw_extreme = poison(blocks, masks, 1e100)
    first = clone(template).fit(raw_nan, y, source_masks=masks)
    second = clone(template).fit(raw_extreme, y, source_masks=masks)
    np.testing.assert_array_equal(first.predict(raw_nan, source_masks=masks), second.predict(raw_extreme, source_masks=masks))
    np.testing.assert_array_equal(first.model_.coef_, second.model_.coef_)


def test_both_fit_and_transform_see_only_present_dataframe_rows(multimodal_data, monkeypatch):
    blocks, y = multimodal_data
    masks = masks_for(blocks)
    expected = blocks[-1].index[masks["tabular"]].tolist()
    seen = []
    original_fit = ColumnTransformer.fit
    original_transform = ColumnTransformer.transform

    def checked_fit(self, X, y=None, **kwargs):
        assert X.index.tolist() == expected
        np.testing.assert_array_equal(y, multimodal_data[1][masks["tabular"]])
        seen.append("fit")
        return original_fit(self, X, y, **kwargs)

    def checked_transform(self, X, **kwargs):
        assert X.index.tolist() == expected
        seen.append("transform")
        return original_transform(self, X, **kwargs)

    monkeypatch.setattr(ColumnTransformer, "fit", checked_fit)
    monkeypatch.setattr(ColumnTransformer, "transform", checked_transform)
    model = make_regressor().set_params(missing_source_policy="zero_with_indicator").fit(blocks, y, source_masks=masks)
    model.predict(blocks, source_masks=masks)
    assert seen == ["fit", "transform", "transform"]


@pytest.mark.parametrize("kind", ["regressor", "classifier"])
def test_entirely_absent_prediction_sources_skip_transform_and_replay_without_fit(multimodal_data, monkeypatch, kind):
    blocks, y = multimodal_data
    regressor = make_regressor().set_params(missing_source_policy="zero_with_indicator")
    model = regressor if kind == "regressor" else MultimodalClassifier(regressor.transformers, LogisticRegression(), missing_source_policy="zero_with_indicator")
    model.fit(blocks, y if kind == "regressor" else (y > np.median(y)).astype(int))
    masks = {name: np.zeros(len(y), dtype=bool) for name in model.source_names_}
    hidden = poison(blocks, masks, np.nan)
    expected = model.model_.predict(np.zeros((len(y), sum(model.output_widths_.values()))))
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    def forbidden(*args, **kwargs):
        pytest.fail("All-absent replay must neither fit nor call transform(empty)")

    for estimator in (TensorPCA, StandardScaler, ColumnTransformer, Ridge, LogisticRegression):
        monkeypatch.setattr(estimator, "fit", forbidden)
    restored = joblib.load(buffer)
    for transformer in restored.transformers_.values():
        monkeypatch.setattr(transformer, "transform", forbidden)
    np.testing.assert_array_equal(restored.predict(hidden, source_masks=masks), expected)
    if kind == "classifier":
        expected_proba = restored.model_.predict_proba(np.zeros((len(y), sum(restored.output_widths_.values()))))
        np.testing.assert_allclose(restored.predict_proba(hidden, source_masks=masks), expected_proba)


@pytest.mark.parametrize("bad", ["missing", "extra", "list", "integer", "object", "scalar", "rank", "short", "masked"])
def test_bad_masks_fail_before_any_encoder_fit(multimodal_data, monkeypatch, bad):
    blocks, y = multimodal_data
    masks = masks_for(blocks)
    if bad == "missing":
        del masks["tabular"]
    elif bad == "extra":
        masks["unknown"] = np.ones(len(y), dtype=bool)
    elif bad == "list":
        masks = list(masks.values())
    else:
        masks["tabular"] = {
            "integer": np.ones(len(y), dtype=int), "object": np.ones(len(y), dtype=object),
            "scalar": True, "rank": np.ones((len(y), 1), dtype=bool),
            "short": np.ones(len(y) - 1, dtype=bool), "masked": np.ma.array(np.ones(len(y), dtype=bool)),
        }[bad]

    def forbidden(*args, **kwargs):
        pytest.fail("Mask validation must precede every encoder fit")

    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    with pytest.raises(ValueError, match="source_masks"):
        make_regressor().set_params(missing_source_policy="zero_with_indicator").fit(blocks, y, source_masks=masks)


@pytest.mark.parametrize("bad", ["default", "all_absent", "empty_target_intersection", "policy"])
def test_unsupported_absence_fails_before_any_encoder_fit(multimodal_data, monkeypatch, bad):
    blocks, y = multimodal_data
    masks = masks_for(blocks)
    model = make_regressor().set_params(missing_source_policy="zero_with_indicator")
    kwargs = {}
    if bad == "default":
        model.set_params(missing_source_policy="error")
    elif bad == "policy":
        model.set_params(missing_source_policy="impute")
    elif bad == "all_absent":
        masks["tabular"][:] = False
    else:
        model.set_params(target_policy="per_target")
        y = np.column_stack([y, y * 2])
        target_mask = np.ones(y.shape, dtype=bool)
        target_mask[:, 1] = ~masks["tabular"]
        kwargs["target_mask"] = target_mask

    def forbidden(*args, **kwargs):
        pytest.fail("All source/target intersections must be checked before fitting")

    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    with pytest.raises(ValueError, match="missing_source_policy|present training row"):
        model.fit(blocks, y, source_masks=masks, **kwargs)


def test_per_target_masks_intersect_presence_and_keep_independent_chains(multimodal_data):
    blocks, y = multimodal_data
    masks = masks_for(blocks)
    y = np.column_stack([y, y * 2 + 1])
    target_mask = np.column_stack([np.arange(len(y)) % 3 != 0, np.arange(len(y)) % 3 != 1])
    y[~target_mask] = np.nan
    model = make_regressor().set_params(target_policy="per_target", missing_source_policy="zero_with_indicator").fit(blocks, y, target_mask=target_mask, source_masks=masks)
    predictions = model.predict(blocks, source_masks=masks)
    for index, child in enumerate(model.target_models_):
        rows = target_mask[:, index]
        reference = make_regressor().set_params(missing_source_policy="zero_with_indicator").fit(
            [subset(block, rows) for block in blocks], y[rows, index], source_masks={name: mask[rows] for name, mask in masks.items()},
        )
        np.testing.assert_allclose(predictions[:, index], reference.predict(blocks, source_masks=masks))
        assert child.transformers_["image"].seen_shape_[0] == int((rows & masks["image"]).sum())
        assert child.transformers_["image"].target_sum_ == pytest.approx(y[rows & masks["image"], index].sum())
    np.testing.assert_array_equal(model.target_counts_, target_mask.sum(axis=0))
    assert model.target_models_[0].transformers_["image"] is not model.target_models_[1].transformers_["image"]
    model.set_params(target_policy="complete").fit(blocks, np.arange(len(y)), source_masks=masks)
    assert not hasattr(model, "target_models_")
    assert model.missing_source_policy_ == "zero_with_indicator"


def test_zero_weight_annihilates_presence_indicator_and_encoded_features(multimodal_data):
    blocks, y = multimodal_data
    masks = masks_for(blocks)
    model = make_regressor().set_params(missing_source_policy="zero_with_indicator", source_weights__image=0).fit(blocks, y, source_masks=masks)
    changed = {name: mask.copy() for name, mask in masks.items()}
    changed["image"][:] = False
    np.testing.assert_array_equal(model.predict(blocks, source_masks=masks), model.predict(blocks, source_masks=changed))
    encoded = model._encode(blocks, source_masks=masks)
    start = model.output_widths_["nirs"]
    assert not encoded[:, start:start + model.output_widths_["image"]].any()


def test_clone_policy_and_complete_default_preserve_historical_dimensions(multimodal_data):
    blocks, y = multimodal_data
    masks = {name: np.ones(len(y), dtype=bool) for name in make_regressor().transformers}
    default = make_regressor().fit(blocks, y)
    explicit = make_regressor().fit(blocks, y, source_masks=masks)
    np.testing.assert_array_equal(default.predict(blocks), explicit.predict(blocks, source_masks=masks))
    assert default.output_widths_ == explicit.output_widths_
    optin = clone(default).set_params(missing_source_policy="zero_with_indicator")
    assert clone(optin).get_params()["missing_source_policy"] == "zero_with_indicator"
    optin.fit(blocks, y)
    assert optin.output_widths_ == {name: width + 1 for name, width in default.output_widths_.items()}
    bad = dict(masks, image=np.zeros(len(y), dtype=bool))
    with pytest.raises(ValueError, match="missing_source_policy='error'"):
        default.predict(blocks, source_masks=bad)


def test_passthrough_masks_hide_nonfinite_cells():
    x = np.array([[1., 2.], [np.nan, np.nan], [3., 4.], [np.nan, np.nan]])
    mask = {"table": np.array([True, False, True, False])}
    model = MultimodalRegressor({"table": None}, Ridge(), missing_source_policy="zero_with_indicator").fit([x], [1., 2., 3., 4.], source_masks=mask)
    assert model.output_widths_ == {"table": 3}
    assert np.isfinite(model.predict([x], source_masks=mask)).all()
