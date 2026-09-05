"""Source binding is declared before any scientific operation runs."""

import copy
import json

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from nirs4all.pipeline.dagml.operator_routing import route_operator
from nirs4all.pipeline.dagml.source_stacking import lower_source_stacking
from nirs4all.pipeline.dagml_bridge import _json_safe_params, _qualname


def _pipeline(hpo=False):
    model = {"model": Ridge(3), "train_params": {"tol": 0.003}, "refit_params": {"alpha": 7}}
    if hpo:
        model["finetune_params"] = {"approach": "grouped", "n_trials": 2, "sampler": "grid", "model_params": {"alpha": [1, 2]}}
    body = [StandardScaler(), model]
    return [KFold(3, shuffle=True, random_state=9), {"branch": {"by_source": True, "steps": body}},
            {"merge": "predictions"}, {"model": Ridge(0.1)}], body


def test_lowering_preserves_controls_and_declares_exact_source_slices_without_fit(monkeypatch):
    pipeline, body = _pipeline()
    monkeypatch.setattr(StandardScaler, "fit", lambda *args, **kwargs: pytest.fail("lowering fitted preprocessing"))
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("lowering fitted model"))
    lowered, branches, layout = lower_source_stacking(pipeline, body, source_widths=[2, 3], source_names=["NIR", "MIR"])
    assert list(lowered[1]["branch"]) == ["source_0", "source_1"]
    assert lowered[2] == {"merge": "predictions"}
    assert isinstance(branches[0][0], ColumnTransformer)
    assert branches[0][0].transformers[0][2] == [0, 1]
    assert branches[1][0].transformers[0][2] == [2, 3, 4]
    assert [branch[-1]["train_params"] for branch in branches] == [{"tol": 0.003}] * 2
    assert [branch[-1]["refit_params"] for branch in branches] == [{"alpha": 7}] * 2
    assert branches[0][-1]["model"] is not branches[1][-1]["model"]
    assert "by_source" in pipeline[1]["branch"] and len(body) == 2
    assert layout["total_columns"] == 5


def test_source_selector_survives_actual_operator_json_roundtrip():
    pipeline, body = _pipeline()
    _, branches, _ = lower_source_stacking(pipeline, body, source_widths=[2, 3], source_names=["NIR", "MIR"])
    X = np.arange(35).reshape(7, 5)
    for index, branch in enumerate(branches):
        selector = branch[0]
        routed = route_operator("transform", _qualname(selector), params=json.loads(json.dumps(_json_safe_params(selector))))
        expected = X[:, :2] if index == 0 else X[:, 2:]
        np.testing.assert_array_equal(routed.fit_transform(X), expected)


def test_layout_is_deterministic_and_fingerprints_binding_not_only_total_width():
    pipeline, body = _pipeline()
    args = {"source_widths": [2, 3], "source_names": ["same", "same"]}
    first = lower_source_stacking(pipeline, body, **args)
    assert first[2] == lower_source_stacking(pipeline, body, **args)[2]
    assert len(first[0][1]["branch"]) == 2  # Duplicate display names never lose a source.
    changed = lower_source_stacking(pipeline, body, source_widths=[3, 2], source_names=["same", "same"])
    assert first[2]["fingerprint"] != changed[2]["fingerprint"]


def test_grouped_hpo_retains_explicit_outer_split_policy_without_mutating_user_config():
    pipeline, body = _pipeline(hpo=True)
    before = copy.deepcopy(body[-1]["finetune_params"])
    _, branches, _ = lower_source_stacking(pipeline, body, source_widths=[2, 3], source_names=["NIR", "MIR"])
    for branch in branches:
        config = branch[-1]["finetune_params"]
        assert "__dagml_inner_splitter" in config
        assert config["model_params"] == before["model_params"]
    assert body[-1]["finetune_params"] == before


@pytest.mark.parametrize("widths,names", [([], []), ([2], ["NIR"]), ([2, 0], ["NIR", "MIR"]), ([2, 3], ["NIR"]), ([2, True], ["NIR", "MIR"])])
def test_invalid_source_layout_is_rejected(widths, names):
    pipeline, body = _pipeline()
    with pytest.raises(ValueError):
        lower_source_stacking(pipeline, body, source_widths=widths, source_names=names)
