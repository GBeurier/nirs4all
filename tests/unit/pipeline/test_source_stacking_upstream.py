"""Upstream lowering preserves real per-source chains without performing FIT."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data.multimodal import MultimodalSpectroDataset
from nirs4all.pipeline.dagml.detect import _detect_by_source_stacking_branch
from nirs4all.pipeline.dagml.late_tuning import prepare_late_tuning
from nirs4all.pipeline.dagml.source_stacking import lower_source_stacking


def _declaration(prefix):
    body = [StandardScaler(), {"model": Ridge(alpha=0.4)}]
    pipeline = [*prefix, GroupKFold(3), {"branch": {"by_source": True, "steps": body,
                "missing_source_policy": "zero_with_indicator", "target_policy": "per_target"}},
                {"merge": "predictions"}, {"model": Ridge(alpha=0.5)}]
    return pipeline, body


def _dataset():
    ids = [f"sample-{index}" for index in range(12)]
    X = np.arange(36, dtype=float).reshape(12, 3)
    sources = {name: TensorSource(X + offset, ids, representation_id="signal_1d")
               for name, offset in (("NIR", 0), ("MIR", 20))}
    cohort = MultimodalDataset(sources, sample_ids=ids, y=X[:, :2], target_names=["sugar", "protein"],
                              target_mask=np.ones((12, 2), dtype=bool), task_type="regression",
                              groups=[f"group-{index // 2}" for index in range(12)], partitions=["train"] * 12)
    return MultimodalSpectroDataset(cohort)


@pytest.mark.parametrize("spelling", ["bare", "preprocessing"])
@pytest.mark.parametrize("position", ["before_split", "after_split"])
def test_prefix_is_cloned_inside_each_source_in_order_and_removed_from_outer_path(spelling, position, monkeypatch):
    upstream = MinMaxScaler(feature_range=(-2, 3))
    prefix = [upstream] if spelling == "bare" else [{"preprocessing": upstream}]
    pipeline, body = _declaration(prefix)
    if position == "after_split":
        pipeline[0], pipeline[1] = pipeline[1], pipeline[0]
    dataset = _dataset()
    for cls in (MinMaxScaler, StandardScaler, Ridge):
        monkeypatch.setattr(cls, "fit", lambda *args, **kwargs: pytest.fail("lowering performed FIT"))
    detected = _detect_by_source_stacking_branch(pipeline, 2)
    assert detected is not None and detected[0] is body
    lowered, branches, layout = lower_source_stacking(pipeline, body, source_widths=dataset.num_features,
        source_names=list(dataset.source_names), source_descriptors=dataset.cohort.schema_descriptors())
    assert isinstance(lowered[0], GroupKFold)
    assert len(lowered) == 4
    assert layout["schema"] == "nirs4all.source-stacking-layout.v4"
    for branch in branches:
        assert [type(step) for step in branch[:2]] == [MinMaxScaler, StandardScaler]
        assert branch[0].feature_range == (-2, 3)
        assert branch[0] is not upstream
    assert branches[0][0] is not branches[1][0]
    assert len(body) == 2 and len(pipeline) == 5


@pytest.mark.parametrize("bad", [
    {"preprocessing": StandardScaler(), "fit_on_all": True},
    {"preprocessing": StandardScaler(), "force_layout": "3d"},
    {"y_processing": StandardScaler()}, {"merge": {"sources": "concat"}},
    {"sample_augmentation": StandardScaler()}, PCA(n_components=1, random_state=None),
])
def test_unhandled_global_target_layout_or_random_preprocessing_is_refused_before_fit(bad, monkeypatch):
    monkeypatch.setattr(StandardScaler, "fit", lambda *args, **kwargs: pytest.fail("refusal reached FIT"))
    pipeline, body = _declaration([bad])
    assert _detect_by_source_stacking_branch(pipeline, 2) is None
    with pytest.raises(ValueError, match="upstream preprocessing"):
        lower_source_stacking(pipeline, body, source_widths=[3, 3], source_names=["NIR", "MIR"],
                              source_descriptors=_dataset().cohort.schema_descriptors())


def test_transform_between_branch_and_meta_is_not_moved_across_prediction_boundary():
    pipeline, _body = _declaration([MinMaxScaler()])
    pipeline.insert(3, StandardScaler())
    assert _detect_by_source_stacking_branch(pipeline, 2) is None


def test_no_prefix_layout_and_public_branch_addresses_are_conserved():
    pipeline, body = _declaration([])
    dataset = _dataset()
    first = lower_source_stacking(pipeline, body, source_widths=[3, 3], source_names=["NIR", "MIR"],
                                  source_descriptors=dataset.cohort.schema_descriptors())
    second = lower_source_stacking(copy.deepcopy(pipeline), copy.deepcopy(body), source_widths=[3, 3], source_names=["NIR", "MIR"],
                                   source_descriptors=dataset.cohort.schema_descriptors())
    assert first[2] == second[2]
    recipe = prepare_late_tuning(pipeline, dataset, ["branches.NIR.1.alpha"])
    assert recipe.bindings["branches.NIR.1.alpha"]["node_id"] == "branch:0.node:1"


def test_whole_stack_prefix_parameters_address_real_nodes_and_selected_refit_applies_them_once(monkeypatch):
    pipeline, _body = _declaration([MinMaxScaler()])
    for cls in (MinMaxScaler, StandardScaler, Ridge):
        monkeypatch.setattr(cls, "fit", lambda *args, **kwargs: pytest.fail("recipe preparation performed FIT"))
    recipe = prepare_late_tuning(pipeline, _dataset(), ["branches.NIR.0.clip", "branches.MIR.2.alpha"])
    assert recipe.bindings["branches.NIR.0.clip"]["node_id"] == "branch:0.node:0"
    assert recipe.bindings["branches.MIR.2.alpha"]["node_id"] == "branch:1.node:2"
    selected = recipe.selected_pipeline({"branches.NIR.0.clip": True, "branches.MIR.2.alpha": 0.9})
    assert len(selected) == 4 and isinstance(selected[0], GroupKFold)
    body, _meta = _detect_by_source_stacking_branch(selected, 2)
    assert len(body["NIR"]) == 3 and body["NIR"][0].clip is True
    assert body["MIR"][0].clip is False and body["MIR"][-1]["model"].alpha == 0.9
    assert recipe.branches[0][0].clip is False and pipeline[0].clip is False
    _lowered, branches, _layout = lower_source_stacking(selected, body, source_widths=[3, 3],
        source_names=["NIR", "MIR"], source_descriptors=_dataset().cohort.schema_descriptors())
    assert [len(branch) for branch in branches] == [3, 3]


@pytest.mark.parametrize("weighted", [False, True])
def test_experimental_unit_admission_checks_every_expanded_prefix_before_fit(weighted, monkeypatch):
    from nirs4all.pipeline.dagml.experimental_units import require_weighted_context
    from nirs4all.pipeline.dagml.run_paths import _canonical_branch

    prefix = StandardScaler() if weighted else MinMaxScaler()
    pipeline, body = _declaration([prefix])
    dataset = _dataset()
    _lowered, branches, _layout = lower_source_stacking(pipeline, body, source_widths=[3, 3],
        source_names=["NIR", "MIR"], source_descriptors=dataset.cohort.schema_descriptors())
    context = {"metadata": {"experimental_unit": {"schema_version": 1}}, "steps": [
        {"kind": "branch", "mode": "duplication", "branches": [_canonical_branch(branch, index)
            for index, branch in enumerate(branches)]}]}
    from functools import wraps

    def forbid_fit(method):
        @wraps(method)
        def forbidden(*args, **kwargs):
            pytest.fail("capability admission reached FIT")
        return forbidden

    for cls in (MinMaxScaler, StandardScaler, Ridge):
        monkeypatch.setattr(cls, "fit", forbid_fit(cls.fit))
    if weighted:
        assert require_weighted_context(context) is True
    else:
        with pytest.raises(ValueError, match="MinMaxScaler.*sample_weight"):
            require_weighted_context(context)
