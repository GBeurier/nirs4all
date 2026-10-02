"""Source-selection boundary checks before catalogue, optimizer or FIT ownership."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from nirs4all.data import SpectroDataset
from nirs4all.operators.transforms import SNV, SavitzkyGolay
from nirs4all.pipeline.dagml.structural_sources import build_source_selection_step
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure, validate_structural_profile
from nirs4all.pipeline.dagml_bridge import lower_structural_hpo_pipeline


def _source_choice(selectors: Any) -> dict[str, Any]:
    return {"merge": {"sources": {"strategy": "concat", "sources": selectors}}}


def _pipeline() -> list[Any]:
    return [
        {"_or_": [_source_choice([0]), _source_choice([0, 2])]},
        {"_or_": [None, StandardScaler()]},
        {"split": GroupKFold(3), "group_by": "batch"},
        {"model": {"_or_": [Ridge(), PLSRegression(scale=False)]}},
    ]


def _dataset() -> SpectroDataset:
    values = np.arange(234, dtype=np.float32).reshape(18, 13)
    dataset = SpectroDataset("source-selection-boundaries")
    dataset.add_samples([values[:, :6], values[:, 6:9], values[:, 9:]], {"partition": "train"})
    dataset.add_targets(np.arange(18, dtype=float))
    dataset.add_metadata(np.repeat(["a", "b", "c"], 6)[:, None], headers=["batch"])
    dataset.set_task_type("regression")
    return dataset


def _layout() -> dict[str, Any]:
    return {
        "kind": "by_source_concat", "source_ids": ["fixture-source-0", "fixture-source-1", "fixture-source-2"],
        "source_order": ["source_0", "source_1", "source_2"],
        "blocks": [
            {"source_index": 0, "source_id": "fixture-source-0", "source_name": "source_0", "column_start": 0, "column_count": 6},
            {"source_index": 1, "source_id": "fixture-source-1", "source_name": "source_1", "column_start": 6, "column_count": 3},
            {"source_index": 2, "source_id": "fixture-source-2", "source_name": "source_2", "column_start": 9, "column_count": 4},
        ],
    }


def test_ordered_projection_serializes_a_real_constructor_and_one_native_three_stage_generator() -> None:
    pipeline = _pipeline()
    pipeline[0]["_or_"] = [_source_choice([0]), _source_choice([2, 0])]
    original = deepcopy(pipeline[0])
    steps, _splitter = validate_structural_profile(pipeline)
    dsl = lower_structural_hpo_pipeline(steps, source_layout=_layout())
    assert len(dsl["pipeline"]) == 1
    generator = dsl["pipeline"][0]
    assert generator["kind"] == "generator" and generator["mode"] == "cartesian"
    assert [len(stage["branches"]) for stage in generator["stages"]] == [2, 2, 2]
    source_nodes = [branch["steps"][0] for branch in generator["stages"][0]["branches"]]
    node = source_nodes[1]
    params = json.loads(json.dumps(node["params"], allow_nan=False))
    reconstructed = ColumnTransformer(**params)
    assert reconstructed.transformers == [["selected", "passthrough", list(range(9, 13)) + list(range(6))]]
    assert reconstructed.remainder == "drop" and reconstructed.sparse_threshold == 0
    assert "selected" not in params, "deep pseudo-parameters cannot become constructor arguments"
    selection = node["metadata"]["nirs4all_structural_source_selection"]
    assert selection["source_indices"] == [2, 0]
    assert selection["source_order"] == ["source_2", "source_0"]
    assert selection["source_ids"] == ["fixture-source-2", "fixture-source-0"]
    assert selection["input_source_widths"] == [6, 3, 4]
    assert selection["columns"] == list(range(9, 13)) + list(range(6))
    assert selection["input_source_layout"] == _layout()
    assert pipeline[0] == original
    assert "entries" not in dsl and "variant_id" not in dsl


def test_source_aliases_lower_to_the_same_native_selection_as_integer_indices() -> None:
    layout = _layout()
    integer = build_source_selection_step(_source_choice([2, 0]), "witness", layout)
    aliases = build_source_selection_step(_source_choice(["source_2", "source_0"]), "witness", layout)
    assert aliases == integer


@pytest.mark.parametrize("mutation", [
    lambda layout: layout["blocks"][1].__setitem__("column_start", 5),
    lambda layout: layout["blocks"][1].__setitem__("column_count", 0),
    lambda layout: layout["blocks"][1].__setitem__("source_index", True),
    lambda layout: layout["source_ids"].__setitem__(1, layout["source_ids"][0]),
    lambda layout: layout["source_order"].reverse(),
])
def test_noncontiguous_or_ambiguous_layout_is_refused_without_guessing_columns(mutation: Any) -> None:
    layout = _layout()
    mutation(layout)
    with pytest.raises(ValueError, match="(?i)(source|column|layout)"):
        build_source_selection_step(_source_choice([2, 0]), "witness", layout)


def _forbid_native_or_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    import dag_ml
    from n4m.model_selection.optimizer import Optimizer

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid source declaration reached native catalogue, optimizer or FIT")

    for name in ("compile_pipeline_dsl_artifact_with_controllers", "prepare_host_hpo_structural_catalogue"):
        monkeypatch.setattr(dag_ml, name, forbidden)
    monkeypatch.setattr(Optimizer, "__init__", forbidden)
    for cls in (StandardScaler, SNV, SavitzkyGolay, Ridge, PLSRegression):
        monkeypatch.setattr(cls, "fit", forbidden)


@pytest.mark.parametrize("selectors", [[], (), [True], [-1], [0.5], [None], [0, 0], ["source_0", "source_0"], [[0]], {"source": 0}])
def test_invalid_source_selector_shape_is_refused_without_native_owners(selectors: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_native_or_fit(monkeypatch)
    pipeline = _pipeline()
    pipeline[0]["_or_"][0] = _source_choice(selectors)
    with pytest.raises(ValueError, match="(?i)(source|selector|structural)"):
        validate_structural_profile(pipeline)


@pytest.mark.parametrize("branch", [
    {"merge": {"sources": "concat"}},
    {"merge": {"sources": {"strategy": "mean", "sources": [0]}}},
    {"merge": {"sources": {"strategy": "concat", "sources": [0], "unknown": 1}}},
    {"merge": {"sources": {"strategy": "concat", "sources": [0]}, "features": True}},
    {"merge": {"sources": {"strategy": "concat", "sources": [0]}}, "name": "extra"},
    {"sources": [0]},
    None,
    StandardScaler(),
])
def test_unconsumed_source_choice_fields_or_nonconcat_strategy_are_refused(branch: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_native_or_fit(monkeypatch)
    pipeline = _pipeline()
    pipeline[0]["_or_"][0] = deepcopy(branch)
    with pytest.raises(ValueError, match="(?i)(source|concat|structural)"):
        validate_structural_profile(pipeline)


@pytest.mark.parametrize("selectors", [[3], ["source_9"], ["unknown"], [0, "source_0"]])
def test_source_indices_names_and_alias_duplicates_resolve_before_catalogue(selectors: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_native_or_fit(monkeypatch)
    pipeline = _pipeline()
    pipeline[0]["_or_"][0] = _source_choice(selectors)
    with pytest.raises(ValueError, match="(?i)(source|selector|structural)"):
        _prepare_structure(pipeline, _dataset(), {"engine": "n4m", "space": {"model.alpha": [0.1], "model.n_components": [2]}}, {})


@pytest.mark.parametrize("operator,components,match", [
    (SavitzkyGolay(window_length=5, polyorder=2), 2, "window_length.*feature width"),
    (SNV(ddof=4), 2, "ddof.*feature width"),
    (StandardScaler(), 5, "(?i)(component|feature|width)"),
])
def test_all_source_choices_bound_transform_and_model_width_before_catalogue(
    operator: Any, components: int, match: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_native_or_fit(monkeypatch)
    pipeline = _pipeline()
    pipeline[0]["_or_"] = [_source_choice([0, 2]), _source_choice([2])]
    pipeline[1]["_or_"] = [None, operator]
    with pytest.raises(ValueError, match=match):
        _prepare_structure(pipeline, _dataset(), {"engine": "n4m", "space": {"model.alpha": [0.1], "model.n_components": [components]}}, {})


def test_typed_multimodal_adapter_is_not_silently_flattened_into_dense_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from nirs4all_io import MultimodalDataset, TensorSource

    from nirs4all.data.multimodal import MultimodalSpectroDataset

    ids = [f"typed-{index}" for index in range(18)]
    values = np.arange(234, dtype=np.float32).reshape(18, 13)
    cohort = MultimodalDataset(
        {f"source_{index}": TensorSource(block, ids, representation_id="signal_1d")
         for index, block in enumerate([values[:, :6], values[:, 6:9], values[:, 9:]])},
        sample_ids=ids, y=np.arange(18, dtype=float), groups=list(np.repeat(["a", "b", "c"], 6)),
        partitions=["train"] * 18, name="typed-source-profile-refusal", task_type="regression",
    )
    dataset = MultimodalSpectroDataset(cohort)
    _forbid_native_or_fit(monkeypatch)
    with pytest.raises(ValueError, match="(?i)(typed|multimodal|dense|spectrodataset)"):
        _prepare_structure(_pipeline(), dataset, {"engine": "n4m", "space": {"model.alpha": [0.1], "model.n_components": [2]}}, {})
