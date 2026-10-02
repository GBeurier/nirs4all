"""Typed declarations preserve the strict fixed profile and seal native choices."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from sklearn.model_selection import GroupKFold

from nirs4all.pipeline.config.component_serialization import deserialize_component, serialize_component
from nirs4all.pipeline.dagml.methods_multimodal import recipe_from_estimator, source_schemas_from_cohort
from nirs4all.pipeline.dagml.structural_multimodal import lower_typed_choices, validate_alpha_space
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure, validate_structural_profile
from nirs4all.pipeline.dagml.tuning_contracts import parse_tuning_spec

_PATH = Path(__file__).resolve().parents[4] / "examples/user/04_models/U20_structural_hpo_typed_modalities.py"
_SPEC = importlib.util.spec_from_file_location("typed_structural_unit_example", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def test_public_roundtrip_keeps_order_weights_and_full_signed_raw_schemas() -> None:
    pipeline = deserialize_component(json.loads(json.dumps(serialize_component(example.make_pipeline()))))
    steps, splitter = validate_structural_profile(pipeline)
    assert type(splitter) is GroupKFold
    schemas = source_schemas_from_cohort(example.make_dataset())
    dsl = lower_typed_choices(steps, schemas)
    generator = dsl["pipeline"][0]
    assert generator["mode"] == "cartesian" and len(generator["stages"]) == 1
    branches = generator["stages"][0]["branches"]
    orders = [["nir"], ["nir", "image"], ["image", "nir"], ["image", "series", "metadata"]]
    for branch, expected in zip(branches, orders, strict=True):
        assert len(branch["steps"]) == 1
        node = branch["steps"][0]
        recipe = node["operator"]["recipe"]
        assert recipe["source_order"] == expected and set(recipe["encoders"]) == set(expected)
        assert node["params"]["recipe"] == recipe
        assert node["params"]["source_schemas"] == schemas == node["operator"]["source_schemas"]
        assert node["params"]["model__alpha"] == 1.0
        assert set(node["operator"]["source_schemas"]) == {"nir", "image", "series", "metadata"}
    with pytest.raises(ValueError, match="source order"):
        recipe_from_estimator(example.make_model(("image", "nir")))


@pytest.mark.parametrize(
    "space",
    [
        {"model__alpha": [-1.0, 1.0]},
        {"model__alpha": [True, 1.0]},
        {"model__alpha": [float("nan"), 1.0]},
        {"model__alpha": [1.0], "source_weights__image": [0.5]},
    ],
)
def test_alpha_axis_refuses_invalid_numbers_and_external_structure_knobs(space: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_alpha_space(parse_tuning_spec({"engine": "n4m", "space": space}))


@pytest.mark.parametrize("alpha_path", ["model__alpha", "model.alpha"])
def test_public_alpha_spellings_prepare_the_same_native_catalogue(alpha_path: str, tmp_path: Path) -> None:
    tuning = example.make_tuning(tmp_path / "study")
    tuning["space"] = {alpha_path: [0.1, 1.0]}
    prepared = _prepare_structure(example.make_pipeline(), example.make_dataset(), tuning, {})
    assert set(prepared["spec"].space) == {"model.alpha"}
    assert len(prepared["catalogue"]["entries"]) == 4


def test_pca_bounds_and_public_refit_false_fail_before_catalogue_or_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import dag_ml
    from n4m import MultimodalPipeline

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid typed declaration reached catalogue or encoder FIT")

    monkeypatch.setattr(dag_ml, "prepare_host_hpo_structural_catalogue", forbidden)
    monkeypatch.setattr(MultimodalPipeline, "fit", forbidden)
    tuning = example.make_tuning(tmp_path / "study")
    with pytest.raises(ValueError, match="refit"):
        _prepare_structure(example.make_pipeline(), example.make_dataset(), tuning, {"refit": False})
    pipeline = example.make_pipeline()
    pipeline[-1]["model"]["_or_"][1].transformers["image"].n_components = 1000
    with pytest.raises(ValueError, match="PCA components"):
        _prepare_structure(pipeline, example.make_dataset(), tuning, {"refit": True})
