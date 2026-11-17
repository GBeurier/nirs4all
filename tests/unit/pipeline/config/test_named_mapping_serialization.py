"""User-chosen mapping keys are names, never fields of a step envelope.

Source names, branch names, partition names and tag names are free text. A name
that spells a descriptive field (``metadata``), an argument field (``params``)
or a component declaration (``class``) must round-trip exactly like any other.
"""

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nirs4all.operators.filters.y_outlier import YOutlierFilter
from nirs4all.pipeline.config._generator.keywords import GENERATION_KEYWORDS
from nirs4all.pipeline.config.component_serialization import _GENERATOR_KEYS, deserialize_component, serialize_component

DESCRIPTIVE = ("name", "description", "metadata", "_metadata_", "_tags_")
ARGUMENT = ("params", "args", "nir_params")
COMPONENT = ("class", "function", "instance")
STRUCTURAL = ("branch", "steps", "values", "tag")
NAMES = ("nir", *DESCRIPTIVE, *ARGUMENT, *COMPONENT, *STRUCTURAL)
# BranchController never reads a ``_``-prefixed key of a duplication mapping as a branch name.
BRANCH_NAMES = tuple(name for name in NAMES if not name.startswith("_"))
SCALER = "sklearn.preprocessing.StandardScaler"
CANONICAL_SCALER = "sklearn.preprocessing._data.StandardScaler"
RESERVED = {"_metadata_": {"origin": SCALER}, "_tags_": [SCALER]}
LABELS = {"name": SCALER, "description": SCALER, "metadata": {"origin": [SCALER]}, **RESERVED}


def _wire(pipeline: Any) -> Any:
    """Serialize as a host does: canonical form, then a JSON transport."""
    return json.loads(json.dumps(serialize_component(pipeline)))


def _assert_body(body: Any) -> None:
    assert isinstance(body, list) and len(body) == 2, body
    assert isinstance(body[0], StandardScaler), body
    assert isinstance(body[1], Ridge) and body[1].alpha == 0.4, body


@pytest.mark.parametrize("name", NAMES)
def test_by_source_steps_restore_components_for_any_source_name(name: str) -> None:
    pipeline = [{"branch": {"by_source": True, "steps": {name: [StandardScaler(), Ridge(alpha=0.4)], "other": [StandardScaler()]}}}]

    restored = deserialize_component(_wire(pipeline))

    steps = restored[0]["branch"]["steps"]
    assert list(steps) == [name, "other"]
    _assert_body(steps[name])
    assert restored[0]["branch"]["by_source"] is True


@pytest.mark.parametrize("criterion", [{"by_metadata": "site"}, {"by_tag": "y_outlier_iqr"}, {"by_filter": YOutlierFilter(threshold=2.5)}], ids=["by_metadata", "by_tag", "by_filter"])
@pytest.mark.parametrize("name", DESCRIPTIVE)
def test_separation_steps_restore_components_for_any_partition_name(criterion: dict[str, Any], name: str) -> None:
    pipeline = [{"branch": {**criterion, "steps": {name: [StandardScaler(), Ridge(alpha=0.4)]}}}]

    restored = deserialize_component(_wire(pipeline))

    _assert_body(restored[0]["branch"]["steps"][name])


@pytest.mark.parametrize("name", BRANCH_NAMES)
def test_named_branch_bodies_restore_components_for_any_branch_name(name: str) -> None:
    pipeline = [{"branch": {name: [StandardScaler(), Ridge(alpha=0.4)], "other": [StandardScaler()]}}]

    restored = deserialize_component(_wire(pipeline))

    assert list(restored[0]["branch"]) == [name, "other"]
    _assert_body(restored[0]["branch"][name])


@pytest.mark.parametrize("name", NAMES)
def test_by_tag_value_conditions_restore_callables_for_any_group_name(name: str) -> None:
    pipeline = [{"branch": {"by_tag": "score", "values": {name: np.isscalar, "rest": False}, "steps": [StandardScaler()]}}]

    restored = deserialize_component(_wire(pipeline))

    assert restored[0]["branch"]["values"] == {name: np.isscalar, "rest": False}


@pytest.mark.parametrize("name", NAMES)
def test_named_tag_filters_restore_for_any_tag_name(name: str) -> None:
    pipeline = [{"tag": {name: YOutlierFilter(threshold=2.5)}}]

    restored = deserialize_component(_wire(pipeline))

    tagger = restored[0]["tag"][name]
    assert isinstance(tagger, YOutlierFilter) and tagger.threshold == 2.5


@pytest.mark.parametrize("name", NAMES)
def test_serializer_canonicalizes_references_under_any_name(name: str) -> None:
    """Pipeline identity must not depend on how a source, branch or tag is named."""
    by_source = serialize_component([{"branch": {"by_source": True, "steps": {name: [SCALER]}}}])
    tagged = serialize_component([{"tag": {name: SCALER}}])

    assert by_source[0]["branch"]["steps"][name] == [CANONICAL_SCALER]
    assert tagged[0]["tag"][name] == CANONICAL_SCALER
    if name in BRANCH_NAMES:
        assert serialize_component([{"branch": {name: [SCALER]}}])[0]["branch"][name] == [CANONICAL_SCALER]


ENVELOPES: dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    "by_source": (lambda step: [{"branch": {"by_source": True, "steps": {"metadata": [step]}}}], lambda doc: doc[0]["branch"]["steps"]["metadata"][0]),
    "by_source_params": (lambda step: [{"branch": {"by_source": True, "steps": {"params": [step]}}}], lambda doc: doc[0]["branch"]["steps"]["params"][0]),
    "named_branch": (lambda step: [{"branch": {"metadata": [step]}}], lambda doc: doc[0]["branch"]["metadata"][0]),
    "named_branch_args": (lambda step: [{"branch": {"args": [step]}}], lambda doc: doc[0]["branch"]["args"][0]),
    "listed_branch": (lambda step: [{"branch": [{"name": SCALER, "steps": [step]}]}], lambda doc: doc[0]["branch"][0]["steps"][0]),
    "root": (lambda step: {"pipeline": [step], "name": SCALER, "description": SCALER}, lambda doc: doc["pipeline"][0]),
}


@pytest.mark.parametrize("shape", ENVELOPES)
def test_descriptive_fields_of_step_envelopes_stay_literal(shape: str) -> None:
    """A real descriptive field is text, also inside a body keyed by a colliding name."""
    wrap, step_of = ENVELOPES[shape]
    document = wrap({"model": {"class": "sklearn.linear_model.Ridge", "params": {"alpha": 0.4}, "name": SCALER}, **LABELS})

    serialized = serialize_component(document)
    restored = deserialize_component(document)

    for result in (serialized, restored):
        assert {key: step_of(result)[key] for key in LABELS} == LABELS
    assert step_of(serialized)["model"]["name"] == SCALER
    model = step_of(restored)["model"]
    assert isinstance(model, Ridge) and model.alpha == 0.4
    if shape == "listed_branch":
        assert serialized[0]["branch"][0]["name"] == restored[0]["branch"][0]["name"] == SCALER
    if shape == "root":
        assert {key: restored[key] for key in ("name", "description")} == {key: serialized[key] for key in ("name", "description")} == {"name": SCALER, "description": SCALER}


@pytest.mark.parametrize("branch", [
    {"_or_": [[SCALER], ["sklearn.preprocessing.MinMaxScaler"]], **RESERVED},
    {"snv": [SCALER], **RESERVED},
    {"by_source": True, "steps": {"_or_": [[SCALER], ["sklearn.preprocessing.MinMaxScaler"]], **RESERVED}},
], ids=["generated_branches", "named_branches", "generated_source_steps"])
def test_reserved_generator_fields_stay_literal(branch: dict[str, Any]) -> None:
    """Generator annotations are not branch or source names."""
    document = [{"branch": branch}]

    for result in (serialize_component(document), deserialize_component(document)):
        body = result[0]["branch"]
        body = body.get("steps", body)
        assert {key: body[key] for key in RESERVED} == RESERVED


def test_component_declared_under_a_grammar_keyword_is_still_a_component() -> None:
    """A single live step serializes to a declaration, not to a mapping of names."""
    restored = deserialize_component(_wire([{"branch": Ridge(alpha=0.4)}, {"tag": YOutlierFilter(threshold=2.5)}]))

    assert isinstance(restored[0]["branch"], Ridge) and restored[0]["branch"].alpha == 0.4
    assert isinstance(restored[1]["tag"], YOutlierFilter) and restored[1]["tag"].threshold == 2.5


def test_generator_keys_mirror_the_generator_grammar() -> None:
    assert _GENERATOR_KEYS == GENERATION_KEYWORDS
