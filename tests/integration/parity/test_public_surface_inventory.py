"""Keep the reviewed legacy syntax surface tied to the live implementation."""

from __future__ import annotations

import importlib
import json

from nirs4all.controllers.data.merge import MergeController
from nirs4all.pipeline.steps.parser import StepParser

from . import coverage_meter as meter
from ._public_surface import SNAPSHOT_PATH, discover_public_surface, surface_drift


def test_reviewed_public_surface_matches_legacy_implementation() -> None:
    assert not surface_drift(), (
        "Legacy public pipeline surface changed; inspect parser, generator, "
        "controller routes and operator exports, then update parity oracles "
        "and public_surface.json: " + ", ".join(surface_drift())
    )


def test_inventory_includes_routes_missing_from_the_case_keyword_registry() -> None:
    """Aliases and multi-model syntax must remain visible outside case labels."""
    surface = discover_public_surface()
    assert {"merge", "merge_sources", "merge_predictions"} <= set(surface["controller_routes"]["MergeController"]["keywords"])
    assert {"augment_chart", "augmentation_chart"} <= set(surface["controller_routes"]["AugmentationChartController"]["keywords"])
    assert "fold_" in surface["controller_routes"]["FoldChartController"]["prefixes"]


def test_surface_drift_detects_new_parser_keyword(monkeypatch) -> None:
    monkeypatch.setattr(StepParser, "WORKFLOW_KEYWORDS", [*StepParser.WORKFLOW_KEYWORDS, "new_legacy_step"])
    assert "parser" in surface_drift()


def test_surface_drift_detects_new_controller_alias(monkeypatch) -> None:
    monkeypatch.setattr(MergeController, "SUPPORTED_KEYWORDS", {*MergeController.SUPPORTED_KEYWORDS, "new_merge_alias"})
    assert "controller_routes" in surface_drift()


def test_surface_drift_detects_new_exported_operator(monkeypatch) -> None:
    operators = importlib.import_module("nirs4all.operators")
    monkeypatch.setattr(operators, "__all__", [*operators.__all__, "NewPublicOperator"])
    assert "operator_exports" in surface_drift()


def test_snapshot_has_no_duplicate_names() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    for package, names in snapshot["operator_exports"].items():
        assert len(names) == len(set(names)), package


def test_model_checkpoint_positions_inventory_is_complete() -> None:
    complete, open_patterns, invalid = meter.checkpoint_inventory_status()
    assert complete
    assert not open_patterns
    assert not invalid


def test_verified_checkpoint_requires_a_real_public_test(tmp_path) -> None:
    inventory = json.loads(meter.MODEL_CHECKPOINT_COVERAGE_JSON.read_text(encoding="utf-8"))
    inventory["inventory_complete"] = True
    inventory["patterns"][1]["coverage"] = "verified"
    inventory["patterns"][1]["evidence"] = ["tests/integration/parity/test_dagml_sequential_metamodel.py::test_does_not_exist"]
    path = tmp_path / "checkpoint_inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    complete, open_patterns, invalid = meter.checkpoint_inventory_status(path)
    assert complete
    assert "model_then_augmentation_then_model" not in open_patterns
    assert "missing test function" in invalid[0]
