"""Discover the legacy pipeline surface independently of the DAG-ML router.

The checked-in snapshot is a review trigger, not proof that every discovered
shape works. Ordered compositions and per-operator options still need public
dual-engine oracles before ``inventory_complete`` can be set.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from pathlib import Path
from typing import Any

import nirs4all.controllers  # noqa: F401 - registers legacy controllers
from nirs4all.controllers.registry import CONTROLLER_REGISTRY
from nirs4all.pipeline.config._generator.keywords import ALL_KEYWORDS
from nirs4all.pipeline.steps.parser import StepParser

SNAPSHOT_PATH = Path(__file__).with_name("public_surface.json")
_OPERATOR_PACKAGES = (
    "nirs4all.operators",
    "nirs4all.operators.augmentation",
    "nirs4all.operators.filters",
    "nirs4all.operators.methods",
    "nirs4all.operators.models",
    "nirs4all.operators.splitters",
    "nirs4all.operators.transforms",
)


def _literal_strings(node: ast.AST, controller: type) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set().union(*(_literal_strings(item, controller) for item in node.elts))
    if isinstance(node, ast.Attribute) and node.attr == "SUPPORTED_KEYWORDS":
        return set(getattr(controller, "SUPPORTED_KEYWORDS", ()))
    return set()


def _controller_route(controller: type) -> dict[str, Any]:
    method = controller.__dict__.get("matches")
    if method is None:
        raise ValueError(f"{controller.__name__} must declare matches() for inventory review")
    source = textwrap.dedent(inspect.getsource(method.__func__ if isinstance(method, classmethod) else method))
    syntax = ast.parse(source)
    keywords: set[str] = set()
    prefixes: set[str] = set()
    for node in ast.walk(syntax):
        if isinstance(node, ast.Compare):
            for left, right in zip([node.left, *node.comparators[:-1]], node.comparators):
                if isinstance(left, ast.Name) and left.id == "keyword":
                    keywords.update(_literal_strings(right, controller))
                if isinstance(right, ast.Name) and right.id == "keyword":
                    keywords.update(_literal_strings(left, controller))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "keyword"
            and node.func.attr == "startswith"
            and node.args
        ):
            prefixes.update(_literal_strings(node.args[0], controller))
    # The AST digest catches changes to route predicates that the literal
    # extractor cannot interpret, while ignoring formatting and comments.
    digest = hashlib.sha256(ast.dump(syntax, include_attributes=False).encode()).hexdigest()[:16]
    return {"keywords": sorted(keywords), "prefixes": sorted(prefixes), "matches_ast": digest}


def discover_public_surface() -> dict[str, Any]:
    """Return the parser, generator, controller and exported operator surface."""
    import importlib

    return {
        "schema_version": 1,
        "parser": {
            "workflow_keywords": sorted(StepParser.WORKFLOW_KEYWORDS),
            "reserved_keywords": sorted(StepParser.RESERVED_KEYWORDS),
            "serialization_operators": sorted(StepParser.SERIALIZATION_OPERATORS),
        },
        "generator_keywords": sorted(ALL_KEYWORDS),
        "controller_routes": {cls.__name__: _controller_route(cls) for cls in sorted(CONTROLLER_REGISTRY, key=lambda cls: cls.__name__)},
        "operator_exports": {package: sorted(getattr(importlib.import_module(package), "__all__", ())) for package in _OPERATOR_PACKAGES},
    }


def surface_drift() -> list[str]:
    """Name every top-level surface section that differs from the reviewed snapshot."""
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    actual = discover_public_surface()
    return sorted(key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key))
