"""Versioned, in-memory original-source envelopes for isolated preparation.

No source file IO, imports of operator references, generator expansion, storage
or scientific execution occurs here. This is the restrictive interim profile.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from .variant_identity import MAX_DEPTH, MAX_SOURCE_BYTES, PROFILE, IdentityError, Path, RecipeEncoding, _Budget, _check_digest, _json_bytes, _Node, _text, decode_json, encode_recipe

SOURCE_SCHEMA = "nirs4all.recipe-source.v1"
_FIELDS = {"schema", "profile", "representation", "document", "original_text", "conversions"}
_KINDS = {"python_value", "json_bytes", "constructor_parameters"}


def _source_limit(max_bytes: int) -> None:
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_SOURCE_BYTES:
        raise IdentityError("limit_invalid")


def _range(value: tuple[Any, ...]) -> dict[str, Any] | None:
    # Exactly the existing _preserve_finetune_ranges conversion, not a new DSL.
    if len(value) == 2 and all(type(item) in (int, float) for item in value):
        return {"type": "int" if all(type(item) is int for item in value) else "float", "min": value[0], "max": value[1]}
    return None


def _capture(value: Any, path: Path, active: set[int], conversions: list[dict[str, Any]],
             budget: _Budget, memo: dict[tuple[int, bool], _Node], allow_range: bool = False) -> _Node:
    if len(path) > MAX_DEPTH:
        raise IdentityError("depth_limit", path)
    kind = type(value)
    if kind is tuple:
        converted = _range(value) if allow_range else None
        if converted is None:
            raise IdentityError("tuple_not_supported", path)
        try:
            encoded = encode_recipe(converted)
        except IdentityError as error:
            raise IdentityError(error.code, path) from error
        conversion = {"path": list(path), "original": list(value)}
        cost = len(encoded.canonical) - len(PROFILE.encode("ascii")) - 1
        # Conversion records also grow per occurrence, not per object identity.
        record_cost = len(encode_recipe(conversion).canonical) - len(PROFILE.encode("ascii")) - 1
        budget.add(cost + record_cost)
        conversions.append(conversion)
        return _Node(converted, cost)
    if kind not in (list, dict):
        try:
            encoded = encode_recipe(value)
        except IdentityError as error:
            raise IdentityError(error.code, path) from error
        cost = len(encoded.canonical) - len(PROFILE.encode("ascii")) - 1
        budget.add(cost)
        return _Node(value, cost)
    marker = id(value)
    if marker in active:
        raise IdentityError("cycle", path)
    cache_key = (marker, allow_range)
    if cache_key in memo:
        node = memo[cache_key]
        if len(path) + len(node.deepest) > MAX_DEPTH:
            raise IdentityError("depth_limit", (*path, *node.deepest[:MAX_DEPTH - len(path) + 1]))
        budget.add(node.cost)
        return node
    active.add(marker)
    conversion_start = len(conversions)
    try:
        cost = len(b'["list",[]]' if kind is list else b'["mapping",[]]') + max(len(value) - 1, 0)
        budget.add(cost)
        deepest: Path = ()
        if kind is list:
            # The owner conversion deliberately leaves categorical lists alone.
            items: list[Any] = []
            for index, item in enumerate(value):
                child = _capture(item, (*path, index), active, conversions, budget, memo)
                items.append(child.value)
                cost += child.cost
                if len(child.deepest) + 1 > len(deepest):
                    deepest = (index, *child.deepest)
            copied: Any = items
        else:
            copied = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise IdentityError("key_not_supported", path)
                _text(key, path)
                key_cost = len(_json_bytes(key)) + 3
                budget.add(key_cost)
                child = _capture(item, (*path, key), active, conversions, budget, memo, allow_range or key == "finetune_params")
                copied[key] = child.value
                cost += key_cost + child.cost
                if len(child.deepest) + 1 > len(deepest):
                    deepest = (key, *child.deepest)
        node = _Node(copied, cost, deepest)
        # Revisit conversions at their actual source locations. Only completed
        # conversion-free subtrees can safely reuse normalized copies and costs.
        if len(conversions) == conversion_start:
            memo[cache_key] = node
        return node
    finally:
        active.remove(marker)


def _restore(document: Any, conversions: list[Any]) -> Any:
    restored = encode_recipe(document).value()
    seen: set[Path] = set()
    for conversion in conversions:
        if type(conversion) is not dict or set(conversion) != {"path", "original"}:
            raise IdentityError("source_conversion_invalid")
        path = conversion["path"]
        original = conversion["original"]
        if type(path) is not list or not path or any(type(part) not in (str, int) for part in path):
            raise IdentityError("source_conversion_invalid")
        location = tuple(path)
        if location in seen or type(original) is not list:
            raise IdentityError("source_conversion_invalid")
        seen.add(location)
        converted = _range(tuple(original))
        if converted is None:
            raise IdentityError("source_conversion_invalid")
        current = restored
        try:
            for part in path[:-1]:
                if type(current) is list and (type(part) is not int or part < 0):
                    raise IdentityError("source_conversion_invalid")
                if type(current) is dict and type(part) is not str:
                    raise IdentityError("source_conversion_invalid")
                current = current[part]
            key = path[-1]
            if type(current) is list and (type(key) is not int or key < 0):
                raise IdentityError("source_conversion_invalid")
            if type(current) is dict and type(key) is not str:
                raise IdentityError("source_conversion_invalid")
            if encode_recipe(current[key]).canonical != encode_recipe(converted).canonical:
                raise IdentityError("source_conversion_invalid")
            current[key] = tuple(original)
        except (KeyError, IndexError, TypeError) as error:
            raise IdentityError("source_conversion_invalid") from error
    return restored


def _envelope(payload: bytes, digest: str) -> dict[str, Any]:
    value = decode_json(payload, max_bytes=MAX_SOURCE_BYTES)
    encoded = encode_recipe(value, max_bytes=MAX_SOURCE_BYTES)
    if payload != encoded.payload:
        raise IdentityError("noncanonical_payload")
    _check_digest(digest, "sha256:" + hashlib.sha256(payload).hexdigest())
    if type(value) is not dict or set(value) != _FIELDS:
        raise IdentityError("source_schema_invalid")
    if value["schema"] != SOURCE_SCHEMA or value["profile"] != PROFILE:
        raise IdentityError("source_schema_invalid")
    if type(value["representation"]) is not str or value["representation"] not in _KINDS:
        raise IdentityError("source_schema_invalid")
    if type(value["document"]) not in (list, dict) or type(value["conversions"]) is not list:
        raise IdentityError("source_schema_invalid")
    if value["representation"] == "constructor_parameters" and type(value["document"]) is not dict:
        raise IdentityError("source_schema_invalid")
    restored = _restore(value["document"], value["conversions"])
    conversions: list[dict[str, Any]] = []
    normalized = _capture(restored, (), set(), conversions, _Budget(MAX_SOURCE_BYTES), {}).value
    if encode_recipe(normalized).canonical != encode_recipe(value["document"]).canonical or conversions != value["conversions"]:
        raise IdentityError("source_conversion_invalid")
    if value["representation"] == "json_bytes":
        if type(value["original_text"]) is not str or conversions:
            raise IdentityError("source_schema_invalid")
        parsed = decode_json(value["original_text"].encode("utf-8"), max_bytes=MAX_SOURCE_BYTES)
        if encode_recipe(parsed).canonical != encode_recipe(value["document"]).canonical:
            raise IdentityError("source_document_mismatch")
    elif value["original_text"] is not None:
        raise IdentityError("source_schema_invalid")
    return value


@dataclass(frozen=True, slots=True)
class SourceEnvelope:
    """Immutable bytes; each accessor verifies and returns fresh containers."""

    payload: bytes
    digest: str

    def document(self) -> Any:
        value = _envelope(self.payload, self.digest)
        return _restore(value["document"], value["conversions"])

    def normalized_document(self) -> Any:
        return _envelope(self.payload, self.digest)["document"]

    def original_bytes(self) -> bytes | None:
        text = _envelope(self.payload, self.digest)["original_text"]
        return text.encode("utf-8") if text is not None else None

    def conversions(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], _envelope(self.payload, self.digest)["conversions"])


def _build(document: Any, representation: str, original_text: str | None, max_bytes: int) -> SourceEnvelope:
    _source_limit(max_bytes)
    if type(document) not in (list, dict):
        raise IdentityError("source_root_invalid")
    conversions: list[dict[str, Any]] = []
    normalized = _capture(document, (), set(), conversions, _Budget(max_bytes), {}).value
    envelope = {"schema": SOURCE_SCHEMA, "profile": PROFILE, "representation": representation,
                "document": normalized, "original_text": original_text, "conversions": conversions}
    encoded = encode_recipe(envelope, max_bytes=max_bytes)
    return read_source(encoded.payload, "sha256:" + hashlib.sha256(encoded.payload).hexdigest())


def capture_source(document: Any, *, max_bytes: int = MAX_SOURCE_BYTES) -> SourceEnvelope:
    """Snapshot parsed/Python containers before serializer tuple/key loss."""
    return _build(document, "python_value", None, max_bytes)


def capture_json_source(original: bytes, *, max_bytes: int = MAX_SOURCE_BYTES) -> SourceEnvelope:
    """Preserve exact supplied UTF-8 JSON bytes, including whitespace/order."""
    _source_limit(max_bytes)
    document = decode_json(original, max_bytes=max_bytes)
    return _build(document, "json_bytes", original.decode("utf-8"), max_bytes)


def read_source(payload: bytes, digest: str) -> SourceEnvelope:
    """Verify content and conversion provenance without constructing objects."""
    _envelope(payload, digest)
    return SourceEnvelope(payload, digest)


@dataclass(frozen=True, slots=True)
class ConstructorProjection:
    """Explicit owner-supplied parameter snapshot and separate projection."""

    source: SourceEnvelope
    recipe: RecipeEncoding


def project_constructor(parameters: Any, projector: Callable[[Any], Any]) -> ConstructorProjection:
    """Precheck exposed raw parameters before invoking an explicit projector.

    The caller supplies the owner projection; this helper does not introspect,
    resolve references, import classes or promise that arbitrary callbacks are safe.
    """
    if type(parameters) is not dict:
        raise IdentityError("constructor_parameters_unavailable")
    source = _build(parameters, "constructor_parameters", None, MAX_SOURCE_BYTES)
    projected = projector(source.document())
    return ConstructorProjection(source, encode_recipe(projected))
