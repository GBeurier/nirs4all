"""Isolated interim identity codec. No execution, expansion or owner binding.

This restrictive profile is not rollout-ready. Typed-key/tuple completion,
owner seed/choice exposure and qualified dispatch remain mandatory.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from typing import Any

PROFILE = "nirs4all.realized-recipe.ordered-json.v1"
MAX_DEPTH = 128
MAX_RECIPE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
MAX_OCCURRENCES = 10_000
MAX_PREPARED_BYTES = 64 * 1024 * 1024
Path = tuple[str | int, ...]


class IdentityError(ValueError):
    """Refuse an identity claim, without claiming execution is unsupported."""

    def __init__(self, code: str, path: Path = ()) -> None:
        self.code = code
        self.path = path
        super().__init__(f"{code} at {path!r}")


def _limits(max_bytes: int, max_depth: int) -> None:
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_RECIPE_BYTES:
        raise IdentityError("limit_invalid")
    if type(max_depth) is not int or not 0 < max_depth <= MAX_DEPTH:
        raise IdentityError("limit_invalid")


def _text(value: str, path: Path) -> None:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise IdentityError("invalid_unicode", path) from error


@dataclass(frozen=True, slots=True)
class _Node:
    value: Any
    cost: int
    deepest: Path = ()


class _Budget:
    """Charge expanded canonical bytes before allocating expanded containers."""

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.canonical = 0
        self.add(len(PROFILE.encode("ascii")) + 1)

    def add(self, amount: int) -> None:
        self.canonical += amount
        if self.limit is not None and self.canonical > self.limit:
            raise IdentityError("size_limit")


def _scalar(value: Any, budget: _Budget) -> _Node:
    # Payload conversion is checked first, preserving huge-integer refusal.
    _json_bytes(value)
    node = _Node(value, len(_json_bytes(_typed(value))))
    budget.add(node.cost)
    return node


def _snapshot(value: Any, path: Path, active: set[int], max_depth: int, budget: _Budget, memo: dict[int, _Node]) -> _Node:
    if len(path) > max_depth:
        raise IdentityError("depth_limit", path)
    kind = type(value)
    if value is None or kind in (bool, int):
        return _scalar(value, budget)
    if kind is float:
        if not math.isfinite(value):
            raise IdentityError("nonfinite_number", path)
        return _scalar(value, budget)
    if kind is str:
        _text(value, path)
        return _scalar(value, budget)
    if kind is tuple:
        raise IdentityError("tuple_not_supported", path)
    if kind not in (list, dict):
        raise IdentityError("value_not_supported", path)
    marker = id(value)
    if marker in active:
        raise IdentityError("cycle", path)
    if marker in memo:
        node = memo[marker]
        if len(path) + len(node.deepest) > max_depth:
            raise IdentityError("depth_limit", (*path, *node.deepest[:max_depth - len(path) + 1]))
        budget.add(node.cost)
        return node
    active.add(marker)
    try:
        cost = len(b'["list",[]]' if kind is list else b'["mapping",[]]') + max(len(value) - 1, 0)
        budget.add(cost)
        deepest: Path = ()
        if kind is list:
            items: list[Any] = []
            for index, item in enumerate(value):
                child = _snapshot(item, (*path, index), active, max_depth, budget, memo)
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
                key_cost = len(_json_bytes(key)) + 3  # [key,typed-value]
                budget.add(key_cost)
                child = _snapshot(item, (*path, key), active, max_depth, budget, memo)
                copied[key] = child.value
                cost += key_cost + child.cost
                if len(child.deepest) + 1 > len(deepest):
                    deepest = (key, *child.deepest)
        node = _Node(copied, cost, deepest)
        memo[marker] = node
        return node
    finally:
        active.remove(marker)


def _typed(value: Any) -> list[Any]:
    if value is None:
        return ["null"]
    kind = type(value)
    if kind is bool:
        return ["bool", value]
    if kind is int:
        return ["int", str(value)]
    if kind is float:
        return ["float", value.hex()]
    if kind is str:
        return ["string", value]
    if kind is list:
        return ["list", [_typed(item) for item in value]]
    return ["mapping", [[key, _typed(item)] for key, item in value.items()]]


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise IdentityError("invalid_json") from error


def _check_digest(expected: str, actual: str) -> None:
    if type(expected) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None:
        raise IdentityError("digest_invalid")
    if not hmac.compare_digest(expected, actual):
        raise IdentityError("digest_mismatch")


@dataclass(frozen=True, slots=True)
class RecipeEncoding:
    """Exact validated insertion bytes and their typed, ordered digest input."""

    payload: bytes
    canonical: bytes
    digest: str
    profile: str = PROFILE

    def value(self) -> Any:
        """Return an independent verified container tree."""
        return verify_recipe(self.payload, self.digest, profile=self.profile)


def encode_recipe(value: Any, *, max_bytes: int = MAX_RECIPE_BYTES, max_depth: int = MAX_DEPTH) -> RecipeEncoding:
    """Snapshot only supported values; never import, stringify or construct."""
    _limits(max_bytes, max_depth)
    copied = _snapshot(value, (), set(), max_depth, _Budget(max_bytes), {}).value
    payload = _json_bytes(copied)
    if len(payload) > max_bytes:
        raise IdentityError("size_limit")
    canonical = PROFILE.encode("ascii") + b"\0" + _json_bytes(_typed(copied))
    if len(canonical) > max_bytes:
        raise IdentityError("size_limit")
    return RecipeEncoding(payload, canonical, "sha256:" + hashlib.sha256(canonical).hexdigest())


def decode_json(payload: bytes, *, max_bytes: int = MAX_RECIPE_BYTES) -> Any:
    """Strict JSON read, rejecting coercion, duplicates and nonfinite leaves."""
    _limits(max_bytes, MAX_DEPTH)
    if type(payload) is not bytes:
        raise IdentityError("payload_not_bytes")
    if len(payload) > max_bytes:
        raise IdentityError("size_limit")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise IdentityError("invalid_unicode") from error

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise IdentityError("duplicate_key")
            result[key] = value
        return result

    def constant(token: str) -> Any:
        raise IdentityError("nonfinite_number")

    try:
        decoded = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except IdentityError:
        raise
    except RecursionError as error:
        raise IdentityError("depth_limit") from error
    except ValueError as error:
        raise IdentityError("invalid_json") from error
    return _snapshot(decoded, (), set(), MAX_DEPTH, _Budget(None), {}).value


def verify_recipe(payload: bytes, digest: str, *, profile: str = PROFILE) -> Any:
    """Reconstruct digest input from exact stored bytes, without execution."""
    if profile != PROFILE:
        raise IdentityError("profile_unsupported")
    value = decode_json(payload)
    encoded = encode_recipe(value)
    if payload != encoded.payload:
        raise IdentityError("noncanonical_payload")
    _check_digest(digest, encoded.digest)
    return value
