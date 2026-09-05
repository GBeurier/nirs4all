"""Bounded dataset-document normalization for library hosts.

This boundary reads configuration documents and directory names, never matrix
payloads. ConfigNormalizer remains the owner of aliases, sources, variations and
folder conventions. Path authorization remains the caller's responsibility.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from nirs4all.data.parsers.normalizer import ConfigNormalizer


@dataclass(frozen=True)
class DatasetDocumentLimits:
    """Finite admission limits, explicitly adjustable for trusted workloads."""

    max_bytes: int = 2 * 1024 * 1024
    max_depth: int = 64
    max_nodes: int = 100_000
    max_aliases: int = 1024
    max_directory_entries: int = 4096

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")


def _plain_document(value: Any, limits: DatasetDocumentLimits) -> None:
    active: set[int] = set()
    nodes = 0
    string_bytes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes, string_bytes
        nodes += 1
        if nodes > limits.max_nodes or depth > limits.max_depth:
            raise ValueError("Dataset document exceeds node/depth budget")
        if isinstance(item, str):
            string_bytes += len(item.encode("utf-8"))
            if string_bytes > limits.max_bytes:
                raise ValueError("Dataset document exceeds byte budget")
            return
        if item is None or type(item) in (bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) not in (list, dict):
            raise ValueError("Dataset document must contain finite plain JSON values")
        if id(item) in active:
            raise ValueError("Dataset document contains a cycle")
        active.add(id(item))
        if isinstance(item, dict):
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("Dataset document keys must be strings")
                visit(key, depth + 1)
                visit(child, depth + 1)
        else:
            for child in item:
                visit(child, depth + 1)
        active.remove(id(item))

    visit(value, 0)
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) > limits.max_bytes:
        raise ValueError("Dataset document exceeds byte budget")


def _preflight_yaml(content: str, limits: DatasetDocumentLimits) -> None:
    """Bound expanded alias cost before SafeLoader can construct/merge nodes."""
    stack: list[tuple[str | None, int]] = []
    anchors: dict[str, int] = {}
    aliases = 0

    def add(cost: int) -> None:
        if cost > limits.max_nodes:
            raise ValueError("Dataset YAML exceeds expanded node budget")
        if stack:
            anchor, previous = stack[-1]
            if previous + cost > limits.max_nodes:
                raise ValueError("Dataset YAML exceeds expanded node budget")
            stack[-1] = anchor, previous + cost

    for event in yaml.parse(content, Loader=yaml.SafeLoader):
        if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
            if len(stack) >= limits.max_depth:
                raise ValueError("Dataset YAML exceeds depth budget")
            stack.append((event.anchor, 1))
        elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
            anchor, cost = stack.pop()
            if anchor is not None:
                anchors[anchor] = cost
            add(cost)
        elif isinstance(event, yaml.ScalarEvent):
            if event.anchor is not None:
                anchors[event.anchor] = 1
            add(1)
        elif isinstance(event, yaml.AliasEvent):
            aliases += 1
            if aliases > limits.max_aliases or event.anchor not in anchors:
                raise ValueError("Dataset YAML contains a cycle, unresolved alias or excessive aliases")
            add(anchors[event.anchor])


class _BoundedNormalizer(ConfigNormalizer):
    def __init__(self, limits: DatasetDocumentLimits, base_dir: Path):
        super().__init__()
        self.limits = limits
        self.base_dir = base_dir

    def _check_directory(self, path: str | Path) -> None:
        with os.scandir(path) as entries:
            for count, _ in enumerate(entries, 1):
                if count > self.limits.max_directory_entries:
                    raise ValueError("Dataset folder exceeds directory-entry budget")

    def _normalize_string(self, path_str: str) -> tuple[dict[str, Any] | None, str]:
        if Path(path_str).is_dir():
            self._check_directory(path_str)
        return super()._normalize_string(path_str)

    def _normalize_dict(self, config: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        canonical = self._apply_key_aliases(config)
        if canonical.get("folder") is not None:
            folder = Path(canonical["folder"])
            canonical["folder"] = str((folder if folder.is_absolute() else self.base_dir / folder).resolve())
            self._check_directory(canonical["folder"])
        return super()._normalize_dict(canonical)

    def _load_config_file(self, file_path: str) -> tuple[dict[str, Any], str]:
        path = Path(file_path)
        with path.open("rb") as stream:
            content = stream.read(self.limits.max_bytes + 1)
        if len(content) > self.limits.max_bytes:
            raise ValueError("Dataset configuration file exceeds byte budget")
        text = content.decode("utf-8")
        _preflight_yaml(text, self.limits)
        parsed = self._parse_json(text, file_path) if path.suffix.lower() == ".json" else self._parse_yaml(text, file_path)
        if not isinstance(parsed, dict):
            raise ValueError("Dataset configuration must contain an object")
        _plain_document(parsed, self.limits)
        return parsed, str(parsed.get("name", path.stem))


def _explicit_paths(value: Any, base: Path, *, path_value: bool = False) -> Any:
    if isinstance(value, str) and path_value:
        path = Path(value)
        return str((path if path.is_absolute() else base / path).resolve())
    if isinstance(value, list):
        return [_explicit_paths(item, base, path_value=path_value) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            is_path = key in {"path", "file", "input", "index_file", "folds", "train_file", "test_file", "predict_file"}
            is_path |= key in {f"{partition}_{role}" for partition in ("train", "test") for role in ("x", "y", "group")}
            result[key] = _explicit_paths(item, base, path_value=is_path)
        return result
    return value


def normalize_dataset_document(
    document: str | Path | dict[str, Any], *, base_dir: str | Path | None = None,
    limits: DatasetDocumentLimits | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible config with explicit absolute file references.

    Supports existing folder, JSON/YAML, aliases, sources and variations syntax.
    Config bytes/structure and directory enumeration are bounded before parser
    conversion. Matrix files are not opened. Relative references in files use
    the document's directory; dictionary references use ``base_dir`` or cwd.
    This normalizes paths, but does not authorize them or claim an OS sandbox.
    """
    limits = limits or DatasetDocumentLimits()
    if isinstance(document, (str, Path)):
        path = Path(document).resolve()
        base = Path(base_dir).resolve() if base_dir is not None else (path if path.is_dir() else path.parent)
        normalizer = _BoundedNormalizer(limits, base)
        if path.is_dir():
            normalized, name = normalizer.normalize(path)
        else:
            if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                raise ValueError("Dataset document path must name a directory or JSON/YAML config")
            parsed, name = normalizer._load_config_file(str(path))
            normalized, _ = normalizer.normalize(parsed)
    elif isinstance(document, dict):
        _plain_document(document, limits)
        base = Path(base_dir or Path.cwd()).resolve()
        normalizer = _BoundedNormalizer(limits, base)
        normalized, name = normalizer.normalize(document)
    else:
        raise ValueError("Dataset document must be a folder/config path or object")
    if normalized is None:
        raise ValueError("Dataset document could not be normalized")
    # Normalization is not idempotent for expanded source/variation metadata:
    # convert declarations exactly once through their existing owner.
    _plain_document(normalized, limits)
    result = cast(dict[str, Any], _explicit_paths(normalized, base))
    result.setdefault("name", name)
    _plain_document(result, limits)
    return result
