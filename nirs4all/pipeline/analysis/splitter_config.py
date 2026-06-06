"""Splitter (CV) configuration extraction from stored pipeline configs.

Answers "which cross-validation setup did this pipeline use?" from the
``expanded_config`` JSON the workspace store persists per pipeline — the
canonical step list written by :meth:`WorkspaceStore.begin_pipeline`.

This inference previously lived in the nirs4all-studio HTTP layer (BV-07 in
its 2026-06-05 tech-debt closeout): the studio hand-parsed the library's own
storage format and hand-listed splitter names. The parsing of library-written
data belongs to the library; UIs keep only their display vocabulary.

Recognition order: nirs4all's own splitter registry, sklearn
``model_selection`` splitters (by module path or class name), then a token
fallback (``split``/``fold``/``loo``/``holdout``) for custom classes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SplitterConfig",
    "extract_splitter_config",
    "is_splitter_reference",
    "parse_expanded_config_steps",
]

#: Matches non-reconstructable Python repr strings (``<X object at 0x...>``)
#: that older runs serialized via ``json.dumps(default=str)``.
_OBJECT_REPR_RE = re.compile(r"^\s*<(?P<path>.+?)\s+object at 0x[0-9A-Fa-f]+>\s*$")

#: sklearn.model_selection splitter class names (importing sklearn lazily is
#: not worth it for a name check — this is the stable public set).
_SKLEARN_SPLITTERS = frozenset({
    "KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold",
    "RepeatedKFold", "RepeatedStratifiedKFold", "LeaveOneOut", "LeavePOut",
    "LeaveOneGroupOut", "LeavePGroupsOut", "ShuffleSplit",
    "StratifiedShuffleSplit", "GroupShuffleSplit", "TimeSeriesSplit",
    "PredefinedSplit",
})

_TOKEN_FALLBACK = ("split", "fold", "loo", "holdout")


@dataclass
class SplitterConfig:
    """Canonical CV-splitter configuration recovered from a stored pipeline.

    Attributes:
        splitter_class: Leaf class name (e.g. ``"KFold"``).
        reference: The raw step reference as stored.
        n_splits: Fold count when the splitter declares one.
        shuffle: Shuffle flag when declared.
        random_state: Seed when declared.
        test_size: Test fraction when declared (shuffle/holdout splitters).
        group_by: Grouping column when declared (group-aware splitters).
    """

    splitter_class: str
    reference: str
    n_splits: int | None = None
    shuffle: bool | None = None
    random_state: int | None = None
    test_size: float | None = None
    group_by: str | None = None


def _parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def parse_expanded_config_steps(expanded_config: Any) -> list[Any]:
    """Return the canonical step list from a stored ``expanded_config``.

    Accepts the raw column value (JSON string, dict with a ``pipeline`` key,
    or already-parsed list) and normalizes to a list of steps.
    """
    parsed = _parse_json_maybe(expanded_config)
    if isinstance(parsed, dict) and isinstance(parsed.get("pipeline"), list):
        return list(parsed["pipeline"])
    if isinstance(parsed, list):
        return parsed
    if parsed is None:
        return []
    return [parsed]


def _class_name_from_path(class_path: Any) -> str:
    if not isinstance(class_path, str) or not class_path:
        return ""
    normalized = class_path.strip()
    match = _OBJECT_REPR_RE.match(normalized)
    if match:
        normalized = match.group("path").strip()
    if not normalized:
        return ""
    return normalized.rsplit(".", 1)[-1]


def _extract_step_reference(step: Any) -> tuple[str | None, dict[str, Any]]:
    """Return ``(reference, params)`` for a canonical step when possible."""
    if isinstance(step, str):
        return step, {}
    if not isinstance(step, dict):
        return None, {}
    if "class" in step and isinstance(step.get("class"), str):
        params = step.get("params")
        return step["class"], params if isinstance(params, dict) else {}
    return None, {}


def is_splitter_reference(reference: str) -> bool:
    """Whether a step reference denotes a CV/data splitter.

    Checks nirs4all's splitter registry, the sklearn ``model_selection``
    splitter set (by class name or module path), then falls back to name
    tokens for custom splitters.
    """
    class_name = _class_name_from_path(reference)
    if not class_name:
        return False

    from nirs4all.operators import splitters as _splitters_module

    if class_name in getattr(_splitters_module, "__all__", ()):
        return True
    if class_name in _SKLEARN_SPLITTERS:
        return True
    if "model_selection" in reference:
        return True
    lowered = reference.lower()
    return any(token in lowered for token in _TOKEN_FALLBACK)


def extract_splitter_config(expanded_config: Any) -> SplitterConfig | None:
    """Recover the CV-splitter configuration from a stored pipeline config.

    Scans the canonical step list for the first splitter step and returns its
    class plus canonical parameters. Returns ``None`` when the pipeline has no
    recognizable splitter (e.g. predict-only chains).
    """
    for step in parse_expanded_config_steps(expanded_config):
        reference, params = _extract_step_reference(step)
        if not reference:
            continue
        # Non-reconstructable repr strings carry no parseable identity beyond
        # the path inside them; normalize, then test like any reference.
        normalized = _OBJECT_REPR_RE.sub(r"\g<path>", str(reference).strip()).strip()
        if not is_splitter_reference(normalized):
            continue

        config = SplitterConfig(
            splitter_class=_class_name_from_path(normalized) or normalized,
            reference=normalized,
        )

        raw_folds = params.get("n_splits", params.get("cv_folds"))
        if isinstance(raw_folds, (int, float)) and int(raw_folds) > 0:
            config.n_splits = int(raw_folds)
        if isinstance(params.get("shuffle"), bool):
            config.shuffle = params["shuffle"]
        raw_random_state = params.get("random_state")
        if isinstance(raw_random_state, (int, float)):
            config.random_state = int(raw_random_state)
        raw_test_size = params.get("test_size")
        if isinstance(raw_test_size, (int, float)):
            config.test_size = float(raw_test_size)
        for group_key in ("group_by", "groups", "repetition", "aggregate"):
            group_value = params.get(group_key)
            if isinstance(group_value, str) and group_value.strip():
                config.group_by = group_value.strip()
                break

        return config

    return None
