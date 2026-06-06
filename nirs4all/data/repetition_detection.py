"""Repetition (biological replicate) group detection.

Heuristics for recovering "which measurements belong to the same biological
sample" from dataset metadata columns or sample-id naming conventions.

These previously lived in the nirs4all-studio HTTP layer; they are NIRS data
semantics and belong to the library (the studio's 2026-06-05 tech-debt
closeout flagged the boundary violation).

Two entry points:

- :func:`auto_detect_repetition_column` — pick the metadata column that most
  plausibly identifies biological samples (``bio_sample``, ``sample_group``,
  ...), skipping partition/fold bookkeeping and repeat-index columns.
- :func:`detect_repetition_groups` — group sample ids into biological samples
  using an explicit regex or a ladder of common naming conventions
  (``name_rep1``, ``name_2``, ``name-A``, ``name (1)``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RepetitionGroups",
    "auto_detect_repetition_column",
    "detect_repetition_groups",
]

#: Sample-id naming conventions tried (in order) by auto-detection.
DEFAULT_ID_PATTERNS: tuple[str, ...] = (
    r"^(.+?)[-_][Rr]ep\d+$",  # sample_rep1, sample-Rep2
    r"^(.+?)[-_]\d+$",  # sample_1, sample-2
    r"^(.+?)[-_][A-Za-z]$",  # sample_A, sample-b
    r"^(.+?)\s*\(\d+\)$",  # sample (1), sample (2)
)


@dataclass
class RepetitionGroups:
    """Result of sample-id based repetition detection.

    Attributes:
        groups: Mapping of biological-sample id to member sample indices.
        pattern: The regex that produced the grouping (``None`` when no
            convention matched at least one repeated group).
        n_repeated: Number of biological samples with >= 2 measurements.
    """

    groups: dict[str, list[int]] = field(default_factory=dict)
    pattern: str | None = None
    n_repeated: int = 0

    @property
    def has_repetitions(self) -> bool:
        """Whether any biological sample has more than one measurement."""
        return self.n_repeated > 0


def _normalize_metadata_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _looks_like_repeat_index(name: str) -> bool:
    return (
        name in {"rep", "reps"}
        or name.startswith("replicate")
        or name.startswith("repeat")
        or name.startswith("repetition")
        or name.startswith("technicalrep")
    )


def auto_detect_repetition_column(metadata: Mapping[str, Sequence[Any]]) -> str | None:
    """Pick the metadata column that most plausibly groups biological samples.

    Skips partition/fold bookkeeping columns and repeat-index columns, then
    prefers columns whose normalized name marks a biological-sample grouping
    (``bio_sample``, ``biological_sample_id``, ``sample_group``, ...) and that
    actually contain repeated values.

    Args:
        metadata: Mapping of column name to per-sample values.

    Returns:
        The winning column name, or ``None`` when no candidate qualifies.
    """
    candidates: list[tuple[int, int, int, str]] = []
    for column_name, raw_values in metadata.items():
        normalized_name = _normalize_metadata_name(column_name)
        if normalized_name in {"set", "partition", "fold", "foldid"} or _looks_like_repeat_index(normalized_name):
            continue

        counts: dict[str, int] = {}
        for value in raw_values:
            if value is None or value == "":
                continue
            token = str(value)
            counts[token] = counts.get(token, 0) + 1

        repeated_groups = sum(1 for count in counts.values() if count >= 2)
        repeated_measurements = sum(count for count in counts.values() if count >= 2)
        if repeated_groups == 0:
            continue

        is_preferred = int(
            normalized_name in {"biosample", "biosampleid", "biologicalsample", "biologicalsampleid", "samplegroup", "groupid"}
            or ("bio" in normalized_name and "sample" in normalized_name)
            or ("sample" in normalized_name and "group" in normalized_name)
        )
        if not is_preferred:
            continue
        candidates.append((is_preferred, repeated_groups, repeated_measurements, str(column_name)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    return candidates[0][3]


def detect_repetition_groups(
    sample_ids: Sequence[str],
    pattern: str | None = None,
) -> RepetitionGroups:
    """Group sample ids into biological samples by naming convention.

    With an explicit *pattern*, every id is matched against it: group(1) (or
    the whole match) becomes the biological-sample id; non-matching ids form
    their own group. Without a pattern, the :data:`DEFAULT_ID_PATTERNS` ladder
    is tried and the convention producing the most repeated groups wins.

    Args:
        sample_ids: Per-measurement sample identifiers.
        pattern: Optional explicit regex (its group(1) is the biological id).

    Returns:
        A :class:`RepetitionGroups`; ``groups`` is empty only when an explicit
        *pattern* is invalid.

    Raises:
        re.error: If *pattern* is provided and invalid.
    """
    if pattern is not None:
        compiled = re.compile(pattern)
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, sample_id in enumerate(sample_ids):
            match = compiled.match(str(sample_id))
            if match:
                bio_id = match.group(1) if match.groups() else match.group(0)
                groups[bio_id].append(idx)
            else:
                groups[str(sample_id)].append(idx)
        n_repeated = sum(1 for indices in groups.values() if len(indices) >= 2)
        return RepetitionGroups(groups=dict(groups), pattern=pattern, n_repeated=n_repeated)

    best = RepetitionGroups()
    for candidate in DEFAULT_ID_PATTERNS:
        compiled = re.compile(candidate)
        groups = defaultdict(list)
        for idx, sample_id in enumerate(sample_ids):
            match = compiled.match(str(sample_id))
            if match:
                groups[match.group(1)].append(idx)
            else:
                groups[str(sample_id)].append(idx)
        n_repeated = sum(1 for indices in groups.values() if len(indices) >= 2)
        if n_repeated > best.n_repeated:
            best = RepetitionGroups(groups=dict(groups), pattern=candidate, n_repeated=n_repeated)

    return best
