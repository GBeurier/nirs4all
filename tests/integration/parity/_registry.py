"""Pipeline-case registry for the parity oracle.

Each `cases_*` module imports `register` and adds its cases at module top
level. The conftest imports every cases module before pytest collection so
the registry is fully populated when parametrization runs.

Invariants enforced at `register` time:

- case names are globally unique;
- case names match `^[a-z][a-z0-9_]+$` (snake_case);
- every declared keyword belongs to the canonical nirs4all DSL surface
  (`CANONICAL_KEYWORDS`) so the coverage matrix stays in lockstep with the
  CLAUDE.md keyword table;
- `dataset_key` resolves through `_datasets.known_keys()`;
- `task` is the typed `Literal` (regression / classification);
- `expected_min_predictions >= 1`.

Metric tolerances are recorded per case but enforcement happens in Phase 3+
once a gold-standard baseline run is captured.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ._datasets import known_keys

Task = Literal["regression", "classification"]

# `skip_kind` classifies *why* a case is skipped so the runner can decide
# between `pytest.skip` (legitimate fixture / API mismatch) and
# `pytest.xfail(strict=True)` (a known legacy bug that must turn back to PASS
# the moment the bug is fixed — so a fixed bug never goes silent again).
SkipKind = Literal["", "fixture", "unknown_semantics", "legacy_bug"]


# Canonical nirs4all 0.9.x DSL keywords (drawn from CLAUDE.md keyword table +
# the `WORKFLOW_KEYWORDS` list in `pipeline/steps/parser.py`). Cases declare
# only the keyword strings that actually appear inside a dict step in their
# pipeline. Operator-class capabilities (preprocessing class, splitter class,
# model family) live in `capabilities`, not here.
CANONICAL_KEYWORDS: frozenset[str] = frozenset({
    # Workflow keywords (parser-recognized)
    "model",
    "preprocessing",
    "feature_augmentation",
    "auto_transfer_preproc",
    "concat_transform",
    "y_processing",
    "sample_augmentation",
    "branch",
    "merge",
    "tag",
    "exclude",
    "rep_to_sources",
    "rep_to_pp",
    # Per-step hyperparameters
    "finetune_params",
    "train_params",
    "refit_params",
    "fit_on_all",
    "force_layout",
    "na_policy",
    "fill_value",
    "name",
    # Generator keywords
    "_or_",
    "_range_",
    "_log_range_",
    "_grid_",
    "_cartesian_",
    "_zip_",
    "_chain_",
    "_sample_",
    # Generator modifier / constraint keywords (item B surface). `_seed_` makes a
    # stochastic generator (`_sample_`) deterministic; `_mutex_` / `_requires_` /
    # `_exclude_` prune the expanded variant set; `_depends_on_` is a DEAD keyword
    # the constraint engine never consults (documented by a dedicated no-op test).
    "_seed_",
    "_mutex_",
    "_requires_",
    "_exclude_",
    "_depends_on_",
    # Selection / sampling / annotation modifiers. `count` caps the variant set
    # (with `_seed_` for the subsample RNG); `_weights_` weights random selection;
    # `then_pick` / `then_arrange` are the second-order selection keywords;
    # `_tags_` / `_metadata_` are INERT annotations (do not change the variant set).
    "count",
    "_weights_",
    "then_pick",
    "then_arrange",
    "_tags_",
    "_metadata_",
})

# Free-form operator-family / coverage tags. Not the same as DSL keywords:
# these track what controllers and operator classes a case exercises.
COMMON_CAPABILITIES: frozenset[str] = frozenset({
    "preprocessing_transform",
    "y_processing_transform",
    "sklearn_model",
    "pytorch_model",
    "tensorflow_model",
    "jax_model",
    "cross_validator",
    "nirs_splitter",
    "filter",
    "stacking_meta_model",
    "multi_source",
    "classification_model",
    "regression_model",
    "augmentation",
    "generator",
    "session_api",
    "bundle_io",
    "predict_path",
    "explain_path",
    "retrain_path",
})


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]+$")


@dataclass(frozen=True)
class PipelineCase:
    """Frozen contract for one nirs4all pipeline shape.

    Fields:
        name: unique snake_case identifier.
        description: one-line summary.
        keywords: tuple of canonical DSL keywords present in the pipeline dict
            steps (validated against `CANONICAL_KEYWORDS`).
        capabilities: operator-family labels exercised (free-form; cross-checked
            against `COMMON_CAPABILITIES` in `register` to catch typos).
        dataset_key: id resolved via `_datasets.dataset_path()`.
        pipeline_factory: callable returning a fresh pipeline list each call —
            factories avoid sharing mutable operator instances across cases.
        dataset_kwargs: keyword arguments forwarded to `DatasetConfigs(...)`
            (repetition, aggregate, aggregate_method, aggregate_exclude_outliers,
            signal_type, task_type). The bridge must reproduce these.
        task: `"regression"` or `"classification"`.
        expected_min_predictions: lower bound on
            `RunResult.num_predictions` (which counts prediction *entries*
            across all variants, folds, and partitions — see RunResult docs);
            cases use this as a regression alarm, not as a tight assertion.
        metric_tolerances: per-metric absolute tolerance for parity comparison
            once gold-standard baselines exist (e.g. `{"rmse": 0.01, "r2": 0.005}`).
            Empty by default; the bridge parity runner consults this dict.
        tags: free-form labels for pytest filtering (`fast`, `slow`,
            `requires_tf`, etc.).
        skip_reason: if non-empty, the smoke runner skips with this reason.
        skip_kind: classifies the skip — `"fixture"` (missing data),
            `"unknown_semantics"` (API contract not yet confirmed), or
            `"legacy_bug"` (known nirs4all 0.9.x bug). `"legacy_bug"` cases
            are run as `xfail(strict=True)` so they XPASS the moment the bug
            is fixed — they never silently disappear from coverage.
    """

    name: str
    description: str
    keywords: tuple[str, ...]
    capabilities: tuple[str, ...]
    dataset_key: str
    pipeline_factory: Callable[[], list[Any]]
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    task: Task = "regression"
    expected_min_predictions: int = 1
    metric_tolerances: dict[str, float] = field(default_factory=dict)
    tags: frozenset[str] = field(default_factory=frozenset)
    skip_reason: str = ""
    skip_kind: SkipKind = ""

    @property
    def pipeline(self) -> list[Any]:
        """Materialize a fresh pipeline list (factory invoked each access)."""
        return self.pipeline_factory()


_REGISTRY: dict[str, PipelineCase] = {}


def register(case: PipelineCase) -> PipelineCase:
    """Add `case` to the global registry, asserting every invariant.

    Failing fast at import time produces a precise traceback pointing at the
    offending `cases_*` module instead of a generic collection error.
    """
    _validate_name(case)
    _validate_keywords(case)
    _validate_capabilities(case)
    _validate_dataset(case)
    _validate_min_predictions(case)
    if case.name in _REGISTRY:
        prev = _REGISTRY[case.name]
        raise ValueError(
            f"duplicate parity case {case.name!r}: existing description "
            f"{prev.description!r}, new {case.description!r}"
        )
    _REGISTRY[case.name] = case
    return case


def _validate_name(case: PipelineCase) -> None:
    if not _NAME_RE.match(case.name):
        raise ValueError(
            f"parity case name {case.name!r} must match snake_case "
            f"pattern {_NAME_RE.pattern!r}"
        )


def _validate_keywords(case: PipelineCase) -> None:
    unknown = sorted(set(case.keywords) - CANONICAL_KEYWORDS)
    if unknown:
        raise ValueError(
            f"parity case {case.name!r} declares unknown DSL keyword(s) "
            f"{unknown}; canonical list lives in `_registry.CANONICAL_KEYWORDS` "
            f"(if CLAUDE.md is being updated, update both)"
        )


def _validate_capabilities(case: PipelineCase) -> None:
    unknown = sorted(set(case.capabilities) - COMMON_CAPABILITIES)
    if unknown:
        raise ValueError(
            f"parity case {case.name!r} declares unknown capability label(s) "
            f"{unknown}; add to `_registry.COMMON_CAPABILITIES` if intentional"
        )


def _validate_dataset(case: PipelineCase) -> None:
    if case.dataset_key not in known_keys():
        raise ValueError(
            f"parity case {case.name!r} references unknown dataset_key "
            f"{case.dataset_key!r}; declared keys: {known_keys()}"
        )


def _validate_min_predictions(case: PipelineCase) -> None:
    if case.expected_min_predictions < 1:
        raise ValueError(
            f"parity case {case.name!r}: expected_min_predictions must be >= 1, "
            f"got {case.expected_min_predictions}"
        )


def all_cases() -> list[PipelineCase]:
    """Every registered case in insertion order (stable for pytest ids)."""
    return list(_REGISTRY.values())


def by_keyword(keyword: str) -> list[PipelineCase]:
    """Cases that declare `keyword` in their canonical keywords tuple."""
    return [c for c in _REGISTRY.values() if keyword in c.keywords]


def by_capability(cap: str) -> list[PipelineCase]:
    """Cases that declare `cap` in their capability labels."""
    return [c for c in _REGISTRY.values() if cap in c.capabilities]


def by_tag(tag: str) -> list[PipelineCase]:
    """Cases that carry `tag` in their tags set."""
    return [c for c in _REGISTRY.values() if tag in c.tags]


def get(name: str) -> PipelineCase:
    """Lookup a case by name; raise KeyError if absent."""
    return _REGISTRY[name]


def keyword_coverage() -> dict[str, list[str]]:
    """Map every canonical keyword (declared and undeclared) to covering cases.

    Keywords with an empty list need a parity case before the bridge can ship.
    """
    coverage: dict[str, list[str]] = {kw: [] for kw in CANONICAL_KEYWORDS}
    for case in _REGISTRY.values():
        for kw in case.keywords:
            coverage[kw].append(case.name)
    return coverage


def capability_coverage() -> dict[str, list[str]]:
    """Map every known capability to covering cases (undeclared → empty list)."""
    coverage: dict[str, list[str]] = {cap: [] for cap in COMMON_CAPABILITIES}
    for case in _REGISTRY.values():
        for cap in case.capabilities:
            coverage[cap].append(case.name)
    return coverage
