"""Phase D2 declarative context/query sampler.

This module provides a deterministic, bench-only context/query sampler that
sits between :class:`~nirsyntheticpfn.data.latents.CanonicalLatentBatch` /
:class:`~nirsyntheticpfn.data.views.SpectralViewBatch` and
:class:`~nirsyntheticpfn.data.tasks.NIRSPriorTask`. The sampler emits
explicit integer index splits that :meth:`NIRSPriorTask.from_batches`
consumes directly; D1's contract surface is left untouched.

D2 is intentionally narrow:

- No downstream training or evaluation runs here. The sampler only produces
  ``(context_indices, query_indices)`` plus a JSON-serialisable
  ``split_policy`` and a diagnostics dict.
- D3 multi-output regression and classification are accepted: regression
  diagnostics are aggregated per output (no per-sample target values),
  and classification stratification uses the joint label tuple as the
  stratification class for multi-output targets.
- No realism / transfer claim is introduced. Both A3 / B2 risk gates remain
  negative on the resulting :class:`NIRSPriorTask` because they are carried
  by the source latent batch unchanged.

The sampler exposes three deterministic strategies:

``random``
    Uniform permutation of all rows, then a non-empty ``(context, query)``
    prefix split.

``stratified_classification``
    Per-class proportional split for classification targets with
    integer-like labels, using a largest-remainder reconciliation so the
    requested ``n_context`` is met exactly.

``group_holdout``
    Group-disjoint partitioning over ``CanonicalLatentBatch.group_ids``;
    every group lands entirely in either context or query, never both. When
    exact row counts are impossible because groups are atomic, the sampler
    reports requested and actual sizes separately.

Determinism is enforced by routing every random decision through a single
``numpy.random.default_rng(seed)`` instance. No global RNG is read or
modified. The resulting :class:`ContextQuerySplit` carries arrays plus a
non-leaky ``split_policy`` (no per-sample targets / latents) and an
optional diagnostics dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nirsyntheticpfn.data.latents import CanonicalLatentBatch
from nirsyntheticpfn.data.tasks import NIRSPriorTask
from nirsyntheticpfn.data.views import SpectralViewBatch

__all__ = [
    "ContextQuerySplit",
    "ContextQuerySplitConfig",
    "ContextQuerySplitError",
    "sample_context_query_split",
    "sample_nirs_prior_task",
]

_VALID_STRATEGIES: tuple[str, ...] = (
    "random",
    "stratified_classification",
    "group_holdout",
)
_VALID_TARGET_SOURCES: tuple[str, ...] = ("target_clean", "target_noisy")

_NON_LEAKY_DIAGNOSTIC_KEYS: frozenset[str] = frozenset(
    {
        "strategy",
        "n_total",
        "n_outputs",
        "n_context",
        "n_query",
        "requested_n_context",
        "requested_n_query",
        "exact_size_match",
        "indices_disjoint",
        "latent_ids_disjoint",
        "constant_target_context",
        "constant_target_query",
        "shared_label_count",
        "n_classes_context",
        "n_classes_query",
        "n_joint_labels",
        "shared_joint_label_count",
        "n_groups_context",
        "n_groups_query",
        "groups_disjoint",
    }
)


class ContextQuerySplitError(ValueError):
    """Raised when a sampler request fails contract checks."""

    def __init__(self, failures: list[dict[str, str]]) -> None:
        self.failures = failures
        summary = "; ".join(
            f"{failure.get('reason', 'unknown')}:{failure.get('field', '?')}"
            for failure in failures
        )
        super().__init__(summary or "invalid context/query split request")


def _failure(reason: str, field: str, message: str) -> dict[str, str]:
    return {"reason": reason, "field": field, "message": message}


def _is_strict_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, np.integer))


@dataclass(frozen=True)
class ContextQuerySplitConfig:
    """Declarative configuration for a single context/query split.

    Attributes:
        strategy: One of ``"random"``, ``"stratified_classification"``,
            ``"group_holdout"``.
        seed: Mandatory deterministic seed (Python ``int`` or
            ``numpy.integer``). ``None``, ``bool``, and ``float`` are
            rejected.
        n_context: Optional explicit context size (>= 1). Mutually
            exclusive with ``context_fraction``.
        n_query: Optional explicit query size (>= 1). Mutually exclusive
            with ``query_fraction``.
        context_fraction: Optional fraction in ``(0, 1)``. Resolves to
            ``round(n_total * context_fraction)``.
        query_fraction: Optional fraction in ``(0, 1)``. Resolves to
            ``round(n_total * query_fraction)``.
        target_source: Which target array drives stratification and
            diagnostics. Defaults to ``"target_noisy"``.
        shuffle: When ``True`` (default), permute rows before partitioning;
            otherwise process rows in their canonical batch order.
        diagnostics: When ``True`` (default), populate the
            :attr:`ContextQuerySplit.diagnostics` dict.
    """

    strategy: str
    seed: int | np.integer[Any]
    n_context: int | None = None
    n_query: int | None = None
    context_fraction: float | None = None
    query_fraction: float | None = None
    target_source: str = "target_noisy"
    shuffle: bool = True
    diagnostics: bool = True

    def __post_init__(self) -> None:
        failures: list[dict[str, str]] = []

        if self.strategy not in _VALID_STRATEGIES:
            failures.append(
                _failure(
                    "invalid_strategy",
                    "strategy",
                    f"strategy must be one of {_VALID_STRATEGIES}, got {self.strategy!r}",
                )
            )

        if self.seed is None or not _is_strict_int(self.seed):
            failures.append(
                _failure(
                    "invalid_seed",
                    "seed",
                    "seed must be a Python int or numpy.integer (None, bool, float forbidden)",
                )
            )

        if self.target_source not in _VALID_TARGET_SOURCES:
            failures.append(
                _failure(
                    "invalid_target_source",
                    "target_source",
                    f"target_source must be in {_VALID_TARGET_SOURCES}, got {self.target_source!r}",
                )
            )

        if self.n_context is not None:
            if not _is_strict_int(self.n_context):
                failures.append(
                    _failure("invalid_size", "n_context", "n_context must be int or None")
                )
            elif int(self.n_context) < 1:
                failures.append(
                    _failure("invalid_size", "n_context", "n_context must be >= 1 when provided")
                )
        if self.n_query is not None:
            if not _is_strict_int(self.n_query):
                failures.append(
                    _failure("invalid_size", "n_query", "n_query must be int or None")
                )
            elif int(self.n_query) < 1:
                failures.append(
                    _failure("invalid_size", "n_query", "n_query must be >= 1 when provided")
                )

        for name, frac in (
            ("context_fraction", self.context_fraction),
            ("query_fraction", self.query_fraction),
        ):
            if frac is None:
                continue
            if isinstance(frac, bool) or not isinstance(frac, (int, float)):
                failures.append(
                    _failure(
                        "invalid_fraction",
                        name,
                        f"{name} must be a float in (0, 1) or None",
                    )
                )
            elif not (0.0 < float(frac) < 1.0):
                failures.append(
                    _failure(
                        "invalid_fraction",
                        name,
                        f"{name} must be strictly in (0, 1), got {frac!r}",
                    )
                )

        if self.n_context is not None and self.context_fraction is not None:
            failures.append(
                _failure(
                    "conflicting_size",
                    "n_context",
                    "n_context and context_fraction are mutually exclusive",
                )
            )
        if self.n_query is not None and self.query_fraction is not None:
            failures.append(
                _failure(
                    "conflicting_size",
                    "n_query",
                    "n_query and query_fraction are mutually exclusive",
                )
            )

        if not isinstance(self.shuffle, bool):
            failures.append(_failure("invalid_flag", "shuffle", "shuffle must be a bool"))
        if not isinstance(self.diagnostics, bool):
            failures.append(
                _failure("invalid_flag", "diagnostics", "diagnostics must be a bool")
            )

        if failures:
            raise ContextQuerySplitError(failures)


@dataclass(frozen=True)
class ContextQuerySplit:
    """Result of a single context/query split request.

    Attributes:
        context_indices: 1D ``np.intp`` array of unique row indices for the
            context split.
        query_indices: 1D ``np.intp`` array of unique row indices for the
            query split. Disjoint from ``context_indices``.
        split_policy: JSON-serialisable description of the policy that
            produced the split. No per-sample targets / latents / indices
            are embedded.
        diagnostics: Optional non-leaky diagnostics dict (counts, ranges,
            disjointness flags). Empty when
            :attr:`ContextQuerySplitConfig.diagnostics` is ``False``.
    """

    context_indices: np.ndarray
    query_indices: np.ndarray
    split_policy: dict[str, Any]
    diagnostics: dict[str, Any]


# ---------------------------------------------------------------------------
# Size resolution
# ---------------------------------------------------------------------------


def _resolve_sizes(
    config: ContextQuerySplitConfig,
    n_total: int,
) -> tuple[int, int]:
    n_ctx = config.n_context
    n_q = config.n_query

    if n_ctx is None and config.context_fraction is not None:
        n_ctx = int(round(n_total * float(config.context_fraction)))
    if n_q is None and config.query_fraction is not None:
        n_q = int(round(n_total * float(config.query_fraction)))

    if n_ctx is None and n_q is None:
        n_ctx = max(1, n_total // 2)
        n_q = max(1, n_total - n_ctx)
    elif n_ctx is None:
        assert n_q is not None
        n_ctx = n_total - int(n_q)
    elif n_q is None:
        n_q = n_total - int(n_ctx)

    n_ctx = int(n_ctx)
    n_q = int(n_q)

    failures: list[dict[str, str]] = []
    if n_ctx < 1:
        failures.append(
            _failure(
                "infeasible_size",
                "n_context",
                f"resolved n_context={n_ctx} must be >= 1 (n_total={n_total})",
            )
        )
    if n_q < 1:
        failures.append(
            _failure(
                "infeasible_size",
                "n_query",
                f"resolved n_query={n_q} must be >= 1 (n_total={n_total})",
            )
        )
    if n_ctx + n_q > n_total:
        failures.append(
            _failure(
                "infeasible_size",
                "n_context+n_query",
                f"n_context+n_query={n_ctx + n_q} > n_total={n_total}",
            )
        )
    if failures:
        raise ContextQuerySplitError(failures)
    return n_ctx, n_q


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _random_split(
    config: ContextQuerySplitConfig,
    n_total: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_ctx, n_q = _resolve_sizes(config, n_total)
    base = np.arange(n_total, dtype=np.intp)
    if config.shuffle:
        permuted = rng.permutation(base)
    else:
        permuted = base
    ctx = np.sort(permuted[:n_ctx]).astype(np.intp, copy=False)
    q = np.sort(permuted[n_ctx : n_ctx + n_q]).astype(np.intp, copy=False)
    return ctx, q


def _select_target_array(
    latent_batch: CanonicalLatentBatch,
    target_source: str,
) -> np.ndarray:
    """Return the requested target array as ``float`` with 1D / 2D shape.

    Single-column 2D arrays are normalised back to 1D so the
    single-output convention stays unique. Multi-output arrays
    (``shape[1] > 1``) are returned as 2D ``(n_rows, n_outputs)``.
    """
    raw = (
        latent_batch.target_clean
        if target_source == "target_clean"
        else latent_batch.target_noisy
    )
    array = np.asarray(raw, dtype=float)
    if array.ndim not in (1, 2):
        raise ContextQuerySplitError(
            [
                _failure(
                    "shape_mismatch",
                    "target",
                    f"target array must be 1D or 2D, got shape={array.shape}",
                )
            ]
        )
    if array.ndim == 2 and array.shape[1] == 1:
        return array.reshape(-1)
    return array


def _target_n_outputs(target: np.ndarray) -> int:
    return 1 if target.ndim == 1 else int(target.shape[1])


def _encode_joint_labels(
    target: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Encode a 1D / 2D integer-like target as a 1D joint-label array.

    Returns ``(joint, classes)`` where ``joint`` is a 1D ``np.int64``
    array of joint-label ids and ``classes`` is the ordered list of
    distinct label tuples (1-tuple for single-output, n-tuple for
    multi-output) corresponding to those ids.
    """
    if target.ndim == 1:
        int_arr = target.astype(np.int64)
        rows: list[tuple[int, ...]] = [(int(v),) for v in int_arr.tolist()]
    else:
        int_arr = target.astype(np.int64)
        rows = [tuple(int(v) for v in row) for row in int_arr.tolist()]
    classes = sorted(set(rows))
    class_to_id: dict[tuple[int, ...], int] = {key: idx for idx, key in enumerate(classes)}
    joint = np.asarray([class_to_id[r] for r in rows], dtype=np.int64)
    return joint, classes


def _allocate_bounded_largest_remainder(
    *,
    classes: list[int],
    weights: dict[int, float],
    total: int,
    min_per_class: dict[int, int],
    max_per_class: dict[int, int],
) -> dict[int, int]:
    min_total = sum(min_per_class[c] for c in classes)
    max_total = sum(max_per_class[c] for c in classes)
    if total < min_total or total > max_total:
        raise ContextQuerySplitError(
            [
                _failure(
                    "infeasible_stratification",
                    "target",
                    (
                        f"cannot allocate total={total} within per-class bounds "
                        f"[{min_total}, {max_total}]"
                    ),
                )
            ]
        )

    weight_sum = sum(weights[c] for c in classes)
    if weight_sum <= 0.0:
        ideal = {c: float(total) / len(classes) for c in classes}
    else:
        ideal = {c: weights[c] * total / weight_sum for c in classes}

    alloc: dict[int, int] = {}
    for c in classes:
        base = int(np.floor(ideal[c]))
        alloc[c] = max(min_per_class[c], min(max_per_class[c], base))

    diff = total - sum(alloc.values())
    while diff > 0:
        progressed = False
        for c in sorted(classes, key=lambda label: (-(ideal[label] - alloc[label]), label)):
            if diff == 0:
                break
            if alloc[c] < max_per_class[c]:
                alloc[c] += 1
                diff -= 1
                progressed = True
        if not progressed:
            break

    while diff < 0:
        progressed = False
        for c in sorted(classes, key=lambda label: (ideal[label] - alloc[label], label)):
            if diff == 0:
                break
            if alloc[c] > min_per_class[c]:
                alloc[c] -= 1
                diff += 1
                progressed = True
        if not progressed:
            break

    if diff != 0:
        raise ContextQuerySplitError(
            [
                _failure(
                    "infeasible_stratification",
                    "target",
                    f"could not reconcile per-class allocation to total={total}",
                )
            ]
        )
    return alloc


def _stratified_classification_split(
    config: ContextQuerySplitConfig,
    latent_batch: CanonicalLatentBatch,
    n_total: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    target_type = str(latent_batch.target_metadata.get("type", "regression"))
    if target_type != "classification":
        raise ContextQuerySplitError(
            [
                _failure(
                    "strategy_target_mismatch",
                    "strategy",
                    f"stratified_classification requires classification target, got {target_type!r}",
                )
            ]
        )

    target = _select_target_array(latent_batch, config.target_source)
    if target.shape[0] != n_total:
        raise ContextQuerySplitError(
            [
                _failure(
                    "shape_mismatch",
                    "target",
                    f"target length {target.shape[0]} != n_total {n_total}",
                )
            ]
        )
    if not bool(np.all(np.isfinite(target))) or not bool(
        np.all(np.equal(target, np.round(target)))
    ):
        raise ContextQuerySplitError(
            [
                _failure(
                    "invalid_classification_labels",
                    "target",
                    (
                        "stratified_classification requires integer-like, finite "
                        "labels for every output column"
                    ),
                )
            ]
        )

    joint_labels, joint_classes = _encode_joint_labels(target)
    class_ids: list[int] = list(range(len(joint_classes)))
    counts: list[int] = [
        int(np.sum(joint_labels == c).item()) for c in class_ids
    ]
    n_classes = len(class_ids)

    n_ctx, n_q = _resolve_sizes(config, n_total)

    failures: list[dict[str, str]] = []
    if any(c < 2 for c in counts):
        failures.append(
            _failure(
                "infeasible_stratification",
                "target",
                (
                    "every joint class must have >=2 samples; counts="
                    + str(
                        {
                            ",".join(str(v) for v in joint_classes[c]): counts[c]
                            for c in class_ids
                        }
                    )
                ),
            )
        )
    if n_ctx < n_classes:
        failures.append(
            _failure(
                "infeasible_stratification",
                "n_context",
                f"n_context={n_ctx} < n_classes={n_classes}; cannot place every class in context",
            )
        )
    if n_q < n_classes:
        failures.append(
            _failure(
                "infeasible_stratification",
                "n_query",
                f"n_query={n_q} < n_classes={n_classes}; cannot place every class in query",
            )
        )
    if failures:
        raise ContextQuerySplitError(failures)

    class_indices: dict[int, np.ndarray] = {
        c: np.where(joint_labels == c)[0].astype(np.intp) for c in class_ids
    }
    class_sizes: dict[int, int] = {c: int(class_indices[c].size) for c in class_ids}
    context_alloc = _allocate_bounded_largest_remainder(
        classes=class_ids,
        weights={c: float(class_sizes[c]) for c in class_ids},
        total=n_ctx,
        min_per_class=dict.fromkeys(class_ids, 1),
        max_per_class={c: class_sizes[c] - 1 for c in class_ids},
    )
    query_capacity: dict[int, int] = {
        c: class_sizes[c] - context_alloc[c] for c in class_ids
    }
    query_alloc = _allocate_bounded_largest_remainder(
        classes=class_ids,
        weights={c: float(query_capacity[c]) for c in class_ids},
        total=n_q,
        min_per_class=dict.fromkeys(class_ids, 1),
        max_per_class=query_capacity,
    )

    ctx_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    for c in class_ids:
        cls_idx = class_indices[c]
        if config.shuffle:
            cls_idx = rng.permutation(cls_idx)
        else:
            cls_idx = np.sort(cls_idx)
        k_ctx = context_alloc[c]
        k_query = query_alloc[c]
        ctx_parts.append(cls_idx[:k_ctx])
        query_parts.append(cls_idx[k_ctx : k_ctx + k_query])

    ctx = (
        np.sort(np.concatenate(ctx_parts)).astype(np.intp, copy=False)
        if ctx_parts
        else np.empty(0, dtype=np.intp)
    )
    q = (
        np.sort(np.concatenate(query_parts)).astype(np.intp, copy=False)
        if query_parts
        else np.empty(0, dtype=np.intp)
    )

    return ctx, q


def _group_holdout_split(
    config: ContextQuerySplitConfig,
    latent_batch: CanonicalLatentBatch,
    n_total: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    group_ids = latent_batch.group_ids
    if len(group_ids) != n_total:
        raise ContextQuerySplitError(
            [
                _failure(
                    "shape_mismatch",
                    "group_ids",
                    f"len(group_ids)={len(group_ids)} != n_total={n_total}",
                )
            ]
        )

    unique_groups: list[Any] = list(dict.fromkeys(group_ids))
    if len(unique_groups) < 2:
        raise ContextQuerySplitError(
            [
                _failure(
                    "infeasible_grouping",
                    "group_ids",
                    f"group_holdout requires >=2 distinct groups, got {len(unique_groups)}",
                )
            ]
        )

    n_ctx, n_q = _resolve_sizes(config, n_total)

    group_to_indices: dict[Any, list[int]] = {}
    for i, g in enumerate(group_ids):
        group_to_indices.setdefault(g, []).append(i)

    g_count = len(unique_groups)
    if config.shuffle:
        order = rng.permutation(g_count).tolist()
    else:
        order = list(range(g_count))
    shuffled_groups: list[Any] = [unique_groups[i] for i in order]

    cum_sizes: list[int] = []
    running = 0
    for g in shuffled_groups:
        running += len(group_to_indices[g])
        cum_sizes.append(running)

    best_k = 1
    best_score: tuple[float, float, float, int] | None = None
    for k in range(1, g_count):
        actual_context = cum_sizes[k - 1]
        actual_query = n_total - actual_context
        context_diff = abs(actual_context - n_ctx)
        query_diff = abs(actual_query - n_q)
        score = (
            float(context_diff + query_diff),
            float(query_diff),
            float(context_diff),
            k,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_k = k

    ctx_groups = shuffled_groups[:best_k]
    query_groups = shuffled_groups[best_k:]

    ctx_idx_list = sorted(i for g in ctx_groups for i in group_to_indices[g])
    query_idx_list = sorted(i for g in query_groups for i in group_to_indices[g])

    if not ctx_idx_list or not query_idx_list:
        raise ContextQuerySplitError(
            [
                _failure(
                    "infeasible_grouping",
                    "group_ids",
                    f"group_holdout produced empty split: "
                    f"n_context={len(ctx_idx_list)}, n_query={len(query_idx_list)}",
                )
            ]
        )

    ctx = np.asarray(ctx_idx_list, dtype=np.intp)
    q = np.asarray(query_idx_list, dtype=np.intp)
    return ctx, q


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _build_diagnostics(
    *,
    strategy: str,
    ctx: np.ndarray,
    q: np.ndarray,
    n_total: int,
    requested_n_context: int,
    requested_n_query: int,
    latent_batch: CanonicalLatentBatch,
    target_source: str,
    target_type: str,
) -> dict[str, Any]:
    target = _select_target_array(latent_batch, target_source)
    n_outputs = _target_n_outputs(target)

    ctx_set = set(ctx.tolist())
    q_set = set(q.tolist())
    ctx_latent = {latent_batch.latent_ids[i] for i in ctx.tolist()}
    q_latent = {latent_batch.latent_ids[i] for i in q.tolist()}

    if target.ndim == 1:
        ctx_n_unique = int(np.unique(target[ctx]).size) if ctx.size else 0
        q_n_unique = int(np.unique(target[q]).size) if q.size else 0
    else:
        ctx_n_unique = int(np.unique(target[ctx], axis=0).shape[0]) if ctx.size else 0
        q_n_unique = int(np.unique(target[q], axis=0).shape[0]) if q.size else 0

    diag: dict[str, Any] = {
        "strategy": strategy,
        "n_total": int(n_total),
        "n_outputs": int(n_outputs),
        "n_context": int(ctx.size),
        "n_query": int(q.size),
        "requested_n_context": int(requested_n_context),
        "requested_n_query": int(requested_n_query),
        "exact_size_match": bool(
            int(ctx.size) == requested_n_context and int(q.size) == requested_n_query
        ),
        "indices_disjoint": bool(ctx_set.isdisjoint(q_set)),
        "latent_ids_disjoint": bool(ctx_latent.isdisjoint(q_latent)),
        "constant_target_context": bool(ctx.size > 0 and ctx_n_unique == 1),
        "constant_target_query": bool(q.size > 0 and q_n_unique == 1),
    }

    if target_type == "classification" and bool(
        np.all(np.equal(target, np.round(target)))
    ):
        joint_labels, joint_classes = _encode_joint_labels(target)
        if n_outputs == 1:
            ctx_labels, ctx_counts = np.unique(joint_labels[ctx], return_counts=True)
            q_labels, q_counts = np.unique(joint_labels[q], return_counts=True)
            diag["class_counts_context"] = {
                str(int(joint_classes[int(joint_id)][0])): int(count)
                for joint_id, count in zip(
                    ctx_labels.tolist(), ctx_counts.tolist(), strict=True
                )
            }
            diag["class_counts_query"] = {
                str(int(joint_classes[int(joint_id)][0])): int(count)
                for joint_id, count in zip(
                    q_labels.tolist(), q_counts.tolist(), strict=True
                )
            }
            diag["n_classes_context"] = int(ctx_labels.size)
            diag["n_classes_query"] = int(q_labels.size)
            diag["shared_label_count"] = int(
                len(set(ctx_labels.tolist()) & set(q_labels.tolist()))
            )
        else:
            ctx_ids, ctx_counts = np.unique(joint_labels[ctx], return_counts=True)
            q_ids, q_counts = np.unique(joint_labels[q], return_counts=True)

            def _key(joint_id: int) -> str:
                return ",".join(str(int(v)) for v in joint_classes[int(joint_id)])

            diag["joint_label_count_context"] = {
                _key(int(jid)): int(count)
                for jid, count in zip(ctx_ids.tolist(), ctx_counts.tolist(), strict=True)
            }
            diag["joint_label_count_query"] = {
                _key(int(jid)): int(count)
                for jid, count in zip(q_ids.tolist(), q_counts.tolist(), strict=True)
            }
            diag["n_classes_context"] = int(ctx_ids.size)
            diag["n_classes_query"] = int(q_ids.size)
            diag["n_joint_labels"] = int(len(joint_classes))
            diag["shared_joint_label_count"] = int(
                len(set(ctx_ids.tolist()) & set(q_ids.tolist()))
            )
    else:
        if target.ndim == 1:
            if ctx.size:
                diag["target_range_context"] = [
                    float(np.min(target[ctx])),
                    float(np.max(target[ctx])),
                ]
            if q.size:
                diag["target_range_query"] = [
                    float(np.min(target[q])),
                    float(np.max(target[q])),
                ]
        else:
            if ctx.size:
                diag["target_range_context"] = [
                    [float(np.min(target[ctx, j])), float(np.max(target[ctx, j]))]
                    for j in range(n_outputs)
                ]
            if q.size:
                diag["target_range_query"] = [
                    [float(np.min(target[q, j])), float(np.max(target[q, j]))]
                    for j in range(n_outputs)
                ]

    if strategy == "group_holdout":
        group_ids = latent_batch.group_ids
        ctx_groups = [group_ids[i] for i in ctx.tolist()]
        q_groups = [group_ids[i] for i in q.tolist()]
        ctx_unique = list(dict.fromkeys(ctx_groups))
        q_unique = list(dict.fromkeys(q_groups))
        diag["group_counts_context"] = {
            str(g): int(ctx_groups.count(g)) for g in ctx_unique
        }
        diag["group_counts_query"] = {
            str(g): int(q_groups.count(g)) for g in q_unique
        }
        diag["n_groups_context"] = int(len(ctx_unique))
        diag["n_groups_query"] = int(len(q_unique))
        diag["groups_disjoint"] = bool(set(ctx_unique).isdisjoint(set(q_unique)))

    return diag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_context_query_split(
    latent_batch: CanonicalLatentBatch,
    config: ContextQuerySplitConfig,
) -> ContextQuerySplit:
    """Sample a deterministic ``(context, query)`` index split.

    Args:
        latent_batch: Source canonical latent batch.
        config: Declarative split configuration. ``config.seed`` is the
            sole source of randomness.

    Returns:
        A frozen :class:`ContextQuerySplit` with sorted, disjoint
        ``np.intp`` index arrays plus a JSON-serialisable ``split_policy``
        and an optional diagnostics dictionary.

    Raises:
        ContextQuerySplitError: If inputs violate the contract (unknown
            strategy, infeasible sizes, non-classification target for
            stratified, single group for group_holdout, etc.).
    """
    if not isinstance(config, ContextQuerySplitConfig):
        raise ContextQuerySplitError(
            [
                _failure(
                    "invalid_input",
                    "config",
                    "config must be a ContextQuerySplitConfig",
                )
            ]
        )
    if not isinstance(latent_batch, CanonicalLatentBatch):
        raise ContextQuerySplitError(
            [
                _failure(
                    "invalid_input",
                    "latent_batch",
                    "latent_batch must be a CanonicalLatentBatch",
                )
            ]
        )

    n_total = len(latent_batch.latent_ids)
    if n_total < 2:
        raise ContextQuerySplitError(
            [
                _failure(
                    "infeasible_size",
                    "latent_batch",
                    f"latent_batch must have >=2 samples, got {n_total}",
                )
            ]
        )

    rng = np.random.default_rng(int(config.seed))
    target_type = str(latent_batch.target_metadata.get("type", "regression"))

    if config.strategy == "random":
        ctx, q = _random_split(config, n_total, rng)
    elif config.strategy == "stratified_classification":
        ctx, q = _stratified_classification_split(config, latent_batch, n_total, rng)
    elif config.strategy == "group_holdout":
        ctx, q = _group_holdout_split(config, latent_batch, n_total, rng)
    else:  # pragma: no cover - guarded by ContextQuerySplitConfig
        raise ContextQuerySplitError(
            [
                _failure(
                    "invalid_strategy",
                    "strategy",
                    f"unknown strategy {config.strategy!r}",
                )
            ]
        )

    failures: list[dict[str, str]] = []
    if ctx.size == 0:
        failures.append(
            _failure("empty_split", "context_indices", "context split must not be empty")
        )
    if q.size == 0:
        failures.append(
            _failure("empty_split", "query_indices", "query split must not be empty")
        )
    if ctx.size and (int(ctx.min()) < 0 or int(ctx.max()) >= n_total):
        failures.append(
            _failure(
                "invalid_indices",
                "context_indices",
                f"context indices out of range [0, {n_total})",
            )
        )
    if q.size and (int(q.min()) < 0 or int(q.max()) >= n_total):
        failures.append(
            _failure(
                "invalid_indices",
                "query_indices",
                f"query indices out of range [0, {n_total})",
            )
        )
    if len(set(ctx.tolist())) != int(ctx.size):
        failures.append(
            _failure(
                "duplicate_indices",
                "context_indices",
                "context_indices must be unique",
            )
        )
    if len(set(q.tolist())) != int(q.size):
        failures.append(
            _failure(
                "duplicate_indices",
                "query_indices",
                "query_indices must be unique",
            )
        )
    if not set(ctx.tolist()).isdisjoint(set(q.tolist())):
        failures.append(
            _failure(
                "overlapping_indices",
                "context_indices",
                "context and query indices must be disjoint",
            )
        )
    if int(ctx.size) + int(q.size) > n_total:
        failures.append(
            _failure(
                "invalid_size",
                "total",
                f"n_context+n_query={int(ctx.size) + int(q.size)} > n_total={n_total}",
            )
        )
    if failures:
        raise ContextQuerySplitError(failures)

    diagnostics = _build_diagnostics(
        strategy=config.strategy,
        ctx=ctx,
        q=q,
        n_total=n_total,
        requested_n_context=_resolve_sizes(config, n_total)[0],
        requested_n_query=_resolve_sizes(config, n_total)[1],
        latent_batch=latent_batch,
        target_source=config.target_source,
        target_type=target_type,
    )

    diagnostics_summary: dict[str, Any] = {
        key: diagnostics[key]
        for key in diagnostics
        if key in _NON_LEAKY_DIAGNOSTIC_KEYS
    }

    split_policy: dict[str, Any] = {
        "phase": "D2",
        "kind": "context_query_sampler",
        "strategy": config.strategy,
        "seed": int(config.seed),
        "shuffle": bool(config.shuffle),
        "target_source": config.target_source,
        "n_context": int(ctx.size),
        "n_query": int(q.size),
        "requested_n_context": int(diagnostics["requested_n_context"]),
        "requested_n_query": int(diagnostics["requested_n_query"]),
        "exact_size_match": bool(diagnostics["exact_size_match"]),
        "n_total": int(n_total),
        "indices_disjoint": True,
        "latent_ids_disjoint": True,
        "diagnostics_summary": diagnostics_summary,
        "note": (
            "D2 declarative context/query sampler. No realism / transfer claim. "
            "A3 / B2 risk gates remain negative. D3 multi-output regression "
            "and classification are supported; classification stratifies on "
            "joint labels."
        ),
    }

    return ContextQuerySplit(
        context_indices=ctx,
        query_indices=q,
        split_policy=split_policy,
        diagnostics=dict(diagnostics) if config.diagnostics else {},
    )


def sample_nirs_prior_task(
    latent_batch: CanonicalLatentBatch,
    spectral_view: SpectralViewBatch,
    config: ContextQuerySplitConfig,
    *,
    target_name: str | None = None,
) -> NIRSPriorTask:
    """Sample one :class:`NIRSPriorTask` end-to-end via the D2 sampler.

    The function is a thin orchestration: it draws a deterministic
    ``(context, query)`` split through :func:`sample_context_query_split`
    and then delegates to :meth:`NIRSPriorTask.from_batches` with the
    resulting indices, ``split_policy``, and ``task_seed=config.seed``.
    """
    if not isinstance(spectral_view, SpectralViewBatch):
        raise ContextQuerySplitError(
            [
                _failure(
                    "invalid_input",
                    "spectral_view",
                    "spectral_view must be a SpectralViewBatch",
                )
            ]
        )

    split = sample_context_query_split(latent_batch, config)
    return NIRSPriorTask.from_batches(
        latent_batch,
        spectral_view,
        split.context_indices,
        split.query_indices,
        target_source=config.target_source,
        target_name=target_name,
        split_policy=split.split_policy,
        task_seed=int(config.seed),
    )
