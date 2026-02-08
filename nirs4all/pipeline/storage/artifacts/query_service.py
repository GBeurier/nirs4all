"""Shared artifact query/filter helpers used by replay providers.

This module centralizes common artifact filtering semantics so that
``ArtifactLoader`` and provider implementations apply the same rules for:
- branch-aware lookup (including shared pre-branch artifacts),
- source-aware lookup (including source-agnostic artifacts),
- substep/fold/pipeline filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from nirs4all.pipeline.storage.artifacts.types import ArtifactRecord, ArtifactType


@dataclass(frozen=True)
class ArtifactQuerySpec:
    """Normalized query spec for artifact lookup/filtering."""

    step_index: Optional[int] = None
    branch_path: Optional[Sequence[int]] = None
    branch_id: Optional[int] = None
    source_index: Optional[int] = None
    substep_index: Optional[int] = None
    fold_id: Optional[int] = None
    pipeline_id: Optional[str] = None

    def target_branch(self) -> Optional[int]:
        """Return branch discriminator used by current replay contracts."""
        if self.branch_path:
            return int(self.branch_path[0])
        if self.branch_id is not None:
            return int(self.branch_id)
        return None

    def branch_lookup_keys(self) -> List[Tuple[int, ...]]:
        """Return branch keys used by indexed loader lookups.

        Always includes the shared key ``()`` so pre-branch artifacts remain
        visible when a branch-specific query is requested.
        """
        branch_keys: List[Tuple[int, ...]] = [tuple()]
        requested = tuple(int(b) for b in (self.branch_path or []))
        if requested:
            branch_keys.append(requested)
        return branch_keys

    def matches_record(
        self,
        record: ArtifactRecord,
        *,
        shared_branch_ok: bool = True,
        shared_source_ok: bool = True,
        shared_fold_ok: bool = True,
    ) -> bool:
        """Check whether a record matches this query spec."""
        if self.pipeline_id is not None and record.pipeline_id != self.pipeline_id:
            return False
        if self.step_index is not None and int(record.step_index) != int(self.step_index):
            return False

        target_branch = self.target_branch()
        if target_branch is not None:
            artifact_branch = int(record.branch_path[0]) if record.branch_path else None
            if artifact_branch is None and not shared_branch_ok:
                return False
            if artifact_branch is not None and artifact_branch != target_branch:
                return False

        if self.source_index is not None:
            if record.source_index is None and not shared_source_ok:
                return False
            if record.source_index is not None and int(record.source_index) != int(self.source_index):
                return False

        if self.substep_index is not None and record.substep_index is not None:
            if int(record.substep_index) != int(self.substep_index):
                return False

        if self.fold_id is not None:
            if record.fold_id is None and not shared_fold_ok:
                return False
            if record.fold_id is not None and int(record.fold_id) != int(self.fold_id):
                return False

        return True


class ArtifactQueryService:
    """Shared filtering and ordering helpers for artifact lookup paths."""

    MODEL_TYPES = (ArtifactType.MODEL, ArtifactType.META_MODEL)

    @staticmethod
    def gather_indexed_candidate_ids(
        *,
        step_index: int,
        spec: ArtifactQuerySpec,
        by_step_branch: Dict[Tuple[int, Tuple[int, ...]], List[str]],
        by_step_branch_source: Dict[Tuple[int, Tuple[int, ...], Optional[int]], List[str]],
    ) -> List[str]:
        """Collect candidate artifact IDs from loader indexes."""
        candidate_ids: List[str] = []

        for branch_key in spec.branch_lookup_keys():
            if spec.source_index is None:
                candidate_ids.extend(by_step_branch.get((step_index, branch_key), []))
                continue

            # source_index=None records are shared across source-specific lookups.
            candidate_ids.extend(by_step_branch_source.get((step_index, branch_key, None), []))
            candidate_ids.extend(
                by_step_branch_source.get((step_index, branch_key, spec.source_index), [])
            )

        return candidate_ids

    @staticmethod
    def sort_candidate_ids(
        candidate_ids: List[str],
        artifact_order: Dict[str, int],
    ) -> List[str]:
        """Return candidate IDs deduplicated and sorted by manifest order."""
        return sorted(
            set(candidate_ids),
            key=lambda artifact_id: artifact_order.get(artifact_id, -1),
        )

    @staticmethod
    def matches_model_target(
        record: ArtifactRecord,
        *,
        target_sub_index: Optional[int] = None,
        target_model_name: Optional[str] = None,
    ) -> bool:
        """Apply model-target disambiguation for subpipeline prediction."""
        if target_sub_index is None and target_model_name is None:
            return True

        if record.artifact_type not in ArtifactQueryService.MODEL_TYPES:
            return True

        if target_sub_index is not None and record.substep_index is not None:
            return int(record.substep_index) == int(target_sub_index)

        if target_model_name is not None and record.custom_name:
            return record.custom_name == target_model_name

        return True

    @staticmethod
    def sort_by_substep(
        rows: List[Tuple[str, object, Optional[int]]]
    ) -> List[Tuple[str, object, Optional[int]]]:
        """Sort loaded artifact rows deterministically by substep index."""
        return sorted(
            rows,
            key=lambda row: row[2] if row[2] is not None else float("inf"),
        )
