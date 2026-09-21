"""Adapt IO-owned multimodal cohorts to the Python DAG scientific host.

The adapter keeps raw source buffers in IO. It reuses the existing target,
metadata and observation index interfaces, without inventing dense features.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np

from nirs4all.data.dataset import SpectroDataset


class MultimodalSpectroDataset(SpectroDataset):
    """DAG host view of a :class:`nirs4all_io.MultimodalDataset`."""

    def __init__(self, cohort: Any) -> None:
        super().__init__(cohort.name)
        self.cohort = cohort
        self._data_provider_evidence = copy.deepcopy(getattr(cohort, "_data_provider_evidence", None))
        self.sample_ids = tuple(cohort.sample_ids)
        self.source_names: tuple[str, ...] = tuple(cohort.sources)
        for index, partition in enumerate(cohort.partitions):
            self._indexer.add_samples(1, partition="test" if partition == "predict" else partition, sample_indices=[index])
        if cohort.task_type is not None:
            self.set_task_type(cohort.task_type)
        if cohort.y is not None:
            targets = cohort.y
            if not np.all(cohort.target_mask):
                if cohort.task_type != "regression":
                    raise ValueError("partial multimodal targets require an explicit task_type='regression'")
                # The mask remains authoritative; zero is only a finite wire
                # placeholder and must never enter fitting or native scoring.
                targets = np.asarray(np.where(cohort.target_mask, targets, 0), dtype=float)
            train_rows = [index for index, partition in enumerate(cohort.partitions) if partition == "train"]
            # Fit target conversion only on training rows, while preserving the
            # cohort's storage order. A prediction-only cohort uses no learned
            # target state from here: replay owns its captured label decoder.
            self._targets.add_targets(targets, fit_indices=train_rows or None)
        columns: dict[str, Any] = {"sample_id": list(self.sample_ids)}
        if cohort.groups is not None:
            columns["group_id"] = list(cohort.groups)
        import pandas as pd

        self.add_metadata(pd.DataFrame(columns))

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def num_features(self) -> list[int]:
        """Raw scalar counts, not learned embedding dimensions."""
        return [int(np.prod(source.values.shape[1:])) for source in self.cohort.sources.values()]

    @property
    def n_sources(self) -> int:
        return len(self.source_names)

    def features_sources(self) -> int:
        return self.n_sources

    def is_multi_source(self) -> bool:
        return self.n_sources > 1

    def source_name(self, index: int) -> str:
        return self.source_names[index]

    def headers(self, src: int = 0) -> list[str]:
        block = self.cohort.sources[self.source_names[src]]
        return list(block.feature_names or [])

    def x_rows(self, sample_ints: list[int], layout: Any = "2d", concat_source: bool = True) -> Any:
        if concat_source:
            raise ValueError("raw multimodal sources require a source-aware model; fuse them inside the pipeline")
        return self.cohort.source_values(sample_ints)

    def x(self, selector: Any, layout: Any = "2d", concat_source: bool = True, include_augmented: bool = True, include_excluded: bool = False) -> Any:
        return self.x_rows(self.index_column("sample", selector or {}), layout, concat_source)

    def content_hash(self, source_index: int | None = None) -> str:
        """Fingerprint actual typed host buffers, including categorical values."""
        digest = hashlib.sha256()
        items = list(self.cohort.sources.items())
        if source_index is not None:
            items = [items[source_index]]
        for name, source in items:
            values = np.asarray(source.values)
            presence = np.asarray(source.presence_mask)
            if not presence.all():
                values = values.copy()
                values[~presence] = "" if values.dtype.kind in "US" else 0
            digest.update(json.dumps([source.schema_descriptor(name), values.shape], sort_keys=True).encode())
            digest.update(np.ascontiguousarray(presence).tobytes())
            if values.dtype.kind in "OUS":
                if values.dtype.kind == "O":
                    values = np.asarray([cell.item() if isinstance(cell, np.generic) else cell for cell in values.flat], dtype=object).reshape(values.shape)
                digest.update(json.dumps(values.tolist(), ensure_ascii=False, allow_nan=False).encode())
            else:
                digest.update(np.ascontiguousarray(values).tobytes())
        return digest.hexdigest()

    def split_features(self, splitter: Any, pool: list[int]) -> np.ndarray:
        """Expose row indices only to explicitly index/label/group-based splitters."""
        from sklearn import model_selection

        supported = tuple(getattr(model_selection, name) for name in (
            "KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold",
            "ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit",
            "RepeatedKFold", "RepeatedStratifiedKFold", "LeaveOneOut", "LeaveOneGroupOut",
        ))
        if type(splitter) not in supported:
            raise ValueError("raw multimodal input requires an index-, label- or group-based sklearn splitter")
        return np.asarray(pool).reshape(-1, 1)

    def groups_for_rows(self, rows: list[int]) -> np.ndarray | None:
        if self.cohort.groups is None:
            return None
        return np.asarray(self.cohort.groups, dtype=object)[rows]

    def data_schema(self, source_ids: list[str], sample_ids: list[str]) -> dict[str, Any]:
        """Translate the IO descriptors into DAG-ML-Data's typed source schema."""
        from nirs4all.pipeline.dagml.envelope import _target_representation

        sources = []
        for source_id, descriptor in zip(source_ids, self.cohort.descriptors(), strict=True):
            representation = dict(descriptor["native_representation"])
            representation["axes"] = [dict(axis) for axis in representation["axes"]]
            representation["axes"][0]["size"] = len(sample_ids)
            sources.append({
                "id": source_id, "name": descriptor["source_id"], "type_id": representation["type_id"],
                "modality": descriptor["modality"], "native_representation": representation,
                "sample_key": "sample_id", "granularity": "per_sample", "schema": {}, "tags": {},
            })
        return {"dataset_id": f"nirs4all.{self.name}", "sample_ids": sample_ids, "sources": sources,
                "targets": {"y": _target_representation(self, len(sample_ids))}, "metadata": {}}

    def data_plan(self, source_ids: list[str]) -> dict[str, Any]:
        steps = [{
            "kind": "materialize", "source_id": source_id, "adapter_id": None,
            "input_representation": None, "output_representation": source.representation_id,
            "fit_scope": "stateless", "requires_user_choice": False, "metadata": {"output": f"src:{source_id}"},
        } for source_id, source in zip(source_ids, self.cohort.sources.values(), strict=True)]
        steps.append({
            "kind": "join", "source_id": None, "adapter_id": None, "input_representation": None,
            "output_representation": "feature_block_set", "fit_scope": "stateless", "requires_user_choice": False,
            "metadata": {"inputs": [f"src:{source_id}" for source_id in source_ids], "output": "port:X"},
        })
        return {"id": f"plan.{self.name}", "steps": steps, "output_representation": "feature_block_set", "issues": []}

    def source_layout(self, source_ids: list[str]) -> dict[str, Any]:
        return {"kind": "typed_source_blocks", "source_order": list(self.source_names), "source_ids": source_ids,
                "blocks": self.cohort.descriptors(), "alignment": "sample_id"}
