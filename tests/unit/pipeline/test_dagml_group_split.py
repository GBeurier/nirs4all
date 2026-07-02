"""Unit coverage for dag-ml host-side grouped split plumbing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold

from nirs4all.data.config import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.dagml.detect import (
    _detect_by_source_branch,
    _detect_duplication_branch,
    _detect_separation_branch,
    _detect_source_concat_merge,
)
from nirs4all.pipeline.dagml.envelope import sample_relations
from nirs4all.pipeline.dagml.folds import _build_folds, _split_group_grain
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.steps import DagMlSplitStep, _split_pipeline

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SAMPLE_DATA = _PROJECT_ROOT / "examples" / "sample_data"
_SAMPLE_DATASETS = _PROJECT_ROOT / "examples" / "sample_datasets"


def _dataset_path(key: str) -> str:
    paths = {
        "regression": _SAMPLE_DATA / "regression",
        "with_metadata": _SAMPLE_DATASETS / "C04_with_metadata",
    }
    return str(paths[key])


def _train_test_bridge_dataset() -> SpectroDataset:
    """Small repetition dataset where test rows can connect train groups if not scoped."""
    dataset = SpectroDataset(name="train-test-bridge")
    dataset.add_samples(np.arange(16, dtype=float).reshape(4, 4), {"partition": "train"})
    dataset.add_samples(np.arange(8, dtype=float).reshape(2, 4) + 100.0, {"partition": "test"})
    dataset.add_targets(np.linspace(0.0, 1.0, 6))
    dataset.add_metadata(
        pd.DataFrame(
            {
                "sample_id": ["S1", "S1", "S2", "S2", "T1", "T1"],
                "batch": ["B1", "B1", "B2", "B2", "B1", "B2"],
            }
        )
    )
    dataset.set_repetition("sample_id")
    return dataset


def test_split_step_group_by_feeds_group_splitter_and_relations() -> None:
    """``{"split": ..., "group_by": ...}`` carries groups into folds and relations."""
    dataset = DatasetConfigs(_dataset_path("with_metadata")).get_dataset_at(0)
    pool = [int(sample_int) for sample_int in dataset.index_column("sample", {"partition": "train"})]
    steps, splitter = _split_pipeline(
        [
            {"split": GroupKFold(n_splits=3), "group_by": "group"},
            {"model": PLSRegression(n_components=2)},
        ]
    )

    assert isinstance(splitter, DagMlSplitStep)
    assert len(steps) == 1 and isinstance(steps[0], dict) and "model" in steps[0]

    group_by_sample = _split_group_grain(splitter, dataset, pool)
    assert group_by_sample is not None
    assert set(group_by_sample) == set(pool)
    assert len(set(group_by_sample.values())) < len(pool)

    folds = _build_folds(splitter, dataset, pool, set())
    validated = [sample_int for _train, validation in folds for sample_int in validation]
    assert sorted(validated) == sorted(pool)
    for train_ids, validation_ids in folds:
        train_groups = {group_by_sample[sample_int] for sample_int in train_ids}
        validation_groups = {group_by_sample[sample_int] for sample_int in validation_ids}
        assert not (train_groups & validation_groups)

    relations = sample_relations(mint_identity(dataset), sample_ints=pool, group_by_sample=group_by_sample)
    assert [row.get("group_id") for row in relations["rows"]] == [group_by_sample[sample_int] for sample_int in pool]


def test_group_required_splitter_without_group_source_still_fails_loud() -> None:
    """A group-required splitter still refuses native folds when no group source exists."""
    from nirs4all.operators.splitters import BinnedStratifiedGroupKFold

    dataset = DatasetConfigs(_dataset_path("regression")).get_dataset_at(0)
    pool = [int(sample_int) for sample_int in dataset.index_column("sample", {"partition": "train"})]
    with pytest.raises(NotImplementedError, match="#21"):
        _build_folds(BinnedStratifiedGroupKFold(n_splits=3, n_bins=3), dataset, pool, set())


def test_split_step_groups_use_train_context_not_test_connected_components() -> None:
    """Test rows must not connect train groups when resolving repetition + group_by."""
    dataset = _train_test_bridge_dataset()
    pool = [int(sample_int) for sample_int in dataset.index_column("sample", {"partition": "train"})]
    _steps, splitter = _split_pipeline([{"split": GroupKFold(n_splits=2), "group_by": "batch"}])

    group_by_sample = _split_group_grain(splitter, dataset, pool)
    assert group_by_sample == {0: "0", 1: "0", 2: "1", 3: "1"}

    folds = _build_folds(splitter, dataset, pool, set())
    assert len(folds) == 2
    for train_ids, validation_ids in folds:
        assert {group_by_sample[sample_int] for sample_int in train_ids}.isdisjoint(
            {group_by_sample[sample_int] for sample_int in validation_ids}
        )


def test_split_step_group_by_survives_existing_augmented_rows() -> None:
    """Group resolution is aligned to base train rows even after augmentation inserted children."""
    dataset = _train_test_bridge_dataset()
    dataset.add_samples(
        np.arange(8, dtype=float).reshape(2, 4) + 200.0,
        {"partition": "train", "origin": [0, 2], "augmentation": ["unit", "unit"]},
    )
    pool = [int(sample_int) for sample_int in dataset.index_column("sample", {"partition": "train"}) if int(sample_int) < 4]
    _steps, splitter = _split_pipeline([{"split": GroupKFold(n_splits=2), "group_by": "batch"}])

    assert _split_group_grain(splitter, dataset, pool) == {0: "0", 1: "0", 2: "1", 3: "1"}


def test_dict_split_is_seen_by_specialized_dagml_detectors() -> None:
    """Specialized routes accept public ``{"split": ..., "group_by": ...}`` syntax."""
    split_step = {"split": KFold(n_splits=3), "group_by": "batch"}

    separation = [
        split_step,
        {"branch": {"by_metadata": "group", "steps": [{"model": PLSRegression(n_components=2)}]}},
        {"merge": "concat"},
    ]
    assert _detect_separation_branch(separation) is not None

    duplication = [
        split_step,
        {"branch": [[{"model": PLSRegression(n_components=2)}], [{"model": Ridge(alpha=1.0)}]]},
        {"merge": "mean"},
    ]
    assert _detect_duplication_branch(duplication) is not None

    by_source = [
        split_step,
        {"branch": {"by_source": True, "steps": [{"model": PLSRegression(n_components=2)}]}},
        {"merge": "mean"},
    ]
    assert _detect_by_source_branch(by_source, n_sources=2) is not None

    source_concat = [
        {"merge": {"sources": "concat"}},
        split_step,
        {"model": PLSRegression(n_components=2)},
    ]
    assert _detect_source_concat_merge(source_concat, n_sources=2) is not None
