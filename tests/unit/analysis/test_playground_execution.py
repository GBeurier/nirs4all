"""True inline preview results, with scientific owners and preserved lineage."""

import numpy as np
import pytest
from sklearn.model_selection import KFold, ShuffleSplit

from nirs4all.analysis.playground_execution import execute_preview
from nirs4all.analysis.playground_prepare import PreviewStep
from nirs4all.analysis.playground_types import PreviewBatch
from nirs4all.data.selection.sampling import stratified_sample
from nirs4all.operators.augmentation.spectral import GaussianAdditiveNoise
from nirs4all.operators.transforms import Resampler, SavitzkyGolay, StandardNormalVariate


@pytest.fixture
def batch():
    x = np.random.default_rng(9).normal(size=(30, 21))
    return PreviewBatch.from_arrays(x, y=np.arange(30) / 2, wavelengths=np.linspace(1100.001, 1300.003, 21),
                                    sample_ids=[f"sample{i // 2}_rep{i % 2 + 1}" for i in range(30)],
                                    metadata={"bio_sample": [f"sample{i // 2}" for i in range(30)]}, header_unit="nm")


def test_inline_snv_sg_pca_statistics_and_repetitions_are_real(batch):
    result = execute_preview(batch, [PreviewStep("snv", "preprocessing", "StandardNormalVariate", StandardNormalVariate()),
                                     PreviewStep("sg", "preprocessing", "SavitzkyGolay", SavitzkyGolay(window_length=5))])
    expected = SavitzkyGolay(window_length=5).fit_transform(StandardNormalVariate().fit_transform(batch.x))
    assert result["success"] is True
    np.testing.assert_allclose(result["processed"]["spectra"], expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(result["processed"]["statistics"]["mean"], expected.mean(axis=0))
    assert np.asarray(result["pca"]["coordinates"]).shape == (30, 10)
    assert result["repetitions"]["n_with_reps"] == 15
    assert result["repetitions"]["effective_space"] == "pca"
    assert result["evaluation_scope"] == "exploratory_preview"
    assert result["cache"]["used"] is False
    np.testing.assert_array_equal(result["processed"]["y"], batch.y)


def test_visible_subset_origins_remain_full_dataset_indices(batch):
    expected = stratified_sample(batch.x, batch.y, 12, 42)
    result = execute_preview(batch, options={"subset_mode": "visible", "max_samples_displayed": 12})
    np.testing.assert_array_equal(result["original"]["sample_indices"], expected)
    np.testing.assert_array_equal(result["original"]["spectra"], batch.x[expected])
    np.testing.assert_array_equal(result["original"]["y"], batch.y[expected])
    assert result["subset_info"] == {"subset_mode": "visible", "total_samples": 30, "displayed_samples": 12}


def test_first_holdout_is_respected_by_next_cv_split(batch):
    holdout = ShuffleSplit(n_splits=1, test_size=0.2, random_state=2)
    train, test = next(holdout.split(batch.x))
    result = execute_preview(batch, [PreviewStep("holdout", "splitting", "ShuffleSplit", holdout),
                                     PreviewStep("cv", "splitting", "KFold", KFold(n_splits=3))])
    assert result["success"] is True
    assert result["folds"]["kind"] == "cv_folds"
    assert result["processed_partitions"]["test_indices"] == sorted(test.tolist())
    for fold in result["folds"]["folds"]:
        assert set(fold["train_indices"]) <= set(train)
        assert set(fold["test_indices"]) <= set(train)
        assert not set(fold["test_indices"]) & set(test)


def test_split_filter_augmentation_remaps_folds_targets_and_metadata(batch):
    result = execute_preview(batch, [
        PreviewStep("split", "splitting", "KFold", KFold(n_splits=3)),
        PreviewStep("select", "filter", "SampleIndexFilter", params={"indices": list(range(5, 25))}),
        PreviewStep("augment", "augmentation", "GaussianAdditiveNoise", GaussianAdditiveNoise(random_state=3), params={"n_augmented_copies": 1}),
    ])
    assert result["success"] is True
    origins = np.tile(np.arange(5, 25), 2)
    assert result["processed"]["shape"] == [40, 21]
    np.testing.assert_array_equal(result["processed"]["sample_indices"], origins)
    np.testing.assert_array_equal(result["processed"]["metadata"]["bio_sample"], batch.metadata["bio_sample"][origins])
    assert len(result["pca"]["y"]) == len(result["pca"]["fold_labels"]) == 40
    for fold in result["folds"]["folds"]:
        for part in ("train", "test"):
            indices = fold[f"{part}_indices"]
            assert fold[f"{part}_origins"] == origins[indices].tolist()
            assert fold[f"{part}_count"] == len(indices)
            if indices:
                assert fold[f"y_{part}_stats"]["mean"] == pytest.approx(np.mean(batch.y[origins[indices]]))
        assert not set(fold["train_indices"]) & set(fold["test_indices"])


def test_tag_is_non_destructive_and_unknown_step_is_visible(batch):
    tagged = execute_preview(batch, [PreviewStep("tag", "filter", "SampleIndexFilter", params={"indices": [1, 4], "mode": "remove", "filter_mode": "tag"})])
    np.testing.assert_array_equal(tagged["processed"]["spectra"], batch.x)
    assert tagged["filter_info"]["tagged_samples"]["SampleIndexFilter"] == [1, 4]
    failed = execute_preview(batch, [PreviewStep("bad", "typo", "Unknown"),
                                     PreviewStep("snv", "preprocessing", "SNV", StandardNormalVariate())])
    assert failed["success"] is False
    assert failed["step_errors"][0]["step"] == "bad"
    assert failed["execution_trace"][1]["success"] is True
    np.testing.assert_allclose(failed["processed"]["spectra"], StandardNormalVariate().fit_transform(batch.x))


def test_resampling_has_separate_original_processed_exact_axes(batch):
    target = np.linspace(1100.00111, 1300.000123, 13)
    result = execute_preview(batch, [PreviewStep("resample", "preprocessing", "Resampler", Resampler(target_wavelengths=target))])
    assert result["success"] is True
    assert result["original"]["shape"] == [30, 21]
    assert result["processed"]["shape"] == [30, 13]
    np.testing.assert_array_equal(result["original"]["wavelengths"], batch.wavelengths)
    np.testing.assert_array_equal(result["processed"]["wavelengths"], target)


def test_no_hidden_cache_when_y_or_metadata_change(batch):
    original = execute_preview(batch)
    changed = PreviewBatch.from_arrays(batch.x, y=batch.y + 100, metadata={"bio_sample": ["same"] * 30})
    result = execute_preview(changed)
    assert result["pca"]["y"][0] == original["pca"]["y"][0] + 100
    assert result["repetitions"]["n_with_reps"] == 1


def test_no_targets_and_empty_filtered_cohort_do_not_invent_observations(batch):
    unlabeled = execute_preview(PreviewBatch.from_arrays(batch.x))
    assert unlabeled["processed"]["y"] is None
    assert "y" not in unlabeled["pca"]
    empty = execute_preview(batch, [PreviewStep("empty", "filter", "SampleIndexFilter", params={"indices": []})])
    assert empty["success"] is True
    assert empty["processed"]["shape"] == [0, 21]
    assert "error" in empty["pca"]
    assert empty["repetitions"]["has_repetitions"] is False


def test_budget_precedes_projection_array_construction(batch, monkeypatch):
    import nirs4all.analysis.playground_execution as execution

    monkeypatch.setattr(execution, "pca_projection", lambda *args: pytest.fail("projection must not run"))
    with pytest.raises(ValueError, match="before projection"):
        execute_preview(batch, options={"max_response_cells": 1})
