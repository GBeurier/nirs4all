"""Actual file readers, partition/source evidence and historical previews."""

import json

import numpy as np
import pandas as pd
import pytest

from nirs4all.api.dataset_inspection import dataset_statistics, inspect_format_file, preview_dataset


def _csv_dataset(tmp_path, *, width=300, sources=1, rows=150, test_rows=12, metadata=True):
    config = {"name": "inspection", "task_type": "regression", "global_params": {"delimiter": ",", "has_header": True}}
    data = {}
    for partition, count in (("train", rows), ("test", test_rows)):
        if not count:
            continue
        paths = []
        for source in range(sources):
            values = np.arange(count * width, dtype=np.float32).reshape(count, width) / 10 + source * 100
            path = tmp_path / f"{partition}_{source}.csv"
            pd.DataFrame(values, columns=np.arange(width) + 1000 + source * 10000).to_csv(path, index=False)
            paths.append(str(path))
            data[partition, source] = values
        config[f"{partition}_x"] = paths if sources > 1 else paths[0]
        y = tmp_path / f"{partition}_y.csv"
        pd.DataFrame({"protein": np.arange(count) * 0.3 + 1.1}).to_csv(y, index=False)
        config[f"{partition}_y"] = str(y)
        if metadata:
            path = tmp_path / f"{partition}_meta.csv"
            pd.DataFrame({"subject": [f"subject-{i}" for i in range(count)], "batch": [f"batch-{i % 3}" for i in range(count)]}).to_csv(path, index=False)
            config[f"{partition}_group"] = str(path)
    return config, data


def test_real_native_multifile_partition_preview_and_exact_counts(tmp_path, monkeypatch):
    from nirs4all.data.config import DatasetConfigs
    from nirs4all.pipeline.runner import PipelineRunner
    monkeypatch.setattr(DatasetConfigs, "__init__", lambda *a, **k: pytest.fail("CSV must not use oracle assembly"))
    monkeypatch.setattr(PipelineRunner, "run", lambda *a, **k: pytest.fail("inspection must never fit"))
    config, data = _csv_dataset(tmp_path, sources=2)
    result = preview_dataset(config, max_samples=20)
    assert result["reader"]["backend"] == "nirs4all-io.native"
    assert result["reader"]["native_load_limits_applied"] is True
    assert result["summary"]["num_samples"] == 162
    assert result["summary"]["train_samples"] == 150
    assert result["summary"]["test_samples"] == 12
    assert result["summary"]["features_per_source"] == [300, 300]
    assert result["summary"]["metadata_columns"] == ["subject", "batch"]
    assert set(result["spectra_preview_by_partition"]) == {"train", "test", "all"}
    selected = np.linspace(0, 149, 20, dtype=int)
    np.testing.assert_allclose(result["spectra_preview"]["mean_spectrum"], data["train", 0][selected].mean(axis=0), rtol=1e-6)
    assert result["spectra_preview"]["statistics_scope"] == "display_sample"
    assert result["spectra_per_source"][1]["wavelengths"][0] == 11000
    assert result["target_distribution"]["n_samples"] == 150
    assert json.loads(json.dumps(result, allow_nan=False))["success"] is True


def test_native_statistics_match_full_partition_not_display_sample(tmp_path):
    config, data = _csv_dataset(tmp_path, rows=21, test_rows=0)
    result = dataset_statistics(config)
    assert result["global"]["num_samples"] == 21
    assert result["global"]["num_features"] == 300
    assert result["global"]["global_mean"] == pytest.approx(float(data["train", 0].mean()), rel=1e-6)
    preview = preview_dataset(config)
    assert preview["summary"]["test_samples"] == 0
    assert "test" not in preview["spectra_preview_by_partition"]


def test_train_only_preview_has_no_test_targets_or_test_target_read(tmp_path, monkeypatch):
    from nirs4all.data.dataset import SpectroDataset

    original_y = SpectroDataset.y

    def checked_y(self, selector=None, *args, **kwargs):
        if isinstance(selector, dict) and selector.get("partition") == "test":
            pytest.fail("Absent test cohort must not read the empty-index target selector")
        return original_y(self, selector, *args, **kwargs)

    monkeypatch.setattr(SpectroDataset, "y", checked_y)
    config, data = _csv_dataset(tmp_path, rows=21, test_rows=0)
    result = preview_dataset(config, max_samples=21)
    assert result["summary"]["test_samples"] == 0
    assert set(result["target_distribution_by_partition"]) == {"train", "all"}
    assert result["target_distribution_by_partition"]["train"]["n_samples"] == 21
    assert result["target_distribution_by_partition"]["all"]["n_samples"] == 21
    assert result["target_distribution_by_partition"]["train"]["mean"] == pytest.approx((np.arange(21) * 0.3 + 1.1).mean())
    np.testing.assert_allclose(result["spectra_preview"]["sample_spectra"], data["train", 0][:5], rtol=1e-6)


def test_distinct_test_target_cohort_is_not_replaced_by_train_values(tmp_path):
    config, _ = _csv_dataset(tmp_path, rows=21, test_rows=7)
    test_values = np.arange(7) * 0.7 + 100.2
    pd.DataFrame({"protein": test_values}).to_csv(config["test_y"], index=False)
    result = preview_dataset(config)
    train = result["target_distribution_by_partition"]["train"]
    test = result["target_distribution_by_partition"]["test"]
    assert (train["n_samples"], test["n_samples"]) == (21, 7)
    assert train["max"] < test["min"]
    assert test["mean"] == pytest.approx(test_values.mean())
    assert result["target_distribution_by_partition"]["all"]["n_samples"] == 28


def test_preview_does_not_truncate_wide_source_axes(tmp_path):
    config, _ = _csv_dataset(tmp_path, width=8200, sources=2, rows=6, test_rows=3, metadata=False)
    result = preview_dataset(config)
    assert result["summary"]["features_per_source"] == [8200, 8200]
    for source in range(2):
        for partition in ("train", "test", "all"):
            assert len(result["spectra_per_source_by_partition"][source][partition]["mean_spectrum"]) == 8200
    # Historical aliases repeat arrays; retain compatibility, not truncation.
    assert len(json.dumps(result).encode()) < 32 * 1024 * 1024


def test_native_budget_and_failure_never_retry_oracle(tmp_path, monkeypatch):
    from nirs4all.data.config import DatasetConfigs
    monkeypatch.setattr(DatasetConfigs, "__init__", lambda *a, **k: pytest.fail("native failures must propagate"))
    config, _ = _csv_dataset(tmp_path, rows=4, width=3, test_rows=0)
    with pytest.raises(ValueError, match="budget|limit"):
        preview_dataset(config, load_limits={"max_cells": 2})
    with pytest.raises(ValueError, match="raw input byte"):
        preview_dataset(config, max_input_bytes=1)


def test_excel_uses_explicit_existing_loader_and_exposes_limits(tmp_path):
    path = tmp_path / "spectra.xlsx"
    values = np.arange(36, dtype=float).reshape(12, 3) / 10
    pd.DataFrame(values, columns=[1000, 1001, 1002]).to_excel(path, index=False)
    config = {"train_x": str(path), "global_params": {"has_header": True}}
    result = preview_dataset(config)
    assert result["reader"]["backend"] == "nirs4all.loaders"
    assert result["reader"]["native_load_limits_applied"] is False
    assert result["summary"]["num_samples"] == 12
    assert result["summary"]["has_targets"] is False
    file = inspect_format_file(str(path), sample_rows=2)
    assert file["num_rows"] == 12
    assert file["num_columns"] == 3
    assert len(file["sample_data"]) == 2
    with pytest.raises(ValueError, match="cannot be guaranteed"):
        preview_dataset(config, load_limits={"max_cells": 1})


def test_mat_file_uses_registered_loader_without_parsing_in_host(tmp_path):
    from scipy.io import savemat
    path = tmp_path / "spectra.mat"
    savemat(path, {"spectra": np.arange(24, dtype=float).reshape(8, 3)})
    result = inspect_format_file(str(path), params={"variable": "spectra"}, sample_rows=3)
    assert result["num_rows"] == 8
    assert result["num_columns"] == 3
    assert len(result["sample_data"]) == 3


def test_expanded_source_document_can_be_normalized_again_without_losing_metadata(tmp_path):
    from nirs4all.api.dataset_documents import normalize_dataset_document
    config, _ = _csv_dataset(tmp_path, sources=2, rows=5, test_rows=0)
    config["_sources"] = [{"name": "NIR"}, {"name": "MIR"}]
    normalized = normalize_dataset_document(config)
    assert normalized["_sources"] == config["_sources"]
    assert normalize_dataset_document(normalized) == normalized


def test_presentation_budget_rejects_before_projection_arrays(tmp_path, monkeypatch):
    import nirs4all.api.dataset_inspection as inspection
    config, _ = _csv_dataset(tmp_path, rows=4, width=3, test_rows=0)
    monkeypatch.setattr(inspection, "_spectra", lambda *a, **k: pytest.fail("admission must precede projection"))
    with pytest.raises(ValueError, match="before array construction"):
        preview_dataset(config, max_response_bytes=100)
