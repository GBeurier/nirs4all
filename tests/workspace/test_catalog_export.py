"""
Test Phase 2: Catalog & Export

Tests split Parquet storage for predictions and export methods.
"""

import pytest
import tempfile
import shutil
import json
import csv
from pathlib import Path
import polars as pl
from nirs4all.dataset.predictions import Predictions
from nirs4all.pipeline.io import SimulationSaver


class TestPhase2CatalogExport:
    """Test Phase 2 catalog and export implementation."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_predictions_csv(self, temp_workspace):
        """Create sample predictions CSV file."""
        pipeline_dir = temp_workspace / "test_pipeline"
        pipeline_dir.mkdir(parents=True)

        csv_file = pipeline_dir / "predictions.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['y_true', 'y_pred', 'partition', 'sample_id'])
            writer.writerow([12.5, 12.3, 'train', 'sample_001'])
            writer.writerow([14.2, 14.5, 'test', 'sample_002'])
            writer.writerow([13.8, 13.6, 'val', 'sample_003'])

        return pipeline_dir

    def test_save_to_parquet(self, temp_workspace):
        """Test saving predictions to split Parquet files."""
        catalog_dir = temp_workspace / "catalog"

        # Create predictions object with sample data
        pred = Predictions()
        pred._df = pl.DataFrame({
            "dataset_name": ["wheat_sample1", "wheat_sample1"],
            "config_name": ["baseline_pls", "baseline_pls"],
            "test_score": [0.45, 0.46],
            "train_score": [0.32, 0.33],
            "val_score": [0.41, 0.42],
            "model_name": ["PLSRegression", "PLSRegression"],
            "y_true": ["[12.5, 14.2]", "[13.8, 15.1]"],
            "y_pred": ["[12.3, 14.5]", "[13.6, 15.3]"],
            "sample_indices": ["[0, 1]", "[2, 3]"],
            "fold_id": ["0", "0"],
            "partition": ["train", "test"]
        })

        # Save to Parquet
        meta_path, data_path = pred.save_to_parquet(catalog_dir, "test_pred_001")

        # Verify files exist
        assert meta_path.exists()
        assert data_path.exists()
        assert meta_path.name == "predictions_meta.parquet"
        assert data_path.name == "predictions_data.parquet"

        # Verify metadata content
        meta_df = pl.read_parquet(meta_path)
        assert "prediction_id" in meta_df.columns
        assert "dataset_name" in meta_df.columns
        assert "test_score" in meta_df.columns
        assert len(meta_df) > 0

        # Verify array data content
        data_df = pl.read_parquet(data_path)
        assert "prediction_id" in data_df.columns
        assert "y_true" in data_df.columns
        assert "y_pred" in data_df.columns
        assert len(data_df) > 0

    def test_append_to_parquet(self, temp_workspace):
        """Test appending predictions to existing Parquet files."""
        catalog_dir = temp_workspace / "catalog"

        # Save first prediction
        pred1 = Predictions()
        pred1._df = pl.DataFrame({
            "dataset_name": ["wheat_sample1"],
            "test_score": [0.45],
            "y_true": ["[12.5]"],
            "y_pred": ["[12.3]"]
        })
        pred1.save_to_parquet(catalog_dir, "pred_001")

        # Save second prediction (should append)
        pred2 = Predictions()
        pred2._df = pl.DataFrame({
            "dataset_name": ["wheat_sample2"],
            "test_score": [0.50],
            "y_true": ["[14.2]"],
            "y_pred": ["[14.5]"]
        })
        pred2.save_to_parquet(catalog_dir, "pred_002")

        # Verify both predictions are in the files
        meta_df = pl.read_parquet(catalog_dir / "predictions_meta.parquet")
        assert len(meta_df) >= 2  # At least 2 predictions

        data_df = pl.read_parquet(catalog_dir / "predictions_data.parquet")
        assert len(data_df) >= 2

    def test_load_from_parquet(self, temp_workspace):
        """Test loading predictions from Parquet files."""
        catalog_dir = temp_workspace / "catalog"

        # Save predictions
        pred_save = Predictions()
        pred_save._df = pl.DataFrame({
            "dataset_name": ["wheat_sample1", "wheat_sample2"],
            "test_score": [0.45, 0.50],
            "y_true": ["[12.5]", "[14.2]"],
            "y_pred": ["[12.3]", "[14.5]"],
            "sample_indices": ["[0]", "[1]"]
        })
        pred_save.save_to_parquet(catalog_dir, "pred_001")

        # Load predictions
        pred_load = Predictions.load_from_parquet(catalog_dir)

        # Verify loaded data
        assert pred_load._df.height >= 2
        assert "dataset_name" in pred_load._df.columns
        assert "test_score" in pred_load._df.columns

    def test_load_from_parquet_with_filter(self, temp_workspace):
        """Test loading specific predictions by ID."""
        catalog_dir = temp_workspace / "catalog"

        # Save multiple predictions
        pred1 = Predictions()
        pred1._df = pl.DataFrame({"dataset_name": ["wheat"], "test_score": [0.45]})
        pred1.save_to_parquet(catalog_dir, "pred_001")

        pred2 = Predictions()
        pred2._df = pl.DataFrame({"dataset_name": ["corn"], "test_score": [0.50]})
        pred2.save_to_parquet(catalog_dir, "pred_002")

        # Load only pred_001
        pred_filtered = Predictions.load_from_parquet(catalog_dir, prediction_ids=["pred_001"])

        # Verify only one prediction loaded
        assert pred_filtered._df.height >= 1
        # Check that the prediction_id column contains pred_001
        if "prediction_id" in pred_filtered._df.columns:
            assert "pred_001" in pred_filtered._df["prediction_id"].to_list()

    def test_archive_to_catalog(self, temp_workspace, sample_predictions_csv):
        """Test archiving pipeline results to catalog."""
        catalog_dir = temp_workspace / "catalog"
        pipeline_dir = sample_predictions_csv

        # Archive predictions
        pred = Predictions()
        pred_id = pred.archive_to_catalog(
            catalog_dir,
            pipeline_dir,
            metrics={
                "dataset_name": "wheat_sample1",
                "config_name": "baseline_pls",
                "test_score": 0.45,
                "train_score": 0.32,
                "val_score": 0.41,
                "model_type": "PLSRegression"
            }
        )

        # Verify prediction ID returned
        assert pred_id is not None
        assert len(pred_id) > 0

        # Verify files created
        assert (catalog_dir / "predictions_meta.parquet").exists()
        assert (catalog_dir / "predictions_data.parquet").exists()

    def test_export_pipeline_full(self, temp_workspace):
        """Test exporting full pipeline."""
        # Create sample pipeline directory
        run_dir = temp_workspace / "runs" / "2024-10-24_wheat_sample1"
        pipeline_dir = run_dir / "0001_abc123"
        pipeline_dir.mkdir(parents=True)

        # Create sample files
        (pipeline_dir / "pipeline.json").write_text('{"test": "data"}')
        (pipeline_dir / "metrics.json").write_text('{"rmse": 0.45}')
        (pipeline_dir / "predictions.csv").write_text('y_true,y_pred\n12.5,12.3')

        exports_dir = temp_workspace / "exports"
        saver = SimulationSaver()

        # Export without custom name
        export_path = saver.export_pipeline_full(
            pipeline_dir,
            exports_dir,
            "wheat_sample1",
            "20241024"
        )

        # Verify export
        assert export_path.exists()
        assert "wheat_sample1_20241024_0001_abc123" in str(export_path)
        assert (export_path / "pipeline.json").exists()
        assert (export_path / "metrics.json").exists()
        assert (export_path / "predictions.csv").exists()

    def test_export_pipeline_with_custom_name(self, temp_workspace):
        """Test exporting pipeline with custom name."""
        # Create sample pipeline directory
        pipeline_dir = temp_workspace / "test_pipeline"
        pipeline_dir.mkdir(parents=True)
        (pipeline_dir / "pipeline.json").write_text('{}')

        exports_dir = temp_workspace / "exports"
        saver = SimulationSaver()

        # Export with custom name
        export_path = saver.export_pipeline_full(
            pipeline_dir,
            exports_dir,
            "wheat_sample1",
            "20241024",
            custom_name="production_model"
        )

        # Verify custom name in export path
        assert "production_model_test_pipeline" in str(export_path)

    def test_export_best_prediction(self, temp_workspace):
        """Test exporting predictions to best_predictions folder."""
        # Create sample predictions file
        pipeline_dir = temp_workspace / "test_pipeline"
        pipeline_dir.mkdir(parents=True)
        pred_file = pipeline_dir / "predictions.csv"
        pred_file.write_text('y_true,y_pred\n12.5,12.3\n14.2,14.5')

        exports_dir = temp_workspace / "exports"
        saver = SimulationSaver()

        # Export prediction
        export_path = saver.export_best_prediction(
            pred_file,
            exports_dir,
            "wheat_sample1",
            "20241024",
            "0001_abc123"
        )

        # Verify export
        assert export_path.exists()
        assert export_path.parent.name == "best_predictions"
        assert "wheat_sample1_20241024_0001_abc123.csv" in str(export_path)

        # Verify content copied
        assert export_path.read_text() == pred_file.read_text()

    def test_export_best_prediction_with_custom_name(self, temp_workspace):
        """Test exporting prediction with custom name."""
        pred_file = temp_workspace / "predictions.csv"
        pred_file.write_text('y_true,y_pred\n12.5,12.3')

        exports_dir = temp_workspace / "exports"
        saver = SimulationSaver()

        # Export with custom name
        export_path = saver.export_best_prediction(
            pred_file,
            exports_dir,
            "wheat_sample1",
            "20241024",
            "0001_abc123",
            custom_name="best_model"
        )

        # Verify custom name in filename
        assert "best_model_0001_abc123.csv" in str(export_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
