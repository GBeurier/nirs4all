"""
Test TabReportManager aggregation feature.

Tests Phase 2 of the aggregation by sample ID feature:
- generate_best_score_tab_report with aggregate parameter
- Both table string and CSV output formats
- Regression and classification task types
"""

import numpy as np
import pytest

from nirs4all.visualization.reports import TabReportManager


class TestTabReportManagerAggregation:
    """Test TabReportManager with aggregation support."""

    @pytest.fixture
    def regression_partition_data(self):
        """
        Create sample regression data with 4 samples per unique ID.
        Total: 12 samples, 3 unique IDs (ID1, ID2, ID3).
        """
        np.random.seed(42)
        # Simulate 4 measurements per sample ID
        sample_ids = ['ID1', 'ID1', 'ID1', 'ID1', 'ID2', 'ID2', 'ID2', 'ID2', 'ID3', 'ID3', 'ID3', 'ID3']
        y_true = np.array([10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0])
        # Add some noise to predictions
        y_pred = y_true + np.random.normal(0, 0.5, size=len(y_true))

        return {
            'val': {
                'y_true': y_true,
                'y_pred': y_pred,
                'metadata': {'sample_id': sample_ids},
                'task_type': 'regression',
                'n_features': 100
            },
            'test': {
                'y_true': y_true * 1.1,  # Slightly different for test
                'y_pred': y_true * 1.1 + np.random.normal(0, 0.3, size=len(y_true)),
                'metadata': {'sample_id': sample_ids},
                'task_type': 'regression',
                'n_features': 100
            },
            'train': {
                'y_true': y_true * 0.9,
                'y_pred': y_true * 0.9 + np.random.normal(0, 0.2, size=len(y_true)),
                'metadata': {'sample_id': sample_ids},
                'task_type': 'regression',
                'n_features': 100
            }
        }

    @pytest.fixture
    def classification_partition_data(self):
        """
        Create sample classification data with multiple samples per class.
        """
        np.random.seed(42)
        # Simulate classification data: 8 samples, 2 classes
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])  # Some misclassifications

        return {
            'val': {
                'y_true': y_true,
                'y_pred': y_pred,
                'metadata': {},
                'task_type': 'binary_classification',
                'n_features': 50
            },
            'test': {
                'y_true': y_true,
                'y_pred': y_pred,
                'metadata': {},
                'task_type': 'binary_classification',
                'n_features': 50
            }
        }

    def test_generate_report_without_aggregation(self, regression_partition_data):
        """Test that report generation works without aggregation (baseline)."""
        formatted, csv_content = TabReportManager.generate_best_score_tab_report(
            regression_partition_data
        )

        assert formatted is not None
        assert "Cros Val" in formatted
        assert "Train" in formatted
        assert "Test" in formatted
        # Should not have aggregated rows
        assert "Cros Val*" not in formatted
        assert "Train*" not in formatted
        assert "Test*" not in formatted
        # Should not have footer
        assert "Aggregated by" not in formatted

    def test_generate_report_with_string_aggregate(self, regression_partition_data):
        """Test report generation with string column aggregation."""
        formatted, csv_content = TabReportManager.generate_best_score_tab_report(
            regression_partition_data,
            aggregate='sample_id'
        )

        assert formatted is not None
        # Should have raw rows
        assert "Cros Val" in formatted
        assert "Train" in formatted
        assert "Test" in formatted
        # Should have aggregated rows (marked with *)
        assert "Cros Val*" in formatted
        assert "Train*" in formatted
        assert "Test*" in formatted
        # Should have footer note
        assert "Aggregated by sample_id" in formatted

    def test_generate_report_with_bool_aggregate(self, regression_partition_data):
        """Test report generation with True (aggregate by y)."""
        formatted, csv_content = TabReportManager.generate_best_score_tab_report(
            regression_partition_data,
            aggregate=True
        )

        assert formatted is not None
        # Should have aggregated rows
        assert "Cros Val*" in formatted
        # Footer should say "y (target values)"
        assert "Aggregated by y (target values)" in formatted

    def test_aggregated_rows_have_fewer_samples(self, regression_partition_data):
        """Test that aggregated rows show reduced sample count."""
        formatted, csv_content = TabReportManager.generate_best_score_tab_report(
            regression_partition_data,
            aggregate='sample_id'
        )

        # Parse the table to check sample counts
        lines = formatted.split('\n')

        # Find lines with Cros Val and Cros Val*
        val_line = None
        val_agg_line = None
        for line in lines:
            if "Cros Val*" in line:
                val_agg_line = line
            elif "Cros Val" in line and "Cros Val*" not in line:
                val_line = line

        assert val_line is not None, "Should have Cros Val row"
        assert val_agg_line is not None, "Should have Cros Val* row"

        # The raw row should have 12 samples
        assert "12" in val_line

        # The aggregated row should have 3 samples (3 unique IDs)
        assert "3" in val_agg_line

    def test_aggregated_metrics_recalculated(self, regression_partition_data):
        """Test that metrics are recalculated for aggregated data."""
        formatted, csv_content = TabReportManager.generate_best_score_tab_report(
            regression_partition_data,
            aggregate='sample_id'
        )

        # Aggregated R2 should typically be higher (averaging reduces noise)
        # We can't easily parse exact values, but we verify the structure
        assert csv_content is not None
        lines = csv_content.strip().split('\n')

        # Should have header + 3 raw rows + 3 aggregated rows = 7 lines
        assert len(lines) == 7, f"Expected 7 lines, got {len(lines)}"

    def test_csv_includes_aggregated_column(self, regression_partition_data):
        """Test that CSV output includes Aggregated column."""
        formatted, csv_content = TabReportManager.generate_best_score_tab_report(
            regression_partition_data,
            aggregate='sample_id'
        )

        assert csv_content is not None
        # Header should include Aggregated column
        assert "Aggregated" in csv_content

        # Aggregated rows should have the column value
        lines = csv_content.strip().split('\n')
        for line in lines:
            if '*' in line.split(',')[0]:  # Aggregated row names end with *
                assert 'sample_id' in line, f"Aggregated row should have 'sample_id' in Aggregated column: {line}"

    def test_missing_aggregate_column_graceful(self, regression_partition_data):
        """Test graceful handling when aggregate column doesn't exist."""
        # Remove metadata from one partition
        regression_partition_data['val']['metadata'] = {}

        formatted, csv_content = TabReportManager.generate_best_score_tab_report(
            regression_partition_data,
            aggregate='nonexistent_column'
        )

        # Should still produce valid output
        assert formatted is not None
        # Raw rows should be present
        assert "Cros Val" in formatted
        # Aggregated rows may or may not be present depending on which partitions have metadata

    def test_aggregated_stats_columns_blank(self, regression_partition_data):
        """Test that aggregated rows have blank descriptive stats (Mean, Median, etc.)."""
        formatted, csv_content = TabReportManager.generate_best_score_tab_report(
            regression_partition_data,
            aggregate='sample_id'
        )

        # Parse CSV to check that Mean/Median/etc are blank for aggregated rows
        lines = csv_content.strip().split('\n')
        header = lines[0].split(',')

        # Find column indices for stats that should be blank
        mean_idx = header.index('Mean')
        median_idx = header.index('Median')

        for line in lines[1:]:
            cells = line.split(',')
            row_name = cells[0]
            if '*' in row_name:  # Aggregated row
                # Mean and Median should be blank
                assert cells[mean_idx] == '', f"Mean should be blank for aggregated row: {line}"
                assert cells[median_idx] == '', f"Median should be blank for aggregated row: {line}"

    def test_classification_with_aggregation(self, classification_partition_data):
        """Test aggregation works with classification data."""
        # For classification, aggregate=True groups by y (class)
        formatted, csv_content = TabReportManager.generate_best_score_tab_report(
            classification_partition_data,
            aggregate=True  # Aggregate by class
        )

        assert formatted is not None
        # Should have aggregated rows
        assert "*" in formatted
        assert "Aggregated by y (target values)" in formatted

    def test_none_aggregate_same_as_no_aggregate(self, regression_partition_data):
        """Test that aggregate=None produces same output as no aggregate."""
        formatted_none, csv_none = TabReportManager.generate_best_score_tab_report(
            regression_partition_data,
            aggregate=None
        )

        formatted_default, csv_default = TabReportManager.generate_best_score_tab_report(
            regression_partition_data
        )

        assert formatted_none == formatted_default
        assert csv_none == csv_default

    def test_false_aggregate_same_as_none(self, regression_partition_data):
        """Test that aggregate=False produces same output as aggregate=None."""
        formatted_false, csv_false = TabReportManager.generate_best_score_tab_report(
            regression_partition_data,
            aggregate=False
        )

        formatted_none, csv_none = TabReportManager.generate_best_score_tab_report(
            regression_partition_data,
            aggregate=None
        )

        assert formatted_false == formatted_none
        assert csv_false == csv_none

    def test_empty_partition_data(self):
        """Test handling of empty partition data."""
        formatted, csv_content = TabReportManager.generate_best_score_tab_report(
            {},
            aggregate='sample_id'
        )

        assert formatted == "No prediction data available"
        assert csv_content is None

    def test_partial_partition_data(self, regression_partition_data):
        """Test handling when only some partitions have data."""
        # Remove train and test
        partial_data = {'val': regression_partition_data['val']}

        formatted, csv_content = TabReportManager.generate_best_score_tab_report(
            partial_data,
            aggregate='sample_id'
        )

        assert formatted is not None
        assert "Cros Val" in formatted
        assert "Cros Val*" in formatted
        assert "Train" not in formatted
        assert "Test" not in formatted

class TestTabReportAggregationHelper:
    """Test the _aggregate_predictions helper method."""

    def test_aggregate_by_column(self):
        """Test aggregation by metadata column."""
        y_true = np.array([10.0, 10.0, 20.0, 20.0])
        y_pred = np.array([10.5, 9.5, 20.5, 19.5])
        metadata = {'sample_id': ['A', 'A', 'B', 'B']}

        result = TabReportManager._aggregate_predictions(
            y_true, y_pred, 'sample_id', metadata, 'test'
        )

        assert result is not None
        agg_y_true, agg_y_pred = result
        # Should have 2 unique samples
        assert len(agg_y_true) == 2
        assert len(agg_y_pred) == 2
        # Aggregated values should be means
        np.testing.assert_allclose(agg_y_pred, [10.0, 20.0], atol=0.01)

    def test_aggregate_by_y(self):
        """Test aggregation by y_true values."""
        y_true = np.array([10.0, 10.0, 20.0, 20.0])
        y_pred = np.array([10.5, 9.5, 20.5, 19.5])
        metadata = {}

        result = TabReportManager._aggregate_predictions(
            y_true, y_pred, 'y', metadata, 'test'
        )

        assert result is not None
        agg_y_true, agg_y_pred = result
        # Should have 2 unique y values
        assert len(agg_y_true) == 2

    def test_aggregate_missing_column(self):
        """Test graceful failure when column is missing."""
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([10.5, 9.5])
        metadata = {}

        result = TabReportManager._aggregate_predictions(
            y_true, y_pred, 'nonexistent', metadata, 'test'
        )

        assert result is None

    def test_aggregate_length_mismatch(self):
        """Test graceful failure when lengths don't match."""
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([10.5, 9.5])
        metadata = {'sample_id': ['A']}  # Wrong length

        result = TabReportManager._aggregate_predictions(
            y_true, y_pred, 'sample_id', metadata, 'test'
        )

        assert result is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
