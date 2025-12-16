"""
Integration Tests for Transfer Preprocessing Analysis.

Task 3.4: Comprehensive integration tests for the TransferPreprocessingSelector
module with full nirs4all pipelines.

Tests cover:
1. Transfer analysis between two regression datasets (machine transfer scenario)
2. Transfer analysis within a single pipeline (train/test alignment)
3. Integration with nirs4all pipeline operators
4. All presets (fast, balanced, thorough, full)
5. Generator-based preprocessing specification
6. Visualization and export capabilities

Based on specification: bench/SPEC_TRANSFER_PREPROCESSING_SELECTION.md
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.analysis import (
    TransferPreprocessingSelector,
    TransferResult,
    TransferSelectionResults,
    TransferMetricsComputer,
    get_base_preprocessings,
    apply_pipeline,
    apply_stacked_pipeline,
    apply_augmentation,
    PRESETS,
    list_presets,
)
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate,
    FirstDerivative,
    SavitzkyGolay,
    MultiplicativeScatterCorrection,
)

from tests.fixtures.data_generators import TestDataManager, SyntheticNIRSDataGenerator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_data_manager():
    """Create test data manager with multiple datasets for transfer testing."""
    manager = TestDataManager()
    manager.create_regression_dataset("regression_source", n_train=100, n_val=30)
    manager.create_regression_dataset("regression_target", n_train=80, n_val=25)
    manager.create_regression_dataset("regression_3", n_train=60, n_val=20)
    yield manager
    manager.cleanup()


@pytest.fixture
def synthetic_transfer_data():
    """
    Generate synthetic data simulating machine transfer scenario.

    Creates source and target datasets with distributional shift
    (baseline shift, multiplicative scatter) to simulate real transfer scenarios.
    """
    rng = np.random.RandomState(42)
    n_samples_source = 120
    n_samples_target = 100
    n_features = 150

    # Generate base spectral pattern
    wavelengths = np.linspace(0, np.pi * 4, n_features)
    base_pattern = np.sin(wavelengths) + 0.5 * np.sin(2 * wavelengths)

    # Source data (Machine A - reference)
    X_source = np.zeros((n_samples_source, n_features))
    y_source = np.zeros(n_samples_source)
    for i in range(n_samples_source):
        noise = rng.normal(0, 0.08, n_features)
        scale = rng.uniform(0.9, 1.1)
        # Target correlates with spectral features
        y_source[i] = 10 + 30 * scale + rng.normal(0, 1)
        X_source[i] = scale * base_pattern + noise

    # Target data (Machine B - shifted baseline, different scatter)
    X_target = np.zeros((n_samples_target, n_features))
    y_target = np.zeros(n_samples_target)
    for i in range(n_samples_target):
        noise = rng.normal(0, 0.10, n_features)  # Slightly more noise
        scale = rng.uniform(0.85, 1.15)
        baseline_shift = 0.4  # Systematic baseline shift
        scatter = 1 + rng.uniform(-0.15, 0.15)  # More scatter variation
        y_target[i] = 10 + 30 * scale + rng.normal(0, 1.2)
        X_target[i] = scatter * (scale * base_pattern + baseline_shift) + noise

    return X_source, X_target, y_source, y_target


@pytest.fixture
def small_transfer_data():
    """Generate small datasets for quick tests."""
    rng = np.random.RandomState(123)
    X_source = rng.randn(40, 80) + np.sin(np.linspace(0, 2 * np.pi, 80))
    X_target = rng.randn(35, 80) + np.sin(np.linspace(0, 2 * np.pi, 80)) + 0.3
    y_source = rng.randn(40) * 10 + 50
    return X_source, X_target, y_source


# =============================================================================
# Test TransferPreprocessingSelector with Full Pipelines
# =============================================================================


class TestTransferAnalysisIntegration:
    """Integration tests for TransferPreprocessingSelector with nirs4all pipelines."""

    def test_basic_transfer_analysis_fast(self, synthetic_transfer_data):
        """Test basic transfer analysis with fast preset."""
        X_source, X_target, _, _ = synthetic_transfer_data

        selector = TransferPreprocessingSelector(preset="fast", verbose=0)
        results = selector.fit(X_source, X_target)

        # Verify results structure
        assert isinstance(results, TransferSelectionResults)
        assert len(results.ranking) > 0
        assert results.best is not None

        # Verify best result has positive improvement (data has shift)
        # SNV or normalization should help reduce baseline shift
        assert results.best.transfer_score > 0

        # Verify timing info
        assert "stage1" in results.timing
        assert results.timing["stage1"] > 0

    def test_transfer_analysis_balanced_with_stacking(self, synthetic_transfer_data):
        """Test transfer analysis with balanced preset (includes stacking)."""
        X_source, X_target, _, _ = synthetic_transfer_data

        selector = TransferPreprocessingSelector(preset="balanced", verbose=0)
        results = selector.fit(X_source, X_target)

        # Should have both single and stacked results
        pipeline_types = {r.pipeline_type for r in results.ranking}
        assert "single" in pipeline_types
        assert "stacked" in pipeline_types

        # Verify stacking was performed
        assert "stage2" in results.timing

        # Stacked pipelines should have proper structure
        stacked_results = [r for r in results.ranking if r.pipeline_type == "stacked"]
        for r in stacked_results:
            assert ">" in r.name
            assert len(r.components) >= 2

    def test_transfer_analysis_thorough_with_augmentation(self, small_transfer_data):
        """Test thorough preset with feature augmentation."""
        X_source, X_target, _ = small_transfer_data

        selector = TransferPreprocessingSelector(preset="thorough", verbose=0)
        results = selector.fit(X_source, X_target)

        # Should have all three types
        pipeline_types = {r.pipeline_type for r in results.ranking}
        assert "single" in pipeline_types
        assert "stacked" in pipeline_types
        assert "augmented" in pipeline_types

        # Verify stage3 (augmentation) was performed
        assert "stage3" in results.timing

        # Augmented results should have proper structure
        augmented_results = [r for r in results.ranking if r.pipeline_type == "augmented"]
        for r in augmented_results:
            assert "+" in r.name
            assert len(r.components) >= 2

    def test_transfer_analysis_full_with_validation(self, synthetic_transfer_data):
        """Test full preset with supervised validation."""
        X_source, X_target, y_source, _ = synthetic_transfer_data

        selector = TransferPreprocessingSelector(preset="full", verbose=0)
        results = selector.fit(X_source, X_target, y_source=y_source)

        # Should have stage4 validation
        assert "stage4" in results.timing

        # Top results should have signal scores
        validated = [r for r in results.ranking if r.signal_score is not None]
        assert len(validated) > 0

        # Signal scores should be in valid range
        for r in validated:
            assert 0 <= r.signal_score <= 1

    def test_transfer_with_dataset_configs(self, test_data_manager):
        """Test transfer analysis using DatasetConfigs objects."""
        source_path = str(test_data_manager.get_temp_directory() / "regression_source")
        target_path = str(test_data_manager.get_temp_directory() / "regression_target")

        config_source = DatasetConfigs(source_path)
        config_target = DatasetConfigs(target_path)

        # Extract X, y manually using the SpectroDataset API
        # DatasetConfigs returns SpectroDataset objects which use .x() and .y() methods
        source_dataset = config_source.get_dataset_at(0)
        target_dataset = config_target.get_dataset_at(0)

        X_source = np.asarray(source_dataset.x({"partition": "train"}))
        X_target = np.asarray(target_dataset.x({"partition": "train"}))

        selector = TransferPreprocessingSelector(preset="fast", verbose=0)
        results = selector.fit(X_source, X_target)

        assert len(results.ranking) > 0
        assert results.best is not None

    def test_pipeline_spec_output(self, synthetic_transfer_data):
        """Test conversion to nirs4all pipeline specification."""
        X_source, X_target, _, _ = synthetic_transfer_data

        selector = TransferPreprocessingSelector(preset="balanced", verbose=0)
        results = selector.fit(X_source, X_target)

        # Single spec output
        spec = results.to_pipeline_spec(top_k=1)
        assert isinstance(spec, str)
        assert len(spec) > 0

        # Multiple specs as list
        specs = results.to_pipeline_spec(top_k=3)
        assert isinstance(specs, list)
        assert len(specs) == 3

        # Augmentation format
        aug_spec = results.to_pipeline_spec(top_k=2, use_augmentation=True)
        assert isinstance(aug_spec, dict)
        assert "feature_augmentation" in aug_spec
        assert len(aug_spec["feature_augmentation"]) == 2

    def test_results_dataframe_export(self, small_transfer_data):
        """Test export to pandas DataFrame."""
        X_source, X_target, _ = small_transfer_data

        selector = TransferPreprocessingSelector(preset="fast", verbose=0)
        results = selector.fit(X_source, X_target)

        df = results.to_dataframe()

        assert "name" in df.columns
        assert "transfer_score" in df.columns
        assert "improvement_pct" in df.columns
        assert "centroid_distance" in df.columns
        assert len(df) == len(results.ranking)


# =============================================================================
# Test Integration with nirs4all Pipeline
# =============================================================================


class TestPipelineIntegration:
    """Test integration with full nirs4all pipeline runs."""

    def test_recommended_preprocessing_in_pipeline(self, test_data_manager):
        """
        Test using transfer-recommended preprocessing in a full pipeline.

        This simulates the workflow:
        1. Analyze transfer between two datasets
        2. Use recommended preprocessing in pipeline
        3. Train model and verify predictions
        """
        source_path = str(test_data_manager.get_temp_directory() / "regression_source")
        target_path = str(test_data_manager.get_temp_directory() / "regression_target")

        # Step 1: Load data and analyze transfer
        config_source = DatasetConfigs(source_path)
        config_target = DatasetConfigs(target_path)

        # Extract X, y manually using the SpectroDataset API
        source_dataset = config_source.get_dataset_at(0)
        target_dataset = config_target.get_dataset_at(0)

        X_source = np.asarray(source_dataset.x({"partition": "train"}))
        X_target = np.asarray(target_dataset.x({"partition": "train"}))

        selector = TransferPreprocessingSelector(preset="balanced", verbose=0)
        results = selector.fit(X_source, X_target)

        # Get recommended preprocessing
        best_preprocessing = results.best.name
        assert best_preprocessing is not None

        # Step 2: Build pipeline with recommended preprocessing
        preprocessings = get_base_preprocessings()

        if ">" in best_preprocessing:
            # Stacked preprocessing
            components = best_preprocessing.split(">")
            preprocessing_steps = [preprocessings[c] for c in components]
        else:
            # Single preprocessing
            preprocessing_steps = [preprocessings[best_preprocessing]]

        pipeline = [
            MinMaxScaler(),
            *preprocessing_steps,  # Apply recommended preprocessing
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        # Step 3: Run pipeline on source dataset
        pipeline_config = PipelineConfigs(pipeline, "test_transfer_recommended")
        dataset_config = DatasetConfigs(source_path)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        # Verify predictions are valid
        assert predictions.num_predictions > 0
        best_pred = predictions.get_best(ascending=True)
        assert best_pred is not None
        assert np.isfinite(best_pred["val_score"])

    def test_feature_augmentation_from_transfer(self, test_data_manager):
        """
        Test using transfer-recommended augmentation in pipeline.

        Uses the augmentation format output from TransferPreprocessingSelector.
        """
        source_path = str(test_data_manager.get_temp_directory() / "regression_source")

        # Create transfer data for analysis
        generator = SyntheticNIRSDataGenerator(random_state=42)
        X_source, y_source = generator.generate_regression_data(100)
        X_target, _ = generator.generate_regression_data(80)

        # Add baseline shift to target (simulate machine transfer)
        X_target = X_target + 0.3

        # Analyze and get augmentation spec
        selector = TransferPreprocessingSelector(preset="thorough", verbose=0)
        results = selector.fit(X_source, X_target)

        # Get augmentation format spec
        aug_spec = results.to_pipeline_spec(top_k=2, use_augmentation=True)
        assert "feature_augmentation" in aug_spec

        # Build pipeline with augmentation
        pipeline = [
            MinMaxScaler(),
            aug_spec,  # Use the augmentation spec directly
            ShuffleSplit(n_splits=2, test_size=0.25, random_state=42),
            {"model": PLSRegression(n_components=5)},
        ]

        # Run pipeline
        pipeline_config = PipelineConfigs(pipeline, "test_transfer_augmentation")
        dataset_config = DatasetConfigs(source_path)

        runner = PipelineRunner(save_files=False, verbose=0)
        predictions, _ = runner.run(pipeline_config, dataset_config)

        assert predictions.num_predictions > 0


# =============================================================================
# Test Generator-Based Preprocessing Specification
# =============================================================================


class TestGeneratorIntegration:
    """Test generator-based preprocessing specification."""

    def test_generator_simple_or_spec(self, small_transfer_data):
        """Test generator with simple _or_ specification."""
        X_source, X_target, _ = small_transfer_data

        selector = TransferPreprocessingSelector(
            preset="fast",
            preprocessing_spec={"_or_": ["snv", "msc", "d1"]},
            verbose=0,
        )
        results = selector.fit(X_source, X_target)

        # Should only have the 3 specified preprocessings
        single_results = [r for r in results.ranking if r.pipeline_type == "single"]
        assert len(single_results) == 3

        # Names are the full class names when using string specs (resolved to objects)
        names = {r.name for r in single_results}
        assert names == {"StandardNormalVariate", "MultiplicativeScatterCorrection", "FirstDerivative"}

    def test_generator_arrange_spec(self, small_transfer_data):
        """Test generator with arrange (stacking) specification."""
        X_source, X_target, _ = small_transfer_data

        selector = TransferPreprocessingSelector(
            preset="fast",
            preprocessing_spec={
                "_or_": ["snv", "msc", "d1"],
                "arrange": 2,
            },
            verbose=0,
        )
        results = selector.fit(X_source, X_target)

        # Should have stacked pipelines from generator
        stacked = [r for r in results.ranking if r.pipeline_type == "stacked"]
        assert len(stacked) > 0

        # All stacked should have ">" in name
        for r in stacked:
            assert ">" in r.name

    def test_generator_pick_spec(self, small_transfer_data):
        """Test generator with pick (augmentation) specification."""
        X_source, X_target, _ = small_transfer_data

        selector = TransferPreprocessingSelector(
            preset="fast",
            preprocessing_spec={
                "_or_": ["snv", "msc", "d1"],
                "pick": 2,
            },
            verbose=0,
        )
        results = selector.fit(X_source, X_target)

        # Should have augmented combinations
        augmented = [r for r in results.ranking if r.pipeline_type == "augmented"]
        assert len(augmented) > 0

        for r in augmented:
            assert "+" in r.name

    def test_generator_with_mutex_constraint(self, small_transfer_data):
        """Test generator with mutual exclusion constraints."""
        X_source, X_target, _ = small_transfer_data

        selector = TransferPreprocessingSelector(
            preset="fast",
            preprocessing_spec={
                "_or_": ["snv", "d1", "d2"],
                "arrange": 2,
                "_mutex_": [["d1", "d2"]],  # Don't combine derivatives
            },
            verbose=0,
        )
        results = selector.fit(X_source, X_target)

        # d1>d2 and d2>d1 should NOT be present
        stacked_names = {r.name for r in results.ranking if r.pipeline_type == "stacked"}
        assert "d1>d2" not in stacked_names
        assert "d2>d1" not in stacked_names


# =============================================================================
# Test Visualization Methods
# =============================================================================


class TestVisualizationIntegration:
    """Test visualization methods for transfer analysis results."""

    def test_plot_ranking(self, small_transfer_data):
        """Test ranking plot generation."""
        X_source, X_target, _ = small_transfer_data

        selector = TransferPreprocessingSelector(preset="balanced", verbose=0)
        results = selector.fit(X_source, X_target)

        fig = results.plot_ranking(top_k=8)
        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_metrics_comparison(self, small_transfer_data):
        """Test metrics comparison plot."""
        X_source, X_target, _ = small_transfer_data

        selector = TransferPreprocessingSelector(preset="fast", verbose=0)
        results = selector.fit(X_source, X_target)

        fig = results.plot_metrics_comparison(top_k=5)
        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_improvement_heatmap(self, small_transfer_data):
        """Test improvement heatmap plot."""
        X_source, X_target, _ = small_transfer_data

        selector = TransferPreprocessingSelector(preset="fast", verbose=0)
        results = selector.fit(X_source, X_target)

        fig = results.plot_improvement_heatmap(top_k=8)
        assert fig is not None

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_summary_generation(self, small_transfer_data):
        """Test summary text generation."""
        X_source, X_target, _ = small_transfer_data

        selector = TransferPreprocessingSelector(preset="balanced", verbose=0)
        results = selector.fit(X_source, X_target)

        summary = results.summary()

        assert isinstance(summary, str)
        assert "TRANSFER PREPROCESSING SELECTION RESULTS" in summary
        assert results.best.name in summary
        assert "Improvement" in summary


# =============================================================================
# Test All Presets
# =============================================================================


class TestAllPresets:
    """Test all preset configurations work correctly."""

    @pytest.mark.parametrize("preset", ["fast", "balanced", "thorough", "full"])
    def test_preset_execution(self, small_transfer_data, preset):
        """Test each preset executes without error."""
        X_source, X_target, y_source = small_transfer_data

        selector = TransferPreprocessingSelector(preset=preset, verbose=0)

        # Include y_source for presets that use it
        if preset == "full":
            results = selector.fit(X_source, X_target, y_source=y_source)
        else:
            results = selector.fit(X_source, X_target)

        assert len(results.ranking) > 0
        assert results.best is not None

    def test_preset_timing_comparison(self, small_transfer_data):
        """Verify presets have expected relative performance."""
        X_source, X_target, y_source = small_transfer_data

        times = {}
        for preset in ["fast", "balanced"]:
            selector = TransferPreprocessingSelector(preset=preset, verbose=0)
            results = selector.fit(X_source, X_target)
            times[preset] = sum(results.timing.values())

        # Balanced should take longer than fast
        assert times["balanced"] > times["fast"] * 0.8  # Allow some margin


# =============================================================================
# Test Reproducibility
# =============================================================================


class TestReproducibility:
    """Test reproducibility of transfer analysis."""

    def test_deterministic_results(self, small_transfer_data):
        """Test that results are deterministic with same random state."""
        X_source, X_target, _ = small_transfer_data

        selector1 = TransferPreprocessingSelector(
            preset="balanced", verbose=0, random_state=42
        )
        selector2 = TransferPreprocessingSelector(
            preset="balanced", verbose=0, random_state=42
        )

        results1 = selector1.fit(X_source, X_target)
        results2 = selector2.fit(X_source, X_target)

        # Same ranking order
        names1 = [r.name for r in results1.ranking]
        names2 = [r.name for r in results2.ranking]
        assert names1 == names2

        # Same scores
        assert np.isclose(
            results1.best.transfer_score,
            results2.best.transfer_score,
            rtol=1e-5,
        )


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_datasets(self):
        """Test with minimum viable dataset sizes."""
        rng = np.random.RandomState(42)
        X_source = rng.randn(5, 50)  # Minimum samples
        X_target = rng.randn(5, 50)

        selector = TransferPreprocessingSelector(preset="fast", verbose=0)
        results = selector.fit(X_source, X_target)

        assert len(results.ranking) > 0

    def test_identical_datasets(self):
        """Test when source and target are identical."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 100)

        selector = TransferPreprocessingSelector(preset="fast", verbose=0)
        results = selector.fit(X, X)

        # Should still work, with minimal improvement
        assert len(results.ranking) > 0

        # Raw centroid distance should be ~0
        assert results.raw_metrics["centroid_distance"] < 1e-5

    def test_high_dimensional_data(self):
        """Test with high-dimensional spectral data."""
        rng = np.random.RandomState(42)
        X_source = rng.randn(50, 500)  # High features
        X_target = rng.randn(40, 500) + 0.2

        selector = TransferPreprocessingSelector(preset="fast", verbose=0)
        results = selector.fit(X_source, X_target)

        assert len(results.ranking) > 0

    def test_invalid_preset_raises_error(self):
        """Test that invalid preset raises appropriate error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            TransferPreprocessingSelector(preset="invalid_preset")


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
