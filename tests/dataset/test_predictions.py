"""Tests for the PredictionBlock class."""
import numpy as np
import polars as pl
import pytest
from nirs4all.dataset.predictions import PredictionBlock


@pytest.fixture
def comprehensive_prediction_block():
    """Create a PredictionBlock with multiple predictions."""
    pb = PredictionBlock()

    # First prediction: model1, fold0, train
    preds1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    meta1 = {
        'model': 'model1',
        'fold': 0,
        'repeat': 0,
        'partition': 'train',
        'processing': 'raw',
        'seed': 42
    }
    pb.add_prediction(preds1, meta1)

    # Second prediction: model1, fold0, val
    preds2 = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
    meta2 = {
        'model': 'model1',
        'fold': 0,
        'repeat': 0,
        'partition': 'val',
        'processing': 'raw',
        'seed': 42
    }
    pb.add_prediction(preds2, meta2)

    # Third prediction: model2, fold1, train
    preds3 = np.array([[1.1, 1.2]], dtype=np.float32)
    meta3 = {
        'model': 'model2',
        'fold': 1,
        'repeat': 0,
        'partition': 'train',
        'processing': 'processed',
        'seed': 123
    }
    pb.add_prediction(preds3, meta3)

    return pb


class TestPredictionBlockBasics:
    """Test basic PredictionBlock functionality."""

    def test_empty_prediction_block(self):
        """Test empty PredictionBlock state and representation."""
        pb = PredictionBlock()

        repr_str = repr(pb)
        assert 'PredictionBlock' in repr_str, "Repr should include class name"

    def test_add_basic_prediction(self):
        """Test adding a basic prediction."""
        pb = PredictionBlock()
        preds = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        meta = {
            'model': 'test_model',
            'fold': 0,
            'repeat': 0,
            'partition': 'train',
            'processing': 'raw',
            'seed': 42
        }

        pb.add_prediction(preds, meta)

        assert pb.table is not None, "Table should be set after adding prediction"
        assert pb.table.height == 2, "Should have 2 prediction rows"

    def test_add_prediction_preserves_metadata(self):
        """Test that prediction metadata is preserved correctly."""
        pb = PredictionBlock()
        preds = np.array([[1.0]], dtype=np.float32)
        meta = {
            'model': 'complex_model_name',
            'fold': 5,
            'repeat': 3,
            'partition': 'val',
            'processing': 'preprocessed_v2',
            'seed': 12345
        }

        pb.add_prediction(preds, meta)

        table = pb.table
        assert table['model'][0] == 'complex_model_name', "Model name should be preserved"
        assert table['fold'][0] == 5, "Fold should be preserved"
        assert table['repeat'][0] == 3, "Repeat should be preserved"
        assert table['partition'][0] == 'val', "Partition should be preserved"
        assert table['processing'][0] == 'preprocessed_v2', "Processing should be preserved"
        assert table['seed'][0] == 12345, "Seed should be preserved"


class TestPredictionBlockErrorHandling:
    """Test PredictionBlock error handling."""

    def test_add_prediction_missing_required_metadata(self):
        """Test error when required metadata fields are missing."""
        pb = PredictionBlock()
        preds = np.array([[0.5]], dtype=np.float32)

        # Test missing each required field
        required_fields = ['model', 'fold', 'repeat', 'partition', 'processing', 'seed']

        for missing_field in required_fields:
            incomplete_meta = {
                'model': 'test',
                'fold': 0,
                'repeat': 0,
                'partition': 'train',
                'processing': 'raw',
                'seed': 42
            }
            del incomplete_meta[missing_field]

            with pytest.raises(ValueError, match=f".*{missing_field}.*"):
                pb.add_prediction(preds, incomplete_meta)

    def test_add_prediction_empty_array(self):
        """Test adding empty prediction array."""
        pb = PredictionBlock()
        preds = np.array([], dtype=np.float32).reshape(0, 2)
        meta = {
            'model': 'test',
            'fold': 0,
            'repeat': 0,
            'partition': 'train',
            'processing': 'raw',
            'seed': 42
        }

        # Should handle empty arrays gracefully
        try:
            pb.add_prediction(preds, meta)
            assert pb.table.height == 0, "Should have no rows for empty prediction"
        except ValueError:
            # Rejecting empty arrays is also acceptable
            pass

    def test_prediction_access_without_data(self):
        """Test error when accessing predictions without data."""
        pb = PredictionBlock()

        with pytest.raises(ValueError):
            pb.prediction({'model': 'nonexistent'})

    def test_inverse_transform_without_data(self):
        """Test error when applying inverse transform without data."""
        pb = PredictionBlock()

        with pytest.raises(ValueError):
            pb.inverse_transform_prediction([], {'model': 'nonexistent'})


class TestPredictionBlockDataAccess:
    """Test PredictionBlock data access functionality."""

    def test_prediction_filter_by_model(self, comprehensive_prediction_block):
        """Test filtering predictions by model."""
        pb = comprehensive_prediction_block

        model1_preds = pb.prediction({'model': 'model1'})

        assert isinstance(model1_preds, np.ndarray), "Should return numpy array"
        assert model1_preds.shape == (5, 2), "Should stack all model1 predictions (3+2 samples)"

        # Check stacking order (should match insertion order)
        expected = np.array([
            [0.1, 0.2], [0.3, 0.4], [0.5, 0.6],  # First prediction
            [0.7, 0.8], [0.9, 1.0]                # Second prediction
        ], dtype=np.float32)
        np.testing.assert_array_equal(model1_preds, expected, "Model1 predictions should match expected stacking")

    def test_prediction_filter_by_partition(self, comprehensive_prediction_block):
        """Test filtering predictions by partition."""
        pb = comprehensive_prediction_block

        train_preds = pb.prediction({'partition': 'train'})
        val_preds = pb.prediction({'partition': 'val'})

        assert train_preds.shape == (4, 2), "Should have 4 train samples (3 from model1 + 1 from model2)"
        assert val_preds.shape == (2, 2), "Should have 2 val samples (from model1)"

        # Check val predictions content
        expected_val = np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(val_preds, expected_val, "Val predictions should match")

    def test_prediction_filter_by_multiple_criteria(self, comprehensive_prediction_block):
        """Test filtering by multiple criteria."""
        pb = comprehensive_prediction_block

        specific_preds = pb.prediction({
            'model': 'model1',
            'partition': 'train'
        })

        assert specific_preds.shape == (3, 2), "Should match only model1 train predictions"
        expected = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        np.testing.assert_array_equal(specific_preds, expected, "Filtered predictions should match")

    def test_prediction_filter_no_matches(self, comprehensive_prediction_block):
        """Test filtering with criteria that matches nothing."""
        pb = comprehensive_prediction_block

        no_match = pb.prediction({'model': 'nonexistent_model'})

        assert isinstance(no_match, np.ndarray), "Should still return numpy array"
        assert no_match.shape[0] == 0, "Should have no rows"
        # Shape[1] should match the number of outputs from existing data
        assert no_match.shape[1] == 2, "Should preserve number of output columns"

    def test_prediction_filter_by_fold_and_repeat(self, comprehensive_prediction_block):
        """Test filtering by fold and repeat."""
        pb = comprehensive_prediction_block

        fold0_preds = pb.prediction({'fold': 0})
        fold1_preds = pb.prediction({'fold': 1})

        assert fold0_preds.shape == (5, 2), "Fold 0 should have 5 samples"
        assert fold1_preds.shape == (1, 2), "Fold 1 should have 1 sample"

        repeat0_preds = pb.prediction({'repeat': 0})
        assert repeat0_preds.shape == (6, 2), "Repeat 0 should have all 6 samples"


class TestPredictionBlockTransforms:
    """Test PredictionBlock transform functionality."""

    @pytest.fixture
    def simple_prediction_block(self):
        """Create a simple PredictionBlock for transform testing."""
        pb = PredictionBlock()
        preds = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        meta = {
            'model': 'transform_test',
            'fold': 0,
            'repeat': 0,
            'partition': 'test',
            'processing': 'raw',
            'seed': 1
        }
        pb.add_prediction(preds, meta)
        return pb

    def test_inverse_transform_single_function(self, simple_prediction_block):
        """Test inverse transform with single function."""
        pb = simple_prediction_block

        def multiply_by_2(x):
            return x * 2

        transformed = pb.inverse_transform_prediction([multiply_by_2], {'model': 'transform_test'})

        original = pb.prediction({'model': 'transform_test'})
        expected = original * 2

        np.testing.assert_array_equal(transformed, expected, "Single transform should be applied correctly")

    def test_inverse_transform_multiple_functions(self, simple_prediction_block):
        """Test inverse transform with multiple functions (chain)."""
        pb = simple_prediction_block

        def add_10(x):
            return x + 10

        def multiply_by_3(x):
            return x * 3

        transforms = [add_10, multiply_by_3]
        transformed = pb.inverse_transform_prediction(transforms, {'model': 'transform_test'})

        # Transforms should be applied in order: first add_10, then multiply_by_3
        original = pb.prediction({'model': 'transform_test'})
        expected = multiply_by_3(add_10(original))

        np.testing.assert_array_equal(transformed, expected, "Multiple transforms should be chained correctly")

    def test_inverse_transform_empty_function_list(self, simple_prediction_block):
        """Test inverse transform with empty function list."""
        pb = simple_prediction_block

        transformed = pb.inverse_transform_prediction([], {'model': 'transform_test'})
        original = pb.prediction({'model': 'transform_test'})

        np.testing.assert_array_equal(transformed, original, "Empty transform list should return original data")

    def test_inverse_transform_with_filtering(self, comprehensive_prediction_block):
        """Test inverse transform with filtered predictions."""
        pb = comprehensive_prediction_block

        def square(x):
            return x ** 2

        # Transform only val partition predictions
        transformed = pb.inverse_transform_prediction([square], {'partition': 'val'})

        original_val = pb.prediction({'partition': 'val'})
        expected = original_val ** 2

        np.testing.assert_array_equal(transformed, expected, "Transform should apply to filtered predictions")

    def test_inverse_transform_preserves_dtype(self, simple_prediction_block):
        """Test that inverse transform preserves array dtype."""
        pb = simple_prediction_block

        def add_half(x):
            return x + 0.5

        transformed = pb.inverse_transform_prediction([add_half], {'model': 'transform_test'})
        original = pb.prediction({'model': 'transform_test'})

        assert transformed.dtype == original.dtype, "Transform should preserve original dtype"

    def test_inverse_transform_with_complex_functions(self, simple_prediction_block):
        """Test inverse transform with more complex functions."""
        pb = simple_prediction_block

        def normalize_columns(x):
            """Normalize each column independently."""
            return (x - x.mean(axis=0)) / x.std(axis=0)

        def clip_values(x):
            """Clip values to [-1, 1] range."""
            return np.clip(x, -1, 1)

        transforms = [normalize_columns, clip_values]
        transformed = pb.inverse_transform_prediction(transforms, {'model': 'transform_test'})

        assert transformed.shape == (3, 2), "Transformed shape should match original"
        assert np.all(np.abs(transformed) <= 1), "Values should be clipped to [-1, 1]"


class TestPredictionBlockEdgeCases:
    """Test PredictionBlock edge cases."""

    def test_predictions_with_different_shapes(self):
        """Test adding predictions with different output dimensions."""
        pb = PredictionBlock()

        # First prediction: 2 outputs
        preds1 = np.array([[1.0, 2.0]], dtype=np.float32)
        meta1 = {
            'model': 'model1',
            'fold': 0,
            'repeat': 0,
            'partition': 'train',
            'processing': 'raw',
            'seed': 1
        }
        pb.add_prediction(preds1, meta1)

        # Second prediction: 3 outputs (different shape)
        preds2 = np.array([[3.0, 4.0, 5.0]], dtype=np.float32)
        meta2 = {
            'model': 'model2',
            'fold': 0,
            'repeat': 0,
            'partition': 'train',
            'processing': 'raw',
            'seed': 1
        }

        # Behavior depends on implementation - might raise error or handle gracefully
        try:
            pb.add_prediction(preds2, meta2)

            # If it accepts different shapes, test retrieval
            model1_preds = pb.prediction({'model': 'model1'})
            model2_preds = pb.prediction({'model': 'model2'})

            assert model1_preds.shape[1] == 2, "Model1 should have 2 outputs"
            assert model2_preds.shape[1] == 3, "Model2 should have 3 outputs"

        except ValueError:
            # Rejecting different shapes is also acceptable
            pass

    def test_predictions_with_nan_values(self):
        """Test predictions containing NaN values."""
        pb = PredictionBlock()
        preds = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]], dtype=np.float32)
        meta = {
            'model': 'nan_test',
            'fold': 0,
            'repeat': 0,
            'partition': 'test',
            'processing': 'raw',
            'seed': 1
        }

        pb.add_prediction(preds, meta)

        retrieved = pb.prediction({'model': 'nan_test'})
        assert retrieved.shape == (3, 2), "Should preserve shape with NaN values"
        assert np.isnan(retrieved[0, 1]), "NaN values should be preserved"
        assert np.isnan(retrieved[1, 0]), "NaN values should be preserved"
        assert retrieved[2, 0] == 3.0, "Non-NaN values should be preserved"

    def test_predictions_with_extreme_values(self):
        """Test predictions with extreme values."""
        pb = PredictionBlock()
        preds = np.array([
            [1e10, -1e10],
            [1e-10, 0.0],
            [np.finfo(np.float32).max, np.finfo(np.float32).min]
        ], dtype=np.float32)
        meta = {
            'model': 'extreme_test',
            'fold': 0,
            'repeat': 0,
            'partition': 'test',
            'processing': 'raw',
            'seed': 1
        }

        pb.add_prediction(preds, meta)

        retrieved = pb.prediction({'model': 'extreme_test'})
        assert retrieved.shape == (3, 2), "Should handle extreme values"
        assert not np.any(np.isinf(retrieved)), "Should not produce infinite values"

    def test_large_number_of_predictions(self):
        """Test with large number of prediction entries."""
        pb = PredictionBlock()

        n_models = 100
        for i in range(n_models):
            preds = np.random.rand(10, 2).astype(np.float32)
            meta = {
                'model': f'model_{i}',
                'fold': i % 5,
                'repeat': i % 3,
                'partition': 'train' if i % 2 == 0 else 'val',
                'processing': 'raw',
                'seed': i
            }
            pb.add_prediction(preds, meta)

        assert pb.table.height == n_models * 10, f"Should have {n_models * 10} prediction rows"

        # Test filtering on large dataset
        train_preds = pb.prediction({'partition': 'train'})
        val_preds = pb.prediction({'partition': 'val'})

        assert train_preds.shape[0] + val_preds.shape[0] == n_models * 10, "Train + val should equal total"

    def test_predictions_with_special_metadata_values(self):
        """Test predictions with special metadata values."""
        pb = PredictionBlock()
        preds = np.array([[1.0, 2.0]], dtype=np.float32)

        # Test with various special values in metadata
        special_meta = {
            'model': 'model_with_special-chars_123',
            'fold': -1,  # Negative fold
            'repeat': 0,
            'partition': 'val_special',
            'processing': 'preprocessing_v2.1_final',
            'seed': 2**31 - 1  # Large seed value
        }

        pb.add_prediction(preds, special_meta)

        retrieved = pb.prediction({'model': 'model_with_special-chars_123'})
        assert retrieved.shape == (1, 2), "Should handle special metadata values"

        # Test filtering by special values
        negative_fold = pb.prediction({'fold': -1})
        assert negative_fold.shape == (1, 2), "Should filter by negative fold"

    def test_concurrent_prediction_operations(self):
        """Test that prediction operations are consistent."""
        pb = PredictionBlock()

        # Add same prediction multiple times to test consistency
        base_preds = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        base_meta = {
            'model': 'consistency_test',
            'fold': 0,
            'repeat': 0,
            'partition': 'test',
            'processing': 'raw',
            'seed': 42
        }

        pb.add_prediction(base_preds, base_meta)

        # Multiple retrievals should be identical
        pred1 = pb.prediction({'model': 'consistency_test'})
        pred2 = pb.prediction({'model': 'consistency_test'})
        pred3 = pb.prediction({'model': 'consistency_test'})

        np.testing.assert_array_equal(pred1, pred2, "Multiple retrievals should be identical")
        np.testing.assert_array_equal(pred2, pred3, "Multiple retrievals should be identical")

        # Transform operations should also be consistent
        def identity_transform(x):
            return x

        transform1 = pb.inverse_transform_prediction([identity_transform], {'model': 'consistency_test'})
        transform2 = pb.inverse_transform_prediction([identity_transform], {'model': 'consistency_test'})

        np.testing.assert_array_equal(transform1, transform2, "Transform operations should be consistent")
