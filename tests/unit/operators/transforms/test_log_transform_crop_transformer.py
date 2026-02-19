"""Regression tests for LogTransform (fitted-offset) and CropTransformer (state integrity).

These tests lock in the corrected behavior after critical bug fixes:
- LogTransform: fitted offset must be computed from training data, not test data.
- CropTransformer: transform() must not mutate the `end` parameter on the instance.
"""

import numpy as np
import pytest
from sklearn.base import clone

# ---------------------------------------------------------------------------
# LogTransform regression tests (O-01)
# ---------------------------------------------------------------------------

class TestLogTransformFittedOffset:
    """Regression tests for LogTransform fitted-offset correctness."""

    def test_fitted_offset_computed_from_train_data(self):
        """_fitted_offset is computed from fit() data, not transform() data."""
        from nirs4all.operators.transforms import LogTransform

        # Training data with a value below zero
        X_train = np.array([[-5.0, 0.0, 5.0]])
        # Test data that is all positive – would not need any offset on its own
        X_test = np.array([[1.0, 2.0, 3.0]])

        t = LogTransform(auto_offset=True)
        t.fit(X_train)

        # The fitted offset must be at least abs(min_train) + min_value
        # so that X_train + offset >= min_value
        assert t._fitted_offset > 0.0, "Expected a positive fitted offset for negative training data"

        # Applying the fitted offset to test data should not raise
        X_out = t.transform(X_test)
        assert X_out.shape == X_test.shape
        assert np.all(np.isfinite(X_out))

    def test_fitted_offset_not_recomputed_at_transform_time(self):
        """Fitting on data with negatives, then transforming positive data uses the trained offset."""
        from nirs4all.operators.transforms import LogTransform

        X_train = np.array([[-3.0, 0.0, 4.0]])
        X_test = np.array([[10.0, 20.0, 30.0]])

        t = LogTransform(auto_offset=True)
        t.fit(X_train)
        saved_offset = t._fitted_offset

        t.transform(X_test)  # Must not modify _fitted_offset
        assert t._fitted_offset == saved_offset, (
            "_fitted_offset changed during transform(). "
            "Offset must be fixed at fit() time."
        )

    def test_no_offset_for_all_positive_train_data(self):
        """When all training values are > 0, auto_offset should add zero offset."""
        from nirs4all.operators.transforms import LogTransform

        X_train = np.array([[1.0, 2.0, 3.0]])
        t = LogTransform(auto_offset=True)
        t.fit(X_train)
        assert t._fitted_offset == 0.0

    def test_manual_offset_stored_as_fitted_offset(self):
        """Explicit offset kwarg is honoured in _fitted_offset."""
        from nirs4all.operators.transforms import LogTransform

        X_train = np.array([[1.0, 2.0, 3.0]])
        t = LogTransform(offset=1.5, auto_offset=False)
        t.fit(X_train)
        assert t._fitted_offset == 1.5

    def test_inverse_transform_round_trips(self):
        """inverse_transform recovers the original values (using fitted offset)."""
        from nirs4all.operators.transforms import LogTransform

        X_train = np.array([[-1.0, 0.0, 2.0]])
        t = LogTransform(auto_offset=True)
        t.fit(X_train)
        X_fwd = t.transform(X_train.copy())
        X_back = t.inverse_transform(X_fwd)
        np.testing.assert_allclose(X_back, X_train + t._fitted_offset - t._fitted_offset, atol=1e-8)

    def test_train_test_offset_consistency(self):
        """Offset applied during transform must match the offset from fit, not from transform input."""
        from nirs4all.operators.transforms import LogTransform

        # Training data has min=-10 → auto_offset must be about 10 + min_value
        X_train = np.array([[-10.0, 5.0, 10.0]])
        # Test data is all large positives
        X_test = np.array([[100.0, 200.0, 300.0]])

        t = LogTransform(auto_offset=True)
        t.fit(X_train)

        # With the training offset applied, test values should be shifted by the same amount
        expected_log_first = np.log(100.0 + t._fitted_offset)
        X_out = t.transform(X_test)
        assert abs(X_out[0, 0] - expected_log_first) < 1e-8

    def test_clone_resets_fitted_offset(self):
        """A cloned (unfitted) transformer has _fitted_offset reset to 0.0."""
        from nirs4all.operators.transforms import LogTransform

        X_train = np.array([[-5.0, 1.0, 3.0]])
        t = LogTransform(auto_offset=True)
        t.fit(X_train)
        assert t._fitted_offset != 0.0

        t2 = clone(t)
        assert t2._fitted_offset == 0.0

    def test_base10_log(self):
        """LogTransform with base=10 uses log10 correctly."""
        from nirs4all.operators.transforms import LogTransform

        X = np.array([[10.0, 100.0, 1000.0]])
        t = LogTransform(base=10, auto_offset=False)
        t.fit(X)
        X_out = t.transform(X.copy())
        np.testing.assert_allclose(X_out, [[1.0, 2.0, 3.0]], atol=1e-8)

# ---------------------------------------------------------------------------
# CropTransformer regression tests (O-02)
# ---------------------------------------------------------------------------

class TestCropTransformerStateIntegrity:
    """Regression tests for CropTransformer state integrity after transform()."""

    def test_end_attribute_not_mutated_when_clamped(self):
        """transform() must not modify self.end even if the requested end > n_features."""
        from nirs4all.operators.transforms import CropTransformer

        X = np.ones((5, 10))  # 10 features
        t = CropTransformer(start=0, end=50)  # end beyond n_features

        original_end = t.end
        t.transform(X)
        assert t.end == original_end, (
            f"CropTransformer.end changed from {original_end} to {t.end} "
            "after transform(). State must not be mutated."
        )

    def test_end_attribute_unchanged_on_repeated_transforms(self):
        """Repeated calls to transform() on arrays of different widths must not change self.end."""
        from nirs4all.operators.transforms import CropTransformer

        t = CropTransformer(start=2, end=8)
        original_end = t.end

        for n_features in (6, 10, 15, 5):
            X = np.ones((3, n_features))
            t.transform(X)
            assert t.end == original_end

    def test_transform_output_shape_correct(self):
        """Output has the right number of columns for various end values."""
        from nirs4all.operators.transforms import CropTransformer

        X = np.arange(30).reshape(3, 10)

        # Normal crop within range
        t = CropTransformer(start=2, end=6)
        out = t.transform(X)
        assert out.shape == (3, 4)

        # end beyond range: should clip to n_features
        t = CropTransformer(start=0, end=100)
        out = t.transform(X)
        assert out.shape == (3, 10)

    def test_start_zero_end_none(self):
        """Default start=0, end=None returns the full array."""
        from nirs4all.operators.transforms import CropTransformer

        X = np.arange(20).reshape(4, 5)
        t = CropTransformer()
        out = t.transform(X)
        np.testing.assert_array_equal(out, X)

    def test_non_array_input_raises(self):
        """transform() raises ValueError for non-ndarray input."""
        from nirs4all.operators.transforms import CropTransformer

        t = CropTransformer(start=0, end=3)
        with pytest.raises(ValueError, match="numpy array"):
            t.transform([[1, 2, 3, 4]])

    def test_fit_returns_self(self):
        """fit() is a no-op that returns self."""
        from nirs4all.operators.transforms import CropTransformer

        t = CropTransformer(start=1, end=4)
        X = np.ones((5, 10))
        result = t.fit(X)
        assert result is t

    def test_clone_produces_equivalent_transformer(self):
        """clone() produces an unfitted copy with the same hyperparameters."""
        from nirs4all.operators.transforms import CropTransformer

        t = CropTransformer(start=3, end=7)
        t2 = clone(t)
        assert t2.start == 3
        assert t2.end == 7

    def test_fit_transform_matches_transform(self):
        """fit_transform() and transform() produce identical output."""
        from nirs4all.operators.transforms import CropTransformer

        X = np.random.default_rng(0).random((8, 20))
        t = CropTransformer(start=2, end=10)
        out1 = t.fit_transform(X)
        out2 = t.transform(X)
        np.testing.assert_array_equal(out1, out2)
