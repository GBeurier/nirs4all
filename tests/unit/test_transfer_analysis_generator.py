"""
Unit tests for TransferPreprocessingSelector generator mode with custom (non-sklearn) transforms.

Covers the edge cases identified in audit AN-M-02: generator mode combined with
transformer objects that do not implement the sklearn ``get_params`` interface.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CustomTransform:
    """Minimal transformer with no sklearn interface (no get_params)."""

    def fit(self, X, y=None):
        self._mean = X.mean(axis=0)
        return self

    def transform(self, X):
        return X - self._mean

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class _CustomScaleTransform:
    """Another minimal transformer without sklearn interface."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.scale

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

@pytest.fixture
def datasets():
    """Small source and target spectra suitable for transfer metric computation."""
    rng = np.random.default_rng(0)
    X_source = rng.normal(0.5, 0.1, size=(20, 30)).astype(np.float64)
    X_target = rng.normal(0.6, 0.12, size=(15, 30)).astype(np.float64)
    return X_source, X_target

# ---------------------------------------------------------------------------
# get_transform_name / get_transform_signature with custom objects
# ---------------------------------------------------------------------------

class TestGetTransformNameCustom:
    """get_transform_name should handle objects without get_params."""

    def test_name_is_class_name(self):
        from nirs4all.analysis.transfer_utils import get_transform_name
        t = _CustomTransform()
        assert get_transform_name(t) == "_CustomTransform"

    def test_name_for_list_of_custom_transforms(self):
        from nirs4all.analysis.transfer_utils import get_transform_name
        transforms = [_CustomTransform(), _CustomScaleTransform()]
        name = get_transform_name(transforms)
        assert "_CustomTransform" in name
        assert "_CustomScaleTransform" in name
        assert ">" in name

class TestGetTransformSignatureCustom:
    """get_transform_signature should gracefully fall back for non-sklearn objects."""

    def test_signature_without_get_params(self):
        from nirs4all.analysis.transfer_utils import get_transform_signature
        t = _CustomTransform()
        sig = get_transform_signature(t)
        # Should be the class name (no params to introspect)
        assert sig == "_CustomTransform"

    def test_signature_with_attribute_but_no_get_params(self):
        from nirs4all.analysis.transfer_utils import get_transform_signature
        t = _CustomScaleTransform(scale=2.0)
        sig = get_transform_signature(t)
        # No get_params -> falls back to class name only
        assert sig == "_CustomScaleTransform"

# ---------------------------------------------------------------------------
# apply_pipeline with custom transforms
# ---------------------------------------------------------------------------

class TestApplyPipelineCustomTransforms:
    """apply_pipeline should work end-to-end with non-sklearn transforms."""

    def test_single_custom_transform(self, datasets):
        from nirs4all.analysis.transfer_utils import apply_pipeline
        X_source, _ = datasets
        result = apply_pipeline(X_source, [_CustomTransform()])
        assert result.shape == X_source.shape
        assert np.isfinite(result).all()

    def test_two_custom_transforms_chained(self, datasets):
        from nirs4all.analysis.transfer_utils import apply_pipeline
        X_source, _ = datasets
        transforms = [_CustomTransform(), _CustomScaleTransform(scale=2.0)]
        result = apply_pipeline(X_source, transforms)
        assert result.shape == X_source.shape
        assert np.isfinite(result).all()

    def test_input_not_mutated(self, datasets):
        from nirs4all.analysis.transfer_utils import apply_pipeline
        X_source, _ = datasets
        X_copy = X_source.copy()
        apply_pipeline(X_source, [_CustomTransform()])
        np.testing.assert_array_equal(X_source, X_copy)

# ---------------------------------------------------------------------------
# TransferPreprocessingSelector — generator mode with custom transform objects
# ---------------------------------------------------------------------------

class TestSelectorGeneratorModeCustomTransforms:
    """Generator mode of TransferPreprocessingSelector with non-sklearn transforms."""

    def _make_selector_with_custom_preprocessings(self):
        """Build a selector whose preprocessings dict uses custom transforms."""
        from nirs4all.analysis import TransferPreprocessingSelector
        custom_preprocessings = {
            "center": _CustomTransform(),
            "scale2x": _CustomScaleTransform(scale=2.0),
        }
        return TransferPreprocessingSelector(
            preset=None,
            preprocessings=custom_preprocessings,
            run_stage2=False,
            run_stage3=False,
            run_stage4=False,
            n_components=5,
            k_neighbors=3,
            n_jobs=1,
            verbose=0,
        )

    def test_fit_with_custom_preprocessings_dict(self, datasets):
        """Selector fits and returns results when preprocessings dict contains custom transforms."""
        X_source, X_target = datasets
        selector = self._make_selector_with_custom_preprocessings()
        results = selector.fit(X_source, X_target)
        assert results is not None
        assert len(results.ranking) > 0

    def test_results_have_valid_transfer_scores(self, datasets):
        """All results from custom transforms should have finite transfer scores."""
        X_source, X_target = datasets
        selector = self._make_selector_with_custom_preprocessings()
        results = selector.fit(X_source, X_target)
        for r in results.ranking:
            assert r.transfer_score is not None
            assert np.isfinite(r.transfer_score)

    def test_generator_spec_with_custom_object_list(self, datasets):
        """Generator mode using preprocessing_spec containing custom transform objects."""
        from nirs4all.analysis import TransferPreprocessingSelector

        custom_transforms = [_CustomTransform(), _CustomScaleTransform(scale=0.5)]

        # preprocessing_spec can directly list transform objects via _or_
        selector = TransferPreprocessingSelector(
            preset=None,
            preprocessing_spec={"_or_": custom_transforms},
            run_stage2=False,
            run_stage3=False,
            run_stage4=False,
            n_components=5,
            k_neighbors=3,
            n_jobs=1,
            verbose=0,
        )

        X_source, X_target = datasets
        results = selector.fit(X_source, X_target)
        assert results is not None
        assert len(results.ranking) > 0

    def test_generator_mode_produces_no_nan_scores_for_custom_objects(self, datasets):
        """Custom transform objects in generator mode should not produce NaN scores."""
        from nirs4all.analysis import TransferPreprocessingSelector

        custom_transforms = [_CustomTransform(), _CustomScaleTransform(scale=1.5)]

        selector = TransferPreprocessingSelector(
            preset=None,
            preprocessing_spec={"_or_": custom_transforms},
            n_components=5,
            k_neighbors=3,
            n_jobs=1,
            verbose=0,
        )

        X_source, X_target = datasets
        results = selector.fit(X_source, X_target)
        nan_results = [r for r in results.ranking if np.isnan(r.transfer_score)]
        assert len(nan_results) == 0, f"Unexpected NaN scores: {nan_results}"

    def test_transform_name_stored_in_result(self, datasets):
        """Results should record transform class names when using custom objects."""
        from nirs4all.analysis import TransferPreprocessingSelector

        selector = TransferPreprocessingSelector(
            preset=None,
            preprocessing_spec={"_or_": [_CustomTransform()]},
            n_components=5,
            k_neighbors=3,
            n_jobs=1,
            verbose=0,
        )

        X_source, X_target = datasets
        results = selector.fit(X_source, X_target)
        names = [r.name for r in results.ranking]
        assert any("CustomTransform" in n for n in names)
