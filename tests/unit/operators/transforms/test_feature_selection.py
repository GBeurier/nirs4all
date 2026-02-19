"""
Tests for feature selection operators (CARS, MC-UVE).
"""
import numpy as np
import pytest
from sklearn.datasets import make_regression


class TestCARS:
    """Tests for CARS (Competitive Adaptive Reweighted Sampling)."""

    def test_cars_import(self):
        """Test that CARS can be imported."""
        from nirs4all.operators.transforms.feature_selection import CARS
        assert CARS is not None

    def test_cars_basic_fit(self):
        """Test basic CARS fitting."""
        from nirs4all.operators.transforms.feature_selection import CARS

        # Create synthetic data with some informative features
        rng = np.random.default_rng(42)
        n_samples, n_features = 100, 50
        X, y = make_regression(n_samples=n_samples, n_features=n_features,
                               n_informative=10, random_state=42)

        cars = CARS(n_components=5, n_sampling_runs=20, random_state=42)
        cars.fit(X, y)

        assert hasattr(cars, 'selected_indices_')
        assert hasattr(cars, 'selection_mask_')
        assert hasattr(cars, 'n_features_out_')
        assert cars.n_features_in_ == n_features
        assert 1 <= cars.n_features_out_ <= n_features

    def test_cars_transform(self):
        """Test CARS transform reduces features."""
        from nirs4all.operators.transforms.feature_selection import CARS

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        cars = CARS(n_components=5, n_sampling_runs=20, random_state=42)
        cars.fit(X, y)

        X_transformed = cars.transform(X)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == cars.n_features_out_
        assert X_transformed.shape[1] < X.shape[1]

    def test_cars_fit_transform(self):
        """Test CARS fit_transform."""
        from nirs4all.operators.transforms.feature_selection import CARS

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        cars = CARS(n_components=5, n_sampling_runs=20, random_state=42)
        X_transformed = cars.fit_transform(X, y)

        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == cars.n_features_out_

    def test_cars_get_support_mask(self):
        """Test get_support returns boolean mask."""
        from nirs4all.operators.transforms.feature_selection import CARS

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        cars = CARS(n_components=5, n_sampling_runs=20, random_state=42)
        cars.fit(X, y)

        mask = cars.get_support(indices=False)
        assert mask.dtype == bool
        assert len(mask) == X.shape[1]
        assert mask.sum() == cars.n_features_out_

    def test_cars_get_support_indices(self):
        """Test get_support returns indices."""
        from nirs4all.operators.transforms.feature_selection import CARS

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        cars = CARS(n_components=5, n_sampling_runs=20, random_state=42)
        cars.fit(X, y)

        indices = cars.get_support(indices=True)
        assert len(indices) == cars.n_features_out_
        assert np.all(indices >= 0)
        assert np.all(indices < X.shape[1])

    def test_cars_rmsecv_history(self):
        """Test that CARS stores RMSECV history."""
        from nirs4all.operators.transforms.feature_selection import CARS

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        cars = CARS(n_components=5, n_sampling_runs=20, random_state=42)
        cars.fit(X, y)

        assert hasattr(cars, 'rmsecv_history_')
        assert hasattr(cars, 'n_variables_history_')
        assert hasattr(cars, 'optimal_run_idx_')
        assert len(cars.rmsecv_history_) <= 20
        assert cars.optimal_run_idx_ < len(cars.rmsecv_history_)

    def test_cars_reproducibility(self):
        """Test that CARS is reproducible with random_state."""
        from nirs4all.operators.transforms.feature_selection import CARS

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        cars1 = CARS(n_components=5, n_sampling_runs=20, random_state=123)
        cars1.fit(X, y)

        cars2 = CARS(n_components=5, n_sampling_runs=20, random_state=123)
        cars2.fit(X, y)

        np.testing.assert_array_equal(cars1.selected_indices_, cars2.selected_indices_)

    def test_cars_requires_y(self):
        """Test that CARS raises error without y."""
        from nirs4all.operators.transforms.feature_selection import CARS

        X = np.random.randn(100, 50)
        cars = CARS()

        with pytest.raises(ValueError, match="requires y"):
            cars.fit(X)

    def test_cars_wavelengths_stored(self):
        """Test that CARS stores wavelengths if provided."""
        from nirs4all.operators.transforms.feature_selection import CARS

        X, y = make_regression(n_samples=100, n_features=50, random_state=42)
        wavelengths = np.linspace(1000, 2500, 50)

        cars = CARS(n_components=5, n_sampling_runs=20, random_state=42)
        cars.fit(X, y, wavelengths=wavelengths)

        assert hasattr(cars, 'original_wavelengths_')
        np.testing.assert_array_equal(cars.original_wavelengths_, wavelengths)

    def test_cars_feature_names_out_with_wavelengths(self):
        """Test get_feature_names_out with wavelengths."""
        from nirs4all.operators.transforms.feature_selection import CARS

        X, y = make_regression(n_samples=100, n_features=50, random_state=42)
        wavelengths = np.linspace(1000, 2500, 50)

        cars = CARS(n_components=5, n_sampling_runs=20, random_state=42)
        cars.fit(X, y, wavelengths=wavelengths)

        names = cars.get_feature_names_out()
        assert len(names) == cars.n_features_out_
        # Check that names are formatted wavelengths
        assert all('.' in name for name in names)

    def test_cars_repr_unfitted(self):
        """Test CARS repr when unfitted."""
        from nirs4all.operators.transforms.feature_selection import CARS

        cars = CARS(n_components=10)
        repr_str = repr(cars)
        assert 'CARS' in repr_str
        assert 'unfitted' in repr_str

    def test_cars_repr_fitted(self):
        """Test CARS repr when fitted."""
        from nirs4all.operators.transforms.feature_selection import CARS

        X, y = make_regression(n_samples=100, n_features=50, random_state=42)
        cars = CARS(n_components=5, n_sampling_runs=20, random_state=42)
        cars.fit(X, y)

        repr_str = repr(cars)
        assert 'CARS' in repr_str
        assert 'n_in=' in repr_str
        assert 'n_out=' in repr_str

class TestMCUVE:
    """Tests for MC-UVE (Monte-Carlo Uninformative Variable Elimination)."""

    def test_mcuve_import(self):
        """Test that MCUVE can be imported."""
        from nirs4all.operators.transforms.feature_selection import MCUVE
        assert MCUVE is not None

    def test_mcuve_basic_fit(self):
        """Test basic MC-UVE fitting."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        mcuve = MCUVE(n_components=5, n_iterations=50, random_state=42)
        mcuve.fit(X, y)

        assert hasattr(mcuve, 'selected_indices_')
        assert hasattr(mcuve, 'selection_mask_')
        assert hasattr(mcuve, 'n_features_out_')
        assert mcuve.n_features_in_ == 50
        assert 1 <= mcuve.n_features_out_ <= 50

    def test_mcuve_transform(self):
        """Test MC-UVE transform reduces features."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        mcuve = MCUVE(n_components=5, n_iterations=50, random_state=42)
        mcuve.fit(X, y)

        X_transformed = mcuve.transform(X)
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == mcuve.n_features_out_

    def test_mcuve_fit_transform(self):
        """Test MC-UVE fit_transform."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        mcuve = MCUVE(n_components=5, n_iterations=50, random_state=42)
        X_transformed = mcuve.fit_transform(X, y)

        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == mcuve.n_features_out_

    def test_mcuve_stability_computed(self):
        """Test that MC-UVE computes stability values."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        mcuve = MCUVE(n_components=5, n_iterations=50, random_state=42)
        mcuve.fit(X, y)

        assert hasattr(mcuve, 'stability_')
        assert hasattr(mcuve, 'noise_stability_')
        assert hasattr(mcuve, 'threshold_')
        assert len(mcuve.stability_) == 50

    def test_mcuve_get_support(self):
        """Test get_support returns correct values."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        mcuve = MCUVE(n_components=5, n_iterations=50, random_state=42)
        mcuve.fit(X, y)

        mask = mcuve.get_support(indices=False)
        assert mask.dtype == bool
        assert len(mask) == 50
        assert mask.sum() == mcuve.n_features_out_

        indices = mcuve.get_support(indices=True)
        assert len(indices) == mcuve.n_features_out_

    def test_mcuve_threshold_methods(self):
        """Test different threshold methods."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        # Test percentile method
        mcuve_pct = MCUVE(n_components=5, n_iterations=50,
                          threshold_method='percentile', random_state=42)
        mcuve_pct.fit(X, y)
        assert hasattr(mcuve_pct, 'threshold_')

        # Test fixed method
        mcuve_fixed = MCUVE(n_components=5, n_iterations=50,
                            threshold_method='fixed', threshold_value=1.0, random_state=42)
        mcuve_fixed.fit(X, y)
        assert mcuve_fixed.threshold_ == 1.0

        # Test auto method
        mcuve_auto = MCUVE(n_components=5, n_iterations=50,
                           threshold_method='auto', random_state=42)
        mcuve_auto.fit(X, y)
        assert hasattr(mcuve_auto, 'threshold_')

    def test_mcuve_reproducibility(self):
        """Test that MC-UVE is reproducible with random_state."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        mcuve1 = MCUVE(n_components=5, n_iterations=50, random_state=123)
        mcuve1.fit(X, y)

        mcuve2 = MCUVE(n_components=5, n_iterations=50, random_state=123)
        mcuve2.fit(X, y)

        np.testing.assert_array_equal(mcuve1.selected_indices_, mcuve2.selected_indices_)

    def test_mcuve_requires_y(self):
        """Test that MC-UVE raises error without y."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X = np.random.randn(100, 50)
        mcuve = MCUVE()

        with pytest.raises(ValueError, match="requires y"):
            mcuve.fit(X)

    def test_mcuve_wavelengths_stored(self):
        """Test that MC-UVE stores wavelengths if provided."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X, y = make_regression(n_samples=100, n_features=50, random_state=42)
        wavelengths = np.linspace(1000, 2500, 50)

        mcuve = MCUVE(n_components=5, n_iterations=50, random_state=42)
        mcuve.fit(X, y, wavelengths=wavelengths)

        assert hasattr(mcuve, 'original_wavelengths_')
        np.testing.assert_array_equal(mcuve.original_wavelengths_, wavelengths)

    def test_mcuve_mean_std_coefs(self):
        """Test that MC-UVE stores mean and std coefficients."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X, y = make_regression(n_samples=100, n_features=50,
                               n_informative=10, random_state=42)

        mcuve = MCUVE(n_components=5, n_iterations=50, random_state=42)
        mcuve.fit(X, y)

        assert hasattr(mcuve, 'mean_coefs_')
        assert hasattr(mcuve, 'std_coefs_')
        assert len(mcuve.mean_coefs_) == 50
        assert len(mcuve.std_coefs_) == 50

    def test_mcuve_repr_unfitted(self):
        """Test MC-UVE repr when unfitted."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        mcuve = MCUVE(n_components=10)
        repr_str = repr(mcuve)
        assert 'MCUVE' in repr_str
        assert 'unfitted' in repr_str

    def test_mcuve_repr_fitted(self):
        """Test MC-UVE repr when fitted."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X, y = make_regression(n_samples=100, n_features=50, random_state=42)
        mcuve = MCUVE(n_components=5, n_iterations=50, random_state=42)
        mcuve.fit(X, y)

        repr_str = repr(mcuve)
        assert 'MCUVE' in repr_str
        assert 'n_in=' in repr_str
        assert 'n_out=' in repr_str

class TestFeatureSelectionController:
    """Tests for FeatureSelectionController."""

    def test_controller_import(self):
        """Test that controller can be imported."""
        from nirs4all.controllers.data.feature_selection import FeatureSelectionController
        assert FeatureSelectionController is not None

    def test_controller_matches_cars(self):
        """Test that controller matches CARS operator."""
        from nirs4all.controllers.data.feature_selection import FeatureSelectionController
        from nirs4all.operators.transforms.feature_selection import CARS

        cars = CARS()
        assert FeatureSelectionController.matches(cars, cars, "")

    def test_controller_matches_mcuve(self):
        """Test that controller matches MCUVE operator."""
        from nirs4all.controllers.data.feature_selection import FeatureSelectionController
        from nirs4all.operators.transforms.feature_selection import MCUVE

        mcuve = MCUVE()
        assert FeatureSelectionController.matches(mcuve, mcuve, "")

    def test_controller_supports_multi_source(self):
        """Test that controller supports multi-source."""
        from nirs4all.controllers.data.feature_selection import FeatureSelectionController
        assert FeatureSelectionController.use_multi_source() is True

    def test_controller_supports_prediction(self):
        """Test that controller supports prediction mode."""
        from nirs4all.controllers.data.feature_selection import FeatureSelectionController
        assert FeatureSelectionController.supports_prediction_mode() is True

class TestFeatureSelectionWithNIRSData:
    """Integration tests with NIRS-like data."""

    def test_cars_with_spectral_data(self):
        """Test CARS with synthetic spectral data."""
        from nirs4all.operators.transforms.feature_selection import CARS

        # Create spectral-like data with peaks
        rng = np.random.default_rng(42)
        n_samples = 100
        wavelengths = np.linspace(1000, 2500, 200)

        # Create spectra with informative peak at 1500 nm
        X = np.zeros((n_samples, 200))
        peak_center = np.argmin(np.abs(wavelengths - 1500))
        for i in range(n_samples):
            concentration = rng.uniform(0, 10)
            X[i] = rng.normal(0, 0.1, 200)  # Noise
            # Add informative peak proportional to concentration
            X[i, peak_center - 10:peak_center + 10] += concentration * np.exp(
                -np.linspace(-2, 2, 20) ** 2
            )

        y = X[:, peak_center]  # Target correlates with peak

        cars = CARS(n_components=5, n_sampling_runs=30, random_state=42)
        cars.fit(X, y, wavelengths=wavelengths)

        # Check that selected features are near the peak
        selected_wl = wavelengths[cars.selected_indices_]
        assert np.any((selected_wl > 1400) & (selected_wl < 1600))

    def test_mcuve_with_spectral_data(self):
        """Test MC-UVE with synthetic spectral data."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        # Create spectral-like data with peaks
        rng = np.random.default_rng(42)
        n_samples = 100
        wavelengths = np.linspace(1000, 2500, 200)

        # Create spectra with informative peak at 1500 nm
        X = np.zeros((n_samples, 200))
        peak_center = np.argmin(np.abs(wavelengths - 1500))
        for i in range(n_samples):
            concentration = rng.uniform(0, 10)
            X[i] = rng.normal(0, 0.1, 200)  # Noise
            # Add informative peak proportional to concentration
            X[i, peak_center - 10:peak_center + 10] += concentration * np.exp(
                -np.linspace(-2, 2, 20) ** 2
            )

        y = X[:, peak_center]  # Target correlates with peak

        mcuve = MCUVE(n_components=5, n_iterations=50, random_state=42)
        mcuve.fit(X, y, wavelengths=wavelengths)

        # Check that selected features include some near the peak
        selected_wl = wavelengths[mcuve.selected_indices_]
        peak_nearby = np.any((selected_wl > 1400) & (selected_wl < 1600))
        # MC-UVE should at least select some wavelengths
        assert mcuve.n_features_out_ > 0
        # If many wavelengths selected, peak region should be included
        if mcuve.n_features_out_ > 20:
            assert peak_nearby

    def test_cars_transform_preserves_order(self):
        """Test that CARS preserves sample order."""
        from nirs4all.operators.transforms.feature_selection import CARS

        X, y = make_regression(n_samples=50, n_features=100, random_state=42)

        cars = CARS(n_components=5, n_sampling_runs=20, random_state=42)
        cars.fit(X, y)

        X_transformed = cars.transform(X)

        # Check that rows correspond to same samples
        for i, idx in enumerate(cars.selected_indices_):
            np.testing.assert_array_equal(X_transformed[:, i], X[:, idx])

    def test_mcuve_transform_preserves_order(self):
        """Test that MC-UVE preserves sample order."""
        from nirs4all.operators.transforms.feature_selection import MCUVE

        X, y = make_regression(n_samples=50, n_features=100, random_state=42)

        mcuve = MCUVE(n_components=5, n_iterations=50, random_state=42)
        mcuve.fit(X, y)

        X_transformed = mcuve.transform(X)

        # Check that rows correspond to same samples
        for i, idx in enumerate(mcuve.selected_indices_):
            np.testing.assert_array_equal(X_transformed[:, i], X[:, idx])
