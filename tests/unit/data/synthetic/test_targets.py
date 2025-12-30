"""
Unit tests for TargetGenerator class.
"""

import pytest
import numpy as np

from nirs4all.data.synthetic.targets import (
    TargetGenerator,
    ClassSeparationConfig,
    generate_regression_targets,
    generate_classification_targets,
)


class TestTargetGeneratorInit:
    """Tests for TargetGenerator initialization."""

    def test_default_init(self):
        """Test default initialization."""
        gen = TargetGenerator()
        assert gen.rng is not None

    def test_with_random_state(self):
        """Test initialization with random state."""
        gen = TargetGenerator(random_state=42)
        assert gen._random_state == 42


class TestRegressionTargets:
    """Tests for regression target generation."""

    def test_basic_regression(self):
        """Test basic regression target generation."""
        gen = TargetGenerator(random_state=42)
        y = gen.regression(n_samples=30)

        assert len(y) == 30
        assert y.dtype in [np.float32, np.float64]

    def test_with_concentrations(self):
        """Test regression with concentrations."""
        gen = TargetGenerator(random_state=42)
        C = np.random.default_rng(42).dirichlet(np.ones(5), size=30)

        y = gen.regression(n_samples=30, concentrations=C)

        assert len(y) == 30

    def test_target_range(self):
        """Test target range scaling."""
        gen = TargetGenerator(random_state=42)
        C = np.random.default_rng(42).dirichlet(np.ones(5), size=30)

        # Without noise, range should be exact
        y = gen.regression(
            n_samples=30,
            concentrations=C,
            range=(0, 100),
            noise=0,  # Disable noise for strict range test
            correlation=1.0,
        )

        assert y.min() >= 0
        assert y.max() <= 100

    def test_single_component(self):
        """Test using single component as target."""
        gen = TargetGenerator(random_state=42)
        C = np.random.default_rng(42).uniform(0, 1, size=(30, 5))

        y = gen.regression(n_samples=30, concentrations=C, component=0)

        # Should be correlated with first component
        correlation = np.corrcoef(C[:, 0], y)[0, 1]
        assert abs(correlation) > 0.5

    def test_component_by_name(self):
        """Test using component by name."""
        gen = TargetGenerator(random_state=42)
        C = np.random.default_rng(42).uniform(0, 1, size=(30, 3))
        names = ["water", "protein", "lipid"]

        y = gen.regression(
            n_samples=30,
            concentrations=C,
            component="protein",
            component_names=names
        )

        correlation = np.corrcoef(C[:, 1], y)[0, 1]
        assert abs(correlation) > 0.5

    def test_distributions(self):
        """Test different distribution types."""
        gen = TargetGenerator(random_state=42)
        C = np.random.default_rng(42).uniform(0, 1, size=(30, 5))

        for dist in ["uniform", "normal", "lognormal", "bimodal"]:
            y = gen.regression(n_samples=30, concentrations=C, distribution=dist)
            assert len(y) == 30

    def test_log_transform(self):
        """Test log transformation."""
        gen = TargetGenerator(random_state=42)
        y = gen.regression(n_samples=30, range=(1, 100), transform="log")

        # Log-transformed values should be different from raw
        assert y.min() >= 0

    def test_sqrt_transform(self):
        """Test sqrt transformation."""
        gen = TargetGenerator(random_state=42)
        y = gen.regression(n_samples=30, range=(0, 100), transform="sqrt")

        assert y.min() >= 0


class TestClassificationTargets:
    """Tests for classification target generation."""

    def test_binary_classification(self):
        """Test binary classification."""
        gen = TargetGenerator(random_state=42)
        y = gen.classification(n_samples=30, n_classes=2)

        assert len(y) == 30
        assert set(y) == {0, 1}

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        gen = TargetGenerator(random_state=42)
        y = gen.classification(n_samples=30, n_classes=5)

        assert len(y) == 30
        assert set(y) == {0, 1, 2, 3, 4}

    def test_class_weights(self):
        """Test imbalanced class weights."""
        gen = TargetGenerator(random_state=42)
        y = gen.classification(
            n_samples=1000,
            n_classes=3,
            class_weights=[0.6, 0.3, 0.1]
        )

        unique, counts = np.unique(y, return_counts=True)
        # Class 0 should be most frequent
        assert counts[0] > counts[1] > counts[2]

    def test_separation_methods(self):
        """Test different separation methods."""
        gen = TargetGenerator(random_state=42)
        C = np.random.default_rng(42).dirichlet(np.ones(5), size=30)

        for method in ["component", "threshold", "cluster"]:
            y = gen.classification(
                n_samples=30,
                concentrations=C,
                n_classes=3,
                separation_method=method
            )
            assert len(y) == 30
            assert len(set(y)) <= 3

    def test_high_separation(self):
        """Test high class separation creates distinct classes."""
        gen = TargetGenerator(random_state=42)
        C = np.random.default_rng(42).dirichlet(np.ones(5), size=60)

        y = gen.classification(
            n_samples=60,
            concentrations=C,
            n_classes=3,
            separation=3.0
        )

        # All classes should be represented
        assert len(set(y)) == 3

    def test_return_proba(self):
        """Test returning probabilities."""
        gen = TargetGenerator(random_state=42)
        C = np.random.default_rng(42).dirichlet(np.ones(5), size=30)

        y, proba = gen.classification(
            n_samples=30,
            concentrations=C,
            n_classes=3,
            return_proba=True
        )

        assert len(y) == 30
        assert proba.shape == (30, 3)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_invalid_n_classes(self):
        """Test error on invalid n_classes."""
        gen = TargetGenerator(random_state=42)
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            gen.classification(n_samples=30, n_classes=1)

    def test_invalid_class_weights_length(self):
        """Test error on mismatched class weights."""
        gen = TargetGenerator(random_state=42)
        with pytest.raises(ValueError, match="class_weights length"):
            gen.classification(
                n_samples=30,
                n_classes=3,
                class_weights=[0.5, 0.5]
            )

    def test_invalid_class_weights_sum(self):
        """Test error on class weights not summing to 1."""
        gen = TargetGenerator(random_state=42)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            gen.classification(
                n_samples=30,
                n_classes=3,
                class_weights=[0.5, 0.3, 0.3]
            )


class TestReproducibility:
    """Tests for reproducibility."""

    def test_regression_reproducible(self):
        """Test regression is reproducible."""
        C = np.random.default_rng(42).dirichlet(np.ones(5), size=30)

        gen1 = TargetGenerator(random_state=42)
        gen2 = TargetGenerator(random_state=42)

        y1 = gen1.regression(n_samples=30, concentrations=C)
        y2 = gen2.regression(n_samples=30, concentrations=C)

        np.testing.assert_array_equal(y1, y2)

    def test_classification_reproducible(self):
        """Test classification is reproducible."""
        C = np.random.default_rng(42).dirichlet(np.ones(5), size=30)

        gen1 = TargetGenerator(random_state=42)
        gen2 = TargetGenerator(random_state=42)

        y1 = gen1.classification(n_samples=30, concentrations=C, n_classes=3)
        y2 = gen2.classification(n_samples=30, concentrations=C, n_classes=3)

        np.testing.assert_array_equal(y1, y2)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_regression_targets(self):
        """Test regression convenience function."""
        y = generate_regression_targets(
            n_samples=30,
            random_state=42,
            distribution="lognormal",
            range=(0, 100)
        )

        assert len(y) == 30

    def test_generate_classification_targets(self):
        """Test classification convenience function."""
        y = generate_classification_targets(
            n_samples=30,
            random_state=42,
            n_classes=4,
            separation=2.0
        )

        assert len(y) == 30
        assert len(set(y)) == 4


class TestClassSeparationConfig:
    """Tests for ClassSeparationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ClassSeparationConfig()
        assert config.separation == 1.5
        assert config.method == "component"
        assert config.noise == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ClassSeparationConfig(
            separation=2.5,
            method="threshold",
            noise=0.2
        )
        assert config.separation == 2.5
        assert config.method == "threshold"
