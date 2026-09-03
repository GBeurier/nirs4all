"""
Unit tests for nirs4all.generate API.
"""

import builtins

import numpy as np
import pytest

from nirs4all.api.advanced_capabilities import advanced_api_capability_ledger
from nirs4all.pipeline.dagml.rt import RtError


@pytest.fixture(autouse=True)
def _explicit_legacy_generate_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing synthesis assertions exercise the explicit V1 rollback lane."""
    monkeypatch.setenv("N4A_ENGINE", "legacy")
    monkeypatch.delenv("N4A_GENERATE_PLUGIN", raising=False)


class TestGenerateFunction:
    """Tests for the main generate() function."""

    def test_basic_generation(self):
        """Test basic dataset generation."""
        import nirs4all

        dataset = nirs4all.generate(n_samples=100, random_state=42)

        from nirs4all.data import SpectroDataset
        assert isinstance(dataset, SpectroDataset)
        assert dataset.num_samples == 100

    @pytest.mark.parametrize("engine", ["native", "dag-ml", "dual"])
    def test_non_legacy_engine_is_refused_before_constructing_a_dataset(self, monkeypatch: pytest.MonkeyPatch, engine: str):
        """Generation has no native capability yet and must not run implicitly."""
        import nirs4all
        import nirs4all.synthesis

        monkeypatch.setattr(
            nirs4all.synthesis,
            "SyntheticDatasetBuilder",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("generator was constructed")),
        )

        with pytest.raises((NotImplementedError, ValueError), match="nirs4all.generate|dual"):
            nirs4all.generate(n_samples=1, engine=engine)

    @pytest.mark.parametrize(
        "call",
        [
            lambda api, path: api.regression(n_samples=1),
            lambda api, path: api.classification(n_samples=1),
            lambda api, path: api.builder(n_samples=1),
            lambda api, path: api.multi_source(n_samples=1),
            lambda api, path: api.to_folder(path / "dataset", n_samples=1),
            lambda api, path: api.to_csv(path / "dataset.csv", n_samples=1),
            lambda api, path: api.product("dairy", n_samples=1),
            lambda api, path: api.category("dairy", n_samples=1),
            lambda api, path: api.from_template(np.zeros((1, 1)), n_samples=1),
        ],
        ids=[
            "regression",
            "classification",
            "builder",
            "multi_source",
            "to_folder",
            "to_csv",
            "product",
            "category",
            "from_template",
        ],
    )
    def test_generate_convenience_paths_refuse_native_default_before_work(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        call,
    ) -> None:
        """Every public generation entry point obeys the engine boundary."""
        import nirs4all

        monkeypatch.setenv("N4A_ENGINE", "native")
        with pytest.raises(NotImplementedError, match="nirs4all.generate"):
            call(nirs4all.generate, tmp_path)

    def test_generate_as_arrays(self):
        """Test generation returning arrays."""
        import nirs4all

        X, y = nirs4all.generate(n_samples=100, as_dataset=False, random_state=42)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == 100

    def test_generate_reproducibility(self):
        """Test reproducibility with random_state."""
        import nirs4all

        X1, y1 = nirs4all.generate(n_samples=50, as_dataset=False, random_state=42)
        X2, y2 = nirs4all.generate(n_samples=50, as_dataset=False, random_state=42)

        np.testing.assert_allclose(X1, X2)
        np.testing.assert_allclose(y1, y2)

    def test_generate_complexity_simple(self):
        """Test simple complexity generation."""
        import nirs4all

        X, y = nirs4all.generate(
            n_samples=50,
            complexity="simple",
            as_dataset=False,
            random_state=42,
        )

        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(y))

    def test_generate_complexity_realistic(self):
        """Test realistic complexity generation."""
        import nirs4all

        X, y = nirs4all.generate(
            n_samples=50,
            complexity="realistic",
            as_dataset=False,
            random_state=42,
        )

        assert np.all(np.isfinite(X))

    def test_generate_complexity_complex(self):
        """Test complex complexity generation."""
        import nirs4all

        X, y = nirs4all.generate(
            n_samples=50,
            complexity="complex",
            as_dataset=False,
            random_state=42,
        )

        assert np.all(np.isfinite(X))

    def test_generate_wavelength_range(self):
        """Test custom wavelength range."""
        import nirs4all

        X, _ = nirs4all.generate(
            n_samples=50,
            wavelength_range=(1200, 2000),
            as_dataset=False,
            random_state=42,
        )

        # Fewer wavelengths in narrower range
        assert X.shape[1] < 751  # Default is 1000-2500 with step 2

    def test_generate_components(self):
        """Test specifying predefined components."""
        import nirs4all

        X, y = nirs4all.generate(
            n_samples=50,
            components=["water", "protein"],
            as_dataset=False,
            random_state=42,
        )

        # Should have 2 targets (one per component)
        assert y.shape == (50, 2) or y.ndim == 1

    def test_generate_target_range(self):
        """Test target range scaling."""
        import nirs4all

        X, y = nirs4all.generate(
            n_samples=100,
            target_range=(0, 100),
            as_dataset=False,
            random_state=42,
        )

        assert y.min() >= 0
        assert y.max() <= 100

    def test_generate_train_ratio(self):
        """Test train ratio partitioning."""
        import nirs4all

        dataset = nirs4all.generate(
            n_samples=100,
            train_ratio=0.7,
            random_state=42,
        )

        partition_values = dataset._indexer.get_column_values("partition")
        train_count = sum(1 for p in partition_values if p == "train")
        test_count = sum(1 for p in partition_values if p == "test")

        assert train_count == 70
        assert test_count == 30

    def test_generate_custom_name(self):
        """Test custom dataset name."""
        import nirs4all

        dataset = nirs4all.generate(
            n_samples=50,
            name="my_synthetic_data",
            random_state=42,
        )

        assert dataset.name == "my_synthetic_data"

class TestGenerateRegression:
    """Tests for generate.regression() convenience function."""

    def test_regression_basic(self):
        """Test basic regression dataset generation."""
        import nirs4all

        dataset = nirs4all.generate.regression(n_samples=100, random_state=42)

        from nirs4all.data import SpectroDataset
        assert isinstance(dataset, SpectroDataset)
        assert dataset.name == "synthetic_regression"

    def test_regression_as_arrays(self):
        """Test regression returning arrays."""
        import nirs4all

        X, y = nirs4all.generate.regression(
            n_samples=100,
            as_dataset=False,
            random_state=42,
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_regression_target_range(self):
        """Test regression with target range."""
        import nirs4all

        X, y = nirs4all.generate.regression(
            n_samples=100,
            target_range=(0, 100),
            as_dataset=False,
            random_state=42,
        )

        assert y.min() >= 0
        assert y.max() <= 100

    def test_regression_single_target(self):
        """Test regression with single target component."""
        import nirs4all

        X, y = nirs4all.generate.regression(
            n_samples=100,
            target_component=0,
            as_dataset=False,
            random_state=42,
        )

        # Single target should be 1D
        assert y.ndim == 1

    def test_regression_lognormal_distribution(self):
        """Test regression with lognormal distribution."""
        import nirs4all

        X, y = nirs4all.generate.regression(
            n_samples=100,
            distribution="lognormal",
            as_dataset=False,
            random_state=42,
        )

        assert np.all(np.isfinite(y))

class TestGenerateClassification:
    """Tests for generate.classification() convenience function."""

    def test_classification_basic(self):
        """Test basic classification dataset generation."""
        import nirs4all

        dataset = nirs4all.generate.classification(n_samples=100, random_state=42)

        from nirs4all.data import SpectroDataset
        assert isinstance(dataset, SpectroDataset)
        assert dataset.name == "synthetic_classification"

    def test_classification_binary(self):
        """Test binary classification."""
        import nirs4all

        X, y = nirs4all.generate.classification(
            n_samples=100,
            n_classes=2,
            as_dataset=False,
            random_state=42,
        )

        assert set(np.unique(y)) == {0, 1}

    def test_classification_multiclass(self):
        """Test multiclass classification."""
        import nirs4all

        X, y = nirs4all.generate.classification(
            n_samples=100,
            n_classes=4,
            as_dataset=False,
            random_state=42,
        )

        assert len(np.unique(y)) == 4

    def test_classification_imbalanced(self):
        """Test imbalanced classification."""
        import nirs4all

        X, y = nirs4all.generate.classification(
            n_samples=1000,
            n_classes=3,
            class_weights=[0.6, 0.3, 0.1],
            as_dataset=False,
            random_state=42,
        )

        counts = np.bincount(y.astype(int))
        # Class 0 should have most samples
        assert counts[0] > counts[1] > counts[2]

    def test_classification_separation(self):
        """Test class separation parameter."""
        import nirs4all

        X, y = nirs4all.generate.classification(
            n_samples=100,
            n_classes=2,
            class_separation=2.0,
            as_dataset=False,
            random_state=42,
        )

        assert X.shape[0] == 100
        assert set(np.unique(y)) == {0, 1}

class TestGenerateBuilder:
    """Tests for generate.builder() convenience function."""

    def test_builder_returns_builder(self):
        """Test that builder() returns a SyntheticDatasetBuilder."""
        import nirs4all
        from nirs4all.synthesis import SyntheticDatasetBuilder

        builder = nirs4all.generate.builder(n_samples=100, random_state=42)

        assert isinstance(builder, SyntheticDatasetBuilder)

    def test_builder_configuration(self):
        """Test builder configuration and building."""
        import nirs4all

        dataset = (
            nirs4all.generate.builder(n_samples=100, random_state=42)
            .with_features(complexity="realistic")
            .with_targets(distribution="lognormal")
            .with_partitions(train_ratio=0.8)
            .build()
        )

        from nirs4all.data import SpectroDataset
        assert isinstance(dataset, SpectroDataset)
        assert dataset.num_samples == 100

    def test_builder_full_chain(self):
        """Test full builder method chain."""
        import nirs4all

        dataset = (
            nirs4all.generate.builder(n_samples=200, random_state=42)
            .with_features(
                wavelength_range=(1100, 2400),
                complexity="realistic",
                components=["water", "protein"],
            )
            .with_targets(
                distribution="lognormal",
                range=(0, 100),
            )
            .with_partitions(train_ratio=0.75)
            .with_batch_effects(n_batches=2)
            .build()
        )

        assert dataset.num_samples == 200

class TestGenerateNamespace:
    """Tests for the generate namespace functionality."""

    def test_generate_is_callable(self):
        """Test that generate is directly callable."""
        import nirs4all

        # Should not raise - generate should be callable
        dataset = nirs4all.generate(n_samples=50, random_state=42)
        assert dataset is not None

    def test_generate_has_methods(self):
        """Test that generate has method attributes."""
        import nirs4all

        assert hasattr(nirs4all.generate, 'regression')
        assert hasattr(nirs4all.generate, 'classification')
        assert hasattr(nirs4all.generate, 'builder')

    def test_generate_methods_callable(self):
        """Test that generate methods are callable."""
        import nirs4all

        assert callable(nirs4all.generate.regression)
        assert callable(nirs4all.generate.classification)
        assert callable(nirs4all.generate.builder)

    def test_generate_repr(self):
        """Test generate namespace string representation."""
        import nirs4all

        repr_str = repr(nirs4all.generate)
        assert "generate" in repr_str
        assert "regression" in repr_str
        assert "classification" in repr_str
        assert "builder" in repr_str

class TestIntegrationWithPipeline:
    """Tests for integration with nirs4all pipeline."""

    def test_generate_with_run(self):
        """Test using generated data with nirs4all.run()."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import ShuffleSplit
        from sklearn.preprocessing import MinMaxScaler

        import nirs4all

        # Generate synthetic data with single target
        dataset = nirs4all.generate.regression(
            n_samples=200,
            target_component=0,  # Single target for regression
            complexity="simple",
            random_state=42,
        )

        # Run a simple pipeline with cross-validation
        result = nirs4all.run(
            pipeline=[MinMaxScaler(), ShuffleSplit(n_splits=2, test_size=0.2, random_state=42), PLSRegression(n_components=3)],
            dataset=dataset,
            verbose=0,
        )

        assert result is not None

    def test_classification_with_run(self):
        """Test classification data with pipeline."""
        from sklearn.model_selection import ShuffleSplit
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler

        import nirs4all

        # Generate classification data
        dataset = nirs4all.generate.classification(
            n_samples=200,
            n_classes=2,
            complexity="simple",
            random_state=42,
        )

        # Run classification pipeline with cross-validation
        result = nirs4all.run(
            pipeline=[StandardScaler(), ShuffleSplit(n_splits=2, test_size=0.2, random_state=42), KNeighborsClassifier(n_neighbors=3)],
            dataset=dataset,
            engine="legacy",
            verbose=0,
        )

        assert result is not None

class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_sample_count(self):
        """Test with very small sample count."""
        import nirs4all

        X, y = nirs4all.generate(n_samples=10, as_dataset=False, random_state=42)

        assert X.shape[0] == 10

    def test_large_sample_count(self):
        """Test with larger sample count."""
        import nirs4all

        X, y = nirs4all.generate(n_samples=1000, as_dataset=False, random_state=42)

        assert X.shape[0] == 1000
        assert np.all(np.isfinite(X))

    def test_all_train_no_test(self):
        """Test with train_ratio=1.0 (no test set)."""
        import nirs4all

        dataset = nirs4all.generate(
            n_samples=100,
            train_ratio=1.0,
            random_state=42,
        )

        partition_values = dataset._indexer.get_column_values("partition")
        train_count = sum(1 for p in partition_values if p == "train")
        test_count = sum(1 for p in partition_values if p == "test")

        assert train_count == 100
        assert test_count == 0


class TestGenerateApi005Boundary:
    """Fail-closed native/plugin decisions for every public synthesis surface."""

    def test_default_refuses_before_synthesis_import_or_write(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        """The default native profile performs no Python import or output write."""
        import nirs4all

        monkeypatch.delenv("N4A_ENGINE", raising=False)
        output = tmp_path / "must-not-exist.csv"
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "nirs4all.synthesis" or name.startswith("nirs4all.synthesis."):
                raise AssertionError("native generate refusal imported Python synthesis")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(RtError) as caught:
            nirs4all.generate.to_csv(output, n_samples=1)

        assert caught.value.unsupported_capability == "native_generate"
        assert caught.value.verb == "generate"
        assert not output.exists()

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("regression", ()),
            ("classification", ()),
            ("builder", ()),
            ("multi_source", ()),
            ("to_folder", ("must-not-exist",)),
            ("to_csv", ("must-not-exist.csv",)),
            ("product", ("missing-template",)),
            ("category", (["missing-template"],)),
            ("from_template", ("missing-dataset",)),
        ],
    )
    def test_all_namespace_surfaces_preflight_before_data_or_compute(
        self,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        args: tuple,
    ) -> None:
        """Every frozen convenience signature shares the same refusal boundary."""
        import nirs4all

        monkeypatch.delenv("N4A_ENGINE", raising=False)
        with pytest.raises(RtError) as caught:
            getattr(nirs4all.generate, method)(*args)
        assert caught.value.unsupported_capability == "native_generate"

    def test_main_surface_plugin_and_fallback_are_typed_refusals(self) -> None:
        """No plugin or fallback selector can silently enter Python synthesis."""
        import nirs4all

        with pytest.raises(RtError) as plugin_error:
            nirs4all.generate(n_samples=1, plugin="nirs-synthesis-provider")
        assert plugin_error.value.unsupported_capability == "generate_plugin"

        with pytest.raises(RtError) as fallback_error:
            nirs4all.generate(n_samples=1, engine="legacy", allow_fallback=True)
        assert fallback_error.value.unsupported_capability == "implicit_legacy_fallback"

    def test_explicit_legacy_main_selector_is_consumed_before_synthesis(self) -> None:
        """The frozen ``**kwargs`` keeps explicit rollback source-compatible."""
        import nirs4all

        X, y = nirs4all.generate(
            n_samples=4,
            as_dataset=False,
            random_state=7,
            engine="legacy",
        )
        assert X.shape[0] == y.shape[0] == 4

    def test_preflight_and_capability_ledger_are_additive_and_detached(self) -> None:
        """Callers can inspect honest availability without supplying data."""
        import nirs4all

        native = nirs4all.generate.preflight(engine="native")
        assert native.executable is False
        assert native.contract is None
        assert native.unsupported_capability == "native_generate"

        legacy = nirs4all.generate.preflight(engine="legacy")
        assert legacy.executable is True
        assert legacy.contract == "nirs4all.python.synthesis"

        ledger = advanced_api_capability_ledger()
        assert ledger["explain"]["native"]["executable"] is False
        assert ledger["generate"]["plugin"]["executable"] is False
        ledger["generate"]["native"]["executable"] = True
        assert advanced_api_capability_ledger()["generate"]["native"]["executable"] is False
