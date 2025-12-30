"""
Unit tests for DatasetExporter and CSVVariationGenerator.

Tests export functionality for synthetic NIRS datasets to various
file formats and folder structures.
"""

import numpy as np
import pytest
from pathlib import Path


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from nirs4all.data.synthetic import ExportConfig

        config = ExportConfig()
        assert config.format == "standard"
        assert config.separator == ";"
        assert config.float_precision == 6
        assert config.include_headers is True
        assert config.include_index is False
        assert config.compression is None
        assert config.file_extension == ".csv"

    def test_custom_config(self):
        """Test custom configuration values."""
        from nirs4all.data.synthetic import ExportConfig

        config = ExportConfig(
            format="single",
            separator=",",
            float_precision=4,
            include_index=True,
        )
        assert config.format == "single"
        assert config.separator == ","
        assert config.float_precision == 4
        assert config.include_index is True


class TestDatasetExporter:
    """Tests for DatasetExporter class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 50))
        y = rng.random(100) * 100
        wavelengths = np.linspace(1000, 2000, 50)
        return X, y, wavelengths

    def test_init_default(self):
        """Test default initialization."""
        from nirs4all.data.synthetic import DatasetExporter

        exporter = DatasetExporter()
        assert exporter.config.format == "standard"
        assert exporter.config.separator == ";"

    def test_init_with_config(self):
        """Test initialization with custom config."""
        from nirs4all.data.synthetic import DatasetExporter, ExportConfig

        config = ExportConfig(separator=",", float_precision=4)
        exporter = DatasetExporter(config)
        assert exporter.config.separator == ","
        assert exporter.config.float_precision == 4

    def test_to_folder_standard(self, tmp_path, sample_data):
        """Test export to standard folder structure."""
        from nirs4all.data.synthetic import DatasetExporter

        X, y, wavelengths = sample_data
        exporter = DatasetExporter()

        path = exporter.to_folder(
            tmp_path / "export_test",
            X, y,
            train_ratio=0.8,
            wavelengths=wavelengths,
            random_state=42,
        )

        # Check files created
        assert path.exists()
        assert (path / "Xcal.csv").exists()
        assert (path / "Ycal.csv").exists()
        assert (path / "Xval.csv").exists()
        assert (path / "Yval.csv").exists()

        # Check content
        import pandas as pd
        X_cal = pd.read_csv(path / "Xcal.csv", sep=";")
        assert X_cal.shape[0] == 80  # 80% of 100

    def test_to_folder_single_format(self, tmp_path, sample_data):
        """Test export to single file format."""
        from nirs4all.data.synthetic import DatasetExporter

        X, y, wavelengths = sample_data
        exporter = DatasetExporter()

        path = exporter.to_folder(
            tmp_path / "single_test",
            X, y,
            train_ratio=0.8,
            wavelengths=wavelengths,
            format="single",
            random_state=42,
        )

        # Check single file created
        assert (path / "data.csv").exists()

        # Check content
        import pandas as pd
        df = pd.read_csv(path / "data.csv", sep=";")
        assert "partition" in df.columns
        assert df.shape[0] == 100

    def test_to_folder_fragmented(self, tmp_path, sample_data):
        """Test export to fragmented format."""
        from nirs4all.data.synthetic import DatasetExporter

        X, y, wavelengths = sample_data
        exporter = DatasetExporter()

        path = exporter.to_folder(
            tmp_path / "frag_test",
            X, y,
            train_ratio=0.8,
            wavelengths=wavelengths,
            format="fragmented",
            random_state=42,
        )

        # Check folder structure
        assert (path / "train").exists()
        assert (path / "test").exists()

        # Check multiple files in train
        train_files = list((path / "train").glob("X_part*.csv"))
        assert len(train_files) >= 1

    def test_to_csv(self, tmp_path, sample_data):
        """Test export to single CSV file."""
        from nirs4all.data.synthetic import DatasetExporter

        X, y, wavelengths = sample_data
        exporter = DatasetExporter()

        filepath = exporter.to_csv(
            tmp_path / "data.csv",
            X, y,
            wavelengths=wavelengths,
        )

        assert filepath.exists()

        import pandas as pd
        df = pd.read_csv(filepath, sep=";")
        assert df.shape[0] == 100
        assert "target" in df.columns

    def test_to_csv_without_targets(self, tmp_path, sample_data):
        """Test export to CSV without target column."""
        from nirs4all.data.synthetic import DatasetExporter

        X, y, wavelengths = sample_data
        exporter = DatasetExporter()

        filepath = exporter.to_csv(
            tmp_path / "features_only.csv",
            X, y,
            wavelengths=wavelengths,
            include_targets=False,
        )

        import pandas as pd
        df = pd.read_csv(filepath, sep=";")
        assert "target" not in df.columns

    def test_to_numpy(self, tmp_path, sample_data):
        """Test export to numpy format."""
        from nirs4all.data.synthetic import DatasetExporter

        X, y, wavelengths = sample_data
        exporter = DatasetExporter()

        filepath = exporter.to_numpy(
            tmp_path / "data",
            X, y,
            wavelengths=wavelengths,
            compressed=True,
        )

        assert filepath.exists()
        assert filepath.suffix == ".npz"

        # Load and verify
        data = np.load(filepath)
        assert "X" in data
        assert "y" in data
        assert "wavelengths" in data
        np.testing.assert_array_equal(data["X"], X)

    def test_to_folder_validates_inputs(self, tmp_path):
        """Test that exporter validates input shapes."""
        from nirs4all.data.synthetic import DatasetExporter

        X = np.random.random((100, 50))
        y = np.random.random(50)  # Wrong size

        exporter = DatasetExporter()

        with pytest.raises(ValueError, match="same number of samples"):
            exporter.to_folder(tmp_path / "bad", X, y)

    def test_multitarget_export(self, tmp_path):
        """Test export with multiple target columns."""
        from nirs4all.data.synthetic import DatasetExporter

        X = np.random.random((100, 50))
        y = np.random.random((100, 3))  # 3 targets

        exporter = DatasetExporter()
        path = exporter.to_folder(
            tmp_path / "multi",
            X, y,
            train_ratio=0.8,
        )

        import pandas as pd
        Y_cal = pd.read_csv(path / "Ycal.csv", sep=";")
        assert Y_cal.shape[1] == 3


class TestCSVVariationGenerator:
    """Tests for CSVVariationGenerator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        rng = np.random.default_rng(42)
        X = rng.random((50, 30))
        y = rng.random(50) * 100
        wavelengths = np.linspace(1000, 1500, 30)
        return X, y, wavelengths

    def test_generate_all_variations(self, tmp_path, sample_data):
        """Test generating all CSV variations."""
        from nirs4all.data.synthetic import CSVVariationGenerator

        X, y, wavelengths = sample_data
        generator = CSVVariationGenerator()

        paths = generator.generate_all_variations(
            tmp_path / "variations",
            X, y,
            wavelengths=wavelengths,
            random_state=42,
        )

        # Check all variations created
        assert "standard_semicolon" in paths
        assert "comma_separated" in paths
        assert "tab_separated" in paths
        assert "no_headers" in paths
        assert "with_index" in paths
        assert "single_file" in paths
        assert "fragmented" in paths
        assert "low_precision" in paths
        assert "high_precision" in paths

        # Verify each exists
        for name, path in paths.items():
            assert path.exists(), f"{name} path does not exist"

    def test_semicolon_delimiter(self, tmp_path, sample_data):
        """Test semicolon delimiter export."""
        from nirs4all.data.synthetic import CSVVariationGenerator

        X, y, wavelengths = sample_data
        generator = CSVVariationGenerator()

        path = generator.with_semicolon_delimiter(
            tmp_path / "semi",
            X, y,
            wavelengths=wavelengths,
        )

        import pandas as pd
        df = pd.read_csv(path / "Xcal.csv", sep=";")
        assert df.shape[1] == 30

    def test_comma_delimiter(self, tmp_path, sample_data):
        """Test comma delimiter export."""
        from nirs4all.data.synthetic import CSVVariationGenerator

        X, y, wavelengths = sample_data
        generator = CSVVariationGenerator()

        path = generator.with_comma_delimiter(
            tmp_path / "comma",
            X, y,
            wavelengths=wavelengths,
        )

        import pandas as pd
        df = pd.read_csv(path / "Xcal.csv", sep=",")
        assert df.shape[1] == 30

    def test_tab_delimiter(self, tmp_path, sample_data):
        """Test tab delimiter export."""
        from nirs4all.data.synthetic import CSVVariationGenerator

        X, y, wavelengths = sample_data
        generator = CSVVariationGenerator()

        path = generator.with_tab_delimiter(
            tmp_path / "tab",
            X, y,
            wavelengths=wavelengths,
        )

        # Check TSV files
        assert (path / "Xcal.tsv").exists()

    def test_without_headers(self, tmp_path, sample_data):
        """Test export without headers."""
        from nirs4all.data.synthetic import CSVVariationGenerator

        X, y, _ = sample_data
        generator = CSVVariationGenerator()

        path = generator.without_headers(
            tmp_path / "no_headers",
            X, y,
        )

        # Load and check (no header row)
        data = np.loadtxt(path / "Xcal.csv", delimiter=";")
        assert data.shape[0] == 40  # 80% of 50

    def test_precision_levels(self, tmp_path, sample_data):
        """Test different precision levels."""
        from nirs4all.data.synthetic import CSVVariationGenerator

        X, y, wavelengths = sample_data
        generator = CSVVariationGenerator()

        # Low precision
        path_low = generator.with_precision(
            tmp_path / "low",
            X, y,
            wavelengths=wavelengths,
            precision=2,
        )

        # High precision
        path_high = generator.with_precision(
            tmp_path / "high",
            X, y,
            wavelengths=wavelengths,
            precision=10,
        )

        # Check file sizes differ (more precision = larger files)
        low_size = (path_low / "Xcal.csv").stat().st_size
        high_size = (path_high / "Xcal.csv").stat().st_size
        assert high_size > low_size


class TestExportFunctions:
    """Tests for convenience export functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        rng = np.random.default_rng(42)
        X = rng.random((50, 30))
        y = rng.random(50)
        wavelengths = np.linspace(1000, 1500, 30)
        return X, y, wavelengths

    def test_export_to_folder(self, tmp_path, sample_data):
        """Test export_to_folder convenience function."""
        from nirs4all.data.synthetic import export_to_folder

        X, y, wavelengths = sample_data

        path = export_to_folder(
            tmp_path / "quick",
            X, y,
            wavelengths=wavelengths,
            train_ratio=0.7,
            random_state=42,
        )

        assert path.exists()
        assert (path / "Xcal.csv").exists()

    def test_export_to_csv(self, tmp_path, sample_data):
        """Test export_to_csv convenience function."""
        from nirs4all.data.synthetic import export_to_csv

        X, y, wavelengths = sample_data

        path = export_to_csv(
            tmp_path / "data.csv",
            X, y,
            wavelengths=wavelengths,
        )

        assert path.exists()


class TestBuilderExport:
    """Tests for SyntheticDatasetBuilder export methods."""

    def test_builder_export(self, tmp_path):
        """Test export method on builder."""
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        builder = SyntheticDatasetBuilder(n_samples=100, random_state=42)
        builder.with_features(complexity="simple")

        path = builder.export(tmp_path / "builder_export")

        assert path.exists()
        assert (path / "Xcal.csv").exists()
        assert (path / "Ycal.csv").exists()

    def test_builder_export_to_csv(self, tmp_path):
        """Test export_to_csv method on builder."""
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        builder = SyntheticDatasetBuilder(n_samples=100, random_state=42)
        builder.with_features(complexity="simple")

        path = builder.export_to_csv(tmp_path / "data.csv")

        assert path.exists()


class TestGenerateExportFunctions:
    """Tests for generate.to_folder and generate.to_csv."""

    def test_generate_to_folder(self, tmp_path):
        """Test nirs4all.generate.to_folder."""
        from nirs4all.api.generate import to_folder

        path = to_folder(
            tmp_path / "gen_folder",
            n_samples=50,
            random_state=42,
            complexity="simple",
        )

        assert path.exists()
        assert (path / "Xcal.csv").exists()

    def test_generate_to_csv(self, tmp_path):
        """Test nirs4all.generate.to_csv."""
        from nirs4all.api.generate import to_csv

        path = to_csv(
            tmp_path / "gen_data.csv",
            n_samples=50,
            random_state=42,
            complexity="simple",
        )

        assert path.exists()
