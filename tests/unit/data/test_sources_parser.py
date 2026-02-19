"""
Tests for the SourcesParser and multi-source configuration (Phase 6).

Tests the parser classes and schema models for multi-source datasets.
"""

import tempfile
from pathlib import Path

import pytest

from nirs4all.data.parsers import (
    ConfigNormalizer,
    SourcesParser,
    normalize_config,
)
from nirs4all.data.schema import (
    DatasetConfigSchema,
    LoadingParams,
    PartitionType,
    SharedMetadataConfig,
    SharedTargetsConfig,
    SourceConfig,
    SourceFileConfig,
)


class TestSourceConfig:
    """Test suite for SourceConfig schema model."""

    def test_source_with_train_x(self):
        """Test source with direct train_x path."""
        source = SourceConfig(
            name="NIR",
            train_x="data/NIR_train.csv"
        )
        assert source.name == "NIR"
        assert source.train_x == "data/NIR_train.csv"
        assert source.get_train_paths() == ["data/NIR_train.csv"]

    def test_source_with_test_x(self):
        """Test source with direct test_x path."""
        source = SourceConfig(
            name="NIR",
            test_x="data/NIR_test.csv"
        )
        assert source.name == "NIR"
        assert source.test_x == "data/NIR_test.csv"
        assert source.get_test_paths() == ["data/NIR_test.csv"]

    def test_source_with_both_paths(self):
        """Test source with both train and test paths."""
        source = SourceConfig(
            name="NIR",
            train_x="data/NIR_train.csv",
            test_x="data/NIR_test.csv"
        )
        assert source.get_train_paths() == ["data/NIR_train.csv"]
        assert source.get_test_paths() == ["data/NIR_test.csv"]

    def test_source_with_files_list(self):
        """Test source with files list instead of direct paths."""
        source = SourceConfig(
            name="NIR",
            files=[
                {"path": "data/NIR_train.csv", "partition": "train"},
                {"path": "data/NIR_test.csv", "partition": "test"},
            ]
        )
        assert len(source.files) == 2
        assert source.get_train_paths() == ["data/NIR_train.csv"]
        assert source.get_test_paths() == ["data/NIR_test.csv"]

    def test_source_with_string_files(self):
        """Test source with simple string file paths."""
        source = SourceConfig(
            name="NIR",
            files=["data/Xcal.csv", "data/Xval.csv"]
        )
        # Should infer partition from path
        assert source.get_train_paths() == ["data/Xcal.csv"]
        assert source.get_test_paths() == ["data/Xval.csv"]

    def test_source_with_params(self):
        """Test source with loading parameters."""
        source = SourceConfig(
            name="NIR",
            train_x="data/NIR_train.csv",
            params=LoadingParams(
                header_unit="nm",
                signal_type="absorbance"
            )
        )
        assert source.params.header_unit.value == "nm"
        assert source.params.signal_type.value == "absorbance"

    def test_source_with_link_by(self):
        """Test source with link_by column."""
        source = SourceConfig(
            name="NIR",
            train_x="data/NIR_train.csv",
            link_by="sample_id"
        )
        assert source.link_by == "sample_id"

    def test_source_requires_data(self):
        """Test that source requires either files or direct paths."""
        with pytest.raises(ValueError, match="must have either"):
            SourceConfig(name="empty_source")

    def test_source_cannot_have_both_formats(self):
        """Test that source cannot have both files and direct paths."""
        with pytest.raises(ValueError, match="cannot have both"):
            SourceConfig(
                name="conflict",
                files=["data/file.csv"],
                train_x="data/other.csv"
            )

class TestSourceFileConfig:
    """Test suite for SourceFileConfig schema model."""

    def test_simple_file_config(self):
        """Test simple file configuration."""
        file = SourceFileConfig(path="data/file.csv")
        assert file.path == "data/file.csv"
        assert file.partition is None

    def test_file_config_with_partition(self):
        """Test file configuration with partition."""
        file = SourceFileConfig(
            path="data/file.csv",
            partition=PartitionType.TRAIN
        )
        assert file.partition == PartitionType.TRAIN

    def test_file_config_with_params(self):
        """Test file configuration with loading params."""
        file = SourceFileConfig(
            path="data/file.csv",
            params=LoadingParams(delimiter=";")
        )
        assert file.params.delimiter == ";"

class TestSharedTargetsConfig:
    """Test suite for SharedTargetsConfig schema model."""

    def test_simple_targets_config(self):
        """Test simple targets configuration."""
        targets = SharedTargetsConfig(path="data/targets.csv")
        assert targets.path == "data/targets.csv"

    def test_targets_with_link_by(self):
        """Test targets configuration with link_by."""
        targets = SharedTargetsConfig(
            path="data/targets.csv",
            link_by="sample_id"
        )
        assert targets.link_by == "sample_id"

    def test_targets_with_columns(self):
        """Test targets with column selection."""
        targets = SharedTargetsConfig(
            path="data/targets.csv",
            columns=[0, 1]
        )
        assert targets.columns == [0, 1]

    def test_targets_with_partition(self):
        """Test targets for specific partition."""
        targets = SharedTargetsConfig(
            path="data/train_targets.csv",
            partition=PartitionType.TRAIN
        )
        assert targets.partition == PartitionType.TRAIN

class TestSharedMetadataConfig:
    """Test suite for SharedMetadataConfig schema model."""

    def test_simple_metadata_config(self):
        """Test simple metadata configuration."""
        metadata = SharedMetadataConfig(path="data/metadata.csv")
        assert metadata.path == "data/metadata.csv"

    def test_metadata_with_link_by(self):
        """Test metadata configuration with link_by."""
        metadata = SharedMetadataConfig(
            path="data/metadata.csv",
            link_by="sample_id"
        )
        assert metadata.link_by == "sample_id"

class TestSourcesParser:
    """Test suite for SourcesParser."""

    def test_can_parse_sources_syntax(self):
        """Test that parser recognizes sources syntax."""
        parser = SourcesParser()
        config = {
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"}
            ]
        }
        assert parser.can_parse(config) is True

    def test_cannot_parse_empty_sources(self):
        """Test that parser rejects empty sources list."""
        parser = SourcesParser()
        config = {"sources": []}
        assert parser.can_parse(config) is False

    def test_cannot_parse_no_sources(self):
        """Test that parser rejects config without sources."""
        parser = SourcesParser()
        config = {"train_x": "X.csv"}
        assert parser.can_parse(config) is False

    def test_parse_single_source(self):
        """Test parsing single source configuration."""
        parser = SourcesParser()
        config = {
            "name": "test_dataset",
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"}
            ]
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.source_type == "sources"
        assert result.config is not None
        assert len(result.config.sources) == 1
        assert result.config.sources[0].name == "NIR"

    def test_parse_multiple_sources(self):
        """Test parsing multiple source configuration."""
        parser = SourcesParser()
        config = {
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"},
                {"name": "MIR", "train_x": "data/MIR_train.csv"},
            ]
        }

        result = parser.parse(config)

        assert result.success is True
        assert len(result.config.sources) == 2
        assert result.config.sources[0].name == "NIR"
        assert result.config.sources[1].name == "MIR"

    def test_parse_with_shared_targets(self):
        """Test parsing sources with shared targets."""
        parser = SourcesParser()
        config = {
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"},
            ],
            "targets": {
                "path": "data/targets.csv",
                "link_by": "sample_id"
            }
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.config.shared_targets is not None
        assert result.config.shared_targets.path == "data/targets.csv"
        assert result.config.shared_targets.link_by == "sample_id"

    def test_parse_with_string_targets(self):
        """Test parsing sources with simple string targets path."""
        parser = SourcesParser()
        config = {
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"},
            ],
            "targets": "data/targets.csv"
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.config.shared_targets.path == "data/targets.csv"

    def test_parse_with_shared_metadata(self):
        """Test parsing sources with shared metadata."""
        parser = SourcesParser()
        config = {
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"},
            ],
            "metadata": {
                "path": "data/metadata.csv",
                "link_by": "sample_id"
            }
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.config.shared_metadata is not None
        assert result.config.shared_metadata.path == "data/metadata.csv"

    def test_parse_with_global_params(self):
        """Test parsing sources with global parameters."""
        parser = SourcesParser()
        config = {
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"},
            ],
            "global_params": {
                "delimiter": ";",
                "has_header": True
            }
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.config.global_params is not None
        assert result.config.global_params.delimiter == ";"

    def test_parse_source_with_per_source_params(self):
        """Test parsing sources with per-source parameters."""
        parser = SourcesParser()
        config = {
            "sources": [
                {
                    "name": "NIR",
                    "train_x": "data/NIR_train.csv",
                    "params": {
                        "header_unit": "nm",
                        "signal_type": "absorbance"
                    }
                },
                {
                    "name": "MIR",
                    "train_x": "data/MIR_train.csv",
                    "params": {
                        "header_unit": "cm-1",
                        "signal_type": "absorbance"
                    }
                },
            ]
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.config.sources[0].params.header_unit.value == "nm"
        assert result.config.sources[1].params.header_unit.value == "cm-1"

    def test_parse_source_with_files_list(self):
        """Test parsing source with files list."""
        parser = SourcesParser()
        config = {
            "sources": [
                {
                    "name": "NIR",
                    "files": [
                        {"path": "data/NIR_train.csv", "partition": "train"},
                        {"path": "data/NIR_test.csv", "partition": "test"},
                    ]
                }
            ]
        }

        result = parser.parse(config)

        assert result.success is True
        assert len(result.config.sources[0].files) == 2

    def test_parse_fails_duplicate_source_names(self):
        """Test that duplicate source names cause error."""
        parser = SourcesParser()
        config = {
            "sources": [
                {"name": "NIR", "train_x": "data/NIR1_train.csv"},
                {"name": "NIR", "train_x": "data/NIR2_train.csv"},
            ]
        }

        result = parser.parse(config)

        assert result.success is False
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_parse_auto_names_sources(self):
        """Test that sources without names get auto-named."""
        parser = SourcesParser()
        config = {
            "sources": [
                {"train_x": "data/source1.csv"},
                {"train_x": "data/source2.csv"},
            ]
        }

        result = parser.parse(config)

        assert result.success is True
        assert result.config.sources[0].name == "source_0"
        assert result.config.sources[1].name == "source_1"

class TestDatasetConfigSchemaSourcesMethods:
    """Test suite for DatasetConfigSchema sources-related methods."""

    def test_is_sources_format(self):
        """Test is_sources_format method."""
        config = DatasetConfigSchema(
            sources=[
                SourceConfig(name="NIR", train_x="data/NIR_train.csv")
            ]
        )
        assert config.is_sources_format() is True

    def test_is_not_sources_format(self):
        """Test is_sources_format with legacy config."""
        config = DatasetConfigSchema(train_x="data/train.csv")
        assert config.is_sources_format() is False

    def test_is_multi_source_with_sources(self):
        """Test is_multi_source with sources format."""
        config = DatasetConfigSchema(
            sources=[
                SourceConfig(name="NIR", train_x="data/NIR_train.csv"),
                SourceConfig(name="MIR", train_x="data/MIR_train.csv"),
            ]
        )
        assert config.is_multi_source() is True

    def test_is_multi_source_single_source(self):
        """Test is_multi_source with single source."""
        config = DatasetConfigSchema(
            sources=[
                SourceConfig(name="NIR", train_x="data/NIR_train.csv"),
            ]
        )
        assert config.is_multi_source() is False

    def test_get_source_names(self):
        """Test get_source_names method."""
        config = DatasetConfigSchema(
            sources=[
                SourceConfig(name="NIR", train_x="data/NIR_train.csv"),
                SourceConfig(name="MIR", train_x="data/MIR_train.csv"),
            ]
        )
        assert config.get_source_names() == ["NIR", "MIR"]

    def test_get_source_count(self):
        """Test get_source_count method."""
        config = DatasetConfigSchema(
            sources=[
                SourceConfig(name="NIR", train_x="data/NIR_train.csv"),
                SourceConfig(name="MIR", train_x="data/MIR_train.csv"),
            ]
        )
        assert config.get_source_count() == 2

    def test_to_legacy_format(self):
        """Test conversion to legacy format."""
        config = DatasetConfigSchema(
            name="multi_source_test",
            sources=[
                SourceConfig(
                    name="NIR",
                    train_x="data/NIR_train.csv",
                    test_x="data/NIR_test.csv"
                ),
                SourceConfig(
                    name="MIR",
                    train_x="data/MIR_train.csv",
                    test_x="data/MIR_test.csv"
                ),
            ],
            shared_targets=SharedTargetsConfig(path="data/targets.csv"),
        )

        legacy = config.to_legacy_format()

        assert legacy['name'] == "multi_source_test"
        assert legacy['train_x'] == ["data/NIR_train.csv", "data/MIR_train.csv"]
        assert legacy['test_x'] == ["data/NIR_test.csv", "data/MIR_test.csv"]
        assert legacy['train_y'] == "data/targets.csv"
        assert legacy['test_y'] == "data/targets.csv"
        assert '_sources' in legacy
        assert len(legacy['_sources']) == 2

    def test_to_legacy_format_single_source(self):
        """Test conversion with single source returns single path, not list."""
        config = DatasetConfigSchema(
            sources=[
                SourceConfig(name="NIR", train_x="data/NIR_train.csv"),
            ]
        )

        legacy = config.to_legacy_format()

        # Single source should not be wrapped in list
        assert legacy['train_x'] == "data/NIR_train.csv"

class TestConfigNormalizerWithSources:
    """Test suite for ConfigNormalizer with sources format."""

    def test_normalize_sources_config(self):
        """Test normalizing sources configuration."""
        normalizer = ConfigNormalizer()
        config = {
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"},
                {"name": "MIR", "train_x": "data/MIR_train.csv"},
            ]
        }

        result, name = normalizer.normalize(config)

        # Should return legacy format
        assert result is not None
        assert 'train_x' in result
        assert isinstance(result['train_x'], list)
        assert len(result['train_x']) == 2

    def test_normalize_sources_with_targets(self):
        """Test normalizing sources with shared targets."""
        normalizer = ConfigNormalizer()
        config = {
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"},
            ],
            "targets": "data/targets.csv"
        }

        result, name = normalizer.normalize(config)

        assert result is not None
        assert result['train_y'] == "data/targets.csv"

    def test_normalize_sources_with_metadata(self):
        """Test normalizing sources with shared metadata."""
        normalizer = ConfigNormalizer()
        config = {
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"},
            ],
            "metadata": "data/metadata.csv"
        }

        result, name = normalizer.normalize(config)

        assert result is not None
        assert result['train_group'] == "data/metadata.csv"

    def test_normalize_extracts_dataset_name(self):
        """Test that dataset name is extracted from sources config."""
        normalizer = ConfigNormalizer()
        config = {
            "name": "my_multisource_dataset",
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"},
            ]
        }

        result, name = normalizer.normalize(config)

        assert name == "my_multisource_dataset"

    def test_normalize_generates_name_from_sources(self):
        """Test that name is generated if not specified."""
        normalizer = ConfigNormalizer()
        config = {
            "sources": [
                {"name": "NIR", "train_x": "data/NIR_train.csv"},
            ]
        }

        result, name = normalizer.normalize(config)

        assert "NIR" in name or "multisource" in name.lower()

class TestSourcesParserIntegration:
    """Integration tests for sources parser with file creation."""

    @pytest.fixture
    def sample_multisource_folder(self, tmp_path):
        """Create a sample multi-source folder with data files."""
        # Create NIR files
        (tmp_path / "NIR_train.csv").write_text("1,2,3\n4,5,6")
        (tmp_path / "NIR_test.csv").write_text("7,8,9")
        # Create MIR files
        (tmp_path / "MIR_train.csv").write_text("10,20,30\n40,50,60")
        (tmp_path / "MIR_test.csv").write_text("70,80,90")
        # Create shared targets
        (tmp_path / "targets.csv").write_text("1\n2")
        # Create shared metadata
        (tmp_path / "metadata.csv").write_text("sample_id,group\nA,1\nB,2")

        return tmp_path

    def test_normalize_with_absolute_paths(self, sample_multisource_folder):
        """Test normalizing sources config with absolute paths."""
        normalizer = ConfigNormalizer()
        config = {
            "name": "integration_test",
            "sources": [
                {
                    "name": "NIR",
                    "train_x": str(sample_multisource_folder / "NIR_train.csv"),
                    "test_x": str(sample_multisource_folder / "NIR_test.csv"),
                },
                {
                    "name": "MIR",
                    "train_x": str(sample_multisource_folder / "MIR_train.csv"),
                    "test_x": str(sample_multisource_folder / "MIR_test.csv"),
                },
            ],
            "targets": str(sample_multisource_folder / "targets.csv"),
            "metadata": str(sample_multisource_folder / "metadata.csv"),
        }

        result, name = normalizer.normalize(config)

        assert result is not None
        assert name == "integration_test"
        assert len(result['train_x']) == 2
        assert len(result['test_x']) == 2
        assert result['train_y'] == str(sample_multisource_folder / "targets.csv")
        assert result['train_group'] == str(sample_multisource_folder / "metadata.csv")
