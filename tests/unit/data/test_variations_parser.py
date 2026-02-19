"""
Tests for the VariationsParser and feature variations configuration (Phase 7).

Tests the parser classes and schema models for feature variation datasets.
"""

import tempfile
from pathlib import Path

import pytest

from nirs4all.data.parsers import (
    ConfigNormalizer,
    VariationsParser,
    normalize_config,
)
from nirs4all.data.schema import (
    DatasetConfigSchema,
    LoadingParams,
    PartitionType,
    PreprocessingApplied,
    SharedMetadataConfig,
    SharedTargetsConfig,
    VariationConfig,
    VariationFileConfig,
    VariationMode,
)


class TestVariationConfig:
    """Test suite for VariationConfig schema model."""

    def test_variation_with_train_x(self):
        """Test variation with direct train_x path."""
        variation = VariationConfig(
            name="raw",
            train_x="data/X_raw_train.csv"
        )
        assert variation.name == "raw"
        assert variation.train_x == "data/X_raw_train.csv"
        assert variation.get_train_paths() == ["data/X_raw_train.csv"]

    def test_variation_with_test_x(self):
        """Test variation with direct test_x path."""
        variation = VariationConfig(
            name="raw",
            test_x="data/X_raw_test.csv"
        )
        assert variation.name == "raw"
        assert variation.test_x == "data/X_raw_test.csv"
        assert variation.get_test_paths() == ["data/X_raw_test.csv"]

    def test_variation_with_both_paths(self):
        """Test variation with both train and test paths."""
        variation = VariationConfig(
            name="snv",
            train_x="data/X_snv_train.csv",
            test_x="data/X_snv_test.csv"
        )
        assert variation.get_train_paths() == ["data/X_snv_train.csv"]
        assert variation.get_test_paths() == ["data/X_snv_test.csv"]

    def test_variation_with_files_list(self):
        """Test variation with files list instead of direct paths."""
        variation = VariationConfig(
            name="derivative",
            files=[
                {"path": "data/X_deriv_train.csv", "partition": "train"},
                {"path": "data/X_deriv_test.csv", "partition": "test"},
            ]
        )
        assert len(variation.files) == 2
        assert variation.get_train_paths() == ["data/X_deriv_train.csv"]
        assert variation.get_test_paths() == ["data/X_deriv_test.csv"]

    def test_variation_with_string_files(self):
        """Test variation with simple string file paths."""
        variation = VariationConfig(
            name="msc",
            files=["data/Xcal_msc.csv", "data/Xval_msc.csv"]
        )
        # Should infer partition from path
        assert variation.get_train_paths() == ["data/Xcal_msc.csv"]
        assert variation.get_test_paths() == ["data/Xval_msc.csv"]

    def test_variation_with_params(self):
        """Test variation with loading parameters."""
        variation = VariationConfig(
            name="raw",
            train_x="data/X_raw.csv",
            params=LoadingParams(
                header_unit="nm",
                signal_type="absorbance"
            )
        )
        assert variation.params.header_unit.value == "nm"
        assert variation.params.signal_type.value == "absorbance"

    def test_variation_with_description(self):
        """Test variation with description."""
        variation = VariationConfig(
            name="snv",
            description="Standard Normal Variate preprocessed spectra",
            train_x="data/X_snv.csv"
        )
        assert variation.description == "Standard Normal Variate preprocessed spectra"

    def test_variation_with_preprocessing_applied(self):
        """Test variation with preprocessing provenance."""
        variation = VariationConfig(
            name="snv_sg",
            description="SNV followed by SG derivative",
            train_x="data/X_snv_sg.csv",
            preprocessing_applied=[
                PreprocessingApplied(
                    type="SNV",
                    description="Standard Normal Variate",
                    software="OPUS 8.0"
                ),
                PreprocessingApplied(
                    type="SG_derivative",
                    params={"window": 15, "polyorder": 2, "deriv": 1}
                )
            ]
        )
        assert len(variation.preprocessing_applied) == 2
        assert variation.preprocessing_applied[0].type == "SNV"
        assert variation.preprocessing_applied[1].params["window"] == 15

    def test_variation_requires_data(self):
        """Test that variation requires either files or direct paths."""
        with pytest.raises(ValueError, match="must have either"):
            VariationConfig(name="empty_variation")

    def test_variation_cannot_have_both_formats(self):
        """Test that variation cannot have both files and direct paths."""
        with pytest.raises(ValueError, match="cannot have both"):
            VariationConfig(
                name="conflict",
                files=["data/file.csv"],
                train_x="data/other.csv"
            )

class TestVariationFileConfig:
    """Test suite for VariationFileConfig schema model."""

    def test_simple_file_config(self):
        """Test simple file configuration."""
        file = VariationFileConfig(path="data/file.csv")
        assert file.path == "data/file.csv"
        assert file.partition is None

    def test_file_config_with_partition(self):
        """Test file configuration with partition."""
        file = VariationFileConfig(
            path="data/file.csv",
            partition=PartitionType.TRAIN
        )
        assert file.partition == PartitionType.TRAIN

    def test_file_config_with_params(self):
        """Test file configuration with loading params."""
        file = VariationFileConfig(
            path="data/file.csv",
            params=LoadingParams(delimiter=";")
        )
        assert file.params.delimiter == ";"

class TestPreprocessingApplied:
    """Test suite for PreprocessingApplied schema model."""

    def test_simple_preprocessing(self):
        """Test simple preprocessing metadata."""
        preprocessing = PreprocessingApplied(type="SNV")
        assert preprocessing.type == "SNV"

    def test_preprocessing_with_description(self):
        """Test preprocessing with description."""
        preprocessing = PreprocessingApplied(
            type="MSC",
            description="Multiplicative Scatter Correction"
        )
        assert preprocessing.description == "Multiplicative Scatter Correction"

    def test_preprocessing_with_software(self):
        """Test preprocessing with software info."""
        preprocessing = PreprocessingApplied(
            type="Baseline",
            software="OPUS 8.0"
        )
        assert preprocessing.software == "OPUS 8.0"

    def test_preprocessing_with_params(self):
        """Test preprocessing with parameters."""
        preprocessing = PreprocessingApplied(
            type="SG_smooth",
            params={"window": 15, "polyorder": 2}
        )
        assert preprocessing.params["window"] == 15
        assert preprocessing.params["polyorder"] == 2

class TestVariationMode:
    """Test suite for VariationMode enum."""

    def test_variation_modes(self):
        """Test all variation modes."""
        assert VariationMode.SEPARATE.value == "separate"
        assert VariationMode.CONCAT.value == "concat"
        assert VariationMode.SELECT.value == "select"
        assert VariationMode.COMPARE.value == "compare"

class TestVariationsParser:
    """Test suite for VariationsParser."""

    def test_can_parse_variations_syntax(self):
        """Test that parser recognizes variations syntax."""
        parser = VariationsParser()

        # Should parse variations syntax
        assert parser.can_parse({
            "variations": [
                {"name": "raw", "train_x": "X_raw.csv"}
            ]
        }) is True

        # Should not parse non-variations syntax
        assert parser.can_parse({
            "train_x": "X.csv"
        }) is False

        assert parser.can_parse({
            "variations": []  # Empty list
        }) is False

        assert parser.can_parse("path/to/folder") is False

    def test_parse_single_variation(self):
        """Test parsing configuration with single variation."""
        parser = VariationsParser()
        result = parser.parse({
            "name": "test_dataset",
            "variations": [
                {
                    "name": "raw",
                    "train_x": "data/X_raw_train.csv",
                    "test_x": "data/X_raw_test.csv"
                }
            ]
        })

        assert result.success is True
        assert result.source_type == "variations"
        assert result.config is not None
        assert len(result.config.variations) == 1
        assert result.config.variations[0].name == "raw"

    def test_parse_multiple_variations(self):
        """Test parsing configuration with multiple variations."""
        parser = VariationsParser()
        result = parser.parse({
            "name": "spectral_study",
            "variations": [
                {"name": "raw", "train_x": "X_raw.csv"},
                {"name": "snv", "train_x": "X_snv.csv"},
                {"name": "derivative", "train_x": "X_deriv.csv"}
            ],
            "variation_mode": "separate"
        })

        assert result.success is True
        assert len(result.config.variations) == 3
        assert result.config.variation_mode == VariationMode.SEPARATE

    def test_parse_with_variation_select(self):
        """Test parsing with variation_select for mode=select."""
        parser = VariationsParser()
        result = parser.parse({
            "variations": [
                {"name": "raw", "train_x": "X_raw.csv"},
                {"name": "snv", "train_x": "X_snv.csv"},
                {"name": "derivative", "train_x": "X_deriv.csv"}
            ],
            "variation_mode": "select",
            "variation_select": ["raw", "snv"]
        })

        assert result.success is True
        assert result.config.variation_mode == VariationMode.SELECT
        assert result.config.variation_select == ["raw", "snv"]

    def test_parse_with_concat_mode(self):
        """Test parsing with concat mode."""
        parser = VariationsParser()
        result = parser.parse({
            "variations": [
                {"name": "temp", "train_x": "X_temp.csv"},
                {"name": "humidity", "train_x": "X_humidity.csv"}
            ],
            "variation_mode": "concat",
            "variation_prefix": True
        })

        assert result.success is True
        assert result.config.variation_mode == VariationMode.CONCAT
        assert result.config.variation_prefix is True

    def test_parse_with_shared_targets(self):
        """Test parsing with shared targets."""
        parser = VariationsParser()
        result = parser.parse({
            "variations": [
                {"name": "raw", "train_x": "X_raw.csv"}
            ],
            "targets": {
                "path": "data/Y.csv",
                "link_by": "sample_id"
            }
        })

        assert result.success is True
        assert result.config.shared_targets is not None
        assert result.config.shared_targets.path == "data/Y.csv"
        assert result.config.shared_targets.link_by == "sample_id"

    def test_parse_with_preprocessing_provenance(self):
        """Test parsing variations with preprocessing provenance."""
        parser = VariationsParser()
        result = parser.parse({
            "variations": [
                {
                    "name": "snv_sg",
                    "description": "SNV followed by SG derivative",
                    "train_x": "X_snv_sg.csv",
                    "preprocessing_applied": [
                        {
                            "type": "SNV",
                            "software": "OPUS 8.0"
                        },
                        {
                            "type": "SG_derivative",
                            "params": {"window": 15, "polyorder": 2}
                        }
                    ]
                }
            ]
        })

        assert result.success is True
        var = result.config.variations[0]
        assert var.description == "SNV followed by SG derivative"
        assert len(var.preprocessing_applied) == 2
        assert var.preprocessing_applied[0].type == "SNV"

    def test_parse_duplicate_names_error(self):
        """Test that duplicate variation names cause an error."""
        parser = VariationsParser()
        result = parser.parse({
            "variations": [
                {"name": "raw", "train_x": "X1.csv"},
                {"name": "raw", "train_x": "X2.csv"}  # Duplicate
            ]
        })

        assert result.success is False
        assert any("Duplicate" in e for e in result.errors)

    def test_parse_with_global_params(self):
        """Test that global_params are merged with variation params."""
        parser = VariationsParser()
        result = parser.parse({
            "variations": [
                {"name": "raw", "train_x": "X.csv"}
            ],
            "global_params": {
                "delimiter": ";",
                "header_unit": "nm"
            }
        })

        assert result.success is True
        var = result.config.variations[0]
        assert var.params.delimiter == ";"
        assert var.params.header_unit.value == "nm"

class TestDatasetConfigSchemaVariations:
    """Test suite for DatasetConfigSchema variation methods."""

    def test_is_variations_format(self):
        """Test is_variations_format method."""
        schema = DatasetConfigSchema(
            variations=[
                VariationConfig(name="raw", train_x="X.csv")
            ]
        )
        assert schema.is_variations_format() is True

    def test_is_not_variations_format(self):
        """Test is_variations_format returns False for legacy config."""
        schema = DatasetConfigSchema(
            train_x="X.csv"
        )
        assert schema.is_variations_format() is False

    def test_get_variation_names(self):
        """Test get_variation_names method."""
        schema = DatasetConfigSchema(
            variations=[
                VariationConfig(name="raw", train_x="X_raw.csv"),
                VariationConfig(name="snv", train_x="X_snv.csv"),
            ]
        )
        assert schema.get_variation_names() == ["raw", "snv"]

    def test_get_variation_count(self):
        """Test get_variation_count method."""
        schema = DatasetConfigSchema(
            variations=[
                VariationConfig(name="raw", train_x="X_raw.csv"),
                VariationConfig(name="snv", train_x="X_snv.csv"),
                VariationConfig(name="deriv", train_x="X_deriv.csv"),
            ]
        )
        assert schema.get_variation_count() == 3

    def test_get_selected_variations_all(self):
        """Test get_selected_variations returns all for non-select mode."""
        schema = DatasetConfigSchema(
            variations=[
                VariationConfig(name="raw", train_x="X_raw.csv"),
                VariationConfig(name="snv", train_x="X_snv.csv"),
            ],
            variation_mode=VariationMode.SEPARATE
        )
        selected = schema.get_selected_variations()
        assert len(selected) == 2

    def test_get_selected_variations_select_mode(self):
        """Test get_selected_variations filters for select mode."""
        schema = DatasetConfigSchema(
            variations=[
                VariationConfig(name="raw", train_x="X_raw.csv"),
                VariationConfig(name="snv", train_x="X_snv.csv"),
                VariationConfig(name="deriv", train_x="X_deriv.csv"),
            ],
            variation_mode=VariationMode.SELECT,
            variation_select=["raw", "deriv"]
        )
        selected = schema.get_selected_variations()
        assert len(selected) == 2
        assert selected[0].name == "raw"
        assert selected[1].name == "deriv"

    def test_variations_to_legacy_format_separate(self):
        """Test variations_to_legacy_format for separate mode."""
        schema = DatasetConfigSchema(
            name="test_dataset",
            variations=[
                VariationConfig(name="raw", train_x="X_raw.csv", test_x="X_raw_test.csv"),
                VariationConfig(name="snv", train_x="X_snv.csv"),
            ],
            variation_mode=VariationMode.SEPARATE
        )
        legacy = schema.variations_to_legacy_format()

        # Should only include first variation for separate mode
        assert legacy["train_x"] == "X_raw.csv"
        assert legacy["test_x"] == "X_raw_test.csv"
        assert "_variations" in legacy
        assert len(legacy["_variations"]) == 2

    def test_variations_to_legacy_format_concat(self):
        """Test variations_to_legacy_format for concat mode."""
        schema = DatasetConfigSchema(
            name="concat_dataset",
            variations=[
                VariationConfig(name="temp", train_x="X_temp.csv"),
                VariationConfig(name="humidity", train_x="X_humidity.csv"),
            ],
            variation_mode=VariationMode.CONCAT
        )
        legacy = schema.variations_to_legacy_format()

        # Should include all variations concatenated
        assert legacy["train_x"] == ["X_temp.csv", "X_humidity.csv"]
        assert legacy["_variation_mode"] == "concat"

    def test_validation_select_mode_requires_variation_select(self):
        """Test that select mode requires variation_select."""
        with pytest.raises(ValueError, match="variation_select"):
            DatasetConfigSchema(
                variations=[
                    VariationConfig(name="raw", train_x="X.csv")
                ],
                variation_mode=VariationMode.SELECT
                # Missing variation_select
            )

    def test_validation_invalid_variation_select(self):
        """Test that invalid variation names in variation_select are rejected."""
        with pytest.raises(ValueError, match="unknown variation"):
            DatasetConfigSchema(
                variations=[
                    VariationConfig(name="raw", train_x="X.csv")
                ],
                variation_mode=VariationMode.SELECT,
                variation_select=["raw", "nonexistent"]  # "nonexistent" doesn't exist
            )

class TestConfigNormalizerVariations:
    """Test ConfigNormalizer with variations format."""

    def test_normalize_variations_config(self):
        """Test that ConfigNormalizer handles variations format."""
        normalizer = ConfigNormalizer()
        config, name = normalizer.normalize({
            "name": "variation_test",
            "variations": [
                {"name": "raw", "train_x": "X_raw.csv"},
                {"name": "snv", "train_x": "X_snv.csv"}
            ],
            "variation_mode": "separate"
        })

        assert config is not None
        assert name == "variation_test"
        # Should be converted to legacy format
        assert "train_x" in config
        assert "_variations" in config

    def test_normalize_variations_concat_mode(self):
        """Test normalization of concat mode variations."""
        normalizer = ConfigNormalizer()
        config, name = normalizer.normalize({
            "variations": [
                {"name": "var1", "train_x": "X1.csv"},
                {"name": "var2", "train_x": "X2.csv"},
                {"name": "var3", "train_x": "X3.csv"}
            ],
            "variation_mode": "concat"
        })

        assert config is not None
        # All paths should be combined for concat
        assert config["train_x"] == ["X1.csv", "X2.csv", "X3.csv"]

    def test_normalize_variations_with_targets(self):
        """Test normalization of variations with shared targets."""
        normalizer = ConfigNormalizer()
        config, name = normalizer.normalize({
            "variations": [
                {"name": "raw", "train_x": "X.csv"}
            ],
            "targets": "Y.csv"
        })

        assert config is not None
        assert config.get("train_y") == "Y.csv"
        assert config.get("test_y") == "Y.csv"
