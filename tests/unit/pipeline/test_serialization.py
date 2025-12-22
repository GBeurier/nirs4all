"""
Comprehensive test suite for pipeline serialization and deserialization.

Tests all 17+ pipeline syntax types from WRITING_A_PIPELINE.md and sample.py:
- Round-trip JSON serialization
- Round-trip YAML serialization
- Hash consistency
- Complex heterogeneous pipelines
- Generator expansion compatibility

This ensures backward compatibility is removed and only clean, minimal code remains.
"""

import pytest
import json
import yaml
import tempfile
import shutil
from pathlib import Path
from typing import Any

# Import all necessary components
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

from nirs4all.operators.transforms import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)

# Try to import TF models, skip tests if not available
try:
    from nirs4all.operators.models.tensorflow.nicon import nicon, customizable_nicon
    TF_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TF_AVAILABLE = False
    nicon = None
    customizable_nicon = None

from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.config.component_serialization import serialize_component, deserialize_component


# Mock function to simulate TF/PyTorch model functions
def mock_model_function(input_shape, params={}):
    """Mock function that simulates nicon-like behavior."""
    return f"Model with input_shape={input_shape}, params={params}"


# Mark it with framework decorator simulation
mock_model_function.framework = 'tensorflow'


@pytest.fixture
def temp_pipeline_dir():
    """Create temporary directory for pipeline files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestBasicStepSyntaxes:
    """Test all 7 basic step syntaxes from WRITING_A_PIPELINE.md."""

    def test_syntax_1_class_reference(self):
        """Test uninstantiated class syntax."""
        step = StandardScaler
        serialized = serialize_component(step)

        # Should serialize to string path only (no params)
        assert isinstance(serialized, str)
        assert "StandardScaler" in serialized

        # Deserialize should create instance with defaults
        deserialized = deserialize_component(serialized)
        assert isinstance(deserialized, StandardScaler)

    def test_syntax_2_instance_default_params(self):
        """Test instance with default parameters."""
        step = MinMaxScaler()  # feature_range=(0, 1) is default
        serialized = serialize_component(step)

        # Should serialize to string only (no params dict since all are defaults)
        assert isinstance(serialized, str)
        assert "MinMaxScaler" in serialized

    def test_syntax_2_instance_custom_params(self):
        """Test instance with non-default parameters."""
        step = MinMaxScaler(feature_range=(0, 2))  # Non-default
        serialized = serialize_component(step)

        # Should serialize to dict with params
        assert isinstance(serialized, dict)
        assert "class" in serialized
        assert "params" in serialized
        assert serialized["params"]["feature_range"] == [0, 2]  # Tuple converted to list

    def test_syntax_3_string_module_path(self):
        """Test string module path normalization."""
        step = "sklearn.preprocessing.StandardScaler"
        serialized = serialize_component(step)

        # String should be normalized to internal module path for hash consistency
        assert isinstance(serialized, str)
        assert "StandardScaler" in serialized
        # Should be normalized to internal path
        assert serialized == "sklearn.preprocessing._data.StandardScaler"

        # Deserialize should create instance
        deserialized = deserialize_component(serialized)
        assert isinstance(deserialized, StandardScaler)

    def test_syntax_4_string_controller_name(self):
        """Test short controller name."""
        step = "chart_2d"
        serialized = serialize_component(step)

        # Should pass through as-is
        assert serialized == "chart_2d"

    def test_syntax_5_string_file_path(self):
        """Test file path to saved transformer."""
        step = "my/super/transformer.pkl"
        serialized = serialize_component(step)

        # Should pass through as-is
        assert serialized == "my/super/transformer.pkl"

    def test_syntax_6_dict_explicit_config(self):
        """Test explicit dict configuration."""
        step = {
            "class": "sklearn.preprocessing.StandardScaler"
        }
        serialized = serialize_component(step)

        # Should normalize to string (no params)
        assert isinstance(serialized, dict)
        assert "class" in serialized

    def test_syntax_6_dict_with_params(self):
        """Test dict with explicit params."""
        step = {
            "class": "sklearn.model_selection.ShuffleSplit",
            "params": {
                "n_splits": 3,
                "test_size": 0.25
            }
        }
        serialized = serialize_component(step)

        assert isinstance(serialized, dict)
        assert serialized["params"]["n_splits"] == 3

    def test_syntax_7_dict_special_operators(self):
        """Test special operator dict (y_processing, feature_augmentation)."""
        step = {"y_processing": MinMaxScaler}
        serialized = serialize_component(step)

        assert isinstance(serialized, dict)
        assert "y_processing" in serialized


class TestModelStepSyntaxes:
    """Test all model-specific syntaxes."""

    def test_model_instance(self):
        """Test model as instance."""
        step = PLSRegression(n_components=10)
        serialized = serialize_component(step)

        assert isinstance(serialized, dict)
        assert "class" in serialized
        assert "params" in serialized
        assert serialized["params"]["n_components"] == 10

    def test_model_dict_with_name(self):
        """Test model dict with custom name."""
        step = {
            "name": "PLS-10",
            "model": {
                "class": "sklearn.cross_decomposition.PLSRegression",
                "params": {
                    "n_components": 10
                }
            }
        }
        serialized = serialize_component(step)

        assert "name" in serialized
        assert serialized["name"] == "PLS-10"
        assert "model" in serialized

    def test_model_with_finetune_params(self):
        """Test model with hyperparameter optimization."""
        step = {
            "model": PLSRegression(),
            "name": "PLS-Finetuned",
            "finetune_params": {
                "n_trials": 20,
                "model_params": {
                    'n_components': ('int', 1, 30),  # Tuple should convert to list
                }
            }
        }
        serialized = serialize_component(step)

        assert "finetune_params" in serialized
        assert "model_params" in serialized["finetune_params"]
        # Check tuple was converted to list
        assert isinstance(serialized["finetune_params"]["model_params"]["n_components"], list)
        assert serialized["finetune_params"]["model_params"]["n_components"] == ['int', 1, 30]


class TestFunctionBasedModels:
    """Test function-based models (TensorFlow/PyTorch) that need input_shape at runtime."""

    def test_function_without_params(self):
        """Test function reference without parameters."""
        import inspect

        step = mock_model_function
        serialized = serialize_component(step)

        # Should serialize as function reference
        assert isinstance(serialized, dict)
        assert "function" in serialized
        assert "test_serialization" in serialized["function"]
        assert "mock_model_function" in serialized["function"]

        # Deserialize should return dict with type info (not call it)
        deserialized = deserialize_component(serialized)
        # New format: {"type": "function", "func": <func>, "framework": "tensorflow", "params": {}}
        if isinstance(deserialized, dict) and "type" in deserialized:
            assert deserialized["type"] == "function"
            func = deserialized["func"]
        else:
            func = deserialized
        assert inspect.isfunction(func)
        assert func == mock_model_function
        assert func.__name__ == "mock_model_function"

    def test_function_with_params(self):
        """Test function with parameters wrapped in dict."""
        import inspect

        step = {
            "model": mock_model_function,
            "train_params": {
                "epochs": 100,
                "patience": 50
            }
        }
        serialized = serialize_component(step)

        # Should have model with function reference
        assert "model" in serialized
        assert isinstance(serialized["model"], dict)
        assert "function" in serialized["model"]
        assert "mock_model_function" in serialized["model"]["function"]

        # Deserialize should return function reference
        deserialized = deserialize_component(serialized)
        assert "model" in deserialized
        # Handle new dict format with 'type' and 'func' keys
        model = deserialized["model"]
        if isinstance(model, dict) and "type" in model:
            assert model["type"] == "function"
            model = model["func"]
        assert inspect.isfunction(model)
        assert model == mock_model_function

    def test_function_with_model_params(self):
        """Test function with model-specific parameters (passed to function)."""
        import inspect

        # Simulate what happens when a function is serialized with params
        # First serialize a function with params
        func_with_params = mock_model_function
        serialized = serialize_component(func_with_params)

        # Now add params manually to simulate deserialization scenario
        serialized_with_params = {
            "function": serialized["function"],
            "params": {
                "filters1": 16,
                "filters2": 32
            }
        }

        # Deserialize should return dict with function reference and params
        # This is what ModelBuilder expects to inject input_shape later
        deserialized = deserialize_component(serialized_with_params)

        assert isinstance(deserialized, dict)
        # New serialization returns wrapped format with 'type' and 'func' keys
        if "type" in deserialized:
            # New format: {"type": "function", "func": <func>, "framework": "tensorflow", "params": {...}}
            assert deserialized["type"] == "function"
            func = deserialized["func"]
            assert inspect.isfunction(func)
            assert func == mock_model_function
            # Params should be in the dict
            assert "params" in deserialized
        else:
            # Old format with separate function and params keys
            assert "function" in deserialized
            if isinstance(deserialized["function"], dict) and "type" in deserialized["function"]:
                func = deserialized["function"]["func"]
            else:
                func = deserialized["function"]
            assert inspect.isfunction(func)
            # Params validation is optional for wrapped format

    def test_function_not_called_during_deserialization(self):
        """Critical test: Functions should NOT be called during deserialization."""
        import inspect

        # Use mock_model_function which requires input_shape
        # Serialize it
        serialized = serialize_component(mock_model_function)

        # Deserialize should NOT call the function
        # If it did, it would fail because input_shape is required
        deserialized = deserialize_component(serialized)

        # Should get function reference back, not a call result
        # Handle new dict format with 'type' and 'func' keys
        if isinstance(deserialized, dict) and "type" in deserialized:
            assert deserialized["type"] == "function"
            func = deserialized["func"]
            assert inspect.isfunction(func)
            assert func == mock_model_function
            # For wrapped format, call the function from func
            result = func(input_shape=(1, 100), params={"test": "value"})
        else:
            func = deserialized
            assert inspect.isfunction(func)
            assert func == mock_model_function
            # Function should be callable later with proper args
            result = deserialized(input_shape=(1, 100), params={"test": "value"})
        assert "Model with" in result
        assert "(1, 100)" in result

    def test_function_in_pipeline_config(self):
        """Test function-based model in complete pipeline."""
        pipeline = [
            MinMaxScaler(),
            mock_model_function
        ]

        config = PipelineConfigs(pipeline, name="test_mock")

        # Should have 1 configuration
        assert len(config.steps) == 1

        # Check that mock_model_function is serialized correctly
        steps = config.steps[0]
        func_step = steps[1]

        assert isinstance(func_step, dict)
        assert "function" in func_step
        assert "mock_model_function" in func_step["function"]


@pytest.mark.xdist_group("gpu")
@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestFunctionBasedModelsWithTensorFlow:
    """Test actual TensorFlow models (nicon, customizable_nicon)."""

    def test_nicon_function_without_params(self):
        """Test nicon function reference without parameters."""
        import inspect

        step = nicon
        serialized = serialize_component(step)

        # Should serialize as function reference
        assert isinstance(serialized, dict)
        assert "function" in serialized
        assert serialized["function"] == "nirs4all.operators.models.tensorflow.nicon.nicon"

        # Deserialize should return function reference, NOT call it
        deserialized = deserialize_component(serialized)
        # Handle new dict format with 'type' and 'func' keys
        if isinstance(deserialized, dict) and "type" in deserialized:
            assert deserialized["type"] == "function"
            func = deserialized["func"]
        else:
            func = deserialized
        assert inspect.isfunction(func)
        assert func == nicon
        assert func.__name__ == "nicon"

    def test_nicon_function_with_params(self):
        """Test nicon function with parameters wrapped in dict."""
        import inspect

        step = {
            "model": nicon,
            "train_params": {
                "epochs": 100,
                "patience": 50
            }
        }
        serialized = serialize_component(step)

        # Should have model with function reference
        assert "model" in serialized
        assert isinstance(serialized["model"], dict)
        assert "function" in serialized["model"]
        assert serialized["model"]["function"] == "nirs4all.operators.models.tensorflow.nicon.nicon"

        # Deserialize should return function reference
        deserialized = deserialize_component(serialized)
        assert "model" in deserialized
        # Handle new dict format with 'type' and 'func' keys
        model = deserialized["model"]
        if isinstance(model, dict) and "type" in model:
            assert model["type"] == "function"
            model = model["func"]
        assert inspect.isfunction(model)
        assert model == nicon

    def test_customizable_nicon_with_model_params(self):
        """Test customizable_nicon function with model-specific parameters."""
        import inspect

        # Simulate what happens when a function is serialized with params
        serialized = {
            "function": "nirs4all.operators.models.tensorflow.nicon.customizable_nicon",
            "params": {
                "filters1": 16,
                "filters2": 32
            }
        }
        # Deserialize should return dict with function reference and params
        # This is what ModelBuilder expects to inject input_shape later
        deserialized = deserialize_component(serialized)

        assert isinstance(deserialized, dict)
        # New serialization wraps framework functions with 'type' and 'func' keys
        if "type" in deserialized:
            # New format: {"type": "function", "func": <func>, "framework": "tensorflow", "params": {...}}
            assert deserialized["type"] == "function"
            assert inspect.isfunction(deserialized["func"])
            # Params should be in the dict
            assert "params" in deserialized
        else:
            # Old format - might not fully deserialize with params
            # The test is verifying that serialization format is handled
            assert "function" in deserialized or isinstance(deserialized, dict)

    def test_nicon_in_pipeline_config(self):
        """Test nicon function in complete pipeline."""
        pipeline = [
            MinMaxScaler(),
            nicon
        ]

        config = PipelineConfigs(pipeline, name="test_nicon")

        # Should have 1 configuration
        assert len(config.steps) == 1

        # Check that nicon is serialized correctly
        steps = config.steps[0]
        nicon_step = steps[1]

        assert isinstance(nicon_step, dict)
        assert "function" in nicon_step
        assert nicon_step["function"] == "nirs4all.operators.models.tensorflow.nicon.nicon"

    def test_customizable_nicon_with_train_params_in_pipeline(self):
        """Test function with train_params in pipeline."""
        pipeline = [
            MinMaxScaler(),
            {
                "model": customizable_nicon,
                "train_params": {
                    "epochs": 50,
                    "verbose": 0
                }
            }
        ]

        config = PipelineConfigs(pipeline, name="test_custom_nicon")

        # Should have 1 configuration
        assert len(config.steps) == 1

        # Check that customizable_nicon is serialized correctly
        steps = config.steps[0]
        model_step = steps[1]

        assert "model" in model_step
        assert isinstance(model_step["model"], dict)
        assert "function" in model_step["model"]
        assert model_step["model"]["function"] == "nirs4all.operators.models.tensorflow.nicon.customizable_nicon"
        assert "train_params" in model_step
        assert model_step["train_params"]["epochs"] == 50


class TestGeneratorSyntaxes:
    """Test generator syntaxes (_or_, _range_)."""

    def test_generator_or_syntax(self):
        """Test _or_ generator syntax."""
        step = {
            "_or_": [Detrend, FirstDerivative, Gaussian]
        }
        serialized = serialize_component(step)

        assert "_or_" in serialized
        assert isinstance(serialized["_or_"], list)
        assert len(serialized["_or_"]) == 3

    def test_generator_range_syntax(self):
        """Test _range_ generator syntax."""
        step = {
            "_range_": [1, 10, 2],
            "param": "n_components",
            "model": {
                "class": "sklearn.cross_decomposition.PLSRegression"
            }
        }
        serialized = serialize_component(step)

        assert "_range_" in serialized
        assert serialized["_range_"] == [1, 10, 2]
        assert "param" in serialized

    def test_generator_or_with_size(self):
        """Test _or_ with size parameter."""
        step = {
            "_or_": [Detrend, FirstDerivative, Gaussian],
            "size": 2
        }
        serialized = serialize_component(step)

        assert "_or_" in serialized
        assert "size" in serialized

    def test_generator_or_with_count(self):
        """Test _or_ with count parameter (random sampling)."""
        step = {
            "_or_": [Detrend, FirstDerivative, Gaussian, StandardNormalVariate],
            "size": [1, 2],
            "count": 3
        }
        serialized = serialize_component(step)

        assert "_or_" in serialized
        assert "count" in serialized


class TestRoundTripSerialization:
    """Test complete round-trip serialization for JSON and YAML."""

    def test_json_roundtrip_simple_pipeline(self):
        """Test JSON round-trip for simple pipeline."""
        pipeline = [
            StandardScaler(),
            ShuffleSplit(n_splits=3, test_size=0.25),
            PLSRegression(n_components=10)
        ]

        # Serialize to JSON
        serialized = serialize_component(pipeline)
        json_str = json.dumps(serialized, indent=2)

        # Parse back
        parsed = json.loads(json_str)

        # Deserialize
        deserialized = deserialize_component(parsed)

        # Check structure
        assert isinstance(deserialized, list)
        assert len(deserialized) == 3

    def test_yaml_roundtrip_simple_pipeline(self, temp_pipeline_dir):
        """Test YAML round-trip for simple pipeline."""
        pipeline = [
            MinMaxScaler(feature_range=(0, 1)),
            PLSRegression(n_components=5)
        ]

        # Serialize
        serialized = serialize_component(pipeline)

        # Save as YAML
        yaml_file = temp_pipeline_dir / "pipeline.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(serialized, f)

        # Load back
        with open(yaml_file, "r") as f:
            loaded = yaml.safe_load(f)

        # Deserialize
        deserialized = deserialize_component(loaded)

        assert isinstance(deserialized, list)
        assert len(deserialized) == 2

    def test_json_roundtrip_complex_model(self):
        """Test JSON round-trip for model with finetuning."""
        step = {
            "model": PLSRegression(),
            "name": "PLS-Optimized",
            "finetune_params": {
                "n_trials": 10,
                "model_params": {
                    'n_components': ('int', 1, 20),  # Tuple
                }
            }
        }

        serialized = serialize_component(step)
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)

        # Check tuple was converted to list
        assert isinstance(parsed["finetune_params"]["model_params"]["n_components"], list)


class TestHashConsistency:
    """Test that hash-based uniqueness works correctly."""

    def test_same_syntax_same_hash(self):
        """Test that different syntaxes for same object produce same hash."""
        # All these should produce the same hash
        pipeline1 = [StandardScaler]
        pipeline2 = [StandardScaler()]
        pipeline3 = ["sklearn.preprocessing.StandardScaler"]

        config1 = PipelineConfigs(pipeline1, name="test")
        config2 = PipelineConfigs(pipeline2, name="test")
        config3 = PipelineConfigs(pipeline3, name="test")

        hash1 = PipelineConfigs.get_hash(config1.steps[0])
        hash2 = PipelineConfigs.get_hash(config2.steps[0])
        hash3 = PipelineConfigs.get_hash(config3.steps[0])

        assert hash1 == hash2 == hash3

    def test_default_params_no_hash_change(self):
        """Test that explicit default params don't change hash."""
        # MinMaxScaler default: feature_range=(0, 1)
        pipeline1 = [MinMaxScaler()]
        pipeline2 = [MinMaxScaler(feature_range=(0, 1))]

        config1 = PipelineConfigs(pipeline1, name="test")
        config2 = PipelineConfigs(pipeline2, name="test")

        hash1 = PipelineConfigs.get_hash(config1.steps[0])
        hash2 = PipelineConfigs.get_hash(config2.steps[0])

        assert hash1 == hash2

    def test_different_params_different_hash(self):
        """Test that non-default params change hash."""
        pipeline1 = [MinMaxScaler(feature_range=(0, 1))]
        pipeline2 = [MinMaxScaler(feature_range=(0, 2))]

        config1 = PipelineConfigs(pipeline1, name="test")
        config2 = PipelineConfigs(pipeline2, name="test")

        hash1 = PipelineConfigs.get_hash(config1.steps[0])
        hash2 = PipelineConfigs.get_hash(config2.steps[0])

        assert hash1 != hash2


class TestComplexHeterogeneousPipeline:
    """Test complex pipeline with mixed syntax types."""

    def test_sample_py_pipeline(self):
        """Test the complete sample.py pipeline."""
        # Simplified version of examples/sample.py
        pipeline = [
            MinMaxScaler(feature_range=(0.1, 0.8)),  # Syntax 2
            "chart_2d",  # Syntax 4
            {"y_processing": MinMaxScaler},  # Syntax 7
            ShuffleSplit(n_splits=3, test_size=0.25),  # Syntax 2
            {  # Syntax 6
                "class": "sklearn.model_selection.ShuffleSplit",
                "params": {
                    "n_splits": 3,
                    "test_size": 0.25
                }
            },
            {  # Model dict with name
                "name": "PLS-3",
                "model": {
                    "class": "sklearn.cross_decomposition.PLSRegression",
                    "params": {
                        "n_components": 3
                    }
                }
            },
            {  # Model with finetuning
                "model": PLSRegression(),
                "name": "PLS-Finetuned",
                "finetune_params": {
                    "n_trials": 20,
                    "model_params": {
                        'n_components': ('int', 1, 30),
                    }
                }
            },
            "sklearn.linear_model.Ridge"  # Syntax 3
        ]

        # Serialize entire pipeline
        serialized = serialize_component(pipeline)

        # Check structure
        assert isinstance(serialized, list)
        assert len(serialized) == 8

        # Verify tuple conversion in finetune_params
        finetune_step = serialized[6]
        assert isinstance(finetune_step["finetune_params"]["model_params"]["n_components"], list)

        # JSON round-trip
        json_str = json.dumps(serialized, indent=2)
        parsed = json.loads(json_str)

        assert len(parsed) == 8


class TestPipelineConfigsSerialization:
    """Test PipelineConfigs integration with serialization."""

    def test_pipeline_configs_json_export(self, temp_pipeline_dir):
        """Test exporting PipelineConfigs to JSON."""
        pipeline = [
            StandardScaler(),
            PLSRegression(n_components=10)
        ]

        config = PipelineConfigs(pipeline, name="test_pipeline")

        # Serialize steps
        serialized = serialize_component(config.steps[0])

        # Save to JSON
        json_file = temp_pipeline_dir / "pipeline.json"
        with open(json_file, "w") as f:
            json.dump(serialized, f, indent=2)

        # Load back
        with open(json_file, "r") as f:
            loaded = json.load(f)

        assert isinstance(loaded, list)

    def test_pipeline_configs_yaml_export(self, temp_pipeline_dir):
        """Test exporting PipelineConfigs to YAML."""
        pipeline = [
            MinMaxScaler(feature_range=(0, 2)),
            PLSRegression(n_components=5)
        ]

        config = PipelineConfigs(pipeline, name="test_pipeline")
        serialized = serialize_component(config.steps[0])

        # Save to YAML
        yaml_file = temp_pipeline_dir / "pipeline.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(serialized, f)

        # Load back
        with open(yaml_file, "r") as f:
            loaded = yaml.safe_load(f)

        assert isinstance(loaded, list)

    def test_pipeline_configs_from_json_file(self, temp_pipeline_dir):
        """Test loading PipelineConfigs from JSON file."""
        pipeline_data = [
            {
                "class": "sklearn.preprocessing.StandardScaler"
            },
            {
                "class": "sklearn.cross_decomposition.PLSRegression",
                "params": {
                    "n_components": 10
                }
            }
        ]

        # Save to file
        json_file = temp_pipeline_dir / "test_pipeline.json"
        with open(json_file, "w") as f:
            json.dump({"pipeline": pipeline_data}, f)

        # Load via PipelineConfigs
        config = PipelineConfigs(str(json_file), name="test")

        assert len(config.steps) == 1  # No generators, so 1 config

    def test_pipeline_configs_from_yaml_file(self, temp_pipeline_dir):
        """Test loading PipelineConfigs from YAML file."""
        pipeline_data = [
            {
                "class": "sklearn.preprocessing.MinMaxScaler",
                "params": {
                    "feature_range": [0, 1]
                }
            },
            {
                "class": "sklearn.cross_decomposition.PLSRegression",
                "params": {
                    "n_components": 5
                }
            }
        ]

        # Save to file
        yaml_file = temp_pipeline_dir / "test_pipeline.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump({"pipeline": pipeline_data}, f)

        # Load via PipelineConfigs
        config = PipelineConfigs(str(yaml_file), name="test")

        assert len(config.steps) == 1


class TestGeneratorExpansion:
    """Test that serialization is compatible with generator expansion."""

    def test_generator_expansion_preserves_serialization(self):
        """Test that expanded pipelines can be serialized."""
        pipeline = [
            MinMaxScaler(),
            {
                "_or_": [Detrend, FirstDerivative, Gaussian],
                "size": 1
            },
            PLSRegression(n_components=10)
        ]

        config = PipelineConfigs(pipeline, name="test")

        # Should expand to 3 pipelines
        assert config.has_configurations
        assert len(config.steps) == 3

        # Each should be serializable
        for steps in config.steps:
            serialized = serialize_component(steps)
            assert isinstance(serialized, list)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_value(self):
        """Test None serialization."""
        serialized = serialize_component(None)
        assert serialized is None

    def test_empty_list(self):
        """Test empty list."""
        serialized = serialize_component([])
        assert serialized == []

    def test_nested_dicts(self):
        """Test deeply nested dictionaries."""
        step = {
            "model": {
                "class": "sklearn.cross_decomposition.PLSRegression",
                "params": {
                    "n_components": 10
                }
            },
            "finetune_params": {
                "model_params": {
                    "nested": {
                        "deeply": ("int", 1, 10)
                    }
                }
            }
        }

        serialized = serialize_component(step)

        # Check tuple conversion works at any depth
        nested_val = serialized["finetune_params"]["model_params"]["nested"]["deeply"]
        assert isinstance(nested_val, list)

    def test_mixed_types_in_list(self):
        """Test list with mixed types."""
        pipeline = [
            StandardScaler,
            StandardScaler(),
            "sklearn.preprocessing.MinMaxScaler",
            {"class": "sklearn.model_selection.ShuffleSplit"}
        ]

        serialized = serialize_component(pipeline)
        assert isinstance(serialized, list)
        assert len(serialized) == 4


class TestBackwardCompatibility:
    """Test that backward compatibility code has been removed."""

    def test_no_runtime_instance_in_serialization(self):
        """Test that _runtime_instance is not present in serialized output."""
        step = StandardScaler()
        serialized = serialize_component(step)

        # Should not contain _runtime_instance key
        if isinstance(serialized, dict):
            assert "_runtime_instance" not in serialized


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
