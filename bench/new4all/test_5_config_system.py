#!/usr/bin/env python3
"""
Test 5: Configuration-based Pipeline System

This test validates the new configuration-based pipeline system including:
- PipelineConfig for storing configurations
- PipelineSerializer for serialization/deserialization
- PipelineBuilder for building operations from config
- Pipeline with configuration support
- Compatibility with sample.py format
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add the new4all directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PipelineConfig import PipelineConfig
from PipelineSerializer import PipelineSerializer
from PipelineBuilder import PipelineBuilder
from Pipeline import Pipeline

# Import required classes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedStratifiedKFold, ShuffleSplit


def test_pipeline_serializer():
    """Test the pipeline serializer with different object types"""
    print("=== Testing Pipeline Serializer ===")

    serializer = PipelineSerializer()

    # Test basic objects
    scaler = StandardScaler()
    rf = RandomForestClassifier(n_estimators=50, random_state=42)

    # Test serialization
    scaler_serialized = serializer._serialize_component(scaler)
    rf_serialized = serializer._serialize_component(rf)

    print(f"StandardScaler serialized: {scaler_serialized}")
    print(f"RandomForest serialized: {rf_serialized}")

    # Test deserialization
    scaler_deserialized = serializer._deserialize_component(scaler_serialized)
    rf_deserialized = serializer._deserialize_component(rf_serialized)

    print(f"StandardScaler deserialized: {type(scaler_deserialized)}")
    print(f"RandomForest deserialized: {type(rf_deserialized)}")

    # Test complex nested structure
    complex_config = {
        'experiment': {
            'action': 'classification',
            'dataset': 'test_data.csv'
        },
        'pipeline': [
            StandardScaler(),
            {'feature_augmentation': [None, StandardScaler()]},
            RandomForestClassifier(n_estimators=100, random_state=42)
        ]
    }

    serialized_complex = serializer.serialize_config(complex_config)
    deserialized_complex = serializer.deserialize_config(serialized_complex)

    print(f"Complex config serialization successful")
    print(f"Deserialized pipeline length: {len(deserialized_complex['pipeline'])}")

    return True


def test_pipeline_config():
    """Test PipelineConfig class"""
    print("\n=== Testing Pipeline Config ===")

    # Test creating config from dict
    config_dict = {
        'name': 'TestPipeline',
        'experiment': {
            'action': 'classification',
            'dataset': 'test_data.csv'
        },
        'pipeline': [
            {'class': 'sklearn.preprocessing.StandardScaler'},
            {'class': 'sklearn.ensemble.RandomForestClassifier', 'params': {'n_estimators': 100}}
        ]
    }

    config = PipelineConfig.from_dict(config_dict)
    print(f"Config created: {config.name}")
    print(f"Pipeline steps: {len(config.pipeline)}")

    # Test validation
    issues = config.validate()
    print(f"Validation issues: {issues}")

    # Test saving/loading
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        config.save(temp_path)
        loaded_config = PipelineConfig.from_file(temp_path)
        print(f"Config saved and loaded successfully: {loaded_config.name}")
    finally:
        os.unlink(temp_path)

    return True


def test_sample_py_compatibility():
    """Test compatibility with sample.py format"""
    print("\n=== Testing Sample.py Compatibility ===")

    # Create a config similar to sample.py
    sample_config = {
        "experiment": {
            "action": "classification",
            "dataset": "data/sample_data.csv"
        },
        "pipeline": [
            MinMaxScaler(),
            {"feature_augmentation": [None, StandardScaler()]},
            ShuffleSplit(),
            {"cluster": KMeans(n_clusters=5, random_state=42)},
            RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),
            "uncluster",
            {
                "dispatch": [
                    {
                        "model": RandomForestClassifier(random_state=42, n_estimators=100),
                        "y_pipeline": StandardScaler()
                    },
                    {
                        "model": SVC(kernel='linear', C=1.0, random_state=42),
                        "y_pipeline": MinMaxScaler(),
                        "finetune_params": {"C": [0.1, 1.0, 10.0]}
                    }
                ]
            }
        ]
    }

    # Test serialization
    serializer = PipelineSerializer()
    serialized = serializer.serialize_config(sample_config)

    print(f"Sample config serialized successfully")
    print(f"Serialized pipeline steps: {len(serialized['pipeline'])}")

    # Test creating config
    config = PipelineConfig.from_python_config(sample_config)
    print(f"Config created from Python config: {config.name}")

    # Test validation
    issues = config.validate()
    if not issues:
        print("✓ Sample config validation passed")
    else:
        print(f"⚠ Sample config validation issues: {issues}")

    return True


def test_pipeline_from_config():
    """Test creating Pipeline from configuration"""
    print("\n=== Testing Pipeline from Config ===")

    # Simple config
    simple_config = {
        "experiment": {
            "action": "classification",
            "dataset": "test_data.csv"
        },
        "pipeline": [
            StandardScaler(),
            RandomForestClassifier(n_estimators=50, random_state=42)
        ]
    }

    # Test creating pipeline from Python config
    pipeline = Pipeline.from_python_config(simple_config, name="TestPipeline")
    print(f"Pipeline created: {pipeline.name}")
    print(f"Operations count: {len(pipeline.operations)}")

    # Test getting config back
    extracted_config = pipeline.to_config()
    print(f"Config extracted: {extracted_config.name}")

    # Test validation
    issues = pipeline.validate()
    if not issues:
        print("✓ Pipeline validation passed")
    else:
        print(f"⚠ Pipeline validation issues: {issues}")

    return True


def test_json_yaml_roundtrip():
    """Test JSON/YAML serialization roundtrip"""
    print("\n=== Testing JSON/YAML Roundtrip ===")

    # Create a complex config
    config_dict = {
        "name": "ComplexPipeline",
        "experiment": {
            "action": "classification",
            "dataset": "data/test.csv"
        },
        "pipeline": [
            {"class": "sklearn.preprocessing.StandardScaler"},
            {"class": "sklearn.ensemble.RandomForestClassifier",
             "params": {"n_estimators": 100, "random_state": 42}},
            {"feature_augmentation": [
                {"class": "sklearn.preprocessing.MinMaxScaler"},
                {"class": "sklearn.preprocessing.RobustScaler"}
            ]}
        ]
    }

    config = PipelineConfig.from_dict(config_dict)

    # Test JSON roundtrip
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name

    try:
        config.save(json_path)
        json_loaded = PipelineConfig.from_file(json_path)
        print(f"✓ JSON roundtrip successful: {json_loaded.name}")
    finally:
        os.unlink(json_path)

    # Test YAML roundtrip
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = f.name

    try:
        config.save(yaml_path)
        yaml_loaded = PipelineConfig.from_file(yaml_path)
        print(f"✓ YAML roundtrip successful: {yaml_loaded.name}")
    finally:
        os.unlink(yaml_path)

    return True


def test_deterministic_serialization():
    """Test that serialization is deterministic and reproducible"""
    print("\n=== Testing Deterministic Serialization ===")

    # Create the same object multiple times
    configs = []
    for i in range(3):
        config = {
            "pipeline": [
                StandardScaler(),
                RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            ]
        }
        configs.append(config)

    serializer = PipelineSerializer()
    serialized_configs = [serializer.serialize_config(config) for config in configs]

    # Check that all serializations are identical
    first_serialized = json.dumps(serialized_configs[0], sort_keys=True)
    all_same = all(json.dumps(sc, sort_keys=True) == first_serialized
                   for sc in serialized_configs)

    if all_same:
        print("✓ Serialization is deterministic")
    else:
        print("✗ Serialization is not deterministic")

    # Test deserialization produces equivalent objects
    deserialized = [serializer.deserialize_config(sc) for sc in serialized_configs]

    # Check that all have same structure
    pipeline_lengths = [len(d['pipeline']) for d in deserialized]
    all_same_length = all(pl == pipeline_lengths[0] for pl in pipeline_lengths)

    if all_same_length:
        print("✓ Deserialization produces consistent results")
    else:
        print("✗ Deserialization produces inconsistent results")

    return all_same and all_same_length


def test_sample_py_full_serialization():
    """Test full serialization/deserialization using the actual sample.py config"""
    print("\n=== Testing Full Sample.py Config Serialization ===")

    # Import the actual sample.py config directly
    try:
        from sample import config as sample_config
        print("✓ Successfully imported sample.py config")
    except ImportError as e:
        print(f"✗ Failed to import sample.py: {e}")
        return False

    # Test 1: Basic serialization
    print("\n--- Test 1: Basic Serialization ---")
    serializer = PipelineSerializer()

    try:
        serialized_config = serializer.serialize_config(sample_config)
        print("✓ Sample config serialized successfully")
        print(f"  Experiment action: {serialized_config.get('experiment', {}).get('action', 'N/A')}")
        print(f"  Pipeline steps: {len(serialized_config.get('pipeline', []))}")
    except Exception as e:
        print(f"✗ Serialization failed: {e}")
        return False

    # Test 2: Deserialization
    print("\n--- Test 2: Deserialization ---")
    try:
        deserialized_config = serializer.deserialize_config(serialized_config)
        print("✓ Sample config deserialized successfully")
        print(f"  Experiment action: {deserialized_config.get('experiment', {}).get('action', 'N/A')}")
        print(f"  Pipeline steps: {len(deserialized_config.get('pipeline', []))}")
    except Exception as e:
        print(f"✗ Deserialization failed: {e}")
        return False

    # Test 3: Round-trip validation
    print("\n--- Test 3: Round-trip Validation ---")
    try:
        # Serialize again and compare structure
        second_serialized = serializer.serialize_config(deserialized_config)

        # Compare key structural elements
        orig_pipeline_len = len(serialized_config.get('pipeline', []))
        roundtrip_pipeline_len = len(second_serialized.get('pipeline', []))

        if orig_pipeline_len == roundtrip_pipeline_len:
            print("✓ Round-trip maintains pipeline structure")
        else:
            print(f"✗ Round-trip changed pipeline length: {orig_pipeline_len} -> {roundtrip_pipeline_len}")
            return False

        # Check experiment section
        orig_exp = serialized_config.get('experiment', {})
        roundtrip_exp = second_serialized.get('experiment', {})
        if orig_exp == roundtrip_exp:
            print("✓ Round-trip maintains experiment configuration")
        else:
            print("✗ Round-trip changed experiment configuration")
            return False

    except Exception as e:
        print(f"✗ Round-trip validation failed: {e}")
        return False

    # Test 4: PipelineConfig creation
    print("\n--- Test 4: PipelineConfig Creation ---")
    try:
        pipeline_config = PipelineConfig.from_python_config(sample_config)
        print(f"✓ PipelineConfig created successfully: {pipeline_config.name}")

        # Validate the config
        validation_issues = pipeline_config.validate()
        if not validation_issues:
            print("✓ Pipeline config validation passed")
        else:
            print(f"⚠ Pipeline config validation issues: {len(validation_issues)} issues found")
            for i, issue in enumerate(validation_issues[:3]):  # Show first 3 issues
                print(f"    {i+1}. {issue}")
            if len(validation_issues) > 3:
                print(f"    ... and {len(validation_issues) - 3} more issues")
    except Exception as e:
        print(f"✗ PipelineConfig creation failed: {e}")
        return False

    # Test 5: File I/O with sample config
    print("\n--- Test 5: File I/O Operations ---")
    try:
        # Save to JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name

        pipeline_config.save(json_path)
        print("✓ Sample config saved to JSON")

        # Load from JSON
        loaded_config = PipelineConfig.from_file(json_path)
        print(f"✓ Sample config loaded from JSON: {loaded_config.name}")

        # Clean up
        os.unlink(json_path)

        # Save to YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name

        pipeline_config.save(yaml_path)
        print("✓ Sample config saved to YAML")

        # Load from YAML
        loaded_yaml_config = PipelineConfig.from_file(yaml_path)
        print(f"✓ Sample config loaded from YAML: {loaded_yaml_config.name}")

        # Clean up
        os.unlink(yaml_path)

    except Exception as e:
        print(f"✗ File I/O operations failed: {e}")
        return False

    # Test 6: Pipeline creation from sample config
    print("\n--- Test 6: Pipeline Object Creation ---")
    try:
        pipeline = Pipeline.from_python_config(sample_config, name="SamplePipeline")
        print(f"✓ Pipeline created from sample config: {pipeline.name}")
        print(f"  Operations count: {len(pipeline.operations)}")

        # Test pipeline validation
        pipeline_issues = pipeline.validate()
        if not pipeline_issues:
            print("✓ Pipeline validation passed")
        else:
            print(f"⚠ Pipeline validation issues: {len(pipeline_issues)} issues found")
            for i, issue in enumerate(pipeline_issues[:2]):  # Show first 2 issues
                print(f"    {i+1}. {issue}")

        # Test config extraction
        extracted_config = pipeline.to_config()
        print(f"✓ Config extracted from pipeline: {extracted_config.name}")

    except Exception as e:
        print(f"✗ Pipeline creation failed: {e}")
        return False

    # Test 7: Complex nested structure handling
    print("\n--- Test 7: Complex Structure Analysis ---")
    try:
        # Analyze the complex structures in sample.py
        pipeline_steps = sample_config.get('pipeline', [])

        complex_structures = []
        for i, step in enumerate(pipeline_steps):
            if isinstance(step, dict):
                if 'dispatch' in step:
                    complex_structures.append(f"Step {i}: Dispatch with {len(step['dispatch'])} branches")
                elif 'feature_augmentation' in step:
                    complex_structures.append(f"Step {i}: Feature augmentation")
                elif 'sample_augmentation' in step:
                    complex_structures.append(f"Step {i}: Sample augmentation")
                elif 'cluster' in step:
                    complex_structures.append(f"Step {i}: Clustering")
                elif 'stack' in step:
                    complex_structures.append(f"Step {i}: Model stacking")
                else:
                    complex_structures.append(f"Step {i}: Other dict structure")

        print(f"✓ Complex structures identified: {len(complex_structures)}")
        for structure in complex_structures:
            print(f"    {structure}")

        # Test serialization of these complex structures
        for i, step in enumerate(pipeline_steps):
            if isinstance(step, dict) and any(key in step for key in ['dispatch', 'stack', 'feature_augmentation']):
                try:
                    step_serialized = serializer._serialize_component(step)
                    step_deserialized = serializer._deserialize_component(step_serialized)
                    print(f"✓ Complex step {i} serialization successful")
                except Exception as e:
                    print(f"⚠ Complex step {i} serialization issue: {e}")

    except Exception as e:
        print(f"✗ Complex structure analysis failed: {e}")
        return False

    print("\n✓ All sample.py full serialization tests completed successfully!")
    return True


def main():
    """Run all configuration system tests"""
    print("Testing Configuration-based Pipeline System\n")

    tests = [
        test_pipeline_serializer,
        test_pipeline_config,
        test_sample_py_compatibility,
        test_pipeline_from_config,
        test_json_yaml_roundtrip,
        test_deterministic_serialization,
        test_sample_py_full_serialization
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"✓ {test.__name__} passed\n")
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}\n")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")

    if passed == total:
        print("\n🎉 All configuration system tests passed!")
    else:
        print(f"\n⚠ {total-passed} tests failed. Check implementation.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
