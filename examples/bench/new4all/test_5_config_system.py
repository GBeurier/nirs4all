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


def main():
    """Run all configuration system tests"""
    print("Testing Configuration-based Pipeline System\n")

    tests = [
        test_pipeline_serializer,
        test_pipeline_config,
        test_sample_py_compatibility,
        test_pipeline_from_config,
        test_json_yaml_roundtrip,
        test_deterministic_serialization
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
