#!/usr/bin/env python3
"""
Test 6: Full API Validation

This test validates the complete API by testing:
1. Load simple_sample.yaml -> build -> execute -> save results -> serialize pipeline to JSON
2. Load simple_sample.json -> build -> serialize -> compare with first serialization
3. Execute pipeline with sample.py config -> save -> serialize -> compare
4. Compare all results (should be identical)
5. Reload first pipeline -> redo prediction -> compare with other results

This comprehensive test ensures:
- Configuration format compatibility (YAML, JSON, Python dict)
- Pipeline serialization/deserialization consistency
- Execution determinism across different config sources
- Result reproducibility
"""

import os
import sys
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pickle

# Add the new4all directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SpectraDataset import SpectraDataset
from Pipeline import Pipeline
from PipelineConfig import PipelineConfig
from PipelineSerializer import PipelineSerializer
from PipelineBuilder import PipelineBuilder
from PipelineContext import PipelineContext

# Import sklearn components for sample.py config
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedStratifiedKFold, ShuffleSplit


def create_test_dataset():
    """Create a consistent test dataset for all tests"""
    print("Creating test dataset...")

    np.random.seed(42)  # Fixed seed for reproducibility
    n_samples = 100
    n_features = 50

    # Create synthetic spectroscopy data
    X = np.random.randn(n_samples, n_features)

    # Add some structure to make it more realistic
    wavelengths = np.linspace(400, 700, n_features)
    for i in range(n_samples):
        # Add some peaks at specific wavelengths
        X[i] += 0.5 * np.exp(-((wavelengths - 500) / 50) ** 2)
        X[i] += 0.3 * np.exp(-((wavelengths - 600) / 30) ** 2)

    # Create binary classification target
    y_continuous = X.mean(axis=1)
    y_threshold = np.median(y_continuous)
    y = ['class_A' if val > y_threshold else 'class_B' for val in y_continuous]

    # Create dataset using the proper SpectraDataset API
    dataset = SpectraDataset(task_type="classification")

    # Add data to dataset
    sample_ids = dataset.add_data(
        features=[X],  # Single source for simplicity
        targets=np.array(y),
        partition="all"
    )

    print(f"Created dataset: {len(sample_ids)} samples, {n_features} features")
    return dataset


def copy_dataset(dataset):
    """Create a copy of the dataset"""
    # Since SpectraDataset doesn't have a copy method, we'll create a new one
    # Create new dataset
    new_dataset = SpectraDataset(task_type=dataset.task_type)

    # Get features and targets from original
    if len(dataset.indices) > 0:
        row_indices = dataset.indices["row"].to_numpy()
        features = dataset.get_features(row_indices, concatenate=False)
        sample_ids = dataset.indices["sample"].to_list()
          # Try to get targets
        try:
            targets = dataset.get_targets(sample_ids)
        except Exception:
            targets = None

        # Add data to new dataset
        new_dataset.add_data(
            features=features,
            targets=targets,
            partition="all"
        )

    return new_dataset


def extract_dataset_results(dataset):
    """Extract results from a dataset for comparison"""
    if len(dataset.indices) == 0:
        return {
            'final_dataset_length': 0,
            'sample_count': 0,
            'feature_sources': 0
        }

    # Get basic info
    row_indices = dataset.indices["row"].to_numpy()
    sample_ids = dataset.indices["sample"].to_list()

    # Get features
    features = dataset.get_features(row_indices, concatenate=True)
      # Try to get targets
    try:
        targets = dataset.get_targets(sample_ids)
        target_stats = {
            'target_shape': targets.shape if hasattr(targets, 'shape') else len(targets),
            'target_type': str(type(targets)),
            'unique_targets': len(np.unique(targets)) if targets is not None else 0
        }
    except Exception:
        target_stats = {
            'target_shape': None,
            'target_type': None,
            'unique_targets': 0
        }

    results = {
        'final_dataset_length': len(dataset),
        'sample_count': len(sample_ids),
        'feature_sources': len(dataset.features.sources) if dataset.features else 0,
        'feature_shape': features.shape if hasattr(features, 'shape') else (0, 0),
        'feature_mean': float(np.mean(features)) if hasattr(features, 'shape') and features.size > 0 else 0.0,
        'feature_std': float(np.std(features)) if hasattr(features, 'shape') and features.size > 0 else 0.0,
        **target_stats,
        'task_type': dataset.task_type,
        'n_classes': dataset.n_classes if hasattr(dataset, 'n_classes') else 0
    }

    return results


def get_sample_py_config():
    """Get the sample.py configuration as a Python dictionary"""
    config = {
        'experiment': {
            'action': 'classification',
            'dataset': 'data/sample_data.csv'
        },
        'pipeline': [
            MinMaxScaler(),
            {
                'feature_augmentation': [
                    None,
                    StandardScaler()
                ]
            },
            {
                'sample_augmentation': [
                    StandardScaler(),
                    StandardScaler()
                ]
            },
            ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
            {
                'dispatch': [
                    {
                        'model': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5)
                    },
                    {
                        'model': RandomForestClassifier(random_state=42, n_estimators=30)
                    }
                ]
            }
        ]
    }
    return config


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def compare_results(results1: Dict[str, Any], results2: Dict[str, Any], name1: str, name2: str) -> bool:
    """Compare two result dictionaries"""
    print(f"\nComparing results: {name1} vs {name2}")

    # Check if both have the same keys
    if set(results1.keys()) != set(results2.keys()):
        print(f"  ❌ Different keys: {set(results1.keys())} vs {set(results2.keys())}")
        return False

    all_equal = True

    for key in results1.keys():
        if isinstance(results1[key], np.ndarray) and isinstance(results2[key], np.ndarray):
            if not np.allclose(results1[key], results2[key], rtol=1e-10, atol=1e-10):
                print(f"  ❌ {key}: Arrays not equal")
                max_diff = np.max(np.abs(results1[key] - results2[key]))
                print(f"    Max difference: {max_diff}")
                all_equal = False
            else:
                print(f"  ✅ {key}: Arrays equal")
        elif isinstance(results1[key], float) and isinstance(results2[key], float):
            if not np.isclose(results1[key], results2[key], rtol=1e-10, atol=1e-10):
                print(f"  ❌ {key}: {results1[key]} != {results2[key]}")
                all_equal = False
            else:
                print(f"  ✅ {key}: Equal")
        elif results1[key] != results2[key]:
            print(f"  ❌ {key}: {results1[key]} != {results2[key]}")
            all_equal = False
        else:
            print(f"  ✅ {key}: Equal")

    if all_equal:
        print(f"  🎉 All results match between {name1} and {name2}")

    return all_equal


def compare_serialized_pipelines(serialized1: Dict[str, Any], serialized2: Dict[str, Any], name1: str, name2: str) -> bool:
    """Compare two serialized pipeline configurations"""
    print(f"\nComparing serialized pipelines: {name1} vs {name2}")

    # Convert to JSON strings for comparison (this normalizes the format)
    json1 = json.dumps(serialized1, sort_keys=True, indent=2)
    json2 = json.dumps(serialized2, sort_keys=True, indent=2)

    if json1 == json2:
        print("  🎉 Serialized pipelines are identical")
        return True
    else:
        print("  ❌ Serialized pipelines differ")
        # Save diff files for debugging
        with open(f'debug_{name1}.json', 'w') as f:
            f.write(json1)
        with open(f'debug_{name2}.json', 'w') as f:
            f.write(json2)
        print(f"  Debug files saved: debug_{name1}.json, debug_{name2}.json")
        return False


def test_step_1_yaml_pipeline():
    """Step 1: Load simple_sample.yaml -> build -> execute -> save results -> serialize pipeline to JSON"""
    print("\n" + "=" * 60)
    print("STEP 1: YAML Pipeline Test")
    print("=" * 60)

    # Create test dataset
    dataset = create_test_dataset()

    # Load simple_sample.yaml
    yaml_path = Path(__file__).parent / "simple_sample.yaml"
    print(f"Loading YAML config from: {yaml_path}")

    config = PipelineConfig.from_file(yaml_path)
    print(f"Loaded config: {config.name}")

    # Build pipeline
    pipeline = Pipeline.from_config(config, name="YAML_Pipeline")
    print(f"Built pipeline with {len(pipeline.operations)} operations")

    # Execute pipeline
    print("Executing pipeline...")
    result_dataset = pipeline.execute(copy_dataset(dataset))

    # Extract results
    results = extract_dataset_results(result_dataset)

    # Save results
    save_results(results, 'results_yaml.pkl')
    print("Saved YAML results")

    # Serialize pipeline to JSON
    serialized = pipeline.to_config().to_dict()
    with open('pipeline_from_yaml.json', 'w') as f:
        json.dump(serialized, f, indent=2)
    print("Serialized YAML pipeline to JSON")

    return results, serialized, pipeline


def test_step_2_json_pipeline(yaml_serialized):
    """Step 2: Load simple_sample.json -> build -> serialize -> compare with first serialization"""
    print("\n" + "=" * 60)
    print("STEP 2: JSON Pipeline Test")
    print("=" * 60)

    # Load simple_sample.json
    json_path = Path(__file__).parent / "simple_sample.json"
    print(f"Loading JSON config from: {json_path}")

    config = PipelineConfig.from_file(json_path)
    print(f"Loaded config: {config.name}")

    # Build pipeline
    pipeline = Pipeline.from_config(config, name="JSON_Pipeline")
    print(f"Built pipeline with {len(pipeline.operations)} operations")

    # Serialize pipeline
    serialized = pipeline.to_config().to_dict()
    with open('pipeline_from_json.json', 'w') as f:
        json.dump(serialized, f, indent=2)
    print("Serialized JSON pipeline")

    # Compare serializations
    serializations_match = compare_serialized_pipelines(
        yaml_serialized, serialized, "YAML", "JSON"
    )

    return serialized, pipeline, serializations_match


def test_step_3_python_pipeline(yaml_results, yaml_serialized):
    """Step 3: Execute pipeline with sample.py config -> save -> serialize -> compare"""
    print("\n" + "=" * 60)
    print("STEP 3: Python Config Pipeline Test")
    print("=" * 60)

    # Create test dataset
    dataset = create_test_dataset()

    # Get sample.py config
    python_config = get_sample_py_config()
    print("Created Python config dictionary")

    # Create pipeline from Python config
    config = PipelineConfig.from_python_config(python_config)
    pipeline = Pipeline.from_config(config, name="Python_Pipeline")
    print(f"Built pipeline with {len(pipeline.operations)} operations")

    # Execute pipeline
    print("Executing pipeline...")
    result_dataset = pipeline.execute(copy_dataset(dataset))

    # Extract results
    results = extract_dataset_results(result_dataset)

    # Save results
    save_results(results, 'results_python.pkl')
    print("Saved Python results")

    # Serialize pipeline
    serialized = pipeline.to_config().to_dict()
    with open('pipeline_from_python.json', 'w') as f:
        json.dump(serialized, f, indent=2)
    print("Serialized Python pipeline")

    # Compare results with YAML
    results_match = compare_results(yaml_results, results, "YAML", "Python")

    # Compare serializations with YAML
    serializations_match = compare_serialized_pipelines(
        yaml_serialized, serialized, "YAML", "Python"
    )

    return results, serialized, pipeline, results_match, serializations_match


def test_step_4_reload_and_predict(yaml_pipeline):
    """Step 4: Reload first pipeline -> redo prediction -> compare with other results"""
    print("\n" + "=" * 60)
    print("STEP 4: Reload and Prediction Test")
    print("=" * 60)

    # Create test dataset
    dataset = create_test_dataset()

    # Reload pipeline from saved JSON
    print("Reloading pipeline from JSON...")
    config = PipelineConfig.from_file('pipeline_from_yaml.json')
    reloaded_pipeline = Pipeline.from_config(config, name="Reloaded_Pipeline")
    print(f"Reloaded pipeline with {len(reloaded_pipeline.operations)} operations")

    # Execute reloaded pipeline
    print("Executing reloaded pipeline...")
    result_dataset = reloaded_pipeline.execute(copy_dataset(dataset))

    # Extract results
    results = extract_dataset_results(result_dataset)

    # Load original YAML results for comparison
    yaml_results = load_results('results_yaml.pkl')

    # Compare results
    results_match = compare_results(yaml_results, results, "Original YAML", "Reloaded")

    # Test prediction mode
    print("\nTesting prediction mode...")
    try:
        predictions = reloaded_pipeline.predict(copy_dataset(dataset))
        print(f"Predictions successful: {len(predictions)} prediction sets")
    except Exception as e:
        print(f"Prediction failed: {e}")
        predictions = None

    return results, results_match, predictions


def test_comprehensive_validation():
    """Run comprehensive validation test"""
    print("🧪 Starting Comprehensive API Validation Test")
    print("=" * 80)

    try:
        # Step 1: YAML Pipeline
        yaml_results, yaml_serialized, yaml_pipeline = test_step_1_yaml_pipeline()

        # Step 2: JSON Pipeline
        json_serialized, json_pipeline, json_yaml_match = test_step_2_json_pipeline(yaml_serialized)

        # Step 3: Python Pipeline
        python_results, python_serialized, python_pipeline, python_yaml_results_match, python_yaml_serialization_match = test_step_3_python_pipeline(yaml_results, yaml_serialized)

        # Step 4: Reload and Predict
        reload_results, reload_yaml_match, predictions = test_step_4_reload_and_predict(yaml_pipeline)

        # Final Summary
        print("\n" + "=" * 80)
        print("FINAL VALIDATION SUMMARY")
        print("=" * 80)

        all_tests_passed = True

        print("Configuration Format Compatibility:")
        if json_yaml_match:
            print("  ✅ YAML ↔ JSON serialization consistency")
        else:
            print("  ❌ YAML ↔ JSON serialization inconsistency")
            all_tests_passed = False

        if python_yaml_serialization_match:
            print("  ✅ YAML ↔ Python config serialization consistency")
        else:
            print("  ❌ YAML ↔ Python config serialization inconsistency")
            all_tests_passed = False

        print("\nExecution Determinism:")
        if python_yaml_results_match:
            print("  ✅ YAML ↔ Python config execution results match")
        else:
            print("  ❌ YAML ↔ Python config execution results differ")
            all_tests_passed = False

        if reload_yaml_match:
            print("  ✅ Original ↔ Reloaded pipeline results match")
        else:
            print("  ❌ Original ↔ Reloaded pipeline results differ")
            all_tests_passed = False

        print("\nPrediction Capability:")
        if predictions is not None:
            print("  ✅ Pipeline prediction mode works")
        else:
            print("  ❌ Pipeline prediction mode failed")
            all_tests_passed = False

        print("\n" + "=" * 80)
        if all_tests_passed:
            print("🎉 ALL TESTS PASSED - API Validation Complete!")
            print("   ✅ Configuration formats are compatible")
            print("   ✅ Pipeline execution is deterministic")
            print("   ✅ Serialization/deserialization works correctly")
            print("   ✅ Pipeline reloading preserves functionality")
            print("   ✅ Prediction mode is operational")
        else:
            print("❌ SOME TESTS FAILED - API Validation Incomplete")
            print("   Check the detailed output above for specific issues")

        return all_tests_passed

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test execution"""
    print("Test 6: Full API Validation")
    print("Testing configuration loading, pipeline execution, serialization, and result comparison")

    # Clean up any existing test files
    test_files = [
        'results_yaml.pkl', 'results_python.pkl',
        'pipeline_from_yaml.json', 'pipeline_from_json.json', 'pipeline_from_python.json',
        'debug_YAML.json', 'debug_JSON.json', 'debug_Python.json'
    ]

    for file in test_files:
        if os.path.exists(file):
            os.remove(file)

    # Run comprehensive validation
    success = test_comprehensive_validation()

    # Clean up test files
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)

    if success:
        print("\n🎉 Test 6 PASSED: Full API validation successful")
        return True
    else:
        print("\n❌ Test 6 FAILED: API validation issues detected")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
