"""
Comprehensive Demo: Enhanced Pipeline Runner with Parallelization and History

This demo showcases:
1. Enhanced pipeline execution with joblib parallelization
2. Comprehensive history tracking and logging
3. Pipeline serialization and saving capabilities
4. Feature augmentation with parallel processing
5. Error handling and recovery mechanisms
6. Multiple export formats (JSON, pickle, zip bundles)
"""
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime
import json

# Mock classes for demonstration
class MockSpectraDataset:
    """Mock SpectraDataset for demonstration"""

    def __init__(self, n_samples=1000, n_features=100):
        self.n_samples = n_samples
        self.n_features = n_features
        self._features = np.random.random((n_samples, n_features))
        self._labels = np.random.random(n_samples)
        self.added_features = []

    def get_features(self, row_indices=None):
        if row_indices is None:
            return self._features
        return self._features[row_indices]

    def get_labels(self, row_indices=None):
        if row_indices is None:
            return self._labels
        return self._labels[row_indices]

    def get_train_indices(self):
        return list(range(int(0.8 * self.n_samples)))  # 80% for training

    def get_test_indices(self):
        return list(range(int(0.8 * self.n_samples), self.n_samples))  # 20% for testing

    def add_features(self, new_features, feature_names=None):
        """Add new features to the dataset"""
        print(f"✅ Added {new_features.shape[1]} new features to dataset")
        self.added_features.append({
            'features': new_features,
            'names': feature_names,
            'timestamp': datetime.now()
        })
        # Update feature matrix
        self._features = np.hstack([self._features, new_features])
        self.n_features = self._features.shape[1]

class MockPipelineContext:
    """Mock PipelineContext for demonstration"""

    def __init__(self):
        self.filters = {}
        self.metadata = {}

    def apply_filters(self, filters):
        self.filters.update(filters)
        print(f"🔍 Applied context filters: {list(filters.keys())}")

class MockTransformer:
    """Mock transformer for feature augmentation"""

    def __init__(self, name, n_new_features=5):
        self.name = name
        self.n_new_features = n_new_features
        self.fitted = False

    def fit(self, X, y=None):
        self.fitted = True
        print(f"🔧 Fitted {self.name} on {X.shape[0]} samples")
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError(f"{self.name} must be fitted before transform")
        new_features = np.random.random((X.shape[0], self.n_new_features))
        print(f"✨ {self.name} generated {self.n_new_features} new features")
        return new_features

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class MockOperation:
    """Mock pipeline operation"""

    def __init__(self, name, operation_type="transformation"):
        self.name = name
        self.operation_type = operation_type
        self.executed = False

    def get_name(self):
        return self.name

    def execute(self, dataset):
        print(f"⚙️ Executing {self.name}")
        self.executed = True
        # Simulate some processing time
        import time
        time.sleep(0.1)


def demo_enhanced_pipeline_features():
    """
    Comprehensive demonstration of enhanced pipeline features
    """
    print("🚀 Enhanced Pipeline Runner Demo")
    print("=" * 50)

    # Import the enhanced modules
    try:
        from PipelineHistory import PipelineHistory
        print("✅ Successfully imported PipelineHistory")
    except ImportError as e:
        print(f"⚠️ Could not import PipelineHistory: {e}")
        return

    # 1. Create sample dataset and context
    print("\n📊 Creating Sample Dataset")
    print("-" * 30)
    dataset = MockSpectraDataset(n_samples=1000, n_features=200)
    context = MockPipelineContext()

    print(f"Dataset: {dataset.n_samples} samples, {dataset.n_features} features")
    print(f"Train samples: {len(dataset.get_train_indices())}")
    print(f"Test samples: {len(dataset.get_test_indices())}")

    # 2. Create pipeline configuration
    print("\n⚙️ Pipeline Configuration")
    print("-" * 30)
    pipeline_config = [
        {
            "type": "context_filter",
            "filters": {
                "wavelength_range": [400, 2500],
                "sample_quality": "high"
            }
        },
        {
            "type": "transformation",
            "sklearn.preprocessing.StandardScaler": {}
        },
        {
            "type": "feature_augmentation",
            "augmenters": [
                {"name": "PCA_Features", "n_components": 10},
                {"name": "Spectral_Derivatives", "order": 2},
                {"name": "Wavelet_Features", "wavelet": "db4"}
            ],
            "parallel": True,
            "n_jobs": 2
        },
        {
            "type": "model",
            "sklearn.ensemble.RandomForestRegressor": {
                "n_estimators": 100,
                "random_state": 42
            }
        }
    ]

    print("Pipeline steps:")
    for i, step in enumerate(pipeline_config, 1):
        print(f"  {i}. {step['type']}")

    # 3. Initialize pipeline history
    print("\n📈 Initializing Pipeline History")
    print("-" * 30)
    history = PipelineHistory()

    # Start execution tracking
    execution_id = history.start_execution(pipeline_config, metadata={
        "dataset_name": "demo_spectra",
        "experiment_id": "enhanced_pipeline_demo",
        "timestamp": datetime.now().isoformat()
    })
    print(f"Started execution tracking: {execution_id}")

    # 4. Simulate pipeline execution with history tracking
    print("\n🔄 Executing Pipeline Steps")
    print("-" * 30)

    # Step 1: Context filtering
    step1 = history.start_step(
        1, "Context Filtering",
        pipeline_config[0],
        step_type="context_filter",
        metadata={"filters_applied": list(pipeline_config[0]["filters"].keys())}
    )

    # Simulate context filtering
    context.apply_filters(pipeline_config[0]["filters"])
    history.complete_step(step1.step_id)

    # Step 2: Data transformation
    step2 = history.start_step(
        2, "Data Standardization",
        pipeline_config[1],
        step_type="transformation",
        operation_name="StandardScaler"
    )

    # Simulate transformation
    operation = MockOperation("StandardScaler", "transformation")
    operation.execute(dataset)
    history.complete_step(step2.step_id)

    # Step 3: Feature augmentation (parallel)
    step3 = history.start_step(
        3, "Feature Augmentation",
        pipeline_config[2],
        step_type="feature_augmentation",
        metadata={
            "parallel": True,
            "n_augmenters": len(pipeline_config[2]["augmenters"]),
            "n_jobs": pipeline_config[2]["n_jobs"]
        }
    )

    # Simulate parallel feature augmentation
    print("  🔀 Running feature augmentation in parallel...")
    train_indices = dataset.get_train_indices()
    train_features = dataset.get_features(train_indices)

    # Create mock augmenters
    augmenters = [
        MockTransformer("PCA_Features", 10),
        MockTransformer("Spectral_Derivatives", 5),
        MockTransformer("Wavelet_Features", 8)
    ]

    # Simulate parallel processing
    from joblib import Parallel, delayed
      def process_augmenter(augmenter, features):
        """Process a single augmenter"""
        # Generate features for the full dataset, not just training
        full_features = dataset.get_features()  # Get all features
        new_features = augmenter.fit_transform(features)  # Fit on train only

        # Transform full dataset (simulate transform on all data)
        full_new_features = augmenter.transform(full_features)

        return {
            'name': augmenter.name,
            'features': full_new_features,  # Return full dataset features
            'n_features': augmenter.n_new_features
        }

    # Run augmenters in parallel
    results = Parallel(n_jobs=2, backend='threading')(
        delayed(process_augmenter)(aug, train_features) for aug in augmenters
    )

    # Add all new features to dataset
    total_new_features = 0
    for result in results:
        dataset.add_features(result['features'], [f"{result['name']}_{i}" for i in range(result['n_features'])])
        total_new_features += result['n_features']

    print(f"  ✨ Added {total_new_features} new features via parallel processing")
    history.complete_step(step3.step_id, metadata={"new_features_count": total_new_features})

    # Step 4: Model training
    step4 = history.start_step(
        4, "Model Training",
        pipeline_config[3],
        step_type="model",
        operation_name="RandomForestRegressor"
    )

    # Simulate model training
    model_op = MockOperation("RandomForestRegressor", "model")
    model_op.execute(dataset)
    history.complete_step(step4.step_id, metadata={
        "final_feature_count": dataset.n_features,
        "model_type": "RandomForestRegressor"
    })

    # Complete execution
    history.complete_execution()
    print(f"✅ Pipeline execution completed in {history.current_execution.total_duration_seconds:.2f} seconds")

    # 5. Demonstrate serialization capabilities
    print("\n💾 Pipeline Serialization Demo")
    print("-" * 30)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save in different formats
        formats = {
            "JSON": temp_path / "pipeline_history.json",
            "Pickle": temp_path / "pipeline_history.pkl",
            "Bundle": temp_path / "pipeline_bundle.zip"
        }

        # Save history in all formats
        history.save_json(formats["JSON"])
        history.save_pickle(formats["Pickle"])
        history.save_bundle(formats["Bundle"])

        print("Saved pipeline history in multiple formats:")
        for format_name, file_path in formats.items():
            size_kb = file_path.stat().st_size / 1024
            print(f"  📄 {format_name}: {file_path.name} ({size_kb:.1f} KB)")

        # Demonstrate loading JSON
        print("\n🔍 Loading and Inspecting Saved History")
        with open(formats["JSON"], 'r') as f:
            loaded_history = json.load(f)

        execution = loaded_history['executions'][0]
        print(f"Loaded execution: {execution['execution_id']}")
        print(f"Status: {execution['status']}")
        print(f"Steps executed: {len(execution['steps'])}")
        print(f"Total duration: {execution['total_duration_seconds']:.2f}s")

        # Show step details
        print("\nStep details:")
        for step in execution['steps']:
            status_icon = "✅" if step['status'] == 'completed' else "❌"
            print(f"  {status_icon} {step['step_description']} ({step['duration_seconds']:.2f}s)")

    # 6. Demonstrate error handling
    print("\n🚨 Error Handling Demo")
    print("-" * 30)

    # Create new execution with intentional failure
    error_config = [
        {"type": "transformation", "NonExistentScaler": {}},
        {"type": "model", "sklearn.linear_model.LinearRegression": {}}
    ]

    error_exec_id = history.start_execution(error_config, metadata={"demo": "error_handling"})

    # Simulate step failure
    error_step = history.start_step(1, "Failing Step", error_config[0])
    history.fail_step(error_step.step_id, "NonExistentScaler not found")

    # Show graceful handling
    print(f"❌ Step failed: {error_step.error_message}")
    print(f"⏱️ Failure time: {error_step.end_time}")

    # Complete with failure
    history.complete_execution(status='failed')
    print("🔚 Execution marked as failed but system remained stable")

    # 7. Summary statistics
    print("\n📊 Final Statistics")
    print("-" * 30)
    print(f"Total executions tracked: {len(history.executions)}")
    print(f"Successful executions: {sum(1 for e in history.executions if e.status == 'completed')}")
    print(f"Failed executions: {sum(1 for e in history.executions if e.status == 'failed')}")
    print(f"Dataset final size: {dataset.n_samples} samples × {dataset.n_features} features")
    print(f"Features added via augmentation: {len(dataset.added_features)} batches")

    print("\n🎉 Enhanced Pipeline Demo Complete!")
    print("=" * 50)


def demo_parallelization_comparison():
    """
    Demonstrate different parallelization strategies
    """
    print("\n🔀 Parallelization Strategy Comparison")
    print("=" * 50)

    # Create test data
    n_jobs_options = [1, 2, 4]
    backends = ['threading', 'loky']  # Skip multiprocessing for demo

    test_data = np.random.random((1000, 100))

    def mock_heavy_computation(data_chunk, delay=0.1):
        """Simulate heavy computation"""
        import time
        time.sleep(delay)
        return np.sum(data_chunk, axis=1)

    print("Testing parallelization with different configurations:")

    for backend in backends:
        print(f"\n🔧 Backend: {backend}")
        for n_jobs in n_jobs_options:
            try:
                start_time = datetime.now()

                # Split data for parallel processing
                chunks = np.array_split(test_data, n_jobs)

                # Process in parallel
                results = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(mock_heavy_computation)(chunk, 0.05) for chunk in chunks
                )

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                print(f"  n_jobs={n_jobs}: {duration:.2f}s ({len(results)} chunks)")

            except Exception as e:
                print(f"  n_jobs={n_jobs}: Failed ({str(e)})")

    print("\n💡 Recommendations:")
    print("- Use 'threading' backend for I/O-bound tasks")
    print("- Use 'loky' backend for CPU-bound tasks")
    print("- Start with n_jobs=2 for moderate parallelization")
    print("- Monitor memory usage with high n_jobs values")


if __name__ == "__main__":
    # Run comprehensive demo
    demo_enhanced_pipeline_features()

    # Run parallelization comparison
    demo_parallelization_comparison()

    print("\n📚 For more information, see:")
    print("- ENHANCED_RUNNER_SUMMARY.md")
    print("- PARALLELIZATION_PERFORMANCE_ANALYSIS.md")
    print("- PIPELINE_SERIALIZER_ROADMAP.md")
