"""
Demo: Enhanced Pipeline Runner with joblib and History Tracking

This demo shows the improved PipelineRunner with:
1. joblib parallelization for better ML performance
2. Execution history tracking with serialization
3. Pipeline saving capabilities
"""
import numpy as np
from typing import Dict, Any

# Import enhanced components
from PipelineRunner_enhanced import PipelineRunner
from PipelineHistory import PipelineHistory

def create_demo_dataset():
    """Create a mock SpectraDataset for demonstration"""
    class MockSpectraDataset:
        def __init__(self):
            self.data = np.random.randn(100, 50)  # 100 samples, 50 features
            self.labels = np.random.randint(0, 3, 100)  # 3 classes
            self.sample_ids = [f"sample_{i}" for i in range(100)]
            self.partitions = {
                'train': list(range(80)),
                'test': list(range(80, 100))
            }

        def select(self, partition=None, **filters):
            if partition == 'train':
                indices = self.partitions['train']
            elif partition == 'test':
                indices = self.partitions['test']
            else:
                indices = list(range(100))

            mock_view = MockSpectraDataset()
            mock_view.data = self.data[indices]
            mock_view.labels = self.labels[indices]
            mock_view.sample_ids = [self.sample_ids[i] for i in indices]
            return mock_view

        def get_features(self):
            return self.data

        def __len__(self):
            return len(self.data)

    return MockSpectraDataset()

def create_demo_pipeline_config():
    """Create a demo pipeline configuration"""
    return [
        # Preprocessing steps
        {
            "context_filter": {
                "partition": "train"
            }
        },

        # Feature augmentation with multiple transformers (will run in parallel)
        {
            "feature_augmentation": [
                {
                    "type": "StandardScaler"
                },
                {
                    "type": "PCA",
                    "params": {"n_components": 10}
                },
                {
                    "type": "MinMaxScaler"
                }
            ]
        },

        # Model training
        {
            "model": {
                "type": "RandomForestClassifier",
                "params": {
                    "n_estimators": 50,
                    "random_state": 42
                }
            }
        }
    ]

def demo_basic_pipeline_execution():
    """Demo 1: Basic pipeline execution with history tracking"""
    print("=" * 60)
    print("DEMO 1: Basic Pipeline Execution with History")
    print("=" * 60)

    # Create runner with joblib parallelization
    runner = PipelineRunner(
        max_workers=2,  # Use 2 workers for demo
        backend='threading',  # Use threading backend for demo
        verbose=1,  # Show joblib verbose output
        continue_on_error=False
    )

    # Create demo data and config
    dataset = create_demo_dataset()
    pipeline_config = create_demo_pipeline_config()

    print(f"📊 Dataset: {len(dataset)} samples, {dataset.get_features().shape[1]} features")
    print(f"⚙️ Pipeline: {len(pipeline_config)} steps")
    print()

    # Run pipeline - now returns dataset AND history
    try:
        result_dataset, execution_history = runner.run_pipeline(pipeline_config, dataset)

        print("✅ Pipeline completed successfully!")
        print(f"📈 Final dataset: {len(result_dataset)} samples")
        print()

        # Show execution summary
        print("📋 EXECUTION SUMMARY:")
        summary = execution_history.get_execution_summary()
        if summary['executions']:
            exec_info = summary['executions'][0]
            print(f"  • Duration: {exec_info['duration']:.2f} seconds")
            print(f"  • Total steps: {exec_info['total_steps']}")
            print(f"  • Completed: {exec_info['completed_steps']}")
            print(f"  • Failed: {exec_info['failed_steps']}")
            print(f"  • Status: {exec_info['status']}")

        return result_dataset, execution_history

    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        return None, None

def demo_pipeline_saving():
    """Demo 2: Pipeline saving and serialization"""
    print("\n" + "=" * 60)
    print("DEMO 2: Pipeline Saving and Serialization")
    print("=" * 60)

    # Run a pipeline first
    runner = PipelineRunner(max_workers=2, backend='threading')
    dataset = create_demo_dataset()
    pipeline_config = create_demo_pipeline_config()

    result_dataset, history = runner.run_pipeline(pipeline_config, dataset)

    if history is None:
        print("❌ Cannot save - pipeline failed")
        return

    # Save pipeline in different formats
    save_formats = [
        ('pipeline_demo.zip', 'ZIP Bundle'),
        ('pipeline_demo.pkl', 'Pickle Bundle'),
        ('pipeline_demo.json', 'JSON Metadata')
    ]

    for filepath, format_name in save_formats:
        try:
            runner.save_pipeline(filepath, include_dataset=True, dataset=result_dataset)
            print(f"✅ Saved as {format_name}: {filepath}")
        except Exception as e:
            print(f"❌ Failed to save {format_name}: {str(e)}")

    print()
    print("💾 Pipeline artifacts saved successfully!")
    print("   These files contain:")
    print("   • Original pipeline configuration")
    print("   • Execution history and logs")
    print("   • Fitted transformers and models")
    print("   • Dataset with applied transformations")

def demo_parallel_performance():
    """Demo 3: Parallel vs Sequential Performance Comparison"""
    print("\n" + "=" * 60)
    print("DEMO 3: Parallel vs Sequential Performance")
    print("=" * 60)

    import time

    # Create a more complex pipeline for performance testing
    complex_config = [
        {
            "feature_augmentation": [
                {"type": "StandardScaler"},
                {"type": "MinMaxScaler"},
                {"type": "PCA", "params": {"n_components": 15}},
                {"type": "PCA", "params": {"n_components": 10}},
                {"type": "PCA", "params": {"n_components": 5}},
            ]
        }
    ]

    dataset = create_demo_dataset()

    # Test configurations
    test_configs = [
        (1, 'sequential', 'Sequential execution'),
        (2, 'threading', 'Parallel (threading)'),
        (2, 'loky', 'Parallel (loky)')
    ]

    results = []

    for max_workers, backend, description in test_configs:
        print(f"\n🧪 Testing: {description}")

        runner = PipelineRunner(
            max_workers=max_workers if max_workers > 1 else 1,
            backend=backend,
            verbose=0  # Quiet for performance testing
        )

        start_time = time.time()
        try:
            result_dataset, history = runner.run_pipeline(complex_config, dataset)
            end_time = time.time()
            duration = end_time - start_time

            print(f"  ⏱️ Duration: {duration:.2f} seconds")
            results.append((description, duration, '✅'))

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"  ❌ Failed after {duration:.2f} seconds: {str(e)}")
            results.append((description, duration, '❌'))

    # Performance summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"{'Method':<20} {'Time (s)':<10} {'Status':<8} {'Speedup':<10}")
    print("-" * 50)

    baseline_time = results[0][1] if results else 1.0

    for description, duration, status in results:
        speedup = f"{baseline_time/duration:.2f}x" if duration > 0 else "N/A"
        print(f"{description:<20} {duration:<10.2f} {status:<8} {speedup:<10}")

def demo_error_handling():
    """Demo 4: Error handling and continue-on-error functionality"""
    print("\n" + "=" * 60)
    print("DEMO 4: Error Handling and Recovery")
    print("=" * 60)

    # Create a pipeline that will have some failures
    error_prone_config = [
        {"type": "StandardScaler"},  # This should work
        {"type": "NonExistentTransformer"},  # This will fail
        {"type": "MinMaxScaler"},  # This should work if continue_on_error=True
    ]

    dataset = create_demo_dataset()

    # Test with continue_on_error=False
    print("🧪 Testing with continue_on_error=False:")
    runner_strict = PipelineRunner(continue_on_error=False)

    try:
        result, history = runner_strict.run_pipeline(error_prone_config, dataset)
        print("  ✅ Completed (unexpected)")
    except Exception as e:
        print(f"  ❌ Failed as expected: {str(e)}")

    # Test with continue_on_error=True
    print("\n🧪 Testing with continue_on_error=True:")
    runner_lenient = PipelineRunner(continue_on_error=True)

    try:
        result, history = runner_lenient.run_pipeline(error_prone_config, dataset)
        print("  ✅ Completed despite errors")

        # Show which steps failed
        if history and history.executions:
            execution = history.executions[-1]
            for step in execution.steps:
                status_icon = "✅" if step.status == 'completed' else "❌"
                print(f"    {status_icon} Step {step.step_number}: {step.step_description}")
                if step.error_message:
                    print(f"      Error: {step.error_message}")
    except Exception as e:
        print(f"  ❌ Still failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Enhanced Pipeline Runner Demo")
    print("Features: joblib parallelization, history tracking, pipeline saving")
    print()

    # Run all demos
    try:
        demo_basic_pipeline_execution()
        demo_pipeline_saving()
        demo_parallel_performance()
        demo_error_handling()

        print("\n" + "=" * 60)
        print("🎉 All demos completed!")
        print("=" * 60)
        print()
        print("Key improvements demonstrated:")
        print("✅ joblib parallelization for better ML performance")
        print("✅ Comprehensive execution history tracking")
        print("✅ Pipeline saving and serialization")
        print("✅ Configurable backends (threading, loky, multiprocessing)")
        print("✅ Robust error handling with continue-on-error")
        print("✅ Performance monitoring and comparison")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
