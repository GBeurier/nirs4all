#!/usr/bin/env python3
"""
Comprehensive test script for NIRS4ALL Parameter Strategies

This script demonstrates and compares different parameter optimization strategies
including the new GLOBAL_AVERAGE approach for cross-validation.

The script tests:
1. Traditional per_fold_best strategy
2. Global_best strategy
3. New global_average strategy (simultaneous fold optimization)
4. Different CV modes (simple, per_fold, nested)

It provides timing comparisons and performance metrics for each approach.
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# Import NIRS4ALL components
from nirs4all.dataset.loader import get_dataset
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.controllers.registry import reset_registry
import nirs4all.controllers


# Sample dataset configuration (adjust paths as needed)
dataset_config = {
    'source': [
        'sample_data/regression/Protein_NIR.xlsx'
    ],
    'y': 'Protein',
    'folds': 5,  # Enable cross-validation
    'train': 0.7,
    'val': 0.15,
    'test': 0.15,
    'random_state': 42
}

def create_test_pipeline(param_strategy, cv_mode='per_fold', n_trials=10, verbose=1):
    """Create a test pipeline configuration with specified parameter strategy."""
    return {
        "name": f"test_{param_strategy}_{cv_mode}",
        "steps": [
            {
                "name": "test_model",
                "controller": "sklearn",
                "model": PLSRegression(),
                "finetune_params": {
                    "cv_mode": cv_mode,
                    "param_strategy": param_strategy,
                    "n_trials": n_trials,
                    "verbose": verbose,
                    "model_params": {
                        "n_components": ("int", 1, 10)
                    }
                }
            }
        ]
    }

def run_strategy_test(param_strategy, cv_mode='per_fold', n_trials=5):
    """Run a single parameter strategy test and return results."""
    print(f"\n{'='*60}")
    print(f"Testing: {param_strategy.upper()} with {cv_mode.upper()} CV")
    print(f"{'='*60}")

    # Load dataset
    data = get_dataset(dataset_config)

    # Create pipeline configuration
    pipeline_config = create_test_pipeline(
        param_strategy=param_strategy,
        cv_mode=cv_mode,
        n_trials=n_trials,
        verbose=1
    )

    # Create and run pipeline
    config = PipelineConfigs(pipeline_config, f"test_{param_strategy}_{cv_mode}")
    runner = PipelineRunner()

    # Time the execution
    start_time = time.time()

    try:
        res_dataset, history, pipeline = runner.run(config, data)
        execution_time = time.time() - start_time

        # Collect results
        predictions = res_dataset._predictions
        num_predictions = len(predictions)

        # Calculate performance metrics if predictions exist
        performance_metrics = {}
        if num_predictions > 0:
            # Get all prediction data
            all_predictions = predictions.list_keys()
            total_samples = 0
            total_mse = 0

            for pred_key in all_predictions:
                parts = pred_key.split('_', 3)
                if len(parts) >= 4:
                    dataset_name, pipeline_name, model_name, partition_name = parts
                    pred_data = predictions.get_prediction_data(
                        dataset_name, pipeline_name, model_name, partition_name
                    )
                    if pred_data and 'test' in partition_name:
                        y_true = pred_data['y_true'].flatten()
                        y_pred = pred_data['y_pred'].flatten()
                        mse = np.mean((y_true - y_pred) ** 2)
                        total_mse += mse * len(y_true)
                        total_samples += len(y_true)

            if total_samples > 0:
                performance_metrics['avg_mse'] = total_mse / total_samples
                performance_metrics['rmse'] = np.sqrt(performance_metrics['avg_mse'])

        return {
            'strategy': param_strategy,
            'cv_mode': cv_mode,
            'execution_time': execution_time,
            'num_predictions': num_predictions,
            'success': True,
            'performance': performance_metrics,
            'error': None
        }

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            'strategy': param_strategy,
            'cv_mode': cv_mode,
            'execution_time': execution_time,
            'num_predictions': 0,
            'success': False,
            'performance': {},
            'error': str(e)
        }

def run_comprehensive_comparison():
    """Run comprehensive comparison of parameter strategies."""
    print("🧪 NIRS4ALL Parameter Strategy Comprehensive Test")
    print("=" * 80)

    # Test configurations: (strategy, cv_mode, n_trials)
    test_configs = [
        # Standard strategies with per_fold CV
        ('per_fold_best', 'per_fold', 5),
        ('global_best', 'per_fold', 5),

        # New global_average strategy
        ('global_average', 'per_fold', 5),  # 5 trials x 5 folds = 25 model trainings per trial

        # Simple CV comparisons
        ('per_fold_best', 'simple', 8),  # Not really per-fold in simple mode
        ('global_average', 'simple', 8),  # Should work with simple too

        # Nested CV (computationally expensive)
        # ('per_fold_best', 'nested', 3),
        # ('global_average', 'nested', 3),  # Very expensive: 3 trials x 5 outer x 3 inner = 45 models per trial
    ]

    results = []

    for strategy, cv_mode, n_trials in test_configs:
        result = run_strategy_test(strategy, cv_mode, n_trials)
        results.append(result)

        # Print immediate results
        if result['success']:
            print(f"✅ {strategy} ({cv_mode}): {result['execution_time']:.1f}s")
            if result['performance']:
                print(f"   RMSE: {result['performance'].get('rmse', 'N/A'):.4f}")
        else:
            print(f"❌ {strategy} ({cv_mode}): FAILED - {result['error']}")

        # Brief pause between tests
        time.sleep(1)

    # Summary analysis
    print(f"\n{'='*80}")
    print("📊 SUMMARY ANALYSIS")
    print(f"{'='*80}")

    successful_results = [r for r in results if r['success']]

    if successful_results:
        print("\nExecution Time Comparison:")
        print("-" * 40)
        for result in successful_results:
            strategy = result['strategy']
            cv_mode = result['cv_mode']
            time_str = f"{result['execution_time']:.1f}s"
            print(f"{strategy:15} ({cv_mode:8}): {time_str:>8}")

        print("\nPerformance Comparison (RMSE):")
        print("-" * 40)
        for result in successful_results:
            if result['performance'] and 'rmse' in result['performance']:
                strategy = result['strategy']
                cv_mode = result['cv_mode']
                rmse = result['performance']['rmse']
                print(f"{strategy:15} ({cv_mode:8}): {rmse:.4f}")

        # Find best performing strategy
        performance_results = [r for r in successful_results if r['performance'] and 'rmse' in r['performance']]
        if performance_results:
            best_result = min(performance_results, key=lambda x: x['performance']['rmse'])
            print(f"\n🏆 Best Performance: {best_result['strategy']} ({best_result['cv_mode']}) - RMSE: {best_result['performance']['rmse']:.4f}")

        # Find fastest strategy
        fastest_result = min(successful_results, key=lambda x: x['execution_time'])
        print(f"⚡ Fastest Execution: {fastest_result['strategy']} ({fastest_result['cv_mode']}) - {fastest_result['execution_time']:.1f}s")

        # Computational cost analysis
        print(f"\nComputational Cost Analysis:")
        print("-" * 40)
        baseline_time = min(r['execution_time'] for r in successful_results)
        for result in successful_results:
            strategy = result['strategy']
            cv_mode = result['cv_mode']
            relative_cost = result['execution_time'] / baseline_time
            print(f"{strategy:15} ({cv_mode:8}): {relative_cost:.1f}x baseline")

    else:
        print("❌ No tests completed successfully!")

    return results

def demonstrate_global_average_concept():
    """Demonstrate the concept behind global_average parameter optimization."""
    print(f"\n{'='*80}")
    print("🔬 GLOBAL_AVERAGE CONCEPT DEMONSTRATION")
    print(f"{'='*80}")

    print("""
The GLOBAL_AVERAGE strategy works differently from traditional approaches:

Traditional Per-Fold Strategy:
1. Fold 1: Try n_components=5 → RMSE=0.25 → Best for fold 1
2. Fold 2: Try n_components=7 → RMSE=0.23 → Best for fold 2
3. Fold 3: Try n_components=6 → RMSE=0.24 → Best for fold 3
Result: Each fold gets different optimal parameters

Global_Average Strategy:
1. Try n_components=5 on ALL folds → Average RMSE=0.26
2. Try n_components=6 on ALL folds → Average RMSE=0.24
3. Try n_components=7 on ALL folds → Average RMSE=0.25
Result: n_components=6 is best on average across all folds

Benefits:
- More generalizable parameters (optimized for average performance)
- Single consistent parameter set across all folds
- Reduces overfitting to individual fold characteristics
- Better for production deployment

Drawbacks:
- Much higher computational cost (each trial trains on all folds)
- May not be optimal for any single fold
- Requires more careful resource management
""")

def run_quick_demo():
    """Run a quick demonstration of the key strategies."""
    print("🚀 Quick Demo: Comparing Key Parameter Strategies")
    print("=" * 60)

    strategies = [
        ('per_fold_best', 'per_fold'),
        ('global_average', 'per_fold')
    ]

    for strategy, cv_mode in strategies:
        print(f"\nTesting {strategy} with {cv_mode} CV...")
        result = run_strategy_test(strategy, cv_mode, n_trials=3)

        if result['success']:
            print(f"✅ Completed in {result['execution_time']:.1f}s")
            if result['performance']:
                print(f"   RMSE: {result['performance'].get('rmse', 'N/A'):.4f}")
        else:
            print(f"❌ Failed: {result['error']}")

if __name__ == "__main__":
    import sys

    print("NIRS4ALL Parameter Strategy Test Suite")
    print("=" * 50)

    mode = sys.argv[1] if len(sys.argv) > 1 else 'demo'

    if mode == 'full':
        print("Running comprehensive comparison (may take several minutes)...")
        run_comprehensive_comparison()
    elif mode == 'concept':
        demonstrate_global_average_concept()
    elif mode == 'demo':
        print("Running quick demo...")
        run_quick_demo()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python test_parameter_strategies.py [demo|full|concept]")
        print("  demo:    Quick demonstration (default)")
        print("  full:    Comprehensive comparison (slow)")
        print("  concept: Explain global_average concept")