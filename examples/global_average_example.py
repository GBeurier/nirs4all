#!/usr/bin/env python3
"""
Simple example demonstrating the new GLOBAL_AVERAGE parameter strategy in NIRS4ALL.

This example shows how to use the global_average strategy which optimizes
parameters by evaluating them simultaneously across all cross-validation folds.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.config import PipelineConfigs

# Example dataset configuration
dataset_config = {
    'source': ['sample_data/regression/Protein_NIR.xlsx'],
    'y': 'Protein',
    'folds': 3,  # Use 3 folds for faster execution
    'train': 0.7,
    'val': 0.15,
    'test': 0.15,
    'random_state': 42
}

def example_global_average():
    """Demonstrate global_average parameter strategy."""
    print("🌍 GLOBAL_AVERAGE Parameter Strategy Example")
    print("=" * 50)

    # Pipeline configuration using global_average strategy
    pipeline_config = {
        "name": "global_average_demo",
        "steps": [
            {
                "name": "pls_model",
                "controller": "sklearn",
                "model": PLSRegression(),
                "finetune_params": {
                    "cv_mode": "per_fold",
                    "param_strategy": "global_average",  # New strategy!
                    "n_trials": 8,  # Fewer trials due to higher computational cost
                    "verbose": 1,
                    "model_params": {
                        "n_components": ("int", 1, 15)  # Optimize number of components
                    },
                    "train_params": {
                        "verbose": 0  # Silent training during optimization
                    }
                }
            }
        ]
    }

    print("""
How GLOBAL_AVERAGE works:
1. Each parameter candidate (e.g., n_components=5) is tested on ALL folds
2. The validation scores from all folds are averaged
3. The parameter set with the best average score is selected
4. This gives a single parameter set optimized for average performance
5. Final models are trained on each fold using these globally optimal parameters

This approach is more generalizable but computationally more expensive.
""")

    # Load dataset and run pipeline
    from nirs4all.dataset.loader import get_dataset
    data = get_dataset(dataset_config)
    config = PipelineConfigs(pipeline_config, "global_avg_example")
    runner = PipelineRunner()

    print("Running optimization... (this may take a moment)")
    res_dataset, history, pipeline = runner.run(config, data)

    # Analyze results
    predictions = res_dataset._predictions
    print(f"\n✅ Optimization completed!")
    print(f"📊 Generated {len(predictions)} prediction sets")

    if len(predictions) > 0:
        # Calculate performance metrics
        from sklearn.metrics import mean_squared_error, r2_score

        # Combine all fold predictions
        combined_pred = res_dataset._predictions.combine_folds(
            "sample_data", config.name, "PLSRegression", "test_fold"
        )

        if combined_pred:
            y_true = combined_pred['y_true'].flatten()
            y_pred = combined_pred['y_pred'].flatten()

            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)

            print(f"\nCross-Validation Performance:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²:   {r2:.4f}")
            print(f"  Samples: {len(y_true)}")

    print("\n🎯 Key Benefits of GLOBAL_AVERAGE:")
    print("  ✓ Single optimal parameter set for all folds")
    print("  ✓ Optimized for average performance across folds")
    print("  ✓ More generalizable than per-fold optimization")
    print("  ✓ Reduces fold-specific overfitting")

def comparison_example():
    """Compare global_average vs per_fold_best strategies."""
    print("\n🔬 Comparison: GLOBAL_AVERAGE vs PER_FOLD_BEST")
    print("=" * 55)

    strategies = [
        ("per_fold_best", "Traditional approach"),
        ("global_average", "New global optimization")
    ]

    results = {}

    for strategy, description in strategies:
        print(f"\nTesting {strategy} ({description})...")

        pipeline_config = {
            "name": f"comparison_{strategy}",
            "steps": [
                {
                    "name": "pls_model",
                    "controller": "sklearn",
                    "model": PLSRegression(),
                    "finetune_params": {
                        "cv_mode": "per_fold",
                        "param_strategy": strategy,
                        "n_trials": 5,
                        "verbose": 0,  # Silent for comparison
                        "model_params": {
                            "n_components": ("int", 1, 10)
                        }
                    }
                }
            ]
        }

        # Run pipeline
        from nirs4all.dataset.loader import get_dataset
        import time

        data = get_dataset(dataset_config)
        config = PipelineConfigs(pipeline_config, f"comp_{strategy}")
        runner = PipelineRunner()

        start_time = time.time()
        try:
            res_dataset, _, _ = runner.run(config, data)
            execution_time = time.time() - start_time

            # Get performance
            combined_pred = res_dataset._predictions.combine_folds(
                "sample_data", config.name, "PLSRegression", "test_fold"
            )

            if combined_pred:
                from sklearn.metrics import mean_squared_error
                y_true = combined_pred['y_true'].flatten()
                y_pred = combined_pred['y_pred'].flatten()
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                results[strategy] = {
                    'time': execution_time,
                    'rmse': rmse,
                    'success': True
                }

                print(f"  ✅ RMSE: {rmse:.4f}, Time: {execution_time:.1f}s")
            else:
                results[strategy] = {'success': False, 'error': 'No predictions generated'}

        except Exception as e:
            results[strategy] = {'success': False, 'error': str(e)}
            print(f"  ❌ Failed: {e}")

    # Summary
    if all(r.get('success', False) for r in results.values()):
        print(f"\n📈 COMPARISON SUMMARY:")

        per_fold_time = results['per_fold_best']['time']
        global_avg_time = results['global_average']['time']
        time_ratio = global_avg_time / per_fold_time

        print(f"  Execution Time:")
        print(f"    per_fold_best:  {per_fold_time:.1f}s")
        print(f"    global_average: {global_avg_time:.1f}s ({time_ratio:.1f}x slower)")

        print(f"  Performance (RMSE):")
        print(f"    per_fold_best:  {results['per_fold_best']['rmse']:.4f}")
        print(f"    global_average: {results['global_average']['rmse']:.4f}")

        if results['global_average']['rmse'] < results['per_fold_best']['rmse']:
            print(f"  🏆 global_average achieved better generalization!")
        else:
            print(f"  📊 per_fold_best achieved better performance on this dataset")

if __name__ == "__main__":
    # Run the main example
    example_global_average()

    # Optionally run comparison
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        comparison_example()
    else:
        print(f"\nTo run comparison: python {sys.argv[0]} compare")