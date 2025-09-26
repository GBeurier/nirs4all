#!/usr/bin/env python3
"""
Example demonstrating the use_full_train_for_final option in NIRS4ALL.

This example shows how to use cross-validation for hyperparameter optimization
but train the final model on the full training dataset instead of individual folds.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.config import PipelineConfig

# Example dataset configuration
dataset_config = {
    'source': ['sample_data/regression/Protein_NIR.xlsx'],
    'y': 'Protein',
    'folds': 3,  # Use 3 folds for optimization
    'train': 0.7,
    'val': 0.15,
    'test': 0.15,
    'random_state': 42
}

def demonstrate_full_train_option():
    """Demonstrate the use_full_train_for_final option."""
    print("🎯 Full Training Data Option Example")
    print("=" * 50)

    print("""
Traditional Approach:
1. Optimize parameters using cross-validation folds
2. Train separate models on each fold with optimized parameters
3. Get predictions from multiple fold-specific models

NEW Full Training Approach:
1. Optimize parameters using cross-validation folds
2. Combine ALL training data from all folds
3. Train a SINGLE model on the combined training data
4. Get predictions from one unified model
""")

    # Configuration with full training option
    full_train_config = {
        "name": "full_train_demo",
        "steps": [
            {
                "name": "pls_full_train",
                "controller": "sklearn",
                "model": PLSRegression(),
                "finetune_params": {
                    "cv_mode": "per_fold",
                    "param_strategy": "global_average",  # Use global average for best generalization
                    "use_full_train_for_final": True,   # ⭐ KEY NEW OPTION
                    "n_trials": 10,
                    "verbose": 1,
                    "model_params": {
                        "n_components": ("int", 1, 12)
                    },
                    "train_params": {
                        "verbose": 0  # Silent during optimization
                    }
                }
            }
        ]
    }

    # Run the pipeline
    from nirs4all.dataset.loader import get_dataset
    data = get_dataset(dataset_config)
    config = PipelineConfig(full_train_config, "full_train_example")
    runner = PipelineRunner()

    print("🚀 Running optimization with full training option...")
    res_dataset, history, pipeline = runner.run(config, data)

    # Analyze results
    predictions = res_dataset._predictions
    print(f"\n✅ Pipeline completed!")
    print(f"📊 Generated {len(predictions)} prediction sets")

    if len(predictions) > 0:
        print(f"\nPrediction keys: {predictions.list_keys()}")

        # Look for the full training prediction
        full_train_preds = [k for k in predictions.list_keys() if 'global_avg' in k]

        if full_train_preds:
            print(f"\nFound full training prediction: {full_train_preds[0]}")

            # Get the prediction data
            pred_key_parts = full_train_preds[0].split('_', 3)
            if len(pred_key_parts) >= 4:
                dataset_name, pipeline_name, model_name, partition_name = pred_key_parts
                pred_data = predictions.get_prediction_data(
                    dataset_name, pipeline_name, model_name, partition_name
                )

                if pred_data:
                    from sklearn.metrics import mean_squared_error, r2_score

                    y_true = pred_data['y_true'].flatten()
                    y_pred = pred_data['y_pred'].flatten()

                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    rmse = np.sqrt(mse)

                    print(f"\n📈 Full Training Model Performance:")
                    print(f"  RMSE: {rmse:.4f}")
                    print(f"  R²:   {r2:.4f}")
                    print(f"  Training samples used: {pred_data['metadata'].get('training_samples', 'N/A')}")
                    print(f"  Test samples: {len(y_true)}")

    print(f"\n🎯 Benefits of use_full_train_for_final=True:")
    print(f"  ✓ Rigorous hyperparameter optimization using CV")
    print(f"  ✓ Maximum training data utilization for final model")
    print(f"  ✓ Single unified model for deployment")
    print(f"  ✓ Often better performance due to more training data")
    print(f"  ✓ Simpler model management and deployment")

def compare_approaches():
    """Compare traditional fold-based training vs full training."""
    print("\n🔬 Comparison: Fold-based vs Full Training")
    print("=" * 50)

    approaches = [
        ("fold_based", False, "Traditional: separate models per fold"),
        ("full_train", True, "NEW: single model on full data")
    ]

    results = {}

    for approach_name, use_full_train, description in approaches:
        print(f"\nTesting {approach_name} ({description})...")

        config = {
            "name": f"comparison_{approach_name}",
            "steps": [{
                "name": "pls_model",
                "controller": "sklearn",
                "model": PLSRegression(),
                "finetune_params": {
                    "cv_mode": "per_fold",
                    "param_strategy": "global_average",
                    "use_full_train_for_final": use_full_train,  # Key difference
                    "n_trials": 5,  # Quick comparison
                    "verbose": 0,
                    "model_params": {
                        "n_components": ("int", 1, 8)
                    }
                }
            }]
        }

        try:
            from nirs4all.dataset.loader import get_dataset
            import time

            data = get_dataset(dataset_config)
            pipeline_config = PipelineConfig(config, f"comp_{approach_name}")
            runner = PipelineRunner()

            start_time = time.time()
            res_dataset, _, _ = runner.run(pipeline_config, data)
            execution_time = time.time() - start_time

            # Analyze predictions
            predictions = res_dataset._predictions
            pred_keys = predictions.list_keys()

            print(f"  ✅ Completed in {execution_time:.1f}s")
            print(f"     Generated {len(pred_keys)} prediction sets")

            if pred_keys:
                # Get performance from first prediction
                first_key = pred_keys[0]
                key_parts = first_key.split('_', 3)
                if len(key_parts) >= 4:
                    pred_data = predictions.get_prediction_data(*key_parts)
                    if pred_data:
                        from sklearn.metrics import mean_squared_error, r2_score
                        y_true = pred_data['y_true'].flatten()
                        y_pred = pred_data['y_pred'].flatten()
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        r2 = r2_score(y_true, y_pred)

                        results[approach_name] = {
                            'rmse': rmse,
                            'r2': r2,
                            'time': execution_time,
                            'predictions': len(pred_keys),
                            'samples': len(y_true)
                        }

                        print(f"     RMSE: {rmse:.4f}, R²: {r2:.4f}")
                        print(f"     Test samples: {len(y_true)}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")

    # Comparison summary
    if len(results) == 2:
        fold_based = results['fold_based']
        full_train = results['full_train']

        print(f"\n📊 COMPARISON SUMMARY:")
        print(f"  Prediction Sets Generated:")
        print(f"    Fold-based:  {fold_based['predictions']} (separate models)")
        print(f"    Full training: {full_train['predictions']} (unified model)")

        print(f"  Performance (RMSE):")
        print(f"    Fold-based:  {fold_based['rmse']:.4f}")
        print(f"    Full training: {full_train['rmse']:.4f}")

        if full_train['rmse'] < fold_based['rmse']:
            improvement = ((fold_based['rmse'] - full_train['rmse']) / fold_based['rmse']) * 100
            print(f"    🏆 Full training is {improvement:.1f}% better!")

        print(f"  Test Samples:")
        print(f"    Fold-based:  {fold_based['samples']} samples")
        print(f"    Full training: {full_train['samples']} samples")

        print(f"\n💡 Full training advantages:")
        print(f"  ✓ Uses all available training data")
        print(f"  ✓ Single model for deployment")
        print(f"  ✓ Often better performance")
        print(f"  ✓ Simpler model management")

if __name__ == "__main__":
    # Run the main demonstration
    demonstrate_full_train_option()

    # Run comparison if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare_approaches()
    else:
        print(f"\nTo run comparison: python {sys.argv[0]} compare")