"""
Performance test for tab report generation.

Tests the performance improvement from O(N*M) to O(N+M) complexity
after adding prediction indexing.
"""

import time
import numpy as np
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.reports import TabReportManager


def create_test_predictions(n_models=50, n_datasets=10, n_folds=3):
    """Create test predictions buffer for performance testing."""
    predictions = Predictions()

    # Generate predictions for multiple models and datasets
    for dataset_idx in range(n_datasets):
        dataset_name = f"dataset_{dataset_idx}"

        for model_idx in range(n_models):
            model_name = f"model_{model_idx}"
            config_name = f"config_{model_idx}"

            # Add CV fold predictions (train, val, test for each fold)
            for fold_idx in range(n_folds):
                n_samples = 100
                y_true = np.random.randn(n_samples)
                y_pred = y_true + np.random.randn(n_samples) * 0.1

                for partition in ["train", "val", "test"]:
                    predictions.add_prediction(
                        dataset_name=dataset_name,
                        config_name=config_name,
                        model_name=model_name,
                        fold_id=str(fold_idx),
                        partition=partition,
                        y_true=y_true,
                        y_pred=y_pred,
                        test_score=0.5 + np.random.rand() * 0.2,
                        val_score=0.5 + np.random.rand() * 0.2,
                        train_score=0.4 + np.random.rand() * 0.2,
                        n_samples=n_samples,
                        n_features=50,
                        metric="rmse",
                        task_type="regression",
                        step_idx=0,
                        metadata={"sample_id": [f"S{i}" for i in range(n_samples)]},
                    )

            # Add final refit prediction
            n_samples = 100
            y_true = np.random.randn(n_samples)
            y_pred = y_true + np.random.randn(n_samples) * 0.1

            for partition in ["train", "test"]:
                score_dict = {
                    "train": {"rmse": 0.4 + np.random.rand() * 0.2},
                    "val": {"rmse": 0.5 + np.random.rand() * 0.2},
                    "test": {"rmse": 0.5 + np.random.rand() * 0.2},
                }
                predictions.add_prediction(
                    dataset_name=dataset_name,
                    config_name=config_name,
                    model_name=model_name,
                    fold_id="final",
                    partition=partition,
                    y_true=y_true,
                    y_pred=y_pred,
                    test_score=0.5 + np.random.rand() * 0.2,
                    train_score=0.4 + np.random.rand() * 0.2,
                    val_score=0.5 + np.random.rand() * 0.2,
                    scores=score_dict,
                    n_samples=n_samples,
                    n_features=50,
                    metric="rmse",
                    task_type="regression",
                    step_idx=0,
                    refit_context="standalone",
                    metadata={"sample_id": [f"S{i}" for i in range(n_samples)]},
                )

    return predictions


def test_performance():
    """Test performance of per-model summary generation."""
    print("Creating test data...")
    n_models = 50
    n_datasets = 10
    predictions = create_test_predictions(n_models=n_models, n_datasets=n_datasets, n_folds=3)

    # Get refit entries
    refit_entries = predictions.filter_predictions(
        fold_id="final",
        partition="test",
        load_arrays=False
    )

    print(f"\nTest setup:")
    print(f"  Models: {n_models}")
    print(f"  Datasets: {n_datasets}")
    print(f"  Total refit entries: {len(refit_entries)}")
    print(f"  Total predictions in buffer: {len(predictions)}")

    # Test without aggregation
    print("\n" + "="*60)
    print("Testing per-model summary generation (no aggregation)...")
    print("="*60)

    start = time.time()
    summary = TabReportManager.generate_per_model_summary(
        refit_entries,
        ascending=True,
        metric="rmse",
        predictions=predictions,
    )
    elapsed = time.time() - start

    print(f"✓ Generated in {elapsed:.3f} seconds")
    print(f"  ({len(refit_entries)} entries × {len(predictions)} buffer scans)")
    print(f"\nSummary preview (first 500 chars):\n{summary[:500]}")

    # Test with aggregation (more expensive)
    print("\n" + "="*60)
    print("Testing per-model summary generation (WITH aggregation)...")
    print("="*60)

    start = time.time()
    summary_agg = TabReportManager.generate_per_model_summary(
        refit_entries,
        ascending=True,
        metric="rmse",
        aggregate="sample_id",
        predictions=predictions,
    )
    elapsed_agg = time.time() - start

    print(f"✓ Generated in {elapsed_agg:.3f} seconds")
    print(f"  ({len(refit_entries)} entries × {len(predictions)} buffer scans)")
    print(f"\nAggregated summary preview (first 500 chars):\n{summary_agg[:500]}")

    # Performance analysis
    print("\n" + "="*60)
    print("Performance Analysis")
    print("="*60)
    theoretical_ops_old = len(refit_entries) * len(predictions) * 3  # 3 different scans per entry
    theoretical_ops_new = len(predictions) + len(refit_entries) * 3  # Build index once, then O(1) lookups

    print(f"\nTheoretical complexity:")
    print(f"  OLD (O(N*M)): {theoretical_ops_old:,} operations")
    print(f"  NEW (O(N+M)): {theoretical_ops_new:,} operations")
    print(f"  Speedup ratio: {theoretical_ops_old / theoretical_ops_new:.1f}x")

    print(f"\nActual runtime:")
    print(f"  Without aggregation: {elapsed:.3f}s")
    print(f"  With aggregation: {elapsed_agg:.3f}s")

    if elapsed < 1.0 and elapsed_agg < 2.0:
        print(f"\n✅ PASS: Performance is acceptable (< 2s for {len(refit_entries)} models)")
    else:
        print(f"\n⚠️  WARNING: Performance may need further optimization")


if __name__ == "__main__":
    test_performance()
