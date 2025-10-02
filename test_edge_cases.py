#!/usr/bin/env python3
"""Test edge cases for the pred string functions"""

from nirs4all.dataset.predictions import Predictions

# Load the predictions file
predictions = Predictions()
predictions.load_from_file("results/regression/predictions.json")

# Get different types of entries
all_results = predictions.top_k(k=3, metric='rmse')

print("Testing different entry types:")
for i, entry in enumerate(all_results):
    print(f"\nEntry {i+1}:")
    print("Model:", entry['model_name'])
    print("With RMSE and R²:", Predictions.pred_short_string(entry, metrics=["rmse", "r2"]))

    # Test error handling with invalid metric
    print("With invalid metric:", Predictions.pred_short_string(entry, metrics=["invalid_metric", "rmse"]))

    if i == 0:  # Only test first entry thoroughly
        break