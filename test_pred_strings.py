#!/usr/bin/env python3
"""Test the updated pred_short_string and pred_long_string functions"""

from nirs4all.dataset.predictions import Predictions

# Load the predictions file
predictions = Predictions()
predictions.load_from_file("results/regression/predictions.json")

# Get a test entry
result = predictions.top_k(k=1, metric='rmse')
if result:
    entry = result[0]

    print("Testing pred_short_string:")
    print("Without metrics:", Predictions.pred_short_string(entry))
    print("With single metric:", Predictions.pred_short_string(entry, metrics="rmse"))
    print("With multiple metrics:", Predictions.pred_short_string(entry, metrics=["rmse", "r2", "mae"]))

    print("\nTesting pred_long_string:")
    print("Without metrics:", Predictions.pred_long_string(entry))
    print("With single metric:", Predictions.pred_long_string(entry, metrics="rmse"))
    print("With multiple metrics:", Predictions.pred_long_string(entry, metrics=["rmse", "r2", "mae"]))
else:
    print("No test data available")