#!/usr/bin/env python3
"""Test default partition filter"""

from nirs4all.dataset.predictions import Predictions

# Load the predictions file
predictions = Predictions()
predictions.load_from_file("results/regression/predictions.json")

# Test top_k with default partition filter
print("Testing top_k with default partition filter...")
result = predictions.top_k(k=3, metric='rmse')
print(f"Results: {len(result)} items")
if result:
    print(f"First partition: {result[0]['partition']}")
    print(f"First computed_score: {result[0].get('computed_score', 'N/A')}")

# Test with explicit train partition
print("\nTesting top_k with explicit train partition...")
result_train = predictions.top_k(k=3, metric='rmse', partition='train')
print(f"Results: {len(result_train)} items")
if result_train:
    print(f"First partition: {result_train[0]['partition']}")

# Test bottom_k
print("\nTesting bottom_k with default partition filter...")
result_bottom = predictions.bottom_k(k=3, metric='rmse')
print(f"Results: {len(result_bottom)} items")
if result_bottom:
    print(f"First partition: {result_bottom[0]['partition']}")
    print(f"First computed_score: {result_bottom[0].get('computed_score', 'N/A')}")