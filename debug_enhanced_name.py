#!/usr/bin/env python3

import json

# Load predictions and check enhanced_model_name format
with open('results/regression/regression_predictions.json', 'r') as f:
    data = json.load(f)

print(f"Found {len(data)} predictions")

# Look for virtual models (avg, w-avg)
virtual_models = []
regular_models = []

for pred in data:
    fold_idx = pred.get('fold_idx', 'N/A')
    if 'avg' in str(fold_idx):
        virtual_models.append(pred)
    else:
        regular_models.append(pred)

print(f"\nRegular models: {len(regular_models)}")
print(f"Virtual models: {len(virtual_models)}")

# Show format examples
print("\n=== REGULAR MODEL EXAMPLES ===")
for i, pred in enumerate(regular_models[:3]):
    enhanced_name = pred.get('enhanced_model_name', 'N/A')
    fold_idx = pred.get('fold_idx', 'N/A')
    step = pred.get('step', 'N/A')
    model_name = pred.get('model', 'N/A')
    print(f"{i+1}. Enhanced: '{enhanced_name}' | Fold: {fold_idx} | Step: {step} | Model: {model_name}")

print("\n=== VIRTUAL MODEL EXAMPLES ===")
for i, pred in enumerate(virtual_models[:3]):
    enhanced_name = pred.get('enhanced_model_name', 'N/A')
    fold_idx = pred.get('fold_idx', 'N/A')
    step = pred.get('step', 'N/A')
    model_name = pred.get('model', 'N/A')
    print(f"{i+1}. Enhanced: '{enhanced_name}' | Fold: {fold_idx} | Step: {step} | Model: {model_name}")