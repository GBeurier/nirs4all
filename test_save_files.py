"""
Test script to verify save_files flag works correctly
"""
import sys
from pathlib import Path
from nirs4all.dataset.dataset_configs import DatasetConfigs
from nirs4all.pipeline.runner import PipelineRunner

# Test pipeline with charts
pipeline_config = {
    "test_save_on": {
        "y_chart": {},
        "spectra_chart": {},
    },
    "test_save_off": {
        "y_chart": {},
        "spectra_chart": {},
    }
}

# Simple dataset config
dataset_config = {
    "regression": {
        "source": "examples/datasets/corn_m5.csv"
    }
}

print("=" * 80)
print("Test 1: save_files=True (should create outputs)")
print("=" * 80)
runner_on = PipelineRunner(save_files=True, results_path="test_results")
configs_on = DatasetConfigs(dataset_config)
try:
    predictions_on, _ = runner_on.run(pipeline_config, configs_on)

    # Check if outputs directory was created
    outputs_dir = Path("test_results/outputs")
    if outputs_dir.exists():
        output_folders = list(outputs_dir.iterdir())
        print(f"✅ Outputs directory created with {len(output_folders)} folder(s)")
        for folder in output_folders:
            files = list(folder.glob("*.png"))
            print(f"   - {folder.name}: {len(files)} chart(s)")
    else:
        print("❌ No outputs directory created!")
except Exception as e:
    print(f"❌ Error with save_files=True: {e}")

print("\n" + "=" * 80)
print("Test 2: save_files=False (should NOT create outputs)")
print("=" * 80)
runner_off = PipelineRunner(save_files=False, results_path="test_results")
configs_off = DatasetConfigs(dataset_config)
try:
    predictions_off, _ = runner_off.run(pipeline_config, configs_off)

    # Count outputs created (should be same as before, not more)
    outputs_dir = Path("test_results/outputs")
    if outputs_dir.exists():
        output_folders = list(outputs_dir.iterdir())
        print(f"📁 Outputs directory has {len(output_folders)} folder(s) (should be same as before)")
        # Check if new folder was created
        new_folders = [f for f in output_folders if "test_save_off" in f.name]
        if new_folders:
            print(f"❌ New outputs created when save_files=False!")
        else:
            print(f"✅ No new outputs created with save_files=False")
    else:
        print("✅ No outputs directory (correct if first test didn't create it)")
except Exception as e:
    print(f"❌ Error with save_files=False: {e}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("✅ Test completed!")
print("Check test_results/outputs/ to see the chart organization")
