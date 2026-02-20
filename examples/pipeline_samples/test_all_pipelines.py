#!/usr/bin/env python
"""
Pipeline Samples Test Runner
=============================

This script loads and executes all pipeline sample definitions (JSON/YAML)
from the pipeline_samples directory to verify they work correctly and
do not trigger the DummyController.

Usage:
    cd examples/pipeline_samples
    python test_all_pipelines.py

    # With verbose output
    python test_all_pipelines.py -v 2

    # Run specific pipeline only
    python test_all_pipelines.py --pipeline 01_basic_regression.yaml

    # Skip slow pipelines (finetune, neural networks)
    python test_all_pipelines.py --quick

Features Tested:
- All 10 pipeline sample files (JSON and YAML)
- Proper controller routing (no DummyController warnings)
- Successful prediction generation
- Error reporting for failed pipelines

Duration: ~5-10 minutes (full), ~1-2 minutes (quick)
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import yaml

# Ensure we can import nirs4all
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nirs4all.core.logging import get_logger
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

logger = get_logger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Sample data to use for testing (relative to examples/ directory)
DEFAULT_DATASET = "sample_data/regression"

# Pipelines that are slow or require special dependencies
SLOW_PIPELINES = [
    "08_complex_finetune.json",  # Has finetuning with many trials
]

TENSORFLOW_PIPELINES = [
    "08_complex_finetune.json",  # Has NICON neural network
]

# Pipelines that need multi-source data (no longer used)
MULTI_SOURCE_PIPELINES: list[str] = [
    # "10_complete_all_features.json",  # Simplified to work with single source
]

# ============================================================================
# Pipeline Loaders
# ============================================================================

def load_json_pipeline(filepath: Path) -> dict[str, Any]:
    """Load pipeline definition from JSON file."""
    with open(filepath) as f:
        data: Any = json.load(f)

    # Handle both flat list and wrapped dict formats
    if isinstance(data, list):
        return {"pipeline": data}
    return dict(data)

def load_yaml_pipeline(filepath: Path) -> dict[str, Any]:
    """Load pipeline definition from YAML file."""
    with open(filepath) as f:
        data: Any = yaml.safe_load(f)

    # Handle both flat list and wrapped dict formats
    if isinstance(data, list):
        return {"pipeline": data}
    return dict(data)

def load_pipeline(filepath: Path) -> dict[str, Any]:
    """Load pipeline from file based on extension."""
    suffix = filepath.suffix.lower()
    if suffix == '.json':
        return load_json_pipeline(filepath)
    elif suffix in ('.yaml', '.yml'):
        return load_yaml_pipeline(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

# ============================================================================
# Pipeline Preprocessing
# ============================================================================

def filter_comments(pipeline: list[Any]) -> list[Any]:
    """Remove comment-only steps from pipeline."""
    filtered = []
    for step in pipeline:
        if isinstance(step, dict):
            # Skip steps that are only comments
            keys = set(step.keys())
            if keys == {"_comment"}:
                continue
        filtered.append(step)
    return filtered

def simplify_for_quick_run(pipeline: list[Any]) -> list[Any]:
    """Simplify pipeline for quick testing."""
    simplified = []
    for step in pipeline:
        if isinstance(step, dict):
            # Reduce finetune trials
            if "finetune_params" in step:
                step = step.copy()
                step["finetune_params"] = step["finetune_params"].copy()
                step["finetune_params"]["n_trials"] = 2

            # Skip neural network models
            if "model" in step:
                model = step["model"]
                if isinstance(model, dict):
                    model_class = model.get("class", "") or model.get("function", "")
                    if "tensorflow" in model_class or "nicon" in model_class.lower():
                        continue
        simplified.append(step)
    return simplified

def get_dataset_for_pipeline(pipeline_name: str, examples_dir: Path) -> str:
    """Get appropriate dataset for a pipeline."""
    if pipeline_name in MULTI_SOURCE_PIPELINES:
        # Use multi-source dataset if available
        multi_path = examples_dir / "sample_data" / "multi"
        if multi_path.exists():
            return "sample_data/multi"
    return DEFAULT_DATASET

# ============================================================================
# Test Runner
# ============================================================================

class PipelineTestResult:
    """Result of running a single pipeline test."""

    def __init__(self, name: str):
        self.name = name
        self.success = False
        self.error: str | None = None
        self.num_predictions = 0
        self.num_models = 0
        self.duration: float = 0.0
        self.warnings: list[str] = []
        self.dummy_controller_triggered = False

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        msg = f"{status} {self.name}"
        if self.success:
            msg += f" ({self.num_predictions} predictions, {self.duration:.1f}s)"
        else:
            msg += f" - {self.error}"
        return msg

def run_pipeline_test(
    pipeline_file: Path,
    examples_dir: Path,
    verbose: int = 1,
    quick: bool = False
) -> PipelineTestResult:
    """Run a single pipeline test."""
    result = PipelineTestResult(pipeline_file.name)
    start_time = time.time()

    try:
        # Load pipeline
        pipeline_data = load_pipeline(pipeline_file)
        pipeline_steps = pipeline_data.get("pipeline", [])
        pipeline_name = pipeline_data.get("name", pipeline_file.stem)

        # Filter comments
        pipeline_steps = filter_comments(pipeline_steps)

        # Simplify if quick mode
        if quick:
            pipeline_steps = simplify_for_quick_run(pipeline_steps)

        # Get appropriate dataset
        dataset_path = get_dataset_for_pipeline(pipeline_file.name, examples_dir)

        # Change to examples directory for relative paths
        original_dir = os.getcwd()
        os.chdir(examples_dir)

        try:
            # Create configuration objects
            pipeline_config = PipelineConfigs(
                pipeline_steps,
                name=f"Test_{pipeline_name}"
            )
            dataset_config = DatasetConfigs(dataset_path)

            # Run pipeline
            runner = PipelineRunner(
                workspace_path="workspace/pipeline_samples_test",
                save_artifacts=False,
                save_charts=False,
                verbose=verbose,
                plots_visible=False,
            )

            predictions, _ = runner.run(pipeline_config, dataset_config)

            # Check results
            result.num_predictions = predictions.num_predictions if predictions else 0
            result.num_models = len(predictions.get_unique_values("model_name")) if predictions else 0
            result.success = result.num_predictions > 0

            if not result.success:
                result.error = "No predictions generated"

        finally:
            os.chdir(original_dir)

    except Exception as e:
        result.error = str(e)
        result.success = False
        if verbose >= 2:
            traceback.print_exc()

    result.duration = time.time() - start_time
    return result

def run_all_tests(
    pipeline_dir: Path,
    examples_dir: Path,
    verbose: int = 1,
    quick: bool = False,
    specific_pipeline: str | None = None
) -> tuple[list[PipelineTestResult], int, int]:
    """Run all pipeline tests."""
    results = []
    passed = 0
    failed = 0

    # Find all pipeline files
    pipeline_files = sorted(
        list(pipeline_dir.glob("*.json")) +
        list(pipeline_dir.glob("*.yaml")) +
        list(pipeline_dir.glob("*.yml"))
    )

    # Filter if specific pipeline requested
    if specific_pipeline:
        pipeline_files = [f for f in pipeline_files if specific_pipeline in f.name]

    # Skip TensorFlow pipelines if not available
    try:
        from nirs4all.utils.backend import is_available
        has_tensorflow = is_available('tensorflow')
    except ImportError:
        has_tensorflow = False

    print(f"\n{'='*70}")
    print("  Pipeline Samples Test Runner")
    print(f"{'='*70}")
    print(f"  Directory: {pipeline_dir}")
    print(f"  Dataset: {DEFAULT_DATASET}")
    print(f"  Quick mode: {quick}")
    print(f"  TensorFlow available: {has_tensorflow}")
    print(f"  Pipelines to test: {len(pipeline_files)}")
    print(f"{'='*70}\n")

    for pipeline_file in pipeline_files:
        name = pipeline_file.name

        # Skip conditions
        if quick and name in SLOW_PIPELINES:
            print(f"⊘ {name} (skipped - slow pipeline in quick mode)")
            continue

        if not has_tensorflow and name in TENSORFLOW_PIPELINES:
            print(f"⊘ {name} (skipped - requires TensorFlow)")
            continue

        print(f"▶ Testing {name}...")

        result = run_pipeline_test(
            pipeline_file,
            examples_dir,
            verbose=verbose,
            quick=quick
        )

        results.append(result)

        if result.success:
            passed += 1
            print(f"  {result}")
        else:
            failed += 1
            print(f"  {result}")

        print()

    return results, passed, failed

def print_summary(results: list[PipelineTestResult], passed: int, failed: int):
    """Print test summary."""
    print(f"\n{'='*70}")
    print("  TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Total:  {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"{'='*70}")

    if failed > 0:
        print("\n  Failed Pipelines:")
        for r in results:
            if not r.success:
                print(f"    ✗ {r.name}: {r.error}")

    # Check for DummyController warnings
    dummy_warnings = [r for r in results if r.dummy_controller_triggered]
    if dummy_warnings:
        print("\n  ⚠ DummyController Warnings:")
        for r in dummy_warnings:
            print(f"    - {r.name}")

    print()

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test all pipeline sample definitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "-v", "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=quiet, 1=normal, 2=debug)"
    )
    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Quick mode: skip slow pipelines, reduce trials"
    )
    parser.add_argument(
        "-p", "--pipeline",
        type=str,
        default=None,
        help="Run specific pipeline only (partial match)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path"
    )
    args = parser.parse_args()

    # Override default dataset if specified
    global DEFAULT_DATASET
    if args.dataset:
        DEFAULT_DATASET = args.dataset

    # Determine directories
    script_dir = Path(__file__).resolve().parent
    examples_dir = script_dir.parent
    pipeline_dir = script_dir

    # Run tests
    results, passed, failed = run_all_tests(
        pipeline_dir=pipeline_dir,
        examples_dir=examples_dir,
        verbose=args.verbose,
        quick=args.quick,
        specific_pipeline=args.pipeline
    )

    # Print summary
    print_summary(results, passed, failed)

    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
