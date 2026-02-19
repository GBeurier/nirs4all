"""
Utility functions for nirs4all examples.

This module provides helper functions used across examples, including
result validation to ensure tests fail properly when issues occur,
and output management for saving generated files.
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

# =============================================================================
# Output Directory Management
# =============================================================================

# Default output directory for example outputs
_EXAMPLES_OUTPUT_DIR = None

def get_examples_output_dir() -> Path:
    """Get the output directory for example files.

    Returns a path to workspace/examples_output/ which is used for saving
    plots, generated data, and other artifacts from examples.

    Returns:
        Path to the examples output directory.
    """
    global _EXAMPLES_OUTPUT_DIR

    if _EXAMPLES_OUTPUT_DIR is None:
        # Find the workspace directory (relative to examples/)
        examples_dir = Path(__file__).parent
        workspace_dir = examples_dir.parent / "workspace" / "examples_output"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        _EXAMPLES_OUTPUT_DIR = workspace_dir

    return _EXAMPLES_OUTPUT_DIR

def get_example_output_path(
    example_name: str,
    filename: str,
    create_subdir: bool = True
) -> Path:
    """Get the output path for a specific example's file.

    Creates a subdirectory for the example if it doesn't exist.

    Args:
        example_name: Name of the example (e.g., "U01_hello_world").
        filename: Name of the file to save (e.g., "spectra.png").
        create_subdir: If True, create a subdirectory for the example.

    Returns:
        Full path where the file should be saved.

    Example:
        >>> path = get_example_output_path("D01_synthetic", "generated_spectra.png")
        >>> plt.savefig(path)
        >>> print(f"Saved: {path}")
    """
    output_dir = get_examples_output_dir()

    if create_subdir:
        example_dir = output_dir / example_name
        example_dir.mkdir(parents=True, exist_ok=True)
        return example_dir / filename
    else:
        return output_dir / filename

def save_array_summary(
    arrays: dict,
    example_name: str,
    filename: str = "data_summary.txt"
) -> Path:
    """Save a summary of generated arrays to a text file.

    Useful for documenting what data was generated in an example.

    Args:
        arrays: Dictionary of {name: np.ndarray} to summarize.
        example_name: Name of the example.
        filename: Output filename.

    Returns:
        Path to the saved summary file.

    Example:
        >>> X, y = generate_data()
        >>> save_array_summary({"X": X, "y": y}, "D01_synthetic")
    """
    path = get_example_output_path(example_name, filename)

    lines = [
        f"Data Summary - {example_name}",
        "=" * 50,
        ""
    ]

    for name, arr in arrays.items():
        if isinstance(arr, np.ndarray):
            lines.extend([
                f"{name}:",
                f"  Shape: {arr.shape}",
                f"  Dtype: {arr.dtype}",
                f"  Range: [{arr.min():.6g}, {arr.max():.6g}]",
                f"  Mean:  {arr.mean():.6g}",
                f"  Std:   {arr.std():.6g}",
                ""
            ])
        else:
            lines.append(f"{name}: {type(arr).__name__}")
            lines.append("")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    return path

def print_output_location(path: str | Path, description: str = "Output") -> None:
    """Print the location of a saved output file.

    Args:
        path: Path to the saved file.
        description: Description of what was saved.
    """
    path = Path(path)
    # Make path relative to workspace for cleaner output
    try:
        rel_path = path.relative_to(Path(__file__).parent.parent)
        print(f"   📁 {description}: {rel_path}")
    except ValueError:
        print(f"   📁 {description}: {path}")

def validate_result(
    result: Any,
    name: str = "",
    allow_nan_ratio: float = 0.0,
    min_predictions: int = 1,
    exit_on_failure: bool = True,
) -> bool:
    """Validate a nirs4all run result for common issues.

    This function should be called after nirs4all.run() in examples to ensure
    that the pipeline executed correctly and produced valid results.

    Args:
        result: RunResult from nirs4all.run().
        name: Optional name for the result (for error messages).
        allow_nan_ratio: Maximum allowed ratio of predictions with NaN metrics (0.0-1.0).
        min_predictions: Minimum number of predictions expected.
        exit_on_failure: If True, exit with code 1 on validation failure.

    Returns:
        True if validation passed, False otherwise.

    Example:
        >>> result = nirs4all.run(pipeline, dataset)
        >>> validate_result(result, "MyPipeline")  # Exits if issues found
    """
    prefix = f"[{name}] " if name else ""
    issues = []

    # Check for empty predictions
    num_predictions = getattr(result, 'num_predictions', 0)
    if num_predictions < min_predictions:
        issues.append(f"Expected at least {min_predictions} prediction(s), got {num_predictions}")

    # Check for NaN metrics using the built-in validate method if available
    if hasattr(result, 'validate'):
        try:
            report = result.validate(
                raise_on_failure=False,
                nan_threshold=allow_nan_ratio
            )
            if not report['valid']:
                issues.extend(report['issues'])
        except Exception as e:
            issues.append(f"Validation error: {e}")
    else:
        # Fallback for older API
        if hasattr(result, 'predictions') and hasattr(result.predictions, 'top'):
            all_preds = result.predictions.top(n=num_predictions) if num_predictions > 0 else []
            nan_count = 0
            for pred in all_preds:
                for metric in ['rmse', 'r2', 'accuracy', 'mse', 'mae', 'test_score']:
                    value = pred.get(metric)
                    if value is not None and isinstance(value, (int, float)) and np.isnan(value):
                        nan_count += 1
                        break

            if nan_count > 0:
                nan_ratio = nan_count / len(all_preds) if all_preds else 0
                if nan_ratio > allow_nan_ratio:
                    issues.append(
                        f"Found {nan_count}/{len(all_preds)} predictions with NaN metrics "
                        f"({nan_ratio:.1%} > {allow_nan_ratio:.1%} threshold)"
                    )

    if issues:
        print(f"\n{'=' * 60}")
        print(f"VALIDATION FAILED: {prefix.strip() if prefix else 'Result'}")
        print("=" * 60)
        for issue in issues:
            print(f"  ✗ {issue}")
        print("=" * 60 + "\n")

        if exit_on_failure:
            sys.exit(1)
        return False

    return True

def validate_results(
    results: list[Any],
    names: list[str] | None = None,
    allow_nan_ratio: float = 0.0,
    min_predictions: int = 1,
    exit_on_failure: bool = True,
) -> bool:
    """Validate multiple nirs4all run results.

    Args:
        results: List of RunResult objects from nirs4all.run().
        names: Optional list of names for each result.
        allow_nan_ratio: Maximum allowed ratio of predictions with NaN metrics.
        min_predictions: Minimum number of predictions expected per result.
        exit_on_failure: If True, exit with code 1 on any validation failure.

    Returns:
        True if all validations passed, False otherwise.
    """
    if names is None:
        names = [f"Result_{i+1}" for i in range(len(results))]

    all_valid = True
    for result, name in zip(results, names, strict=False):
        if not validate_result(
            result,
            name=name,
            allow_nan_ratio=allow_nan_ratio,
            min_predictions=min_predictions,
            exit_on_failure=False,  # Don't exit on individual failures
        ):
            all_valid = False

    if not all_valid and exit_on_failure:
        sys.exit(1)

    return all_valid
