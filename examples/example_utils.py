"""
Utility functions for nirs4all examples.

This module provides helper functions used across examples, including
result validation to ensure tests fail properly when issues occur.
"""

import sys
import numpy as np
from typing import Any, List, Optional


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
    results: List[Any],
    names: Optional[List[str]] = None,
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
    for result, name in zip(results, names):
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
