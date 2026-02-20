"""
Verification script for dataset configuration testing.

This script loads each YAML configuration and verifies that the corresponding
synthetic dataset can be loaded correctly. It reports successes, failures,
and any issues found.

Usage:
    python -m examples.scripts.verify_test_datasets [--configs-dir PATH] [--verbose]
"""

from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


@dataclass
class VerificationResult:
    """Result of verifying a single dataset."""

    name: str
    success: bool
    config_valid: bool = False
    loads_correctly: bool = False
    n_train_samples: int = 0
    n_test_samples: int = 0
    n_features: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

class DatasetVerifier:
    """Verifies all test datasets load correctly."""

    def __init__(
        self,
        configs_dir: Path,
        verbose: bool = False,
    ) -> None:
        self.configs_dir = Path(configs_dir)
        self.verbose = verbose
        self.results: list[VerificationResult] = []

    def verify_all(self) -> bool:
        """Verify all datasets, return True if all pass."""
        config_files = sorted(self.configs_dir.glob("*.yaml"))

        if not config_files:
            print(f"No YAML configs found in {self.configs_dir}")
            return False

        print(f"Found {len(config_files)} configurations to verify\n")

        for config_path in config_files:
            result = self.verify_single(config_path)
            self.results.append(result)

            status = "\033[92mPASS\033[0m" if result.success else "\033[91mFAIL\033[0m"
            print(f"[{status}] {result.name}")

            if result.success and self.verbose:
                print(f"       train={result.n_train_samples}, test={result.n_test_samples}, features={result.n_features}")
            for err in result.errors:
                print(f"       \033[91mERROR:\033[0m {err}")
            for warn in result.warnings:
                print(f"       \033[93mWARN:\033[0m {warn}")

        return all(r.success for r in self.results)

    def verify_single(self, config_path: Path) -> VerificationResult:
        """Verify a single dataset configuration."""
        name = config_path.stem
        result = VerificationResult(name=name, success=False)

        try:
            # Step 1: Load and parse YAML config
            with open(config_path) as f:
                raw_config = yaml.safe_load(f)

            result.config_valid = True

            # Step 2: Try to load the dataset
            from nirs4all.data import DatasetConfigs

            # Resolve relative paths by changing to configs directory
            configs = DatasetConfigs(str(config_path))

            # Get the dataset
            dataset = configs.get_dataset_at(0)
            result.loads_correctly = True

            # Step 3: Verify data shapes
            X_train = dataset.x({"partition": "train"}, layout="2d")
            assert isinstance(X_train, np.ndarray)
            y_train = dataset.y({"partition": "train"})

            result.n_train_samples = X_train.shape[0] if X_train is not None else 0
            result.n_features = X_train.shape[1] if X_train is not None and X_train.ndim == 2 else 0

            # Check for test data
            try:
                X_test = dataset.x({"partition": "test"}, layout="2d")
                assert isinstance(X_test, np.ndarray)
                y_test = dataset.y({"partition": "test"})
                result.n_test_samples = X_test.shape[0] if X_test is not None else 0
            except Exception:
                result.n_test_samples = 0
                result.warnings.append("No test partition found")

            # Validate we have data
            if result.n_train_samples == 0:
                result.errors.append("No training samples loaded")
                result.success = False
            elif result.n_features == 0:
                result.errors.append("No features loaded")
                result.success = False
            else:
                result.success = True

            # Check for expected sample counts based on generation
            expected_train = 48
            expected_test = 12
            if result.n_train_samples != expected_train:
                result.warnings.append(f"Expected {expected_train} train samples, got {result.n_train_samples}")
            if result.n_test_samples != expected_test and result.n_test_samples > 0:
                result.warnings.append(f"Expected {expected_test} test samples, got {result.n_test_samples}")

        except yaml.YAMLError as e:
            result.errors.append(f"YAML parsing error: {e}")
        except ImportError as e:
            result.errors.append(f"Import error: {e}")
        except FileNotFoundError as e:
            result.errors.append(f"File not found: {e}")
        except Exception as e:
            result.errors.append(f"Loading error: {type(e).__name__}: {e}")
            if self.verbose:
                traceback.print_exc()

        return result

    def print_summary(self) -> None:
        """Print verification summary."""
        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)

        print(f"\n{'=' * 60}")
        print(f"Verification Summary: {passed}/{total} passed")
        print(f"{'=' * 60}")

        if passed < total:
            print("\nFailed datasets:")
            for r in self.results:
                if not r.success:
                    print(f"  - {r.name}")
                    for err in r.errors:
                        print(f"      {err}")

        # Category breakdown
        categories = {}
        for r in self.results:
            cat = r.name.split("_")[0]
            if cat not in categories:
                categories[cat] = {"passed": 0, "total": 0}
            categories[cat]["total"] += 1
            if r.success:
                categories[cat]["passed"] += 1

        print("\nBy category:")
        for cat in sorted(categories.keys()):
            info = categories[cat]
            status = "\033[92mOK\033[0m" if info["passed"] == info["total"] else "\033[91mFAIL\033[0m"
            print(f"  {cat}: {info['passed']}/{info['total']} {status}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify test datasets load correctly")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path(__file__).parent.parent / "sample_configs" / "datasets",
        help="Directory containing YAML configs",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )

    args = parser.parse_args()

    verifier = DatasetVerifier(
        configs_dir=args.configs_dir,
        verbose=args.verbose,
    )

    success = verifier.verify_all()
    verifier.print_summary()

    exit(0 if success else 1)

if __name__ == "__main__":
    main()
