"""
Master test runner for nirs4all integration tests.

This script provides convenient ways to run the integration test suite
with different configurations.

Usage:
    python run_integration_tests.py              # Run all tests
    python run_integration_tests.py --fast       # Skip slow/optional tests
    python run_integration_tests.py --core       # Only core features (no TF/SHAP/Optuna)
    python run_integration_tests.py --coverage   # Run with coverage report
"""

import subprocess
import sys
from pathlib import Path


def run_tests(args):
    """Run pytest with the specified arguments."""
    cmd = ["pytest", "tests/integration_tests/", "-v"] + args

    print("=" * 80)
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)

    result = subprocess.run(cmd)
    return result.returncode

def main():
    """Main test runner."""
    # Check if we're in the right directory
    if not Path("tests/integration_tests").exists():
        print("Error: Must run from nirs4all root directory")
        return 1

    args = sys.argv[1:]
    pytest_args = []

    # Parse custom arguments
    if "--fast" in args:
        # Skip slow and heavy tests
        pytest_args.extend(["-m", "not slow and not tensorflow and not shap and not optuna"])
        args.remove("--fast")

    elif "--core" in args:
        # Only core features (no optional dependencies)
        pytest_args.extend(["-m", "not tensorflow and not shap and not optuna"])
        args.remove("--core")

    elif "--coverage" in args:
        # Run with coverage
        pytest_args.extend([
            "--cov=nirs4all",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
        args.remove("--coverage")

    # Add remaining args
    pytest_args.extend(args)

    # Run tests
    return run_tests(pytest_args)

if __name__ == "__main__":
    sys.exit(main())
