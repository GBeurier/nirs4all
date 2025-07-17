"""
Test runner script for dataset and indexer tests.

This script provides convenient ways to run the test suite with different configurations.
"""

import pytest
import sys
from pathlib import Path


def run_all_tests():
    """Run all dataset and indexer tests."""
    test_args = [
        "tests/test_dataset.py",
        "tests/test_indexer.py",
        "-v",
        "--tb=short",
        "--strict-markers"
    ]
    return pytest.main(test_args)


def run_dataset_tests():
    """Run only dataset tests."""
    test_args = [
        "tests/test_dataset.py",
        "-v",
        "--tb=short"
    ]
    return pytest.main(test_args)


def run_indexer_tests():
    """Run only indexer tests."""
    test_args = [
        "tests/test_indexer.py",
        "-v",
        "--tb=short"
    ]
    return pytest.main(test_args)


def run_integration_tests():
    """Run only integration tests."""
    test_args = [
        "tests/",
        "-v",
        "-k", "integration",
        "--tb=short"
    ]
    return pytest.main(test_args)


def run_with_coverage():
    """Run tests with coverage reporting."""
    test_args = [
        "tests/",
        "--cov=nirs4all.dataset",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ]
    return pytest.main(test_args)


def run_quick_tests():
    """Run a subset of quick tests for development."""
    test_args = [
        "tests/",
        "-v",
        "-k", "not integration",
        "--tb=line",
        "-x"  # Stop on first failure
    ]
    return pytest.main(test_args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "all":
            exit_code = run_all_tests()
        elif command == "dataset":
            exit_code = run_dataset_tests()
        elif command == "indexer":
            exit_code = run_indexer_tests()
        elif command == "integration":
            exit_code = run_integration_tests()
        elif command == "coverage":
            exit_code = run_with_coverage()
        elif command == "quick":
            exit_code = run_quick_tests()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: all, dataset, indexer, integration, coverage, quick")
            exit_code = 1
    else:
        print("Running all tests...")
        exit_code = run_all_tests()

    sys.exit(exit_code)
