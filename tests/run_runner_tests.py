"""
Quick test runner script for PipelineRunner test suite.

Usage:
    python run_runner_tests.py [option]

Options:
    all         - Run all PipelineRunner tests
    quick       - Run only critical regression prevention tests
    coverage    - Run with coverage report
    baseline    - Create baseline results for comparison
    compare     - Compare current results with baseline
    verbose     - Run with verbose output
"""

import sys
import subprocess
from pathlib import Path
import json
from datetime import datetime


def run_command(cmd, capture=False):
    """Run shell command."""
    print(f"\n🚀 Running: {cmd}\n")
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    else:
        return subprocess.run(cmd, shell=True).returncode


def run_all_tests():
    """Run all PipelineRunner tests."""
    print("=" * 80)
    print("📦 Running ALL PipelineRunner Tests")
    print("=" * 80)

    cmd = "pytest tests/test_pipeline_runner*.py -v --tb=short"
    return run_command(cmd)


def run_quick_tests():
    """Run only critical regression prevention tests."""
    print("=" * 80)
    print("⚡ Running CRITICAL Regression Prevention Tests")
    print("=" * 80)

    cmd = "pytest tests/test_pipeline_runner_regression_prevention.py -v --tb=short"
    return run_command(cmd)


def run_coverage():
    """Run tests with coverage report."""
    print("=" * 80)
    print("📊 Running Tests with Coverage Analysis")
    print("=" * 80)

    cmd = "pytest tests/test_pipeline_runner*.py --cov=nirs4all.pipeline.runner --cov-report=html --cov-report=term"
    returncode = run_command(cmd)

    if returncode == 0:
        print("\n✅ Coverage report generated in htmlcov/index.html")

    return returncode


def create_baseline():
    """Create baseline results for comparison."""
    print("=" * 80)
    print("📸 Creating Baseline Results")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_file = f"baseline_runner_tests_{timestamp}.txt"

    cmd = f"pytest tests/test_pipeline_runner_regression_prevention.py -v --tb=short > {baseline_file}"
    returncode, stdout, stderr = run_command(cmd, capture=True)

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "test_file": "test_pipeline_runner_regression_prevention.py",
        "results_file": baseline_file,
        "return_code": returncode
    }

    metadata_file = f"baseline_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Baseline saved to: {baseline_file}")
    print(f"📋 Metadata saved to: {metadata_file}")
    print(f"\n💡 To compare later, run: python run_runner_tests.py compare {baseline_file}")

    return returncode


def compare_with_baseline(baseline_file=None):
    """Compare current results with baseline."""
    print("=" * 80)
    print("🔍 Comparing with Baseline")
    print("=" * 80)

    # Find most recent baseline if not specified
    if baseline_file is None:
        baselines = sorted(Path('.').glob('baseline_runner_tests_*.txt'))
        if not baselines:
            print("❌ No baseline files found. Create one first with: python run_runner_tests.py baseline")
            return 1
        baseline_file = str(baselines[-1])
        print(f"📂 Using most recent baseline: {baseline_file}")

    # Run current tests
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_file = f"current_runner_tests_{timestamp}.txt"

    cmd = f"pytest tests/test_pipeline_runner_regression_prevention.py -v --tb=short > {current_file}"
    run_command(cmd, capture=True)

    # Compare
    print(f"\n📊 Comparing {baseline_file} vs {current_file}\n")

    try:
        with open(baseline_file, 'r') as f:
            baseline_content = f.read()
        with open(current_file, 'r') as f:
            current_content = f.read()

        # Extract test results
        baseline_passed = "passed" in baseline_content
        current_passed = "passed" in current_content

        if baseline_passed and current_passed:
            # Count tests
            import re
            baseline_count = len(re.findall(r'PASSED', baseline_content))
            current_count = len(re.findall(r'PASSED', current_content))

            if baseline_count == current_count:
                print(f"✅ SUCCESS: All tests pass ({current_count} tests)")
                print("✅ Test count matches baseline")
                return 0
            else:
                print(f"⚠️  WARNING: Test count changed")
                print(f"   Baseline: {baseline_count} tests")
                print(f"   Current:  {current_count} tests")
                return 1
        else:
            print("❌ FAILURE: Tests failing")
            if not baseline_passed:
                print("   Baseline had failures")
            if not current_passed:
                print("   Current run has failures")
            return 1

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1


def run_verbose():
    """Run tests with maximum verbosity."""
    print("=" * 80)
    print("🔊 Running Tests with VERBOSE Output")
    print("=" * 80)

    cmd = "pytest tests/test_pipeline_runner*.py -vv --tb=long -s"
    return run_command(cmd)


def show_help():
    """Show help message."""
    print(__doc__)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("❌ No option specified\n")
        show_help()
        return 1

    option = sys.argv[1].lower()

    if option == "all":
        return run_all_tests()
    elif option == "quick":
        return run_quick_tests()
    elif option == "coverage":
        return run_coverage()
    elif option == "baseline":
        return create_baseline()
    elif option == "compare":
        baseline_file = sys.argv[2] if len(sys.argv) > 2 else None
        return compare_with_baseline(baseline_file)
    elif option == "verbose":
        return run_verbose()
    elif option in ["help", "-h", "--help"]:
        show_help()
        return 0
    else:
        print(f"❌ Unknown option: {option}\n")
        show_help()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
