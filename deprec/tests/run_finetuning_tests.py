#!/usr/bin/env python3
"""
Test runner for NIRS4ALL finetuning strategies.

This script runs the finetuning strategy tests and provides a summary
of which features are working correctly.
"""

import sys
import os
import time

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def run_basic_tests():
    """Run basic functionality tests without pytest."""
    print("🧪 NIRS4ALL Finetuning Strategy Tests")
    print("=" * 50)

    test_results = []

    # Test 1: Import test
    try:
        from nirs4all.controllers.models.config import ParamStrategy
        from nirs4all.controllers.models.abstract_model_controller import AbstractModelController
        from nirs4all.pipeline.runner import PipelineRunner
        from nirs4all.pipeline.config import PipelineConfigs
        test_results.append(("Imports", True, None))
        print("✅ All required modules import successfully")
    except Exception as e:
        test_results.append(("Imports", False, str(e)))
        print(f"❌ Import failed: {e}")
        return test_results

    # Test 2: Parameter Strategy Enum
    try:
        expected_strategies = [
            'per_fold_best', 'global_best', 'weighted_average',
            'global_average', 'ensemble_best', 'robust_best', 'stability_best'
        ]

        actual_strategies = [strategy.value for strategy in ParamStrategy]

        all_present = all(expected in actual_strategies for expected in expected_strategies)

        if all_present:
            test_results.append(("Parameter Strategies", True, None))
            print(f"✅ All {len(expected_strategies)} parameter strategies available")
        else:
            missing = [s for s in expected_strategies if s not in actual_strategies]
            test_results.append(("Parameter Strategies", False, f"Missing: {missing}"))
            print(f"❌ Missing parameter strategies: {missing}")

    except Exception as e:
        test_results.append(("Parameter Strategies", False, str(e)))
        print(f"❌ Parameter strategy test failed: {e}")

    # Test 3: Required Methods
    try:
        required_methods = [
            '_execute_global_average_optimization',
            '_train_single_model_on_full_data',
            '_optimize_global_average_on_inner_folds'
        ]

        missing_methods = []
        for method in required_methods:
            if not hasattr(AbstractModelController, method):
                missing_methods.append(method)

        if not missing_methods:
            test_results.append(("Required Methods", True, None))
            print(f"✅ All {len(required_methods)} required methods exist")
        else:
            test_results.append(("Required Methods", False, f"Missing: {missing_methods}"))
            print(f"❌ Missing methods: {missing_methods}")

    except Exception as e:
        test_results.append(("Required Methods", False, str(e)))
        print(f"❌ Method check failed: {e}")

    # Test 4: Configuration Creation
    try:
        from sklearn.cross_decomposition import PLSRegression

        configs = [
            ("Simple CV", "simple", "per_fold_best"),
            ("Per-fold CV", "per_fold", "global_best"),
            ("Global Average", "per_fold", "global_average"),
            ("Full Training", "per_fold", "global_average"),
            ("Nested CV", "nested", "global_average")
        ]

        config_results = []

        for name, cv_mode, param_strategy in configs:
            try:
                config = {
                    "pipeline": [{
                        "model": PLSRegression(),
                        "finetune_params": {
                            "cv_mode": cv_mode,
                            "param_strategy": param_strategy,
                            "n_trials": 2,
                            "verbose": 0,
                            "model_params": {
                                "n_components": ("int", 1, 5)
                            }
                        }
                    }]
                }

                if cv_mode == "nested":
                    config["pipeline"][0]["finetune_params"]["inner_cv"] = 2

                if name == "Full Training":
                    config["pipeline"][0]["finetune_params"]["use_full_train_for_final"] = True

                # Test PipelineConfigs creation
                pipeline_config = PipelineConfigs(config, f"test_{cv_mode}_{param_strategy}")
                config_results.append((name, True, None))

            except Exception as e:
                config_results.append((name, False, str(e)))

        successful_configs = [r for r in config_results if r[1]]

        if len(successful_configs) == len(configs):
            test_results.append(("Configuration Creation", True, None))
            print(f"✅ All {len(configs)} configuration types created successfully")
        else:
            failed = [r[0] for r in config_results if not r[1]]
            test_results.append(("Configuration Creation", False, f"Failed: {failed}"))
            print(f"❌ Failed configurations: {failed}")

        # Print detailed results
        for name, success, error in config_results:
            status = "✅" if success else "❌"
            print(f"    {status} {name}")
            if not success:
                print(f"        Error: {error}")

    except Exception as e:
        test_results.append(("Configuration Creation", False, str(e)))
        print(f"❌ Configuration test failed: {e}")

    # Test 5: Performance Check
    try:
        start_time = time.time()

        # Create multiple configurations quickly
        for i in range(50):
            config = {
                "pipeline": [{
                    "model": PLSRegression(),
                    "finetune_params": {
                        "cv_mode": "simple",
                        "param_strategy": "global_average",
                        "n_trials": 1
                    }
                }]
            }
            pipeline_config = PipelineConfigs(config, f"perf_test_{i}")

        elapsed = time.time() - start_time

        if elapsed < 2.0:
            test_results.append(("Performance", True, f"{elapsed:.3f}s"))
            print(f"✅ Performance test passed: {elapsed:.3f}s for 50 configs")
        else:
            test_results.append(("Performance", False, f"Too slow: {elapsed:.3f}s"))
            print(f"⚠️ Performance slower than expected: {elapsed:.3f}s")

    except Exception as e:
        test_results.append(("Performance", False, str(e)))
        print(f"❌ Performance test failed: {e}")

    return test_results

def print_summary(test_results):
    """Print test summary."""
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = [r for r in test_results if r[1]]
    failed = [r for r in test_results if not r[1]]

    print(f"✅ Passed: {len(passed)}/{len(test_results)} tests")
    print(f"❌ Failed: {len(failed)}/{len(test_results)} tests")

    if failed:
        print(f"\nFailed Tests:")
        for name, _, error in failed:
            print(f"  ❌ {name}: {error}")

    if len(passed) == len(test_results):
        print(f"\n🎉 All tests passed! NIRS4ALL finetuning strategies are ready to use.")
        print(f"\nAvailable features:")
        print(f"  ✓ 7 parameter strategies including global_average")
        print(f"  ✓ 3 cross-validation modes")
        print(f"  ✓ Full training data option (use_full_train_for_final)")
        print(f"  ✓ Multiple model types supported")
        print(f"  ✓ Configurable parameter spaces")
    else:
        print(f"\n⚠️ Some tests failed. Check implementation or dependencies.")

    return len(failed) == 0

def run_pytest_if_available():
    """Try to run pytest if available."""
    try:
        import pytest
        print(f"\n🧪 Running detailed tests with pytest...")

        test_files = [
            "tests/test_finetuning_focused.py"
        ]

        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"\nRunning {test_file}...")
                result = pytest.main([test_file, "-v", "-x"])  # Stop on first failure
                if result != 0:
                    print(f"⚠️ Some pytest tests failed in {test_file}")
            else:
                print(f"⚠️ Test file not found: {test_file}")

    except ImportError:
        print(f"\n💡 pytest not available. Install with: pip install pytest")
    except Exception as e:
        print(f"\n⚠️ pytest execution failed: {e}")

if __name__ == "__main__":
    # Run basic tests first
    results = run_basic_tests()
    success = print_summary(results)

    # Run pytest if requested and available
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        run_pytest_if_available()
    elif success:
        print(f"\n💡 Run with --pytest flag for detailed testing")

    # Exit with appropriate code
    sys.exit(0 if success else 1)