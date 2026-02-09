"""Unit tests for stack_params helper function."""

import pytest
from nirs4all.optimization.optuna import stack_params


class TestStackParamsHelper:
    """Tests for stack_params helper function."""

    def test_basic_final_estimator_params(self):
        """Test basic parameter namespacing for final_estimator."""
        params = stack_params(
            final_estimator_params={
                "alpha": ("float", 1e-3, 1e0),
                "fit_intercept": [True, False],
            }
        )

        assert "final_estimator__alpha" in params
        assert "final_estimator__fit_intercept" in params
        assert params["final_estimator__alpha"] == ("float", 1e-3, 1e0)
        assert params["final_estimator__fit_intercept"] == [True, False]

    def test_other_stack_params(self):
        """Test that other Stack parameters are passed through."""
        params = stack_params(
            final_estimator_params={"alpha": (0.1, 1.0)},
            passthrough=True,
            cv=5,
        )

        assert params["passthrough"] is True
        assert params["cv"] == 5
        assert "final_estimator__alpha" in params

    def test_no_final_estimator_params(self):
        """Test with only Stack-level parameters."""
        params = stack_params(passthrough=True, cv=10)

        assert params == {"passthrough": True, "cv": 10}

    def test_empty_params(self):
        """Test with no parameters."""
        params = stack_params()
        assert params == {}

    def test_complex_parameter_specs(self):
        """Test with complex Optuna parameter specs."""
        params = stack_params(
            final_estimator_params={
                "alpha": {"type": "float", "min": 1e-4, "max": 1e0, "log": True},
                "solver": ["auto", "svd", "cholesky", "lsqr"],
                "max_iter": ("int", 100, 1000),
            }
        )

        assert params["final_estimator__alpha"]["type"] == "float"
        assert params["final_estimator__alpha"]["log"] is True
        assert params["final_estimator__solver"] == ["auto", "svd", "cholesky", "lsqr"]
        assert params["final_estimator__max_iter"] == ("int", 100, 1000)

    def test_nested_final_estimator_params(self):
        """Test that nested params in final_estimator are properly namespaced."""
        params = stack_params(
            final_estimator_params={
                "inference_config__PARAM_A": [True, False],
                "inference_config__PARAM_B": (1, 10),
            }
        )

        # Double-nested params should have final_estimator__ prefix
        assert "final_estimator__inference_config__PARAM_A" in params
        assert "final_estimator__inference_config__PARAM_B" in params

    def test_integration_with_finetune_params(self):
        """Test that stack_params output integrates with finetune_params structure."""
        finetune_params = {
            "n_trials": 20,
            "sampler": "tpe",
            "model_params": stack_params(
                final_estimator_params={
                    "alpha": ("float", 1e-3, 1e0),
                    "fit_intercept": [True, False],
                },
                passthrough=True,
            ),
        }

        assert finetune_params["n_trials"] == 20
        assert finetune_params["sampler"] == "tpe"
        assert "final_estimator__alpha" in finetune_params["model_params"]
        assert "passthrough" in finetune_params["model_params"]
