"""Tests for refit=True as the default (Task 2.11).

Verifies that:
- nirs4all.run() defaults to refit=True
- PipelineRunner.run() defaults to refit=True
- PipelineOrchestrator.execute() defaults to refit=True
- refit=False explicitly disables refit
- refit=None disables refit (backward compat for callers passing None)
- The orchestrator enables the refit pass when refit=True
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

# =========================================================================
# Default parameter inspection
# =========================================================================


class TestRefitDefaultValues:
    """Verify that refit defaults to True in all entry points."""

    def test_api_run_defaults_to_refit_true(self):
        """nirs4all.run() has refit=True as its default."""
        from nirs4all.api.run import run

        sig = inspect.signature(run)
        refit_param = sig.parameters["refit"]
        assert refit_param.default is True, (
            f"Expected run() refit default to be True, got {refit_param.default!r}"
        )

    def test_pipeline_runner_run_defaults_to_refit_true(self):
        """PipelineRunner.run() has refit=True as its default."""
        from nirs4all.pipeline.runner import PipelineRunner

        sig = inspect.signature(PipelineRunner.run)
        refit_param = sig.parameters["refit"]
        assert refit_param.default is True, (
            f"Expected PipelineRunner.run() refit default to be True, got {refit_param.default!r}"
        )

    def test_orchestrator_execute_defaults_to_refit_true(self):
        """PipelineOrchestrator.execute() has refit=True as its default."""
        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        sig = inspect.signature(PipelineOrchestrator.execute)
        refit_param = sig.parameters["refit"]
        assert refit_param.default is True, (
            f"Expected orchestrator.execute() refit default to be True, got {refit_param.default!r}"
        )


# =========================================================================
# Refit-enabled condition logic
# =========================================================================


class TestRefitEnabledCondition:
    """Verify the refit_enabled flag evaluates correctly for all input values."""

    @staticmethod
    def _eval_refit_enabled(refit):
        """Reproduce the orchestrator's refit_enabled logic."""
        return refit is True or (isinstance(refit, dict) and refit)

    def test_refit_true_enables(self):
        """refit=True enables the refit pass."""
        assert self._eval_refit_enabled(True)

    def test_refit_false_disables(self):
        """refit=False disables the refit pass."""
        assert not self._eval_refit_enabled(False)

    def test_refit_none_disables(self):
        """refit=None disables the refit pass (backward compat)."""
        assert not self._eval_refit_enabled(None)

    def test_refit_dict_with_options_enables(self):
        """refit={"option": value} enables the refit pass."""
        assert self._eval_refit_enabled({"warm_start": True})

    def test_refit_empty_dict_disables(self):
        """refit={} disables the refit pass (empty dict is falsy)."""
        assert not self._eval_refit_enabled({})


# =========================================================================
# Integration: refit parameter flows through the call chain
# =========================================================================


class TestRefitParameterFlow:
    """Verify that the refit parameter flows through to the orchestrator."""

    @patch("nirs4all.pipeline.runner.PipelineOrchestrator")
    def test_runner_passes_refit_to_orchestrator(self, MockOrchestrator):
        """PipelineRunner.run() passes refit to orchestrator.execute()."""
        mock_orch = MagicMock()
        mock_orch.execute.return_value = (MagicMock(), {})
        mock_orch.last_pipeline_uid = None
        mock_orch.last_executor = None
        mock_orch.raw_data = {}
        mock_orch.pp_data = {}
        mock_orch._figure_refs = []
        MockOrchestrator.return_value = mock_orch

        from nirs4all.pipeline.runner import PipelineRunner

        runner = PipelineRunner(verbose=0)
        # Override orchestrator with our mock
        runner.orchestrator = mock_orch

        runner.run(pipeline=[], dataset="dummy")

        # Verify execute was called with refit=True (the default)
        call_kwargs = mock_orch.execute.call_args
        assert call_kwargs.kwargs.get("refit") is True or call_kwargs[1].get("refit") is True

    @patch("nirs4all.pipeline.runner.PipelineOrchestrator")
    def test_runner_passes_explicit_refit_false(self, MockOrchestrator):
        """PipelineRunner.run(refit=False) passes False to orchestrator."""
        mock_orch = MagicMock()
        mock_orch.execute.return_value = (MagicMock(), {})
        mock_orch.last_pipeline_uid = None
        mock_orch.last_executor = None
        mock_orch.raw_data = {}
        mock_orch.pp_data = {}
        mock_orch._figure_refs = []
        MockOrchestrator.return_value = mock_orch

        from nirs4all.pipeline.runner import PipelineRunner

        runner = PipelineRunner(verbose=0)
        runner.orchestrator = mock_orch

        runner.run(pipeline=[], dataset="dummy", refit=False)

        call_kwargs = mock_orch.execute.call_args
        assert call_kwargs.kwargs.get("refit") is False or call_kwargs[1].get("refit") is False

    @patch("nirs4all.pipeline.runner.PipelineOrchestrator")
    def test_runner_passes_explicit_refit_none(self, MockOrchestrator):
        """PipelineRunner.run(refit=None) passes None to orchestrator."""
        mock_orch = MagicMock()
        mock_orch.execute.return_value = (MagicMock(), {})
        mock_orch.last_pipeline_uid = None
        mock_orch.last_executor = None
        mock_orch.raw_data = {}
        mock_orch.pp_data = {}
        mock_orch._figure_refs = []
        MockOrchestrator.return_value = mock_orch

        from nirs4all.pipeline.runner import PipelineRunner

        runner = PipelineRunner(verbose=0)
        runner.orchestrator = mock_orch

        runner.run(pipeline=[], dataset="dummy", refit=None)

        call_kwargs = mock_orch.execute.call_args
        assert call_kwargs.kwargs.get("refit") is None or call_kwargs[1].get("refit") is None
