"""Tests for ExecutionPhase enum and its integration with RuntimeContext."""

import pytest

from nirs4all.pipeline.config.context import ExecutionPhase, RuntimeContext


class TestExecutionPhase:
    """Tests for the ExecutionPhase enum."""

    def test_cv_value(self):
        """CV member has expected string value."""
        assert ExecutionPhase.CV.value == "cv"

    def test_refit_value(self):
        """REFIT member has expected string value."""
        assert ExecutionPhase.REFIT.value == "refit"

    def test_members(self):
        """Enum has exactly two members."""
        assert set(ExecutionPhase) == {ExecutionPhase.CV, ExecutionPhase.REFIT}

    def test_identity(self):
        """Enum members compare by identity and equality."""
        assert ExecutionPhase.CV is ExecutionPhase.CV
        assert ExecutionPhase.CV == ExecutionPhase.CV
        assert ExecutionPhase.CV != ExecutionPhase.REFIT

class TestRuntimeContextPhase:
    """Tests for the phase field on RuntimeContext."""

    def test_default_phase_is_cv(self):
        """RuntimeContext defaults to ExecutionPhase.CV."""
        rc = RuntimeContext()
        assert rc.phase == ExecutionPhase.CV

    def test_set_phase_to_refit(self):
        """Phase can be set to REFIT without error."""
        rc = RuntimeContext()
        rc.phase = ExecutionPhase.REFIT
        assert rc.phase == ExecutionPhase.REFIT

    def test_construct_with_refit(self):
        """RuntimeContext can be constructed with phase=REFIT."""
        rc = RuntimeContext(phase=ExecutionPhase.REFIT)
        assert rc.phase == ExecutionPhase.REFIT

    def test_phase_accessible_as_attribute(self):
        """Phase is a regular dataclass field accessible via dot notation."""
        rc = RuntimeContext()
        # Simulate what a controller would do
        phase = rc.phase
        assert isinstance(phase, ExecutionPhase)

    def test_deepcopy_preserves_phase(self):
        """RuntimeContext.__deepcopy__ returns self, so phase is preserved."""
        from copy import deepcopy
        rc = RuntimeContext(phase=ExecutionPhase.REFIT)
        rc_copy = deepcopy(rc)
        # RuntimeContext.__deepcopy__ returns self (shared infrastructure)
        assert rc_copy is rc
        assert rc_copy.phase == ExecutionPhase.REFIT

    def test_existing_fields_unaffected(self):
        """Adding phase does not break existing RuntimeContext fields."""
        rc = RuntimeContext(
            pipeline_uid="test-uid",
            save_artifacts=False,
            step_number=3,
        )
        assert rc.pipeline_uid == "test-uid"
        assert rc.save_artifacts is False
        assert rc.step_number == 3
        assert rc.phase == ExecutionPhase.CV
