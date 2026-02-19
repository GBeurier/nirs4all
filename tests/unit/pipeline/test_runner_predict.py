import os
import random
from unittest.mock import MagicMock

import numpy as np
import pytest

from nirs4all.pipeline.config.context import ExecutionContext
from nirs4all.pipeline.runner import PipelineRunner, init_global_random_state


def test_init_global_random_state_controls_entropy():
    """Test that init_global_random_state properly seeds random generators."""
    # Clear any prior hash seed to ensure the function sets it.
    os.environ.pop("PYTHONHASHSEED", None)

    init_global_random_state(123)
    np_val1 = np.random.rand()
    py_val1 = random.random()

    init_global_random_state(123)
    np_val2 = np.random.rand()
    py_val2 = random.random()

    assert np_val1 == np_val2
    assert py_val1 == py_val2
    assert os.environ["PYTHONHASHSEED"] == "123"

# Note: run_step and run_steps methods were removed as they were deprecated
# and not part of the public API. Pipeline execution is now handled through
# the orchestrator/executor architecture.
