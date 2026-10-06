"""VIZ-10: installation smoke success requires the mandatory sklearn lane."""

import builtins
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nirs4all.cli import installation_test
from nirs4all.pipeline import PipelineRunner


@pytest.mark.parametrize('required_passes', [False, True])
@pytest.mark.parametrize('optional_state', ['passed', 'failed', 'unavailable'])
def test_integration_never_counts_optional_skips_as_required_success(monkeypatch, required_passes, optional_state):
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == 'tensorflow' or name == 'optuna' and optional_state == 'unavailable':
            raise ImportError(f'{name} unavailable in smoke fixture')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_module)
    logger = MagicMock()
    monkeypatch.setattr(installation_test, 'logger', logger)
    runs = []

    def run(runner, *args, **kwargs):
        is_required = not runs
        runs.append(is_required)
        if is_required and not required_passes or not is_required and optional_state == 'failed':
            raise RuntimeError('pipeline execution is broken')
        return SimpleNamespace(num_predictions=100), {}

    monkeypatch.setattr(PipelineRunner, 'run', run)
    assert installation_test.test_integration() is required_passes
    warnings = '\n'.join(str(call.args[0]) for call in logger.warning.call_args_list)
    successes = '\n'.join(str(call.args[0]) for call in logger.success.call_args_list)
    assert 'SKIP TensorFlow' in warnings
    assert 'PASS TensorFlow' not in successes
    if optional_state == 'unavailable':
        assert 'SKIP Optuna' in warnings
        assert 'PASS Optuna' not in successes
    if not required_passes:
        assert 'Basic pipeline functionality is working' not in successes
