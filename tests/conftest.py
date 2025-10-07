"""
Pytest configuration for nirs4all tests.

This file configures the test environment, including matplotlib backend
for headless/GUI-less test execution.
"""

import matplotlib


def pytest_configure(config):
    """
    Configure pytest environment before tests run.

    Sets matplotlib to use non-interactive 'Agg' backend to avoid
    GUI-related errors in test environments (CI/CD, headless systems).

    Args:
        config: pytest config object (required by pytest hook)
    """
    # Use non-interactive backend for all tests
    matplotlib.use('Agg')
