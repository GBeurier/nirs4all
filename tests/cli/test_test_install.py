"""
Tests for CLI installation testing functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from nirs4all.cli.test_install import check_dependency, test_installation, full_test_installation, test_integration


class TestInstallationTesting:
    """Tests for installation testing functionality."""

    def test_check_dependency_available(self):
        """Test dependency checking for available package."""
        # Test with numpy which should be available
        available, version = check_dependency('numpy')
        assert available is True
        assert version != "Not installed"

    def test_check_dependency_not_available(self):
        """Test dependency checking for unavailable package."""
        available, version = check_dependency('nonexistent_package_xyz')
        assert available is False
        assert version == "Not installed"

    @patch('nirs4all.cli.test_install.check_dependency')
    def test_test_installation_success(self, mock_check_dependency):
        """Test successful installation test."""
        # Mock all dependencies as available
        mock_check_dependency.return_value = (True, "1.0.0")

        with patch('builtins.print'):  # Suppress output during test
            result = test_installation()

        assert result is True

    @patch('nirs4all.cli.test_install.check_dependency')
    def test_test_installation_failure(self, mock_check_dependency):
        """Test failed installation test."""
        # Mock some dependencies as unavailable
        def side_effect(name, min_version=None):
            if name == 'numpy':
                return (False, "Not installed")
            return (True, "1.0.0")

        mock_check_dependency.side_effect = side_effect

        with patch('builtins.print'):  # Suppress output during test
            result = test_installation()

        assert result is False

    @patch('nirs4all.cli.test_install.test_installation')
    def test_full_test_installation_basic_fail(self, mock_test_installation):
        """Test full installation test when basic test fails."""
        mock_test_installation.return_value = False

        with patch('builtins.print'):  # Suppress output during test
            result = full_test_installation()

        assert result is False

    @patch('nirs4all.cli.test_install.test_installation')
    @patch('nirs4all.core.runner.ExperimentRunner')
    def test_integration_test_basic_fail(self, mock_runner, mock_test_installation):
        """Test integration test when basic installation test fails."""
        mock_test_installation.return_value = False

        with patch('builtins.print'):  # Suppress output during test
            result = test_integration()

        assert result is False
        # ExperimentRunner should not be called if basic test fails
        mock_runner.assert_not_called()