"""
Test load_XY and handle_data with header unit threading - Step 4
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from nirs4all.dataset.loader import load_XY, handle_data


class TestLoadXYHeaderUnit:
    """Test header unit threading through load_XY"""

    def test_load_xy_returns_header_unit(self):
        """Test that load_XY returns header_unit"""
        # Create temporary CSVs
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fx:
            fx.write("4000;5000;6000\n")
            fx.write("1.1;1.2;1.3\n")
            fx.write("2.1;2.2;2.3\n")
            x_path = fx.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fy:
            fy.write("target\n")
            fy.write("10.5\n")
            fy.write("20.3\n")
            y_path = fy.name

        try:
            x_params = {'delimiter': ';', 'has_header': True}
            y_params = {'delimiter': ';', 'has_header': True}

            x, y, m, x_headers, m_headers, x_unit = load_XY(
                x_path, None, x_params,
                y_path, None, y_params
            )

            assert x_unit == "cm-1"  # Default
            assert x_headers == ["4000", "5000", "6000"]
            assert x.shape == (2, 3)
            assert y.shape == (2, 1)
        finally:
            Path(x_path).unlink()
            Path(y_path).unlink()

    def test_load_xy_custom_header_unit(self):
        """Test loading with custom header_unit parameter"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fx:
            fx.write("780;800;850\n")
            fx.write("0.5;0.6;0.7\n")
            x_path = fx.name

        try:
            x_params = {
                'delimiter': ';',
                'has_header': True,
                'header_unit': 'nm'  # Specify nm unit
            }

            x, y, m, x_headers, m_headers, x_unit = load_XY(
                x_path, None, x_params,
                None, None, {}
            )

            assert x_unit == "nm"
            assert x_headers == ["780", "800", "850"]
        finally:
            Path(x_path).unlink()

    def test_load_xy_no_header_unit(self):
        """Test loading with 'none' header unit"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fx:
            fx.write("1.1;1.2;1.3\n")
            fx.write("2.1;2.2;2.3\n")
            x_path = fx.name

        try:
            x_params = {
                'delimiter': ';',
                'has_header': False,
                'header_unit': 'none'
            }

            x, y, m, x_headers, m_headers, x_unit = load_XY(
                x_path, None, x_params,
                None, None, {}
            )

            assert x_unit == "none"
            assert len(x_headers) == 3  # Auto-generated headers
        finally:
            Path(x_path).unlink()


class TestHandleDataHeaderUnit:
    """Test header unit threading through handle_data"""

    def test_handle_data_single_source_returns_unit(self):
        """Test that handle_data returns unit for single source"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fx:
            fx.write("4000;5000\n")
            fx.write("1.1;1.2\n")
            x_path = fx.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fy:
            fy.write("target\n")
            fy.write("10\n")
            y_path = fy.name

        try:
            config = {
                'train_x': x_path,
                'train_y': y_path,
                'global_params': {
                    'delimiter': ';',
                    'has_header': True
                }
            }

            x, y, m, x_headers, m_headers, x_unit = handle_data(config, 'train')

            assert x_unit == "cm-1"  # Default
            assert x_headers == ["4000", "5000"]
        finally:
            Path(x_path).unlink()
            Path(y_path).unlink()

    def test_handle_data_custom_unit_in_config(self):
        """Test that header_unit can be specified in config"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fx:
            fx.write("780;800\n")
            fx.write("0.5;0.6\n")
            x_path = fx.name

        try:
            config = {
                'train_x': x_path,
                'global_params': {
                    'delimiter': ';',
                    'has_header': True,
                    'header_unit': 'nm'  # Specify nm in config
                }
            }

            x, y, m, x_headers, m_headers, x_unit = handle_data(config, 'train')

            assert x_unit == "nm"
            assert x_headers == ["780", "800"]
        finally:
            Path(x_path).unlink()

    def test_handle_data_multi_source_returns_units_list(self):
        """Test that handle_data returns list of units for multi-source"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fx1:
            fx1.write("4000;5000\n")
            fx1.write("1.1;1.2\n")
            x1_path = fx1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fx2:
            fx2.write("780;800\n")
            fx2.write("0.5;0.6\n")
            x2_path = fx2.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as fy:
            fy.write("target\n")
            fy.write("10\n")
            y_path = fy.name

        try:
            config = {
                'train_x': [x1_path, x2_path],
                'train_y': y_path,
                'train_x_params': {
                    'delimiter': ';',
                    'has_header': True,
                    'header_unit': 'cm-1'  # Can specify for all sources
                }
            }

            x, y, m, x_headers, m_headers, x_units = handle_data(config, 'train')

            # Should return list of units
            assert isinstance(x_units, list)
            assert len(x_units) == 2
            assert x_units[0] == "cm-1"
            assert x_units[1] == "cm-1"

            # Headers should be list of lists
            assert isinstance(x_headers, list)
            assert len(x_headers) == 2
        finally:
            Path(x1_path).unlink()
            Path(x2_path).unlink()
            Path(y_path).unlink()

    def test_handle_data_preloaded_arrays_return_default_unit(self):
        """Test that pre-loaded numpy arrays return default unit"""
        x_array = np.random.rand(10, 5)
        y_array = np.random.rand(10, 1)

        config = {
            'train_x': x_array,
            'train_y': y_array
        }

        x, y, m, x_headers, m_headers, x_unit = handle_data(config, 'train')

        assert x_unit == "cm-1"  # Default for pre-loaded arrays
        assert x_headers == ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
