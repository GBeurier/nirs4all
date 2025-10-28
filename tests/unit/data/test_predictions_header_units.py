"""
Step 8 Tests: Visualization with Header Units
Test that visualizations use appropriate axis labels based on header_unit.
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
from nirs4all.dataset.dataset import SpectroDataset
from nirs4all.controllers.chart.op_spectra_charts import SpectraChartController


class TestVisualizationHeaderUnit:
    """Test visualization axis labels with different header units"""

    def test_plot_2d_with_cm1_unit(self):
        """Test that 2D plot uses 'Wavenumber (cm⁻¹)' for cm-1 data"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(10)]

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="cm-1")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Wavenumber (cm⁻¹)'
        plt.close(fig)

    def test_plot_2d_with_nm_unit(self):
        """Test that 2D plot uses 'Wavelength (nm)' for nm data"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(700 + i*50) for i in range(10)]

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="nm")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Wavelength (nm)'
        plt.close(fig)

    def test_plot_2d_with_none_unit(self):
        """Test that 2D plot uses 'Features' for none unit"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [f'f{i}' for i in range(10)]

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="none")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Features'
        plt.close(fig)

    def test_plot_2d_with_text_headers(self):
        """Test that 2D plot uses 'Features' for text headers"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = ['feature_a', 'feature_b', 'feature_c', 'feature_d', 'feature_e',
                   'feature_f', 'feature_g', 'feature_h', 'feature_i', 'feature_j']

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="text")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Features'
        plt.close(fig)

    def test_plot_2d_without_headers(self):
        """Test that 2D plot uses 'Features' when no headers provided"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers=None, header_unit="cm-1")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Features'
        plt.close(fig)

    def test_plot_3d_with_cm1_unit(self):
        """Test that 3D plot uses 'Wavenumber (cm⁻¹)' for cm-1 data"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(10)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        controller._plot_3d_spectra(ax, x_data, y_data, "raw", headers, header_unit="cm-1")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Wavenumber (cm⁻¹)'
        plt.close(fig)

    def test_plot_3d_with_nm_unit(self):
        """Test that 3D plot uses 'Wavelength (nm)' for nm data"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(700 + i*50) for i in range(10)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        controller._plot_3d_spectra(ax, x_data, y_data, "raw", headers, header_unit="nm")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Wavelength (nm)'
        plt.close(fig)

    def test_plot_3d_with_none_unit(self):
        """Test that 3D plot uses 'Features' for none unit"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [f'f{i}' for i in range(10)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        controller._plot_3d_spectra(ax, x_data, y_data, "raw", headers, header_unit="none")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Features'
        plt.close(fig)

    def test_plot_2d_axis_labels_consistency(self):
        """Test that 2D plots have consistent y-axis labels"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(10)]

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="cm-1")

        # Check y-axis label
        assert ax.get_ylabel() == 'Intensity'
        plt.close(fig)

    def test_plot_3d_axis_labels_consistency(self):
        """Test that 3D plots have consistent y and z-axis labels"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(10)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        controller._plot_3d_spectra(ax, x_data, y_data, "raw", headers, header_unit="cm-1")

        # Check axis labels
        assert ax.get_ylabel() == 'y (sorted)'
        assert ax.get_zlabel() == 'Intensity'
        plt.close(fig)

    def test_plot_with_index_unit(self):
        """Test that plot uses 'Features' for index unit"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(i) for i in range(10)]

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="index")

        # Check that x-axis label is correct
        assert ax.get_xlabel() == 'Features'
        plt.close(fig)

    def test_plot_title_includes_processing_name(self):
        """Test that plot title includes processing name"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(10)]
        processing_name = "SavitzkyGolay"

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, processing_name, headers, header_unit="cm-1")

        # Check that title contains processing name and dimensions
        title = ax.get_title()
        assert processing_name in title
        assert "20 samples" in title
        assert "10 features" in title
        plt.close(fig)

    def test_plot_with_mismatched_headers(self):
        """Test that plot falls back to indices when headers don't match data"""
        controller = SpectraChartController()

        x_data = np.random.randn(20, 10)
        y_data = np.linspace(0, 1, 20)
        headers = [str(1000 + i*100) for i in range(5)]  # Only 5 headers for 10 features

        fig, ax = plt.subplots()
        controller._plot_2d_spectra(ax, x_data, y_data, "raw", headers, header_unit="cm-1")

        # When headers don't match, should use 'Features'
        assert ax.get_xlabel() == 'Features'
        plt.close(fig)
