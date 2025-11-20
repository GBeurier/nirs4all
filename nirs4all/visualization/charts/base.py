"""
BaseChart - Abstract base class for all prediction visualization charts.
"""
from abc import ABC, abstractmethod
from typing import Optional
import re
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from nirs4all.visualization.charts.config import ChartConfig


class BaseChart(ABC):
    """Abstract base class for all prediction visualization charts.

    Provides common interface and shared functionality for chart implementations.
    Each chart type should inherit from this class and implement required methods.

    Designed to be operator-ready for future integration with the controller/operator
    pattern (see SpectraChartController for reference pattern).

    Attributes:
        predictions: Predictions object containing prediction data.
        dataset_name_override: Optional dataset name override for display.
        config: ChartConfig instance for customization.
    """

    def __init__(self, predictions, dataset_name_override: Optional[str] = None,
                 config: Optional[ChartConfig] = None):
        """Initialize chart with predictions object.

        Args:
            predictions: Predictions object instance.
            dataset_name_override: Optional dataset name override.
            config: Optional ChartConfig for customization.
        """
        self.predictions = predictions
        self.dataset_name_override = dataset_name_override
        self.config = config or ChartConfig()

    @abstractmethod
    def render(self, **kwargs) -> Figure:
        """Render the chart and return matplotlib Figure.

        This method must be implemented by all chart subclasses.

        Args:
            **kwargs: Chart-specific rendering parameters.

        Returns:
            matplotlib Figure object.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> None:
        """Validate input parameters for the chart.

        This method should be called before rendering to ensure
        all required parameters are present and valid.

        Args:
            **kwargs: Chart-specific parameters to validate.

        Raises:
            ValueError: If validation fails.
        """
        pass

    def _create_empty_figure(self, figsize, message: str) -> Figure:
        """Create empty figure with message.

        Args:
            figsize: Figure size tuple.
            message: Message to display.

        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
        ax.set_title('No Data Available')
        ax.axis('off')
        return fig

    @staticmethod
    def _natural_sort_key(text: str):
        """Generate natural sorting key for strings with numbers.

        E.g., 'PLSRegression_10_cp' will sort after 'PLSRegression_2_cp'.

        Args:
            text: Input string to generate key for.

        Returns:
            List of alternating strings and integers for sorting.
        """
        def convert(part):
            return int(part) if part.isdigit() else part.lower()
        return [convert(c) for c in re.split(r'(\d+)', str(text))]
