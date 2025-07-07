"""DummyController.py - A dummy controller for testing purposes in the nirs4all pipeline."""

from typing import Any, Dict, TYPE_CHECKING
import matplotlib.pyplot as plt
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from sklearn.base import TransformerMixin
if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset

@register_controller
class SpectraChartController(OperatorController):

    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "spectra_charts"

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1
    ):
        print(f"Executing spectra charts for step: {step}, keyword: {context.get('keyword', '')}, source: {source}")

        local_context = context.copy()
        # local_context["partition"] = "train"
        spectra_data = dataset.x(local_context, "3d", source=source)
        y = dataset.y(local_context)
        print(">>>", spectra_data.shape, y.shape)
        for i in range(spectra_data.shape[1]):
            sub_data = spectra_data[:, i, :]
            print(sub_data.shape)
            plt.figure(figsize=(15, 4))
            processing_name = dataset.features.sources[0]._processing_ids[i]
            for j in range(sub_data.shape[0]):
                plt.plot(sub_data[j, :], alpha=0.7)
            plt.title(f'Spectra Chart for {processing_name}: {sub_data.shape[0]} samples')
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.legend()
            plt.grid()
            plt.show()

        return context