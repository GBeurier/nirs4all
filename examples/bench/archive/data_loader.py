import numpy as np
from typing import Any, Sequence
from spectraset import SpectraDataset


def load_data(_: Any) -> SpectraDataset:
    """
    Load the dataset.
    """
    spectraset = SpectraDataset()
    spectra, targets = _generate_random_data(classification=False)
    spectraset.add_spectra(spectra, target=targets, name="random_data")
    return spectraset


def _generate_random_data(classification: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """ Generate random spectra data.   """

    # Generate random spectra data
    num_samples = 1000
    num_features = 1000
    spectra = np.random.rand(num_samples, num_features)
    targets = np.random.randint(0, 2, size=num_samples) if classification else np.random.rand(num_samples)

    return spectra, targets
