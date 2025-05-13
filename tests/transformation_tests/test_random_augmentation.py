
import numpy as np
import pytest
from nirs4all.transformations._random_augmentation import Rotate_Translate, Random_X_Operation


def test_Rotate_Translate(random_data):
    augmenter = Rotate_Translate(random_state=42)
    X_augmented = augmenter.transform(random_data)
    assert X_augmented.shape == random_data.shape


def test_Random_X_Operation(random_data):
    augmenter = Random_X_Operation(random_state=42)
    X_augmented = augmenter.transform(random_data)
    assert X_augmented.shape == random_data.shape


