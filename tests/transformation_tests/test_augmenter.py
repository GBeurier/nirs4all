import numpy as np
from nirs4all.transformations import Augmenter, IdentityAugmenter


def test_identity_augmenter(simple_data):
    augmenter = IdentityAugmenter()
    X_transformed = augmenter.transform(simple_data)
    np.testing.assert_array_equal(X_transformed, simple_data)


def test_augmenter_transform_method(simple_data):
    class DummyAugmenter(Augmenter):
        def augment(self, X, apply_on="samples"):
            return X + 1

    augmenter = DummyAugmenter()
    X_transformed = augmenter.transform(simple_data)
    np.testing.assert_array_equal(X_transformed, simple_data + 1)


def test_augmenter_fit_transform(simple_data):
    class DummyAugmenter(Augmenter):
        def augment(self, X, apply_on="samples"):
            return X * 2

    augmenter = DummyAugmenter()
    X_transformed = augmenter.fit_transform(simple_data)
    np.testing.assert_array_equal(X_transformed, simple_data * 2)


