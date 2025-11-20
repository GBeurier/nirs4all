.. _augmentation:

Augmentation Module
===================


The :mod:`nirs4all.augmentation` module in Nirs4all provides functionalities for data augmentation techniques. Data augmentation is a crucial step in improving the generalization and robustness of machine learning models by increasing the diversity and size of the training data.

This module includes several augmentation functions that can be used to apply specific techniques to NIRS data:

- `Random_X_Spline_Deformation`: Performs random X-spline deformation on the spectra data.
- `Monotonous_Spline_Simplification`: Applies monotonous spline simplification to the spectra.
- `Dependent_Spline_Simplification`: Performs dependent spline simplification on the spectra.
- `Random_Spline_Addition`: Adds random splines to the spectra data.
- `Rotate_Translate`: Applies rotation and translation transformations to the spectra.
- `Random_X_Operation`: Performs random X-operation on the spectra.

In addition, the `nirs4all.augmentation` module provides an abstract Python class called `Augmenter`. This class serves as a base class for implementing custom data augmentation strategies. By subclassing `Augmenter` and overriding its methods, users can define their own augmentation techniques tailored to their specific requirements.

The `Augmenter` class provides a consistent interface for data augmentation, allowing users to apply their custom augmentation methods to NIRS data. By inheriting from `Augmenter`, users can leverage the underlying functionality of the Nirs4all package while extending it with their own augmentation logic.

For more information on the available augmentation functions and the usage of the `Augmenter` class, please refer to the API reference documentation for `nirs4all.augmentation`.