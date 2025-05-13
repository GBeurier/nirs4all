import random

import numpy as np
import scipy.interpolate as interpolate

from .augmenter import Augmenter


def segment_length(x1, y1, x2, y2):
    """
    Compute the length of a line segment given its coordinates.

    Parameters
    ----------
    x1 : float
        x-coordinate of the first point.
    y1 : float
        y-coordinate of the first point.
    x2 : float
        x-coordinate of the second point.
    y2 : float
        y-coordinate of the second point.

    Returns
    -------
    float
        Length of the line segment.
    """

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


v_segment_length = np.vectorize(segment_length)


def X_length(x, y):
    """
    Compute the total length, segment lengths, and cumulative segment lengths of a curve given its coordinates.

    Parameters
    ----------
    x : ndarray
        Array of x-coordinates of the curve.
    y : ndarray
        Array of y-coordinates of the curve.

    Returns
    -------
    tuple
        A tuple containing the total length, segment lengths, and cumulative segment lengths of the curve.
    """

    x1 = x[:-1]
    y1 = y[:-1]
    x2 = x[1:]
    y2 = y[1:]

    SpecLen_seg = v_segment_length(x1, y1, x2, y2)
    SpecLen = np.sum(SpecLen_seg)
    SpecLen_seg_cumsum = np.cumsum(SpecLen_seg)
    return SpecLen, SpecLen_seg, SpecLen_seg_cumsum


def segment_pt_coord(x1, y1, x2, y2, fracL, L):
    """
    Compute the coordinates of a point on a line segment given the fraction of its length.

    Parameters
    ----------
    x1 : float
        x-coordinate of the first point of the line segment.
    y1 : float
        y-coordinate of the first point of the line segment.
    x2 : float
        x-coordinate of the second point of the line segment.
    y2 : float
        y-coordinate of the second point of the line segment.
    fracL : float
        Fraction of the length of the line segment.
    L : float
        Length of the line segment.

    Returns
    -------
    tuple
        A tuple containing the x and y coordinates of the point on the line segment.
    """

    propL = fracL / L
    xp = x1 + propL * (x2 - x1)
    yp = y1 + propL * (y2 - y1)
    return xp, yp


def interval_selection(n_l, CumVect):
    """
    Select the interval indices that bound a given value in an array.

    Parameters
    ----------
    n_l : float
        Value to be bounded.
    CumVect : ndarray
        Cumulative array of values.

    Returns
    -------
    tuple
        A tuple containing the minimum and maximum indices of the bounding interval.
    """

    i1 = np.where(n_l <= CumVect)
    i2 = np.where(n_l >= CumVect)
    return np.min(i1), np.max(i2)


class Spline_Smoothing(Augmenter):
    """
    Class to apply a smoothing spline to a 1D signal.

    Parameters
    ----------
    X : ndarray
        Input data.
    apply_on : str, optional
        Apply augmentation on "samples" or "global" (default: "samples").
    """

    def augment(self, X, apply_on="samples"):
        """
        Apply a smoothing spline to the data.

        Parameters
        ----------
        X : ndarray
            Input data.
        apply_on : str, optional
            Apply augmentation on "samples" or "global" (default: "samples").

        Returns
        -------
        ndarray
            Augmented data.
        """
        length = X.shape[-1]
        x_abs = np.arange(length)
        res = []
        for x in X:
            spl = interpolate.UnivariateSpline(x_abs, x, s=1 / length)
            y_smooth = spl(x_abs)
            res.append(y_smooth)

        print(np.array(res).shape, X.shape)
        return np.array(res)


class Spline_X_Perturbations(Augmenter):
    """
    Class to apply a perturbation to a 1D signal using B-spline interpolation.

    Parameters
    ----------
    X : ndarray
        Input data.
    apply_on : str, optional
        Apply augmentation on "samples" or "global" (default: "samples").
    spline_degree : int, optional
        Degree of the spline. Default is 3 (cubic).
    perturbation_density : float, optional
        Density of perturbation points relative to data size. Default is 0.05.
    perturbation_range : tuple, optional
        Range of perturbation values (min, max). Default is (-10, 10).
    """

    def __init__(self, apply_on="samples", random_state=None, *, copy=True, spline_degree=3, perturbation_density=0.05, perturbation_range=(-10, 10)):
        self.spline_degree = spline_degree
        self.perturbation_density = perturbation_density
        self.perturbation_range = perturbation_range
        super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="samples"):
        """
        Augment the data with a perturbation using B-spline interpolation.

        Parameters
        ----------
        X : ndarray
            Input data to be augmented.
        apply_on : str, optional
            Apply augmentation on "samples" or "global" data. Default is "samples".

        Returns
        -------
        ndarray
            Augmented data.
        """
        if not 0 <= self.perturbation_density <= 1:
            raise ValueError("Perturbation density must be between 0 and 1")

        x_range = np.arange(X.shape[-1])
        res = []

        t, c, k = interpolate.splrep(x_range, X[0], s=0, k=self.spline_degree)
        # Determine the number of perturbation points
        delta_x_size = max(int(np.around(len(t) * self.perturbation_density)), 2)
        delta_x = np.linspace(np.min(x_range), np.max(x_range), delta_x_size)
        delta_y = self.random_gen.uniform(self.perturbation_range[0], self.perturbation_range[1], delta_x_size)
        # Apply perturbation
        delta = np.interp(t, delta_x, delta_y)
        t_perturbed = t + delta

        for x in X:
            if apply_on == "global":
                t, c, _ = interpolate.splrep(x_range, x, s=0, k=self.spline_degree)
                perturbed_spline = interpolate.BSpline(t_perturbed, c, k, extrapolate=True)
                px = perturbed_spline(x_range)
                res.append(px)
            else:
                t, c, k = interpolate.splrep(x_range, X[0], s=0, k=self.spline_degree)
                # Determine the number of perturbation points
                delta_x_size = max(int(np.around(len(t) * self.perturbation_density)), 2)
                delta_x = np.linspace(np.min(x_range), np.max(x_range), delta_x_size)
                delta_y = self.random_gen.uniform(self.perturbation_range[0], self.perturbation_range[1], delta_x_size)
                # Apply perturbation
                delta = np.interp(t, delta_x, delta_y)
                t_perturbed = t + delta
                perturbed_spline = interpolate.BSpline(t_perturbed, c, k, extrapolate=True)
                px = perturbed_spline(x_range)
                res.append(px)

        return np.array(res)


class Spline_Y_Perturbations(Augmenter):
    """
    Augment the data with a perturbation on the y-axis using B-spline interpolation.

    Parameters
    ----------
    X : ndarray
        Input data.
    apply_on : str, optional
        Apply augmentation on "samples" or "global" (default: "samples").
    spline_degree : int, optional
        Degree of the spline. Default is 3 (cubic).
    perturbation_density : float, optional
        Density of perturbation points relative to data size. Default is 0.05.
    perturbation_range : tuple, optional
        Range of perturbation values (min, max). Default is (-10, 10).
    """

    def __init__(self, apply_on="samples", random_state=None, *, copy=True, spline_points=None, perturbation_intensity=0.005):
        self.spline_points = spline_points
        self.perturbation_intensity = perturbation_intensity
        super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="samples"):
        """
        Augment the data with a perturbation on the y-axis using B-spline interpolation.

        Parameters
        ----------
        X : ndarray
            Input data to be augmented.
        apply_on : str, optional
            Apply augmentation on "samples" or "global" data. Default is "samples".

        Returns
        -------
        ndarray
            Augmented data.
        """
        x_range = np.arange(X.shape[-1])
        variation = np.max(X) * self.perturbation_intensity
        res = []
        baseline = self.random_gen.uniform(-variation, variation)
        interval_min = -variation + baseline
        interval_max = variation + baseline
        nb_spline_points = int(X.shape[-1]/2) if self.spline_points is None else self.spline_points
        x_points = np.linspace(0, X.shape[-1], nb_spline_points)

        if apply_on == "global":
            y_points = [self.random_gen.uniform(interval_min, interval_max) for _ in range(nb_spline_points)]
            x = np.asarray(x_points)
            x.sort()
            y = np.asarray(y_points)
            t, c, k = interpolate.splrep(x, y, s=0, k=3)
            spline = interpolate.BSpline(t, c, k, extrapolate=False)
            distor = spline(x_range)
            y_distor = X + distor
            return np.array(y_distor)

        res = []
        for x in X:
            y_points = [self.random_gen.uniform(interval_min, interval_max) for _ in range(nb_spline_points)]
            x_gen = np.asarray(x_points)
            x_gen.sort()
            y = np.asarray(y_points)
            t, c, k = interpolate.splrep(x_gen, y, s=0, k=3)
            spline = interpolate.BSpline(t, c, k, extrapolate=False)
            distor = spline(x_range)
            y_distor = x + distor
            res.append(y_distor)

        return np.array(res)


class Spline_X_Simplification(Augmenter):
    """
    Class to simplify a 1D signal using B-spline interpolation along the x-axis.

    Parameters
    ----------
    X : ndarray
        Input data.
    apply_on : str, optional
        Apply augmentation on "samples" or "global" (default: "samples").
    spline_points : int, optional
        Number of spline points for simplification. Default is None: the length of the sample / 4.
    uniform : bool, optional
        If True, the spline points are uniformly spaced. Default is False.
    """

    def __init__(self, apply_on="samples", random_state=None, *, copy=True, spline_points=None, uniform=False):
        self.spline_points = spline_points
        self.uniform = uniform
        super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="samples"):
        """
        Select randomly spaced points along the x-axis and adjust a spline.

        Parameters
        ----------
        X : ndarray
            Input data.
        apply_on : str, optional
            Apply augmentation on "samples" or "global" (default: "samples").

        Returns
        -------
        ndarray
            Augmented data.
        """

        x_range = np.arange(0, X.shape[-1], 1)
        nb_points = self.spline_points if self.spline_points is not None else int(X.shape[-1] / 4)

        res = []
        if self.uniform:
            ctrl_points = np.linspace(0, X.shape[-1] - 1, nb_points).astype(int)
        else:
            ctrl_points = np.unique(np.concatenate(([0], self.random_gen.choice(range(X.shape[-1]), nb_points, replace=False), [X.shape[-1] - 1])))

        for x in X:
            if apply_on == "samples":
                if self.uniform:
                    ctrl_points = np.linspace(0, X.shape[-1] - 1, nb_points).astype(int)
                else:
                    ctrl_points = np.unique(np.concatenate(([0], self.random_gen.choice(range(X.shape[-1]), nb_points, replace=False), [X.shape[-1] - 1])))

            x_subrange = x_range[ctrl_points]
            y = x[ctrl_points]
            t, c, k = interpolate.splrep(x_subrange, y, s=0, k=3)
            spline = interpolate.BSpline(t, c, k, extrapolate=False)
            res.append(spline(x_range))

        return np.array(res)


class Spline_Curve_Simplification(Augmenter):
    """
    Class to simplify a 1D signal using B-spline interpolation along the curve.

    Parameters
    ----------
    X : ndarray
        Input data.
    apply_on : str, optional
        Apply augmentation on "samples" or "global" (default: "samples").
    spline_points : int, optional
        Number of spline points for simplification. Default is None: the length of the sample / 4.
    uniform : bool, optional
        If True, the spline points are uniformly spaced. Default is False.
    """

    def __init__(self, apply_on="samples", random_state=None, *, copy=True, spline_points=None, uniform=False):
        self.spline_points = spline_points
        self.uniform = uniform
        super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="samples"):
        """
        Select regularly spaced points on the x-axis and adjust a spline.

        Parameters
        ----------
        X : ndarray
            Input data.
        apply_on : str, optional
            Apply augmentation on "samples" or "features" (default: "samples").

        Returns
        -------
        ndarray
            Augmented data.
        """
        nb_points = self.spline_points if self.spline_points is not None else int(X.shape[-1] / 4)

        samples, wavelengths = X.shape
        simplified_X = np.zeros_like(X)

        if self.uniform:
            control_point_indices = np.linspace(0, X.shape[-1] - 1, nb_points).astype(int)
        else:
            control_point_indices = np.unique(np.concatenate(([0], self.random_gen.choice(range(X.shape[-1]), nb_points, replace=False), [X.shape[-1] - 1])))

        for i in range(samples):
            # Choose num_cp control points along the Y-values of the curve
            if apply_on == "samples":
                if self.uniform:
                    control_point_indices = np.linspace(0, X.shape[-1] - 1, nb_points).astype(int)
                else:
                    control_point_indices = np.unique(np.concatenate(([0], self.random_gen.choice(range(X.shape[-1]), nb_points, replace=False), [X.shape[-1] - 1])))

            control_point_indices = np.unique(control_point_indices)

            x = np.arange(wavelengths)
            y = X[i]

            # Fit a cubic B-spline to the control points
            t, c, k = interpolate.splrep(x[control_point_indices], y[control_point_indices], s=0, k=3)

            # Evaluate the B-spline at all wavelengths to get simplified signal
            simplified_X[i] = interpolate.BSpline(t, c, k, extrapolate=False)(x)

        return simplified_X
