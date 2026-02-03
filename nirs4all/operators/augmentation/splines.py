import numpy as np
import scipy.interpolate as interpolate
from sklearn.base import TransformerMixin, BaseEstimator


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


def X_length(x, y):
    """
    Compute the total length, segment lengths, and cumulative segment lengths of a curve.

    Vectorized implementation without np.vectorize.

    Parameters
    ----------
    x : ndarray
        Array of x-coordinates of the curve.
    y : ndarray
        Array of y-coordinates of the curve.

    Returns
    -------
    tuple
        A tuple containing the total length, segment lengths, and cumulative segment lengths.
    """
    x1 = x[:-1]
    y1 = y[:-1]
    x2 = x[1:]
    y2 = y[1:]

    # Vectorized segment length computation (no np.vectorize needed)
    SpecLen_seg = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
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


class Spline_Smoothing(TransformerMixin, BaseEstimator):
    """
    Class to apply a smoothing spline to a 1D signal.

    Parameters
    ----------
    random_state : int or None, optional
        Seed for the random number generator (unused, kept for API consistency).
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, **kwargs):
        """
        Apply a smoothing spline to the data.

        Optimized implementation with pre-allocated output array.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray
            Transformed data.
        """
        n_samples, n_features = X.shape
        x_abs = np.arange(n_features)
        result = np.empty_like(X)
        s_param = 1 / n_features

        for i in range(n_samples):
            spl = interpolate.UnivariateSpline(x_abs, X[i], s=s_param)
            result[i] = spl(x_abs)

        return result


class Spline_X_Perturbations(TransformerMixin, BaseEstimator):
    """
    Class to apply a perturbation to a 1D signal using B-spline interpolation.

    Optimized implementation with pre-generated random parameters.

    Parameters
    ----------
    spline_degree : int, optional
        Degree of the spline. Default is 3 (cubic).
    perturbation_density : float, optional
        Density of perturbation points relative to data size. Default is 0.05.
    perturbation_range : tuple, optional
        Range of perturbation values (min, max). Default is (-10, 10).
    random_state : int or None, optional
        Seed for the random number generator.
    """

    def __init__(self, spline_degree=3, perturbation_density=0.05, perturbation_range=(-10, 10), random_state=None):
        self.spline_degree = spline_degree
        self.perturbation_density = perturbation_density
        self.perturbation_range = perturbation_range
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        """
        Apply a perturbation using B-spline interpolation.

        Optimized with pre-allocated arrays and batch random generation.

        Parameters
        ----------
        X : ndarray
            Input data to be transformed.

        Returns
        -------
        ndarray
            Transformed data.
        """
        if not 0 <= self.perturbation_density <= 1:
            raise ValueError("Perturbation density must be between 0 and 1")

        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))

        n_samples, n_features = X.shape
        x_range = np.arange(n_features)
        result = np.empty_like(X)

        # Get spline representation for first sample to determine perturbation size
        t, c, k = interpolate.splrep(x_range, X[0], s=0, k=self.spline_degree)
        delta_x_size = max(int(np.around(len(t) * self.perturbation_density)), 2)
        delta_x = np.linspace(np.min(x_range), np.max(x_range), delta_x_size)

        # Pre-generate all random perturbations at once
        all_delta_y = rng.uniform(
            self.perturbation_range[0], self.perturbation_range[1],
            size=(n_samples, delta_x_size)
        )

        for i in range(n_samples):
            t_i, c_i, k_i = interpolate.splrep(x_range, X[i], s=0, k=self.spline_degree)
            delta = np.interp(t_i, delta_x, all_delta_y[i])
            t_perturbed = t_i + delta
            perturbed_spline = interpolate.BSpline(t_perturbed, c_i, k_i, extrapolate=True)
            result[i] = perturbed_spline(x_range)

        return result


class Spline_Y_Perturbations(TransformerMixin, BaseEstimator):
    """
    Augment the data with a perturbation on the y-axis using B-spline interpolation.

    Optimized implementation with pre-generated random parameters.

    Parameters
    ----------
    spline_points : int, optional
        Number of spline points. Default is None (uses sample length / 2).
    perturbation_intensity : float, optional
        Intensity of perturbation relative to max value. Default is 0.005.
    random_state : int or None, optional
        Seed for the random number generator.
    """

    def __init__(self, spline_points=None, perturbation_intensity=0.005, random_state=None):
        self.spline_points = spline_points
        self.perturbation_intensity = perturbation_intensity
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        """
        Apply a perturbation on the y-axis using B-spline interpolation.

        Optimized with pre-allocated arrays and batch random generation.

        Parameters
        ----------
        X : ndarray
            Input data to be transformed.

        Returns
        -------
        ndarray
            Transformed data.
        """
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))

        n_samples, n_features = X.shape
        x_range = np.arange(n_features)
        variation = np.max(X) * self.perturbation_intensity
        nb_spline_points = int(n_features / 2) if self.spline_points is None else self.spline_points
        x_points = np.linspace(0, n_features, nb_spline_points)

        # Pre-generate baseline for all samples
        baseline = rng.uniform(-variation, variation)
        interval_min = -variation + baseline
        interval_max = variation + baseline

        # Pre-generate all random y_points at once for all samples
        all_y_points = rng.uniform(
            interval_min, interval_max, size=(n_samples, nb_spline_points)
        )

        result = np.empty_like(X)
        x_gen = np.sort(x_points)

        for i in range(n_samples):
            y_points = all_y_points[i]
            t, c, k = interpolate.splrep(x_gen, y_points, s=0, k=3)
            spline = interpolate.BSpline(t, c, k, extrapolate=False)
            distor = spline(x_range)
            result[i] = X[i] + distor

        return result


class Spline_X_Simplification(TransformerMixin, BaseEstimator):
    """
    Class to simplify a 1D signal using B-spline interpolation along the x-axis.

    Optimized implementation with pre-generated random parameters.

    Parameters
    ----------
    spline_points : int, optional
        Number of spline points for simplification. Default is None: the length of the sample / 4.
    uniform : bool, optional
        If True, the spline points are uniformly spaced. Default is False.
    random_state : int or None, optional
        Seed for the random number generator.
    """

    def __init__(self, spline_points=None, uniform=False, random_state=None):
        self.spline_points = spline_points
        self.uniform = uniform
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        """
        Select randomly spaced points along the x-axis and adjust a spline.

        Optimized with pre-allocated arrays and batch random generation.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray
            Transformed data.
        """
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))

        n_samples, n_features = X.shape
        x_range = np.arange(n_features)
        nb_points = self.spline_points if self.spline_points is not None else int(n_features / 4)

        result = np.empty_like(X)

        if self.uniform:
            # Uniform points are the same for all samples
            ctrl_points = np.linspace(0, n_features - 1, nb_points).astype(int)

            for i in range(n_samples):
                x_subrange = x_range[ctrl_points]
                y = X[i, ctrl_points]
                t, c, k = interpolate.splrep(x_subrange, y, s=0, k=3)
                spline = interpolate.BSpline(t, c, k, extrapolate=False)
                result[i] = spline(x_range)
        else:
            # Each sample gets different random points
            for i in range(n_samples):
                ctrl_points = np.unique(np.concatenate((
                    [0],
                    rng.choice(range(n_features), nb_points, replace=False),
                    [n_features - 1]
                )))
                x_subrange = x_range[ctrl_points]
                y = X[i, ctrl_points]
                t, c, k = interpolate.splrep(x_subrange, y, s=0, k=3)
                spline = interpolate.BSpline(t, c, k, extrapolate=False)
                result[i] = spline(x_range)

        return result


class Spline_Curve_Simplification(TransformerMixin, BaseEstimator):
    """
    Class to simplify a 1D signal using B-spline interpolation along the curve.

    Optimized implementation with pre-allocated output arrays.

    Parameters
    ----------
    spline_points : int, optional
        Number of spline points for simplification. Default is None: the length of the sample / 4.
    uniform : bool, optional
        If True, the spline points are uniformly spaced. Default is False.
    random_state : int or None, optional
        Seed for the random number generator.
    """

    def __init__(self, spline_points=None, uniform=False, random_state=None):
        self.spline_points = spline_points
        self.uniform = uniform
        self.random_state = random_state

    def fit(self, X, y=None, **kwargs):
        self._rng = np.random.default_rng(self.random_state)
        return self

    def transform(self, X, **kwargs):
        """
        Select regularly spaced points on the x-axis and adjust a spline.

        Optimized with pre-allocated output array.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray
            Transformed data.
        """
        rng = getattr(self, '_rng', np.random.default_rng(self.random_state))

        n_samples, n_features = X.shape
        nb_points = self.spline_points if self.spline_points is not None else int(n_features / 4)
        x = np.arange(n_features)

        simplified_X = np.empty_like(X)

        for i in range(n_samples):
            if self.uniform:
                control_point_indices = np.linspace(0, n_features - 1, nb_points).astype(int)
            else:
                control_point_indices = np.unique(np.concatenate((
                    [0],
                    rng.choice(range(n_features), nb_points, replace=False),
                    [n_features - 1]
                )))

            control_point_indices = np.unique(control_point_indices)
            y = X[i]

            # Fit a cubic B-spline to the control points
            t, c, k = interpolate.splrep(x[control_point_indices], y[control_point_indices], s=0, k=3)

            # Evaluate the B-spline at all wavelengths to get simplified signal
            simplified_X[i] = interpolate.BSpline(t, c, k, extrapolate=False)(x)

        return simplified_X
