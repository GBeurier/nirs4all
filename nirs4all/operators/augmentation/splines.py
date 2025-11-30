import numpy as np
import scipy.interpolate as interpolate

from .abc_augmenter import Augmenter


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

        Optimized implementation with pre-allocated output array.

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
        n_samples, n_features = X.shape
        x_abs = np.arange(n_features)
        result = np.empty_like(X)
        s_param = 1 / n_features

        for i in range(n_samples):
            spl = interpolate.UnivariateSpline(x_abs, X[i], s=s_param)
            result[i] = spl(x_abs)

        return result


class Spline_X_Perturbations(Augmenter):
    """
    Class to apply a perturbation to a 1D signal using B-spline interpolation.

    Optimized implementation with pre-generated random parameters.

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

        Optimized with pre-allocated arrays and batch random generation.

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

        n_samples, n_features = X.shape
        x_range = np.arange(n_features)
        result = np.empty_like(X)

        # Get spline representation for first sample to determine perturbation size
        t, c, k = interpolate.splrep(x_range, X[0], s=0, k=self.spline_degree)
        delta_x_size = max(int(np.around(len(t) * self.perturbation_density)), 2)
        delta_x = np.linspace(np.min(x_range), np.max(x_range), delta_x_size)

        if apply_on == "global":
            # Single perturbation for all samples
            delta_y = self.random_gen.uniform(
                self.perturbation_range[0], self.perturbation_range[1], delta_x_size
            )
            delta = np.interp(t, delta_x, delta_y)
            t_perturbed = t + delta

            for i in range(n_samples):
                t_i, c_i, _ = interpolate.splrep(x_range, X[i], s=0, k=self.spline_degree)
                perturbed_spline = interpolate.BSpline(t_perturbed, c_i, k, extrapolate=True)
                result[i] = perturbed_spline(x_range)
        else:
            # Pre-generate all random perturbations at once
            all_delta_y = self.random_gen.uniform(
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


class Spline_Y_Perturbations(Augmenter):
    """
    Augment the data with a perturbation on the y-axis using B-spline interpolation.

    Optimized implementation with pre-generated random parameters.

    Parameters
    ----------
    X : ndarray
        Input data.
    apply_on : str, optional
        Apply augmentation on "samples" or "global" (default: "samples").
    spline_points : int, optional
        Number of spline points. Default is None (uses sample length / 2).
    perturbation_intensity : float, optional
        Intensity of perturbation relative to max value. Default is 0.005.
    """

    def __init__(self, apply_on="samples", random_state=None, *, copy=True, spline_points=None, perturbation_intensity=0.005):
        self.spline_points = spline_points
        self.perturbation_intensity = perturbation_intensity
        super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="samples"):
        """
        Augment the data with a perturbation on the y-axis using B-spline interpolation.

        Optimized with pre-allocated arrays and batch random generation.

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
        n_samples, n_features = X.shape
        x_range = np.arange(n_features)
        variation = np.max(X) * self.perturbation_intensity
        nb_spline_points = int(n_features / 2) if self.spline_points is None else self.spline_points
        x_points = np.linspace(0, n_features, nb_spline_points)

        # Pre-generate baseline for all samples (or single for global)
        baseline = self.random_gen.uniform(-variation, variation)
        interval_min = -variation + baseline
        interval_max = variation + baseline

        if apply_on == "global":
            # Single distortion for all samples
            y_points = self.random_gen.uniform(interval_min, interval_max, nb_spline_points)
            x_gen = np.sort(x_points)
            t, c, k = interpolate.splrep(x_gen, y_points, s=0, k=3)
            spline = interpolate.BSpline(t, c, k, extrapolate=False)
            distor = spline(x_range)
            return X + distor

        # Pre-generate all random y_points at once for all samples
        all_y_points = self.random_gen.uniform(
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


class Spline_X_Simplification(Augmenter):
    """
    Class to simplify a 1D signal using B-spline interpolation along the x-axis.

    Optimized implementation with pre-generated random parameters.

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

        Optimized with pre-allocated arrays and batch random generation.

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
        n_samples, n_features = X.shape
        x_range = np.arange(n_features)
        nb_points = self.spline_points if self.spline_points is not None else int(n_features / 4)

        result = np.empty_like(X)

        if self.uniform:
            # Uniform points are the same for all samples
            ctrl_points = np.linspace(0, n_features - 1, nb_points).astype(int)

            for i in range(n_samples):
                if apply_on == "samples":
                    # Still use same uniform points for each sample
                    pass
                x_subrange = x_range[ctrl_points]
                y = X[i, ctrl_points]
                t, c, k = interpolate.splrep(x_subrange, y, s=0, k=3)
                spline = interpolate.BSpline(t, c, k, extrapolate=False)
                result[i] = spline(x_range)
        else:
            if apply_on == "global":
                # Same random control points for all samples
                ctrl_points = np.unique(np.concatenate((
                    [0],
                    self.random_gen.choice(range(n_features), nb_points, replace=False),
                    [n_features - 1]
                )))

                for i in range(n_samples):
                    x_subrange = x_range[ctrl_points]
                    y = X[i, ctrl_points]
                    t, c, k = interpolate.splrep(x_subrange, y, s=0, k=3)
                    spline = interpolate.BSpline(t, c, k, extrapolate=False)
                    result[i] = spline(x_range)
            else:
                # Pre-generate random control points for all samples
                # Note: Each sample gets different random points
                for i in range(n_samples):
                    ctrl_points = np.unique(np.concatenate((
                        [0],
                        self.random_gen.choice(range(n_features), nb_points, replace=False),
                        [n_features - 1]
                    )))
                    x_subrange = x_range[ctrl_points]
                    y = X[i, ctrl_points]
                    t, c, k = interpolate.splrep(x_subrange, y, s=0, k=3)
                    spline = interpolate.BSpline(t, c, k, extrapolate=False)
                    result[i] = spline(x_range)

        return result


class Spline_Curve_Simplification(Augmenter):
    """
    Class to simplify a 1D signal using B-spline interpolation along the curve.

    Optimized implementation with pre-allocated output arrays.

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

        Optimized with pre-allocated output array.

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
        n_samples, n_features = X.shape
        nb_points = self.spline_points if self.spline_points is not None else int(n_features / 4)
        x = np.arange(n_features)

        simplified_X = np.empty_like(X)

        if self.uniform:
            control_point_indices = np.linspace(0, n_features - 1, nb_points).astype(int)
        else:
            control_point_indices = np.unique(np.concatenate((
                [0],
                self.random_gen.choice(range(n_features), nb_points, replace=False),
                [n_features - 1]
            )))

        for i in range(n_samples):
            if apply_on == "samples" and not self.uniform:
                control_point_indices = np.unique(np.concatenate((
                    [0],
                    self.random_gen.choice(range(n_features), nb_points, replace=False),
                    [n_features - 1]
                )))

            control_point_indices = np.unique(control_point_indices)
            y = X[i]

            # Fit a cubic B-spline to the control points
            t, c, k = interpolate.splrep(x[control_point_indices], y[control_point_indices], s=0, k=3)

            # Evaluate the B-spline at all wavelengths to get simplified signal
            simplified_X[i] = interpolate.BSpline(t, c, k, extrapolate=False)(x)

        return simplified_X
