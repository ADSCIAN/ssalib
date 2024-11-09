"""Math utility functions for Singular Spectrum Analysis."""

# Author: Damien Delforge <damien.delforge@adscian.be>
#         Alice Alonso <alice.alonso@adscian.be>
#
# License: BSD 3 clause

from typing import Literal, Sequence

import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.linalg import toeplitz
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def correlation_weights(
        timeseries_length: int,
        window: int
) -> np.ndarray:
    """Calculate the default weights for the weighted correlation matrix.

    Parameters
    ----------
    timeseries_length : int
        Length of the time series.
    window : int
        Singular Spectrum Analysis window size.

    Returns
    -------
    np.ndarray
        Calculated weights.

    Notes
    -----
    See [1]_ or [2]_ for implementation details.

    References
    ----------
    [1] Hassani, H. (2007). Singular Spectrum Analysis: Methodology and
    Comparison. Journal of Data Science, 5(2), 239–257.
    https://doi.org/10.6339/JDS.2007.05(2).396

    [2] Golyandina, N., & Zhigljavsky, A. (2020). Singular Spectrum Analysis
    for Time Series. Berlin, Heidelberg: Springer.
    https://doi.org/10.1007/978-3-662-62436-4

    """
    k = timeseries_length - window + 1
    ls = min(window, k)
    ks = max(window, k)

    if ls > 1:
        weights = np.concatenate((np.arange(1, ls), np.full(ks - ls + 1, ls),
                                  np.arange(ls - 1, 0, -1)))
    else:
        weights = np.ones(timeseries_length)

    return weights


def weighted_correlation_matrix(
        reconstructed_series: Sequence[float],
        weights: np.ndarray
) -> np.ndarray:
    """Calculate the weighted correlation matrix.

    Parameters
    ----------
    reconstructed_series : array-like
        Input datasets matrix (or datasets frame) containing reconstructed
        time series of length `N` for the desired number of components
        (columns).
    weights : array-like
        Weights for the computation.

    Returns
    -------
    np.ndarray
        Weighted correlation matrix.

    Notes
    -----
    See [1]_ or [2]_ for implementation details.

    References
    ----------
    [1] Hassani, H. (2007). Singular Spectrum Analysis: Methodology and
    Comparison. Journal of Data Science, 5(2), 239–257.
    https://doi.org/10.6339/JDS.2007.05(2).396

    [2] Golyandina, N., & Zhigljavsky, A. (2020). Singular Spectrum Analysis
    for Time Series. Berlin, Heidelberg: Springer.
    https://doi.org/10.1007/978-3-662-62436-4

    """
    reconstructed_series = np.asarray(reconstructed_series)

    # Compute weighted covariance matrix
    weights = weights[np.newaxis, :]
    weighted_series = weights * np.conj(reconstructed_series)
    wcov_matrix = np.dot(weighted_series, reconstructed_series.T)

    # Convert to correlation matrix
    diag_covariances = np.diag(wcov_matrix)
    scales = np.sqrt(1 / np.abs(diag_covariances))
    wcorr_matrix = wcov_matrix * scales[:, np.newaxis] * scales[np.newaxis, :]
    np.fill_diagonal(wcorr_matrix, 1)

    # Fix possible numeric error
    wcorr_matrix = np.clip(wcorr_matrix, -1, 1)

    return wcorr_matrix


def construct_SVD_matrix(
        timeseries: Sequence[float],
        window: int | None = None,
        kind: 'str' = 'BK'
):
    """Construct the matrix for Singular Value Decomposition from time series.

    Parameters
    ----------
    timeseries : array-like
        The time series to be turned into a matrix for SVD, as one-dimensional
        array-like sequence of float.
    window : int | None, default=None
        Window size for the SVD matrix construction.
    kind : str, default='BK'
        Method for matrix construction. Either 'BK_trajectory', 'BK_covariance,
        or 'VG_covariance'. Default is 'BK_trajectory'. See Notes.

    Returns
    -------
    matrix: np.ndarray
        Bi-dimensional matrix of kind BK or VG. See Notes.

    Notes
    -----
    For the Broomhead & King (BK) matrix associated with the Basic Singular
    Spectrum Analysis, see [1]_. For the Vautard and Ghil (VG) matrix associated
    with the Toeplitz Singular Spectrum Analysis, see [2]_. For implementation
    and mathematical details, refer to [3]_.

    References
    ----------

    .. [1] Broomhead, D. S., & King, G. P. (1986). Extracting qualitative
    dynamics from experimental data. Physica D: Nonlinear Phenomena, 20(2),
    217–236. https://doi.org/10.1016/0167-2789(86)90031-X

    .. [2] Vautard, R., & Ghil, M. (1989). Singular spectrum analysis in
    nonlinear dynamics, with applications to paleoclimatic time series.
    Physica D: Nonlinear Phenomena, 35(3), 395–424.
    https://doi.org/10.1016/0167-2789(89)90077-8

    .. [3] Golyandina, N., & Zhigljavsky, A. (2020). Singular Spectrum Analysis
    for Time Series. Berlin, Heidelberg: Springer.
    https://doi.org/10.1007/978-3-662-62436-4

    Examples
    --------

    >>> ts = np.array([1,3,0,-3,-2,-1])
    >>> construct_SVD_matrix(ts, window=4, kind='BK')
    array([[ 1.,  3.,  0.],
           [ 3.,  0., -3.],
           [ 0., -3., -2.],
           [-3., -2., -1.]])

    >>> construct_SVD_matrix(ts, window=3, kind='VG')
    array([[ 4. ,  2.2, -1.5],
           [ 2.2,  4. ,  2.2],
           [-1.5,  2.2,  4. ]])

    """
    if window is None:
        window = len(timeseries) // 2
    if not isinstance(window, int):
        raise TypeError('window must be an integer')
    if not isinstance(kind, str):
        raise TypeError('kind must be a string')

    if kind == 'BK':
        method = construct_BK_trajectory_matrix
    elif kind == 'VG':
        method = construct_VG_covariance_matrix
    else:
        raise ValueError("SVD matrix 'kind' must be either 'BK_trajectory', "
                         "'BK_covariance', or 'VG_covariance'")

    matrix = method(timeseries, window=window)

    return matrix


def construct_BK_trajectory_matrix(
        timeseries: Sequence[float],
        window: int
) -> np.ndarray:
    """Construct Broomhead and King (BK) trajectory matrix from time series.

    Parameters
    ----------
    timeseries : array-like
        Time series to be turned into a BK matrix for SVD.
    window : int
        Window size.

    Returns
    -------
    BK_trajectory_matrix: np.ndarray
        BK lagged trajectory matrix.

    See Also
    --------
    construct_SVD_matrix
        For examples and references.

    """
    timeseries = np.asarray(timeseries)
    k = len(timeseries) - window + 1
    bk_trajectory_matrix = np.zeros(shape=(window, k))
    for i in range(k):
        bk_trajectory_matrix[:, i] = timeseries[i:i + window]
    return bk_trajectory_matrix


def construct_VG_covariance_matrix(
        timeseries: Sequence[float],
        window: int
):
    """Construct Vautard and Ghil lagged covariance matrix from time series.

    Parameters
    ----------
    timeseries : array-like
        Time series to be turned into a VG matrix for SVD.
    window : int
        Window size.

    Returns
    -------
    VG_covariance_matrix : np.ndarray
        VG lagged covariance matrix.

    See Also
    --------
    construct_SVD_matrix
        For examples and references.

    """
    timeseries = np.asarray(timeseries)
    n = len(timeseries)
    diag = np.array(
        [
            np.sum(timeseries[:n - i] * timeseries[i:]) / (n - i)
            for i in range(window)
        ]
    )
    vg_covariance_matrix = toeplitz(diag)
    return vg_covariance_matrix


def average_antidiagonals(matrix: np.ndarray) -> np.ndarray:
    """Average the anti-diagonals of a matrix.

    Averaging anti-diagonals allows to reconstruct a time series with from
    the SSA reconstructed group or component matrices, assuming a Hankel
    structure alike the original trajectory matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The 2D array matrix with anti-diagonals to be averaged.

    Returns
    -------
    timeseries: np.ndarray
        The 1D time series resulting from anti-diagonals averaging.

    Examples
    --------

    >>> mx = np.arange(12).reshape(4, 3)
    >>> mx
    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]])
    >>> average_antidiagonals(mx)
    array([ 0.,  2.,  4.,  7.,  9., 11.])

    """
    timeseries = np.array([
        np.mean(matrix[::-1, :].diagonal(i)) for i in
        range(-matrix.shape[0] + 1, matrix.shape[1])
    ])

    return timeseries


def autoregressive_model_score(
        timeseries: Sequence[float],
        order: int,
        criterion: Literal['aic', 'bic'] = 'bic',
):
    """
    Compute the information criterion score for an autoregressive model.

    This function fits an autoregressive (AR) model of a specified order to the
    given time series data and computes the selected information criterion
    score ('aic' or 'bic'). The AR model is fitted using the `SARIMAX` class
    from the `statsmodels` library, with no differencing or moving average
    components.

    Parameters
    ----------
    timeseries : Sequence[float]
        The time series data to which the autoregressive model is to be fitted.
    order : int
        The order of the autoregressive model.
    criterion : {'aic', 'bic'}, optional
        The information criterion used to evaluate the model. Either 'aic'
        (Akaike Information Criterion) or 'bic' (Bayesian Information Criterion).
        Default is 'bic'.

    Returns
    -------
    score : float
        The computed information criterion score for the fitted autoregressive
        model.

    References
    ----------
    .. [1] "statsmodels.tsa.statespace.sarimax.SARIMAX" `statsmodels` documentation.
           https://www.statsmodels.org/devel/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
    """
    # Input validation
    if len(timeseries) == 0:  # Check for empty sequence
        raise ValueError("timeseries cannot be empty")
    if order < 0:
        raise ValueError("order must be non-negative")
    if len(timeseries) <= order:
        raise ValueError("timeseries length must be greater than order")


    arp = sm.tsa.statespace.SARIMAX(
        timeseries,
        order=(order, 0, 0),
        trend=None
    ).fit()

    if criterion == 'bic':
        score = arp.bic
    elif criterion == 'aic':
        score = arp.aic

    return score


def fit_autoregressive_model(
        timeseries: Sequence[float],
        max_order: int = 1,
        criterion: Literal['aic', 'bic'] = 'bic',
        n_jobs: int | None = None,
) -> SARIMAXResults:
    """
    Fits an autoregressive model to the given time series data using the
    specified criterion to select the best order.

    This function evaluates autoregressive models of orders ranging from 0 to
    `max_order` and selects the model that minimizes the specified information
    criterion ('aic' or 'bic').

    The fitting process can be parallelized across multiple CPU cores if
    `n_jobs` is specified.

    Parameters
    ----------
    timeseries : Sequence[float]
        The time series data to which the autoregressive model is to be fitted.
    max_order : int, optional
        The maximum order of the autoregressive model to be evaluated. Default
        is 1.
    criterion : {'aic', 'bic'}, optional
        The information criterion used to select the best model. Default is
        'bic'.
    n_jobs : int or None, optional
        The number of CPU cores to use for parallel processing. If None, all
        available cores are used. If -1, also uses all available cores. Default
        is None.

    Returns
    -------
    autoregressive_model : statsmodels.tsa.statespace.sarimax.SARIMAXResults
        The fitted SARIMAX model object from the `statsmodels` library.

    References
    ----------
    .. [1] "statsmodels.tsa.statespace.sarimax.SARIMAX" `statsmodels` documentation.
           https://www.statsmodels.org/devel/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    """

    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    if not isinstance(max_order, int):
        raise TypeError("max_order must be an integer")
    if len(timeseries) <= max_order:
        raise ValueError("timeseries length must be greater than max_order")

    if n_jobs is None:
        n_jobs = -1
    order_list = list(range(max_order + 1))
    model_scores = Parallel(n_jobs=n_jobs)(
        delayed(autoregressive_model_score)(
            timeseries,
            order,
            criterion
        ) for order in order_list
    )
    best_order = order_list[np.argmin(model_scores)]
    autoregressive_model = sm.tsa.statespace.SARIMAX(
        timeseries,
        order=(best_order, 0, 0),
        trend=None
    ).fit()

    return autoregressive_model


def generate_autoregressive_surrogate(
        ar_coefficients: Sequence[float],
        n_samples: int,
        scale: float,
        seed: int | None = None,
        burnin: int = 100
):
    """
    Generate a surrogate time series using an autoregressive (AR) process.

    This function generates an autoregressive surrogate time series based on
    specified AR coefficients.

    Parameters
    ----------
    ar_coefficients : Sequence[float]
        The coefficient for autoregressive lag polynomial, including zero lag.
    n_samples : int
        The number of samples to generate in the surrogate time series.
    scale : float
        The standard deviation of the white noise component added to the AR model.
    seed : int or None, optional
        Random seed for reproducibility. If `None`, the random number generator
        is not seeded.
    burnin : int
        Number of initial samples to discard to reduce the effect of initial
        conditions. Default is 100.

    Raises
    ------
    ValueError
        If max_order is negative or greater than the length of timeseries
    TypeError
        If max_order is not an integer

    Returns
    -------
    np.ndarray
        An array containing the generated autoregressive surrogate time series.

    Notes
    -----

    - The function uses `statsmodels.tsa.arima_process.arma_generate_sample`
      [1]_ to generate the AR time series.
    - As noted in [1]_, the AR components should include the coefficient on the
      zero-lag. This is typically 1. Further, the AR parameters should have the
      opposite sign of what you might expect. See the examples below.
    - Standardizing the generated series helps in preventing any scale-related
      issues in further analysis.
    - The function sets a burn-in period of 100 samples to mitigate the
      influence of initial conditions.

    References
    ----------
    .. [1] "ARMA Process." `statsmodels` documentation.
           https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_process.arma_generate_sample.html

    Examples
    --------

    >>> ar_coefficients = [1, -0.9]
    >>> n_samples = 5
    >>> scale = 1.0
    >>> seed = 42
    >>> surrogate = generate_autoregressive_surrogate(ar_coefficients, n_samples, scale, seed)
    >>> print(surrogate)
    [ 1.28271453  0.6648326   0.36050652 -1.39807629 -0.90997736]

    Raises
    ------
    ValueError
        If n_samples or scale is not positive, or if ar_coefficients is empty
        or doesn't start with 1.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if not ar_coefficients:
        raise ValueError("ar_coefficients must not be empty")
    if ar_coefficients[0] != 1:
        raise ValueError("First AR coefficient must be 1")

    if seed is not None:
        np.random.seed(seed)
    surrogate = arma_generate_sample(
        ar_coefficients,
        [1],
        n_samples,
        scale=scale,
        burnin=burnin
    )

    return surrogate


if __name__ == '__main__':
    import doctest

    doctest.testmod()
