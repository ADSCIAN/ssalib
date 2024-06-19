"""Math utility functions for Singular Spectrum Analysis."""

# Author: Damien Delforge <damien.delforge@adscian.be>
#         Alice Alonso <alice.alonso@adscian.be>
#
# License: BSD 3 clause

from typing import Sequence

import numpy as np
from scipy.linalg import toeplitz


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


if __name__ == '__main__':
    import doctest

    doctest.testmod()
