import numpy as np
from numpy.typing import ArrayLike


def correlation_weights(
        timeseries_length: int,
        window: int
) -> np.ndarray:
    """
    Calculate the default weights.

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

    References
    ----------
    [1] Hassani, Hossein. "Singular Spectrum Analysis: Methodology and
    Comparison." MPRA Paper, April 1, 2007.
    https://mpra.ub.uni-muenchen.de/4991/.

    [2] Golyandina, N., & Zhigljavsky, A. (2020). Singular Spectrum Analysis for
    Time Series. Berlin, Heidelberg: Springer.
    https://doi.org/10.1007/978-3-662-62436-4

    """
    K = timeseries_length - window + 1
    Ls = min(window, K)
    Ks = max(window, K)

    if Ls > 1:
        weights = np.concatenate((np.arange(1, Ls), np.full(Ks - Ls + 1, Ls),
                                  np.arange(Ls - 1, 0, -1)))
    else:
        weights = np.ones(timeseries_length)

    return weights


def weighted_correlation_matrix(
        reconstructed_series: ArrayLike,
        weights: ArrayLike
) -> np.ndarray:
    """
    Calculate the weighted correlation matrix for the input matrix `x`.

    Parameters
    ----------
    reconstructed_series : ArrayLike
        Input datasets matrix (or datasets frame) of shape containing reconstructed
        time series of length `N` for the desired number of components
        (columns).
    weights : ArrayLike
        Weights for the computation.

    Returns
    -------
    np.ndarray
        Weighted correlation matrix.

    References
    ----------
    [1] Hassani, Hossein. "Singular Spectrum Analysis: Methodology and
    Comparison." MPRA Paper, April 1, 2007.
    https://mpra.ub.uni-muenchen.de/4991/.

    [2] Golyandina, N., & Zhigljavsky, A. (2020). Singular Spectrum Analysis for
    Time Series. Berlin, Heidelberg: Springer.
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
