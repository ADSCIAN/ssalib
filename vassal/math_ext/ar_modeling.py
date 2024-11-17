"""Autoregressive Modeling Utility Functions for Singular Spectrum Analysis.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def autoregressive_model_score(
        timeseries: ArrayLike,
        order: int,
        criterion: Literal['aic', 'bic'] = 'bic',
):
    """
    Compute the information criterion score for an autoregressive model.

    This function fits an autoregressive (AR) model of a specified order to the
    given timeseries data and computes the selected information criterion
    score ('aic' or 'bic'). The AR model is fitted using the `SARIMAX` class
    from the `statsmodels` library, with no differencing or moving average
    components.

    Parameters
    ----------
    timeseries : ArrayLike
        The time series data to which the autoregressive model is to be fitted.
    order : int
        The order of the autoregressive model.
    criterion : {'aic', 'bic'}, optional
        The information criterion used to evaluate the model. Either 'aic'
        (Akaike Information Criterion) or 'bic' (Bayesian Information
        Criterion). Default is 'bic'.

    Returns
    -------
    score : float
        The computed information criterion score for the fitted autoregressive
        model.

    Raises
    ------
    TypeError
        If `order` is not an integer or `criterion` is not a string.
    ValueError
        If `timeseries` is empty, `order` is negative, the length of
        `timeseries` is less than or equal to `order`, or `criterion` is not
        'aic' or 'bic'.

    References
    ----------
    .. [1] "statsmodels.tsa.statespace.sarimax.SARIMAX" `statsmodels` documentation.
           https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
    """

    if not isinstance(order, int):
        raise TypeError("Argument order must be an integer")
    if not isinstance(criterion, str):
        raise TypeError("Argument criterion must be a string")

    if len(timeseries) == 0:  # Check for empty sequence
        raise ValueError("Argument timeseries cannot be empty")
    if order < 0:
        raise ValueError("Argument order must be non-negative")
    if len(timeseries) <= order:
        raise ValueError(
            "Argument timeseries must have length greater than order")

    arp = sm.tsa.statespace.SARIMAX(
        timeseries,
        order=(order, 0, 0),
        trend=None
    ).fit(disp=False)

    if criterion == 'bic':
        score = arp.bic
    elif criterion == 'aic':
        score = arp.aic
    else:
        raise ValueError(
            "Argument criterion must be either 'aic' or 'bic'")

    return score


def fit_autoregressive_model(
        timeseries: ArrayLike,
        max_order: int = 1,
        criterion: Literal['aic', 'bic'] = 'bic',
        n_jobs: int | None = None,
) -> SARIMAXResults:
    """
    Fits an autoregressive model to the given time series data using the
    specified criterion to select the best order.

    This function evaluates autoregressive models of orders ranging from 0 to
    `max_order` and selects the model that minimizes the specified information
    criterion ('aic' or 'bic'). The fitting process can be parallelized across
    multiple CPU cores if `n_jobs` is specified.

    Parameters
    ----------
    timeseries : ArrayLike
        The time series data to which the autoregressive model is to be fitted.
    max_order : int, optional, default=1
        The maximum order of the autoregressive model to be evaluated. Default
        is 1.
    criterion : {'aic', 'bic'}, optional, default='bic'
        The information criterion used to select the best model. Default is
        'bic'.
    n_jobs : int or None, optional, default=None
        The number of CPU cores to use for parallel processing. If None, all
        available cores are used. If -1, also uses all available cores. Default
        is None.

    Returns
    -------
    autoregressive_model : statsmodels.tsa.statespace.sarimax.SARIMAXResults
        The fitted SARIMAX model object from the `statsmodels` library.

    Raises
    ------
    ValueError
        If `max_order` is negative, or if the length of `timeseries` is less
        than or equal to `max_order`, or if `criterion` is not 'aic' or 'bic'.
    TypeError
        If `max_order` is not an integer.

    References
    ----------
    .. [1] "statsmodels.tsa.statespace.sarimax.SARIMAX" statsmodels
           documentation.
           https://www.statsmodels.org/devel/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    """

    if max_order < 0:
        raise ValueError("Argument max_order must be non-negative")
    if not isinstance(max_order, int):
        raise TypeError("Argument max_order must be an integer")
    if len(timeseries) <= max_order:
        raise ValueError(
            "Argument timeseries must have length greater than max_order")
    if criterion not in ['aic', 'bic']:
        raise ValueError(
            "Argument criterion must be either 'aic' or 'bic'")

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
    ).fit(disp=False)

    return autoregressive_model


def generate_autoregressive_surrogate(
        ar_coefficients: NDArray[float],
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
    ar_coefficients : NDArray[float]
        The coefficient for autoregressive lag polynomial, including zero lag.
    n_samples : int
        The number of samples to generate in the surrogate time series.
    scale : float
        The standard deviation of the white noise component added to the AR model.
    seed : int | None, optional, default=None
        Random seed for reproducibility. If `None`, the random number generator
        is not seeded.
    burnin : int, optional, default=100
        Number of initial samples to discard to reduce the effect of initial
        conditions. Default is 100.

    Returns
    -------
    NDArray[float]
        An array containing the generated autoregressive surrogate time series.

    Raises
    ------
    ValueError
        If `n_samples` or `scale` is not positive, or if `ar_coefficients` is
        empty or does not start with 1.
    TypeError
        If `n_samples` is not an integer or `scale` is not a float or integer.

    Notes
    -----
    - The function uses statsmodels.tsa.arima_process.arma_generate_sample
      [1]_ to generate the AR time series.
    - As noted in [1]_, the AR components should include the coefficient on the
      zero-lag. This is typically 1. Further, the AR parameters should have the
      opposite sign of what you might expect. See the examples below.
    - The function sets a burn-in period of 100 samples to mitigate the
      influence of initial conditions.

    References
    ----------
    .. [1] "ARMA Process." statsmodels documentation.
           https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_process.arma_generate_sample.html

    Examples
    --------

    >>> ar_coefficients = [1, -0.9]
    >>> n_samples = 5
    >>> scale = 1.0
    >>> seed = 42
    >>> surrogate = generate_autoregressive_surrogate(ar_coefficients, n_samples, scale, seed)
    >>> print(surrogate)
    [-2.29389472 -2.48515057 -2.57935003 -3.1236923  -2.97260878]
    """
    # Type checking
    if not isinstance(n_samples, int):
        raise TypeError(f"Argument n_samples must be an integer, got "
                        f"{type(n_samples)}")
    if not isinstance(scale, (int, float)):
        raise TypeError(f"Argument scale must be a number, got {type(scale)}")
    if seed is not None and not np.issubdtype(type(seed), np.integer):
        raise TypeError(f"Argument seed must be None or an integer, got "
                        f"{type(seed)}")
    if not isinstance(burnin, int):
        raise TypeError(f"Argument burnin must be an integer, got "
                        f"{type(burnin)}")

    if n_samples <= 0:
        raise ValueError(f"Argument n_samples must be positive, got "
                         f"{n_samples}")
    if scale <= 0:
        raise ValueError(f"Argument scale must be positive, got {scale}")
    if not ar_coefficients:
        raise ValueError(f"Argument ar_coefficients must not be empty, got "
                         f"{ar_coefficients}")
    if ar_coefficients[0] != 1:
        raise ValueError(
            f"Argument ar_cofficients should have 1 as first element, got "
            f"{ar_coefficients[0]}")

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
