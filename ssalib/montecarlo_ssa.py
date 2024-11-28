"""Monte Carlo Singular Spectrum Analysis"""
from __future__ import annotations

from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from ssalib.error import DecompositionError
from ssalib.math_ext.ar_modeling import (
    fit_autoregressive_model,
    generate_autoregressive_surrogate
)
from ssalib.ssa import SingularSpectrumAnalysis, SSAMatrixType


class MonteCarloSSA(SingularSpectrumAnalysis):
    """Monte Carlo Singular Spectrum Analysis

    Proposed by [1]_, Monte Carlo Singular Spectrum Analysis relies on
    autoregressive surrogate of the original time series [2]_ to test the
    significance of the components' singular (or eigen) values.

    Parameters
    ----------
    timeseries : ArrayLike
        Timeseries data as a one-dimensional array-like sequence of
        float, e.g., a python list, numpy array, or pandas series.
        If timeseries is a pd.Series with a pd.DatetimeIndex, the index
        will be stored to return SSA-decomposed time series as pd.Series
        using the same index.
    window : int, optional
        Window length for the SSA algorithm. Defaults to half the series
        length if not provided.
    svd_matrix: str, default 'BK'
        Matrix to use for the SVD algorithm, either 'BK' or 'VG', with
        defaults to 'BK' (see Notes).
    svd_solver : str, default 'np_svd'
        Method of singular value decomposition to use. Call the
        available_solver method for possible options.
    standardize : bool, default True
        Whether to standardize the timeseries by removing the mean and
        scaling to unit variance.
    n_surrogates : int, default=100
        Number of surrogates to generate.
    n_jobs : int, default=-1
        Number of jobs to run in parallel using joblib.Parallel. -1 means using
        all processors (default). See joblib documentation for further details.
    ar_order_max : int, default=1
        Maximum autoregressive order to consider. Default is 1 corresponding
        to AR1 surrogates (proposed in [1]_).
    criterion : Literal['aic', 'bic'], default='bic'
        Citerion to use for the autoregressive model selection. Either
        'aic' or 'bic'. Default is 'bic'.
    random_seed : int | None, default=None
        Random seed to use for the surrogates generation. If None, no seed is
        used (default), and surrogates will be different each time the class is
        instantiated.

    Attributes
    ----------
    ar_order_max : int
        Maximum order considered for the autoregressive model selection.
    autoregressive_model : SARIMAXResults
        Fitted SARIMAX model results from statsmodels.
    criterion : Literal['aic', 'bic']
        Citerion used for the autoregressive model selection.
    n_jobs : int
        Number of jobs to run in parallel for surrogate computations. -1 means
        using all processors (default).
    n_surrogates: int
        Number of surrogates.
    random_seed: int | None
        Random seed to use for the surrogates generation.
    surrogates : NDArray[float]
        Array of shape (n_surrogates, n) containing the n_surrogate generated
        time series of length n.


    Notes
    -----
    For surrogate generation, MonteCarloSSA fits an autoregressive model,
    selecting automatically the optimal order in a sequence of orders from zero
    (white noise) to ar_max_order. The model is fitted using state space
    modeling [3]_, due to its tolerance to missing data, using the statsmodels
    package and the SARIMAX model [4]_. The model with the lowest AIC or BIC
    is automatically selected. The fitted coefficients are passed to
    the statsmodels arma_generate_sample function to generate the surrogates.

    The algorithm creates covariance matrices from the surrogates, depending on
    the selected approach (either 'BK' or 'VG'). The surrogate matrices
    are projected onto the eigenvectors, and the distribution of surrogate
    strengths (or power) are compared to the singular (or eigen) values
    resulting from the decomposition of the original time series.


    References
    ----------
    .. [1] Allen, M. R. and Smith, L. A. (1996). Monte Carlo SSA: Detecting
           irregular oscillations in the Presence of Colored Noise, Journal of
           Climate, 9, 3373–3404,
           https://doi.org/10.1175/1520-0442(1996)009<3373:MCSDIO>2.0.CO;2
    .. [2] Schreiber, T. and Schmitz, A. (2000). Surrogate time series,
           Physica D: Nonlinear Phenomena, 142, 346–382,
           https://doi.org/10.1016/S0167-2789(00)00043-9
    .. [3] Durbin, J. and Koopman, S. J. (2012). Time series analysis by state
           space methods, 2nd ed., Oxford University Press, Oxford, 346 pp.,
           2012. ISBN: 978-0-19-964117-8
    .. [4] Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and
           statistical modeling with python. 9th Python in Science Conference.


    Example
    -------

    >>> from ssalib.datasets import load_sst
    >>> from ssalib.montecarlo_ssa import MonteCarloSSA
    >>> sst = load_sst()
    >>> mcssa = MonteCarloSSA(sst, n_surrogates=10, random_seed=42)
    >>> _ = mcssa.decompose()
    >>> mcssa.test_significance(n_components=5)
    array([False, False,  True,  True,  True])

    """

    def __init__(
            self,
            *args,
            n_surrogates: int = 100,
            n_jobs: int = -1,
            ar_order_max: int = 1,
            criterion: Literal['aic', 'bic'] = 'bic',
            random_seed: int = None,
            **kwargs,
    ):

        # Init SingularSpectrumAnalysis
        super().__init__(*args, **kwargs)

        # Type checking
        if not isinstance(n_surrogates, int):
            raise TypeError(f"Argument n_surrogates must be an integer, got "
                            f"{type(n_surrogates)}")
        if not isinstance(n_jobs, int):
            raise TypeError(f"Argument n_jobs must be an integer, got "
                            f"{type(n_jobs)}")
        if not isinstance(ar_order_max, int):
            raise TypeError(f"Argument ar_order_max must be an integer, got "
                            f"{type(ar_order_max)}")
        if not isinstance(criterion, str):
            raise TypeError(f"Argument criterion must be a string, got "
                            f"{type(criterion)}")
        if random_seed is not None and not isinstance(random_seed, int):
            raise TypeError(f"Argument random_seed must be None or an integer, "
                            f"got {type(random_seed)}")

        # Value validation
        if n_surrogates <= 0:
            raise ValueError(f"Argument n_surrogates must be positive, got "
                             f"{n_surrogates}")
        if ar_order_max < 0:
            raise ValueError(f"Argument ar_order_max must be non-negative, got "
                             f"{ar_order_max}")
        if criterion not in ['aic', 'bic']:
            raise ValueError(f"Argument criterion must be either 'aic' or "
                             f"'bic', got {criterion}")

        # Init MonteCarloSSA
        self.n_surrogates = n_surrogates
        self.ar_order_max = ar_order_max
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.random_seed = random_seed

        # center timeseries to prevent statsmodels warning
        ts = self._timeseries_pp - self._timeseries_pp.mean()
        # check na
        if any(self.na_mask):
            ts = np.where(self.na_mask, np.nan, ts)  # TODO to test

        self.autoregressive_model = fit_autoregressive_model(
            ts,
            max_order=ar_order_max,
            criterion=self.criterion,
            n_jobs=self.n_jobs,
        )

        # Init and generate surrogates
        self.surrogates = np.zeros(shape=(n_surrogates, self._n))
        self._generate_surrogates()
        # reintroduces the mean, if any
        self.surrogates += self._timeseries_pp.mean()

    def test_significance(
            self,
            n_components: int | None = None,
            confidence_level: float = 0.95,
            two_tailed: bool = True
    ) -> NDArray[bool]:
        """Test if components' singular values are significantly higher than
        surrogates' projected strength distribution

        Parameters
        ----------
        n_components : int | None, default=None
            Number of the first components to test. If None, all components
            are tested. Default is None.
        confidence_level : float, default=0.95
            Confidence level to determine the significance. Default is 0.95.
        two_tailed : bool, default=True
            If true (default), significance level is achieved for singular
            values above percentile 100 - 100 * (1 - confidence_level) / 2.
            If false, significance level is achieved above percentile
            100 * confidence_level.
        """
        if self.n_components is None:
            raise DecompositionError("Method test_significance cannot be called"
                                     "before the decompose method")
        if n_components is None:
            n_components = self.n_components

        upper_percentile = self.get_confidence_interval(
            n_components, confidence_level, two_tailed, return_lower=False
        )

        return np.greater(self.s_[:n_components], upper_percentile)

    def get_confidence_interval(
            self,
            n_components: int | None = None,
            confidence_level: float = 0.95,
            two_tailed: bool = True,
            return_lower: bool = True,
    ) -> tuple[NDArray[float], NDArray[float]] | NDArray[float]:
        """Return the confidence interval for the surrogates' projected
        strengths.

        Parameters
        ----------
        n_components : int | None
            Number of the first components to return. If None, all components
            are tested. Default is None.
        confidence_level : float
            Confidence level to determine the confidence interval. Default is
            0.95.
        two_tailed : bool
            If True (default), the confidence interval is two-tailed, hence
            corresponding to values between percentiles
            100*(1 - confidence_level)/2 and 100 - 100*(1 - confidence_level)/2.
            If False, the confidence interval corresponds to values between the
            minimum and the 100*confidence_level percentile.
        return_lower : bool
            If True (default), return the lower and upper limits of the
            confidence interval. If False, only return the upper limit.

        Returns
        -------
        tuple[NDArray[float], NDArray[float]] | NDArray[float]
            Depending on the value of return_lower, either a tuple of array
            with the upper and lower interval limits, or only the upper limit.

        """
        if not isinstance(n_components, int) and n_components is not None:
            raise TypeError(f"Argument n_components must be either integer or "
                            f"None, got {type(n_components)}")
        if not isinstance(confidence_level, float):
            raise TypeError(f"Argument confidence_level must be a float, "
                            f"got {type(confidence_level)}")
        if not isinstance(two_tailed, bool):
            raise TypeError(f"Argument two_tailed must be a boolean, "
                            f"got {type(two_tailed)}")

        if n_components is None:
            n_components = self.n_components

        if n_components > self.n_components or n_components < 1:
            raise ValueError(f"Argument n_components must be between 1 and "
                             f"the number of components '{self.n_components}', "
                             f"got '{n_components}'")
        if not isinstance(return_lower, bool):
            raise TypeError(f"Argument return_lower must be a boolean, "
                            f"got {type(return_lower)}")
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError(f"Argument confidence_level must be between 0 "
                             f"and 1, got {confidence_level}")

        lower_percentile, upper_percentile = self._get_percentile_interval(
            confidence_level, two_tailed
        )

        surrogate_value_strengths = self._get_surrogate_values(n_components)

        upper = np.percentile(
            surrogate_value_strengths,
            upper_percentile,
            axis=0
        )
        if not return_lower:
            return upper
        else:
            lower = np.percentile(
                surrogate_value_strengths,
                lower_percentile,
                axis=0
            )
            return lower, upper

    def _generate_surrogates(self) -> None:
        """Generate surrogates
        """
        random_seed = self.random_seed
        n_jobs = self.n_jobs
        n = self._n
        m = self.n_surrogates

        if random_seed is not None:
            np.random.seed(random_seed)
        random_seeds = np.random.randint(0, 1e8, m)

        # AR coefficient includes zero-lag and are in the polynomial form.
        # See generate_autoregressive_surrogate docstrings, itself based on
        # statsmodels 'arma_generate_sample' method.
        ar_coefficients = [1] + list(-self.autoregressive_model.params[:-1])
        sigma = np.sqrt(self.autoregressive_model.params[-1])

        surrogates_list = Parallel(n_jobs=n_jobs)(
            delayed(generate_autoregressive_surrogate)(
                ar_coefficients,
                n, sigma,
                random_seeds[i]
            ) for i in range(m)
        )
        surrogates = np.stack(surrogates_list, axis=0)
        self.surrogates = surrogates

    def _get_surrogate_values(
            self,
            n_components: int | None = None
    ) -> NDArray[float]:
        """Project surrogates on eigenvectors and retrieve strength distribution

        Parameters
        ----------
        n_components : int | None
            Number of first component eigenvectors to project onto.

        Returns
        -------
        NDArray[float]

        """
        k = self._n - self._window + 1

        # Process surrogates in parallel
        diagonals = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_surrogate)(
                surrogate,
                self._svd_matrix_kind,
                self._window,
                self.u_,
                n_components
            ) for surrogate in self.surrogates
        )

        # Stack all results at once
        lambda_surrogates = np.stack(diagonals, axis=0)
        if n_components is None:
            n_components = lambda_surrogates.shape[1]

        # Convert to singular values based on method
        if self._svd_matrix_kind in {
            SSAMatrixType.BK_TRAJECTORY,
            SSAMatrixType.BK_COVARIANCE
        }:
            surrogate_value_strengths = np.sqrt(
                np.abs(lambda_surrogates) * k)
        elif self._svd_matrix_kind == SSAMatrixType.VG_COVARIANCE:
            # normalization factors for VG approach
            vg_norms = self._n - np.arange(n_components)
            vg_norms = vg_norms[np.newaxis, :]
            surrogate_value_strengths = np.sqrt(
                np.abs(lambda_surrogates) * vg_norms
            )
        else:
            raise ValueError(
                f"Invalid svd_matrix_kind: {self._svd_matrix_kind}")
        return surrogate_value_strengths

    @staticmethod
    def _process_surrogate(
            surrogate: NDArray,
            svd_matrix_kind: Literal[
                'bk_trajectory', 'bk_covariance', 'vg_covariance'],
            window: int,
            u: NDArray,
            n_components: int | None = None
    ) -> NDArray[float]:
        """Process a single surrogate time series

        Compute the surrogate covariance matrix and project it on the
        eigenvectors and return the diagonal elements.

        Parameters
        ----------
        surrogate : NDArray
            Single surrogate time series
        svd_matrix_kind : str
            Type of SVD matrix ('BK' or 'VG')
        window : int
            Window length
        k : int
            Number of columns in trajectory matrix (n - w + 1)
        u : NDArray
            Left eigenvectors
        n_components : int | None
            Number of first eigenvector components to project onto. If None
            (default), all vectors are used.

        Returns
        -------
        NDArray[float]
            Diagonal elements after projection
        """
        if n_components is not None:
            # Slice the eigenvector matrix to use only first n_components
            u = u[:, :n_components]

        svd_matrix_kind = SSAMatrixType(svd_matrix_kind)

        if svd_matrix_kind == SSAMatrixType.BK_TRAJECTORY:
            covariance_matrix_surr = SSAMatrixType(
                'bk_covariance').construct_svd_matrix(
                timeseries=surrogate,
                window=window
            )
        elif svd_matrix_kind in {
            SSAMatrixType.BK_COVARIANCE,
            SSAMatrixType.VG_COVARIANCE
        }:
            covariance_matrix_surr = svd_matrix_kind.construct_svd_matrix(
                timeseries=surrogate,
                window=window
            )
        else:
            raise ValueError(
                f"Invalid svd_matrix_kind: {svd_matrix_kind}")

        return np.diag(u.T @ covariance_matrix_surr @ u)

    @staticmethod
    def _get_percentile_interval(
            confidence_level: float,
            two_tailed: bool = True,
    ) -> tuple[float, float]:
        """Get the percentile limit based on desired confidence level
        """

        if two_tailed:
            lower_percentile = 100 * (1 - confidence_level) / 2
            upper_percentile = 100 - 100 * (1 - confidence_level) / 2
        else:
            lower_percentile = 0
            upper_percentile = 100 * confidence_level
        return lower_percentile, upper_percentile


if __name__ == '__main__':
    import doctest

    doctest.testmod()
