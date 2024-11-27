"""Singular Spectrum Analysis"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ssalib.error import DecompositionError, ReconstructionError
from ssalib.math_ext.matrix_operations import (
    average_antidiagonals,
    construct_bk_trajectory_matrix,
    construct_bk_covariance_matrix,
    construct_vg_covariance_matrix,
    correlation_weights,
    weighted_correlation_matrix
)
from ssalib.plotting import PlotSSA
from ssalib.svd import SVDHandler, SVDSolverType

logger = logging.getLogger(__name__)

_CONSTRUCT_FUNC_MAPPING = {
    "bk_trajectory": construct_bk_trajectory_matrix,
    "bk_covariance": construct_bk_covariance_matrix,
    "vg_covariance": construct_vg_covariance_matrix
}


class SSAMatrixType(Enum):
    """Available SVD matrix types for timeseries embedding and decomposition.

    Enumeration of supported matrix types, mapping user-facing
    names to their corresponding method identifiers.
    """
    BK_TRAJECTORY = "bk_trajectory"
    BK_COVARIANCE = "bk_covariance"
    VG_COVARIANCE = "vg_covariance"

    def construct_svd_matrix(self, timeseries: ArrayLike, window: int):
        """Constructs an SVD matrix based on the SSAMatrixType."""
        try:
            construct_matrix = _CONSTRUCT_FUNC_MAPPING[self.value]
        except KeyError:
            raise ValueError(
                f"Unsupported matrix type: {self.value}. "
                f"Available matrix types are: "
                f"{', '.join(self.available_matrices())}."
            )

        return construct_matrix(timeseries, window)

    @property
    def is_covariance(self) -> bool:
        return self in {
            SSAMatrixType.BK_COVARIANCE,
            SSAMatrixType.VG_COVARIANCE
        }

    @classmethod
    def available_matrices(cls) -> list[str]:
        return [matrix.value for matrix in cls]


class SingularSpectrumAnalysis(SVDHandler, PlotSSA):
    """Singular Spectrum Analysis (SSA).

    Singular Spectrum Analysis (SSA) provides non-parametric linear
    decomposition of a time series relying on the Singular Value Decomposition
    (SVD) of a matrix constructed from the time series.

    The SingularSpectrumAnalysis class provides a handful API to different
    timeseries embedding approaches, SVD solvers, and plotting options.

    Parameters
    ----------
    timeseries : ArrayLike
        The timeseries data as a one-dimensional array-like sequence of
        float, e.g., a python list, numpy array, or pandas series.
        If timeseries is a pd.Series with a pd.DatetimeIndex, the index
        will be stored to return SSA-decomposed time series as pd.Series
        using the same index.
    window : int, optional
        The window length for the SSA algorithm. Defaults to half the series
        length if not provided.
    svd_matrix_kind: SSAMatrixType | str,
        Matrix to use for the SVD algorithm, either 'bk_trajectory',
        'bk_covariance', or 'vg_covariance', with
        defaults to 'bk_trajectory'.
    svd_solver : SVDSolverType | str, default SVDSolverType.NUMPY_STANDARD
        The method of singular value decomposition to use. Call the
        available_solver method for possible options.
    standardize : bool, default True
        Whether to standardize the timeseries by removing the mean and
        scaling to unit variance.
    na_strategy : str, default 'raise_error'
        Strategy to handle missing values in the timeseries. If 'raise_error'
        (default), ValueError is raised. If 'fill_mean', missing values are
        replaced by the mean of the timeseries (i.e., zeros if standardize
        is True).

    Attributes
    ----------
    mean_ : float
        Mean of the original timeseries. Available after initialization.
    std_ : float
        Standard deviation of the original timeseries. Available after
        initialization.
    n_components : int or None
        Number of components after decomposition. None if decomposition has
        not been performed.
    s_ : ndarray or None
        Singular values from decomposition. None if decomposition has not been
        performed.
    u_ : ndarray or None
        Left singular vectors from decomposition. None if decomposition has not
        been performed.
    vt_ : ndarray or None
        Right singular vectors from decomposition. None if decomposition has
        not been performed.

    Notes
    -----
    - The Broomhead & King 'bk_trajectory' matrix [1]_ is a unit-delay lagged
      embedding of the time series of shape (window, k), where
      the maximum lag k is given by len(timeseries) - window + 1.
      The bk_trajectory matrix in an Hankel matrix, i.e., having its
      antidiagonal elements equal
    - The 'bk_covariance' matrix is the covariance matrix of shape (window,
      window) given by 1/k * bk_trajectory @ bk_trajectory.T. Both matrices
      relate to the equivalent singular system, but the 'bk_covariance' matrix
      can be more efficient for large matrices.
    - For the Vautard and Ghil 'vg_covariance' matrix is a covariance matrix
      of shape (window, window) with a Toeplitz structure, i.e.,
      having its diagonal elements equal [2]_.

    See [3]_ for additional mathematical details.

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

    >>> from ssalib.datasets import load_sst
    >>> from ssalib.ssa import SingularSpectrumAnalysis
    >>> sst = load_sst() # Sea Surface Temperature Data
    >>> ssa = SingularSpectrumAnalysis(sst, window=20) # Initialization
    >>> _ = ssa.decompose()  # Perform SVD
    >>> _ = ssa.reconstruct(groups={'trend': [0], 'seasonal': [1,2]})  # Group components
    >>> ssa['trend'].head(5)  # Access reconstructed groups and show first values
    Date
    1982-01-15   -0.700452
    1982-02-15   -0.714862
    1982-03-15   -0.736786
    1982-04-15   -0.766092
    1982-05-15   -0.795355
    Name: trend, dtype: float64
    """

    _DEFAULT_GROUPS = {
        'ssa_original': '_timeseries',
        'ssa_preprocessed': '_timeseries_pp',
        'ssa_reconstructed': '_ssa_reconstructed',
        'ssa_residuals': '_ssa_residuals'
    }

    def __init__(
            self,
            timeseries: ArrayLike,
            window: int | None = None,
            svd_matrix_kind: SSAMatrixType | str = SSAMatrixType.BK_TRAJECTORY,
            svd_solver: SVDSolverType | str = SVDSolverType.NUMPY_STANDARD,
            standardize: bool = True,
            na_strategy: Literal['raise_error', 'fill_mean'] = 'raise_error',
    ) -> None:
        SVDHandler.__init__(self, svd_solver)

        # Initialize timeseries
        if hasattr(timeseries, 'index'):
            self._ix = timeseries.index
        else:
            self._ix = None

        # Validate na_strategy
        if na_strategy not in ['raise_error', 'fill_mean']:
            raise ValueError(
                f"Argument na_strategy should be either 'raise_error' or "
                f"'fill_mean', got '{na_strategy}' instead"
            )
        self._na_strategy: str = na_strategy

        # Validate timeseries
        self._timeseries: NDArray = self.__validate_timeseries(timeseries)

        # Create na_mask
        if na_strategy == 'raise_error':
            self.na_mask = np.zeros_like(timeseries, dtype=bool)
        else:
            self.na_mask = np.isnan(timeseries)
        self._has_na: bool = any(self.na_mask)

        # Other attributes
        self._n: int = self._timeseries.shape[0]
        self.mean_: float = np.nanmean(self._timeseries)
        self.std_: float = np.nanstd(self._timeseries)
        self._standardized: bool = standardize
        if self._na_strategy == 'fill_mean':
            self._timeseries[self.na_mask] = self.mean_
        if standardize:
            self._timeseries_pp = (self._timeseries - self.mean_) / self.std_
        else:
            self._timeseries_pp = self._timeseries

        # Initialize matrix construction
        self._svd_matrix_kind: SSAMatrixType = self.__validate_svd_matrix_kind(
            svd_matrix_kind
        )
        self._window: int = self.__validate_window(window)

        # Initialize groups for reconstruction
        self._user_groups: dict[str, int | list[int]] | None = None

    def __repr__(self) -> str:
        ts_type = 'Series' if self._ix is not None else 'Array'
        ts_shape = f"shape={self._timeseries.shape}"
        return (f"{self.__class__.__name__}(timeseries=<{ts_type} {ts_shape}>, "
                f"window={self._window}, svd_matrix='{self._svd_matrix_kind}', "
                f"svd_solver='{self.svd_solver}', standardize="
                f"{self._standardized})")

    def __str__(self) -> str:
        ts_format = 'Series' if self._ix is not None else 'Array'
        n, mu, sigma = self._n, self.mean_, self.std_

        # Determine decomposition status
        status = 'Initialized'
        n_components = str(self.n_components)
        if self.n_components is not None:
            status = 'Decomposed'
        if self._user_groups is not None:
            status = 'Reconstructed'

        # Format parameters section
        params = [
            f"timeseries: {ts_format}, n={n}, mean={mu:.2f}, std={sigma:.2f}",
            f"svd_matrix_kind: {self._svd_matrix_kind.value} "
            f"{self.svd_matrix.shape}",
            f"window: {self._window}",
            f"standardize: {self._standardized}",
            f"svd_solver: {self.svd_solver}",
            f"status: {status}",
            f"n_components: {n_components}",
            f"groups: {self._user_groups or 'None'}"
        ]
        print_str = f"""
# {self.__class__.__name__}
{chr(10).join(params)
        }"""

        return print_str

    def __getitem__(
            self,
            item: int | slice | list[int] | str
    ) -> NDArray | pd.Series:
        """API to access SSA timeseries data."""
        self.__validate_item_keys(item)

        if isinstance(item, str):
            if item in self._DEFAULT_GROUPS.keys():
                default_attr = self._DEFAULT_GROUPS[item]
                timeseries = getattr(self, default_attr)
            else:
                timeseries = self.__get_user_timeseries_by_group_name(item)
        else:
            timeseries = self._reconstruct_group_timeseries(item)

        if self._ix is not None:
            name = item if isinstance(item, str) else None
            timeseries = pd.Series(index=self._ix, data=timeseries, name=name)

        return timeseries

    def __get_user_timeseries_by_group_name(
            self,
            group_name: str
    ):
        """Get time series by default- or user-group name.
        """
        return self._reconstruct_group_timeseries(self.groups[group_name])

    def __validate_int_key(self, key: int) -> None:
        """Validate __getitem__ key for an integer key.
        """
        if self.n_components is None:
            raise DecompositionError(
                "Cannot retrieve components by indices prior to "
                "decomposition. Call the 'decompose' "
                "method first"
            )
        elif key < 0 or key >= self.n_components:
            raise KeyError(f"Integer key '{key}' is is out of range")

    def __validate_item_keys(self, key: Any) -> None:
        """Validate __getitem__ key.
        """
        if isinstance(key, str):
            self.__validate_string_key(key)
        elif isinstance(key, int):
            self.__validate_int_key(key)
        elif isinstance(key, slice):
            self.__validate_slice_key(key)
        elif isinstance(key, list):
            self.__validate_list_key(key)
        else:
            raise KeyError(
                f"Key '{key}' is not a valid key type. Make sure to "
                f"retrieve timeseries by using integer indices, list of integer "
                f"indices, slices of integer indices, or group name strings"
            )

    def __validate_list_key(self, key: list) -> None:
        """Validate __getitem__ key for a list key.
        """
        if self.n_components is None:
            raise DecompositionError(
                "Cannot retrieve components by list prior to decomposition. "
                "Call the 'decompose' method first"
            )
        if not all(isinstance(x, int) for x in key):
            raise KeyError("All indices in the list must be integers")
        if any(x < 0 or x >= self.n_components for x in key):
            raise KeyError(f"Indices in the list {key} are out of range.")

    def __validate_slice_key(self, key: slice) -> None:
        """Validate __getitem__ key for a slice key.
        """
        if self.n_components is None:
            raise DecompositionError(
                "Cannot retrieve components by slice prior to decomposition. "
                "Call the 'decompose' method first"
            )

        # Check start index
        if key.start is not None:
            if key.start < 0 or key.start >= self.n_components:
                raise KeyError(
                    f"Slice start index '{key.start}' is out of range. It must "
                    f"be between 0 and {self.n_components - 1}.")

        # Check stop index
        if key.stop is not None:
            if key.stop < 0 or key.stop > self.n_components:
                raise KeyError(
                    f"Slice stop index '{key.stop}' is out of range. It must "
                    f"be between 0 and {self.n_components}.")

            # If both start and stop are specified, ensure start < stop
            if key.start is not None and key.start >= key.stop:
                raise KeyError(
                    f"Slice start index '{key.start}' must be less than "
                    f"stop index '{key.stop}'.")

    def __validate_string_key(self, key: str) -> None:
        """Validate __getitem__ key for a string key.
        """
        if key in ["ssa_reconstructed", "ssa_residuals"]:
            if self.n_components is None:
                raise DecompositionError(
                    f"Cannot access '{key}' prior to decomposition. "
                    f"Call the 'decompose' method first"
                )
        elif key not in self.groups.keys():
            if self.n_components is None:
                raise DecompositionError(
                    f"Cannot access user-defined key '{key}' prior to "
                    f"decomposition and reconstruction. Call the "
                    f"decompose and reconstruct method first"
                )
            elif self._user_groups is None:
                raise ReconstructionError(
                    "Cannot access user-defined key prior to group "
                    "reconstruction. Call the reconstruct method first")
            else:
                raise KeyError(
                    f"Key '{key}' is not a valid group name. Valid group names "
                    f"are {', '.join(self.groups.keys())}"
                )

    @staticmethod
    def __validate_svd_matrix_kind(
            svd_matrix_kind: SSAMatrixType | str
    ) -> SSAMatrixType:
        """Validates SVD matrix kind.
        """
        if isinstance(svd_matrix_kind, str):
            try:
                svd_matrix_kind = SSAMatrixType(svd_matrix_kind)
            except ValueError:
                valid_matrices = SSAMatrixType.available_matrices()
                raise ValueError(
                    f"Invalid svd_matrix_kind '{svd_matrix_kind}'. "
                    f"Valid matrices are: {', '.join(valid_matrices)}."
                )
        elif not isinstance(svd_matrix_kind, SSAMatrixType):
            raise TypeError(
                f"Argument svd_matrix_kind must be of type str or "
                f"SSAMatrixType, got {type(svd_matrix_kind)}"
            )

        return svd_matrix_kind

    def __validate_timeseries(
            self,
            timeseries: ArrayLike
    ) -> NDArray[float]:
        """Validates the timeseries data.
        """
        timeseries = np.squeeze(np.array(timeseries))
        if timeseries.ndim != 1:
            raise ValueError("Argument timeseries must be one-dimensional")

        if not np.issubdtype(timeseries.dtype, np.number):
            raise ValueError(
                "All timeseries elements must be integers or floats"
            )
        if (self._na_strategy == 'raise_error' and
                (any(np.isinf(timeseries)) or any(np.isnan(timeseries)))):
            raise ValueError(
                "Argument timeseries cannot inf or NaN values with na_strategy "
                "set to 'raise_error'"
            )
        return timeseries

    def __validate_user_groups(
            self,
            groups: dict[str, int | list[int]]
    ) -> None:
        """Validates the user_groups dictionary.

        Parameters
        ----------
        groups : dict[str, int | list[int]]
            A dictionary where keys are strings and values are either int or
            list of int representing eigentriple components to label as a group.

        Raises
        ------
        ValueError
            If any key is not a string, if any key is in self.__default_groups,
            if any value is not an int or list of int, or if any int value is
            negative or equal to or above self.n_components.

        Warnings
        --------
        Warns if duplicate integers are found in the combined values of all
        entries.

        """
        if not isinstance(groups, dict):
            raise TypeError(
                f"Argument groups must be a dictionary, got {type(groups)}"
            )

        all_values = []

        for key, value in groups.items():
            # Validate key
            if not isinstance(key, str):
                raise TypeError(
                    f"Key types in groups dictionary should be string, "
                    f"got {type(key)}"
                )
            if key in self._DEFAULT_GROUPS:
                raise ValueError(
                    f"Group name '{key}' is reserved for default group names. "
                    f"Use a different group name"
                )

            # Validates value and collect all integers
            if isinstance(value, int):
                value = [value]
            elif not (isinstance(value, list) and all(
                    isinstance(i, int) for i in value)):
                raise ValueError(
                    f"Value for key '{key}' must be an int or list of int, "
                    f"got {type(value)}"
                )

            if any(i < 0 or i >= self.n_components for i in value):
                raise ValueError(
                    f"Values for key '{key}' must be in the range 0 <= value < "
                    f"{self.n_components}."
                )
            all_values.extend(value)

        # Check for duplicate integer indices
        unique_values, counts = np.unique(all_values, return_counts=True)
        duplicates = unique_values[counts > 1]
        if duplicates.size > 0:
            logger.warning(
                f"Reconstructed groups contain duplicate indices: "
                f"{duplicates.tolist()}"
            )

    def __validate_window(
            self,
            window: int | None
    ) -> int:
        """Validates the embedding window parameter.
        """
        if window is None:
            window = self._n // 2
        elif not isinstance(window, int):
            raise TypeError(
                f"Argument window must be integer, got {type(window)}"
            )
        elif (window < 2) or (window > self._n - 2):
            raise ValueError(
                f'Invalid window size {window}. Valid size must be '
                f'between 2 and {self._n - 2} (n-2). Recommended window size '
                f'is between 2 and {self._n // 2} (n/2)')
        return window

    def decompose(
            self,
            n_components: int | None = None,
            **kwargs: Any
    ) -> "SingularSpectrumAnalysis":
        """Perform Singular Value Decomposition (SVD) of the SVD matrix.

        SVD is applied on the SingularSpectrumAnalysis.svd_matrix using the
        svd_solver selected at initialization.

        Decomposition enables plotting features for exploration prior to
        reconstruction. It also enables access to singular values and
        eigenvectors attributes.

        Parameters
        ----------
        n_components: int | None, default=None
            Number of components, i.e., dimensionality, of the resulting SVD
            decomposition. The argument n_components is required for
            truncated SVD solvers and ignored for full SVD.

        Other Parameters
        ----------------
        **kwargs: Any
            Additional arguments passed to the SVD solver. Available
            arguments depend on the chosen solver type.

        Returns
        -------
        self : SingularSpectrumAnalysis
            The instance itself for method chaining.

        See Also
        --------
        SVDHandler
            For details about available solvers and SVD algorithms.

        Examples
        --------

        >>> from ssalib.datasets import load_sst
        >>> from ssalib import SingularSpectrumAnalysis
        >>> sst = load_sst() # Sea Surface Temperature data
        >>> ssa = SingularSpectrumAnalysis(sst).decompose()

        After decomposition, the following attributes become available:
        - n_components : Number of components (dimensionality)
        - s_ : Singular values
        - u_ : Left singular vectors
        - vt_ : Right singular vectors

        >>> print(ssa.n_components)
        252

        The reconstruct method becomes available, and the reconstructed
        components can also be reconstructed on the fly using the getitem API,
        e.g., by indices or slices (or group name after reconstrion).

        The reconstructed timeseries are returned as the same type than the
        input timeseries.

        >>> ssa[0].head() # using component integer indices and show first values
        Date
        1982-01-15   -1.183962
        1982-02-15   -1.196059
        1982-03-15   -1.208318
        1982-04-15   -1.220207
        1982-05-15   -1.229758
        dtype: float64
        """
        # Input check is done by self.svd
        self.svd(self.svd_matrix, n_components, **kwargs)
        p, _, _ = self.decomposition_results

        # Extra-processing for covariance matrices
        if self._svd_matrix_kind.is_covariance:
            S = self._trajectory_matrix.T @ p
            s = np.linalg.norm(S, axis=0)
            ix_sorted = np.argsort(s)[::-1]
            q = S / s
            p, s, q = p[:, ix_sorted], s[ix_sorted], q[ix_sorted]
            self.decomposition_results = p, s, q.T
        return self

    def get_confidence_interval(
            self,
            n_components: int | None = None,
            confidence_level: float = 0.95,
            two_tailed: bool = True,
            return_lower: bool = True,
    ) -> tuple[NDArray[float], NDArray[float]] | NDArray[
        float]:
        raise NotImplementedError(
            "Method get_confidence_interval is only available for class "
            "MonteCarloSSA"
        )

    def reconstruct(
            self,
            groups: dict[str, int | list[int]]
    ) -> "SingularSpectrumAnalysis":
        """Reconstruct components based on eigentriple indices.

        Define user groups for the signal reconstruction.

        Parameters
        ----------
        groups : dict[str, int | list[int]]
            User-defined groups of component indices for reconstruction. Keys
            represents user-defined group names (str) and values are single
            (int) or multiple (list of int) indices of eigen triples to use
            for the reconstruction.

        Returns
        -------
        self: SingularSpectrumAnalysis
            The Singular Spectrum Analysis object with decomposition method
            achieved.
        """

        n_components = self.n_components
        if n_components is None:
            raise DecompositionError(
                "Decomposition must be performed before reconstruction. "
                "Call the decompose method prior to the reconstruct method"
            )

        # Validate and set or update user groups
        self.__validate_user_groups(groups)
        self._user_groups = groups

        return self

    def test_significance(
            self,
            n_components: int | None = None,
            confidence_level: float = 0.95,
            two_tailed: bool = True
    ) -> NDArray[bool]:
        raise NotImplementedError("Method test_significance is only"
                                  "available for class MonteCarloSSA")

    def to_frame(
            self,
            include: list[str] | None = None,
            exclude: list[str] | None = None,
            rescale: bool = False
    ) -> pd.DataFrame:
        """Return signals as a pandas.DataFrame.

        Return `pandas.DataFrame` with all signals unless specified otherwise
        with the 'include' or 'exclude' parameters. If the
        `SingularSpectrumAnalysis` object was instantiated with and
        `pandas.Series` the returned DataFrame will have the same index.

        Parameters
        ----------
        include : list[str] | None, default=None
            Group names to include as column in the return `pandas.DataFrame`.
            If None, all groups will be included unless the 'exclude' parameter
            is specified.
        exclude : list[str] | None, default=None
            Group names to exclude as column in the return `pandas.DataFrame`.
            If None, all groups will be included unless the 'include' parameter
            is specified.
        rescale : bool, default=False
            If True, rescale the signals relative to the original signal's
            standard deviation and reintroduce the original mean.

        Returns
        -------
        signals: pd.DataFrame
            Data frame with the requested default and grouped signals as
            columns.

        Raises
        ------
        ValueError
            If include or exclude contains unknown group names.
        ValueError
            If both exclude and include are specified.
        """
        if include is not None and exclude is not None:
            raise ValueError(
                "Cannot specify both include and exclude parameters.")

        group_names = list(self.groups.keys())

        if include is not None:
            if any(name not in group_names for name in include):
                raise ValueError(f"Parameter include contains unknown group "
                                 f"names. Valid group names are "
                                 f"{', '.join(self.groups.keys())}")
            group_names = [name for name in group_names if name in include]

        if exclude is not None:
            if any(name not in group_names for name in exclude):
                raise ValueError(f"Parameter exclude contains unknown group "
                                 f"names. Valid group names are "
                                 f"{', '.join(self.groups.keys())}")
            group_names = [name for name in group_names if name not in exclude]

        signals = pd.DataFrame(
            {name: self.__getitem__(name) for name in group_names})

        if self._ix is not None:
            signals.set_index(self._ix, inplace=True)

        if rescale:
            signals.apply(self._rescale)

        return signals

    @property
    def groups(self) -> dict[str, list[int] | None] | None:
        """ Return tgroup names and their eigentriple indices.

        Any group name registered in `groups` is a key to get the values of
        the corresponding signal using a `__getitem__` method.

        If no grouping was done using `reconstruct` method, group names
        are limited to groups defined by default:

        * 'ssa_original': the original signal passed to instantiate the
          SingularSpectrumAnalysis` object. This group is not related to
          eigentriple indices.
        * 'ssa_preprocessed': the preprocessed signal after the instantiation
          of the `SingularSpectrumAnalysis` object. This group is not related
          to eigentriple indices.
        * 'ssa_reconstructed': the signal reconstructed from all available
          eigentriple indices. The reconstructed signal may differ from the
          original one if the singular value decomposition is truncated.

        Reconstructing the signal makes new user groups available based on their
        user-defined group names. It also adds another default group:

        * 'ssa_residuals': the residual signal key and its corresponding
          eigentriple indices.

        Examples
        --------

        >>> from ssalib.datasets import load_sst
        >>> timeseries = load_sst()
        >>> ssa = SingularSpectrumAnalysis(timeseries).decompose()
        >>> _ = ssa.reconstruct(groups={'main': [0,1]})
        >>> ssa.groups['main']
        [0, 1]

        >>> ssa['main'].head(5)
        Date
        1982-01-15   -0.993569
        1982-02-15   -1.006196
        1982-03-15   -1.024155
        1982-04-15   -1.045457
        1982-05-15   -1.065604
        Name: main, dtype: float64

        See Also
        --------
        :meth: SingularSpectrumAnalysis.reconstruct
            Method used for reconstructing components based on eigentriple
            indices.
        """

        all_names = list(self._DEFAULT_GROUPS.keys())
        all_indices = [None, None]

        if self.n_components is not None:
            all_indices += [None]

        if self._user_groups is not None:
            # Get user defined group names
            user_groups_names = list(self._user_groups.keys())
            all_names += user_groups_names

            # Get user defined grouped indexes of singular values
            user_groups_indices = list(self._user_groups.values())
            residuals_indices = None
            all_indices += [residuals_indices]
            all_indices += user_groups_indices

        groups = dict(zip(all_names, all_indices))

        return groups

    @property
    def squared_frobenius_norm(self) -> np.ndarray:
        """Squared Frobenius norm of the trajectory matrix.

        Returns
        -------
        squared_frobenius_norm : float
            The squared Frobenius norm of the trajectory matrix (see Notes).

        Notes
        -----
        The squared Frobenius norm of a matrix is equal to the sum of its
        eigenvalues. The squared Frobenius norm is useful to scale the norm of
        the SSA components, especially when SSA relies on truncated SVD
        algorithms.

        """
        return np.linalg.norm(self._trajectory_matrix, 'fro') ** 2

    @property
    def svd_matrix(self) -> NDArray[float]:
        """Matrix decomposed with SVD

        Returns
         -------
        svd_matrix : NDArray[float]
            The matrix constructed for SVD decomposition, either trajectory or
            covariance matrix depending on parameter svd_matrix_kind.

        """
        svd_matrix = self._svd_matrix_kind.construct_svd_matrix(
            self._timeseries_pp,
            window=self._window
        )
        return svd_matrix

    @property
    def _trajectory_matrix(self) -> NDArray[float]:
        """Return bk_trajectory matrix"""
        trajectory_matrix = SSAMatrixType('bk_trajectory').construct_svd_matrix(
            self._timeseries_pp,
            window=self._window
        )
        return trajectory_matrix

    @property
    def _ssa_reconstructed(self) -> NDArray[float]:
        """Return the reconstructed timeseries signal.
        """
        if self.s_ is None:
            ssa_reconstructed = None
        else:
            full_range = range(self.n_components)
            ssa_reconstructed = self._reconstruct_group_timeseries(full_range)
        return ssa_reconstructed

    @property
    def _ssa_residuals(self) -> NDArray[float]:
        """Return the residual timeseries signal.
        """
        if self.s_ is None:
            ssa_residuals = None
        else:
            user_indices = list(self._user_indices)
            ssa_residuals = (self._timeseries_pp -
                             self._reconstruct_group_timeseries(user_indices))
        return ssa_residuals

    @property
    def _user_indices(self) -> set:
        """Return the set of user indices.
        """
        if self._user_groups is None:
            raise ReconstructionError("Cannot retrieve user indices without "
                                      "defined user groups. Define user groups "
                                      "using the 'reconstruct' method first.")
        else:
            user_indices = set()
            for user_indices_values in self._user_groups.values():
                if isinstance(user_indices_values, int):
                    user_indices_values = [user_indices_values]
                user_indices.update(set(user_indices_values))
        return user_indices

    def wcorr(
            self,
            n_components: int
    ) -> NDArray[float]:
        """Calculate the weighted correlation matrix for a number of components.

        Parameters
        ----------
        n_components : int
            The number of components used to compute the weighted correlation
            matrix.

        Returns
        -------
        wcorr : np.ndarray
            The weighted correlation matrix.

        Raises
        ------
        DecompositionError
            If called before decomposition is performed.
        ValueError
            If n_components is not a positive integer or exceeds available
            components.

        See Also
        --------
        ssalib.math_ext.weighted_correlation_matrix
            For examples and references.

        """
        timeseries = np.array(
            [self._reconstruct_group_timeseries([i]) for i in
             range(n_components)])
        weights = correlation_weights(self._n, self._window)
        wcorr = weighted_correlation_matrix(timeseries, weights=weights)
        return wcorr

    def _rescale(
            self,
            timeseries: pd.Series
    ) -> pd.Series:
        """Rescale the timeseries signal to its original standard deviation
        and reintroduce the original mean.
        """
        if self._standardized and timeseries.name != 'ssa_original':
            timeseries *= self.std_
            timeseries += self.mean_
        return timeseries

    def _reconstruct_group_matrix(
            self,
            group_indices: int | slice | range | list[int]
    ) -> NDArray[float]:
        """ Reconstructs a group matrix using components group indices.

        Parameters
        ----------
        group_indices : int | slice | range | list of int
            Eigentriple indices used to group and reconstruct the time series.
            Time series can be reconstructed using a slice `group_indices`.


        Returns
        -------
        NDArray[float]
            The reconstructed matrix.

        """
        u, s, v = self.u_, self.s_, self.vt_

        if isinstance(group_indices, int):
            group_indices = [group_indices]
        if isinstance(group_indices, slice):
            start, stop, step = group_indices.indices(self.n_components)
            group_indices = list(range(start, stop, step))

        u_selected = u[:, group_indices]
        s_selected = np.diag(s[group_indices])
        v_selected = v[group_indices, :]

        if self._svd_matrix_kind == SSAMatrixType.BK_TRAJECTORY:
            reconstructed_group_matrix = u_selected @ s_selected @ v_selected
        elif self._svd_matrix_kind.is_covariance:
            X = SSAMatrixType('bk_trajectory').construct_svd_matrix(
                self._timeseries_pp,
                window=self._window
            )
            S = X.T @ u_selected
            reconstructed_group_matrix = u_selected @ S.T
        else:
            raise TypeError(
                f"Cannot reconstruct matrix type '{self._svd_matrix_kind.value}'"
            )

        return reconstructed_group_matrix

    def _reconstruct_group_timeseries(
            self,
            group_indices: int | slice | range | list[int]
    ) -> NDArray[float]:
        """ Reconstructs a time series using the group component indices.

        Parameters
        ----------
        group_indices : int | slice | range | list of int
            Eigentriple indices used to group and reconstruct the time series.
            Time series can be reconstructed using a slice group_indices.

        Returns
        -------
        reconstructed_timeseries : NDArray[float]
            The reconstructed time series.

        """

        reconstructed_group_matrix = self._reconstruct_group_matrix(
            group_indices)

        # Anti-diagonal averaging
        reconstructed_timeseries = average_antidiagonals(
            reconstructed_group_matrix
        )
        return reconstructed_timeseries


if __name__ == '__main__':
    from doctest import testmod

    testmod()
