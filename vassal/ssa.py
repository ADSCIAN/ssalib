"""Singular Spectrum Analysis"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from vassal.log_and_error import DecompositionError, ReconstructionError
from vassal.math_ext.matrix_operations import (
    average_antidiagonals,
    construct_SVD_matrix,
    construct_BK_trajectory_matrix,
    construct_BK_covariance_matrix,
    construct_VG_covariance_matrix,
    correlation_weights,
    weighted_correlation_matrix
)
from vassal.plotting import PlotSSA
from vassal.svd import SVDHandler


class SingularSpectrumAnalysis(SVDHandler, PlotSSA):
    """Singular Spectrum Analysis (SSA).

    #TODO: document methods and attributes

    Singular Spectrum Analysis (SSA) provides non-parametric linear
    decomposition of a time series relying on the Singular Value Decomposition
    (SVD) of a matrix constructed from the time series. The SVD decomposed
    matrix is either a lagged embedding, following the Broomhead and King (BK)
    trajectory matrix approach, or a Toeplitz lagged covariance matrix,
    following the Vautard and Ghil (VG) approach.

    Parameters
    ----------
    timeseries : ArrayLike
        The timeseries data as a one-dimensional array-like sequence of
        float, e.g, a python list,  numpy array, or pandas series.
        If timeseries is a pd.Series with a pd.DatetimeIndex, the index
        will be stored to return SSA-decomposed time series as pd.Series
        using the same index.
    window : int, optional
        The window length for the SSA algorithm. Defaults to half the series
        length if not provided.
    svd_matrix: str, default 'BK'
        Matrix to use for the SVD algorithm, either 'BK' or 'VG', with
        defaults to 'BK' (see Notes).
    svd_solver : str, default 'numpy_standard'
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

    Notes
    -----
    The basic SSA version relies on the SVD of a lagged trajectory matrix,
    an approach proposed by Broomhead & King (1986), referred to as the
    BK approach [1]_. The alternative propose by Vautard and Ghil (1989),
    hereafter VG [2]_, relies instead on the SVD of a lagged covariance
    matrix showing a Toeplitz structure.

    Both implementation are explained in [3]_.

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

    """

    def __init__(
            self,
            timeseries: ArrayLike,
            window: int | None = None,
            svd_matrix: Literal['BK', 'VG'] = 'BK',
            svd_solver: str = 'numpy_standard',
            standardize: bool = True,
            na_strategy: Literal['raise_error', 'fill_mean'] = 'raise_error',
            # TODO test na_strategy
    ) -> None:

        SVDHandler.__init__(self, svd_solver)

        # Initialize timeseries
        if hasattr(timeseries, 'index'):
            self._ix = timeseries.index
        else:
            self._ix = None

        if na_strategy not in ['raise_error', 'fill_mean']:
            raise ValueError(f"Argument na_strategy should be either "
                             f"'raise_error' or 'fill_mean', got {na_strategy} "
                             f"instead.")

        if na_strategy == 'raise_error':
            allow_na = False
            self.na_mask = np.zeros_like(timeseries, dtype=bool)
        else:
            allow_na = True
            self.na_mask = np.isnan(timeseries)

        self._na_strategy: str = na_strategy
        self._timeseries: NDArray = self.__validate_timeseries(timeseries,
                                                               allow_na)
        self._has_na: bool = np.isnan(self._timeseries).any()

        self._n: int = self._timeseries.shape[0]
        self.mean_: float = np.nanmean(self._timeseries)
        self.std_: float = np.nanstd(self._timeseries)
        if self._na_strategy == 'fill_mean':
            self._timeseries[self.na_mask] = self.mean_

        if standardize:
            self._timeseries_pp = (self._timeseries - self.mean_) / self.std_
        else:
            self._timeseries_pp = self._timeseries

        self._standardized: bool = standardize

        # Initialize decomposition
        self._svd_matrix_kind: str = self.__validate_svd_matrix_kind(svd_matrix)
        self._w: int = self.__validate_window(window)

        # Initialize groups
        self._default_groups = {
            'ssa_original': '_timeseries',
            'ssa_preprocessed': '_timeseries_pp',
            'ssa_reconstructed': '_ssa_reconstructed',
            'ssa_residuals': '_ssa_residuals'
        }
        self._user_groups: dict[str, int | list[int]] | None = None

    def __repr__(self) -> str:
        ts_format = 'Series' if self._ix is not None else 'Array'
        ts_str = pd.Series(self._timeseries, index=self._ix).__str__() if (
                ts_format == 'Series') else self._timeseries.__str__()
        repr_str = f"""
{self.__class__.__name__}(
    timeseries={ts_str},
    window={self._w},
    svd_matrix='{self._svd_matrix_kind}',
    svd_solver='{self.svd_solver}',
    standardize={self._standardized}
)
        """
        return repr_str

    def __str__(self) -> str:
        ts_format = 'Series' if self._ix is not None else 'Array'
        ts_str = pd.Series(self._timeseries, index=self._ix).__str__() if (
                ts_format == 'Series') else self._timeseries.__str__()
        n, mu, sigma = self._n, self.mean_, self.std_
        status = 'None'
        n_components = str(self.n_components)
        groups = 'None'
        if self.n_components is not None:
            status = 'Decomposed'
        if self._user_groups is not None:
            status = 'Reconstructed'
            groups = self._user_groups
        ssa_str = f"""
Singular Spectrum Analysis
--------------------------
# Parameters
kind: {self._svd_matrix_kind}
window: {self._w}
standardize: {self._standardized}
svd_solver: {self.svd_solver}
status: {status}
n_components: {n_components}
groups: {groups}

# Timeseries ({ts_format}, n={n}, mean={mu:.2f}, std={sigma:.2f})
{ts_str}
        """
        return ssa_str

    def __getitem__(
            self,
            item: int | slice | list[int] | str
    ) -> NDArray | pd.Series:
        """API to access SSA timeseries data."""
        self.__validate_item_keys(item)

        if isinstance(item, str):
            if item in self._default_groups.keys():
                default_attr = self._default_groups[item]
                timeseries = getattr(self, default_attr)
            else:
                timeseries = self.__get_user_timeseries_by_group_name(item)
        else:
            timeseries = self._reconstruct_group_timeseries(item)

        if self._ix is not None:
            name = item if isinstance(item, str) else None
            timeseries = pd.Series(index=self._ix, data=timeseries, name=name)

        return timeseries

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
            raise KeyError(f"Key '{key}' is not a valid key type. Make sure to "
                           f"retrieve timeseries by indices or group name.")

    def __validate_string_key(self, key: str) -> None:
        """Validate __getitem__ key for a string key.
        """
        if key in ["ssa_reconstructed", "ssa_residuals"]:
            if self.n_components is None:
                raise DecompositionError(
                    f"Cannot access '{key}' prior to decomposition. Make sure "
                    f"to call the 'decompose' method first."
                )
        elif key not in self.groups.keys():
            if self.n_components is None:
                raise DecompositionError(
                    f"Cannot access user-defined key '{key}' prior to "
                    f"decomposition and reconstruction. Make sure to call the "
                    f"'decompose' and 'reconstruct' method first."
                )
            elif self._user_groups is None:
                raise ReconstructionError(
                    "Cannot access user-defined key prior to group "
                    "reconstruction. Make sure to define user groups using the "
                    "'reconstruct' method first.")
            else:
                raise KeyError(
                    f"Key '{key}' is not a valid group name. Valid group names "
                    f"are {', '.join(self.groups.keys())}."
                )

    def __validate_int_key(self, key: int) -> None:
        """Validate __getitem__ key for a integer key.
        """
        if self.n_components is None:
            raise DecompositionError(
                "Cannot retrieve components by indices prior to "
                "decomposition. Make sure to call the 'decompose' "
                "method first."
            )
        elif key < 0 or key >= self.n_components:
            raise KeyError(f"Integer key '{key}' is is out of range.")

    def __validate_slice_key(self, key: slice) -> None:
        """Validate __getitem__ key for a slice key.
        """
        if self.n_components is None:
            raise DecompositionError(
                "Cannot retrieve components by slice prior to decomposition. "
                "Make sure to call the 'decompose' method first."
            )
        start, stop, step = key.indices(self.n_components)
        if stop > self.n_components:
            raise KeyError(f"Slice '{key}' is out of range.")

    def __validate_list_key(self, key: list) -> None:
        """Validate __getitem__ key for a list key.
        """
        if self.n_components is None:
            raise DecompositionError(
                "Cannot retrieve components by list prior to decomposition. "
                "Make sure to call the 'decompose' method first.")
        if not all(isinstance(x, int) for x in key):
            raise KeyError("All indices in the list must be integers.")
        if any(x < 0 or x >= self.n_components for x in key):
            raise KeyError(f"Indices in the list {key} are out of range.")

    @staticmethod
    def __validate_svd_matrix_kind(
            svd_matrix_kind: str
    ) -> str:
        """Validates SVD matrix kind.
        """
        if not isinstance(svd_matrix_kind, str):
            raise TypeError("svd_matrix must be a string.")
        if svd_matrix_kind not in ['BK', 'VG']:
            raise ValueError("svd_matrix must be 'BK' or 'VG'.")
        return svd_matrix_kind

    @staticmethod
    def __validate_timeseries(
            timeseries: ArrayLike,
            allow_na: bool = False
    ) -> NDArray:
        """Validates the timeseries data.
        """
        timeseries = np.squeeze(np.array(timeseries))
        if not np.issubdtype(timeseries.dtype, np.number):
            raise ValueError("All elements must be integers or floats.")
        if not allow_na and (
                np.any(np.isinf(timeseries)) or np.any(np.isnan(timeseries))):
            raise ValueError("The array contains inf or NaN values.")
        if timeseries.ndim != 1:
            raise ValueError("Timeseries must be one-dimensional.")
        return timeseries

    def __validate_window(
            self,
            window: int | None
    ) -> int:
        """Validates the embedding window parameter.
        """
        if window is None:
            window = self._n // 2
        elif not isinstance(window, int):
            raise ValueError("Invalid window type. Parameter 'window' must be "
                             "an integer.")
        elif (window < 2) or (window > self._n - 2):
            raise ValueError(
                f'Invalid window size. Current size is {window}, but it must be'
                f' between 2 and {self._n - 2} (n-2). Recommended window size '
                f'is between 2 and {self._n // 2} (n/2).')
        return window

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
            raise TypeError("user_groups must be a dictionary.")

        all_values = []
        seen_keys = set()

        for key, value in groups.items():
            # Validate key
            if not isinstance(key, str):
                raise ValueError(f"Key '{key}' is not a string.")
            if key in seen_keys:
                raise ValueError(f"Duplicate key '{key}' found.")
            if key in self._default_groups:
                raise ValueError(f"Unauthorized group name '{key}', name is "
                                 f"reserved for use in the default groups. "
                                 f"Please use a different group name.")
            seen_keys.add(key)

            # Validates value and collect all integers
            if isinstance(value, int):
                value = [value]
            elif not (isinstance(value, list) and all(
                    isinstance(i, int) for i in value)):
                raise ValueError(
                    f"Value for key '{key}' must be an int or list of int.")

            if any(i < 0 or i >= self.n_components for i in value):
                raise ValueError(
                    f"Values for key '{key}' must be in the range 0 <= value <"
                    f" {self.n_components}.")

            all_values.extend(value)

        # Check for duplicate integer indices
        unique_values, counts = np.unique(all_values, return_counts=True)
        duplicates = unique_values[counts > 1]
        if duplicates.size > 0:
            logger = logging.getLogger(self.__module__)
            log_func = getattr(logger, 'warning')
            log_func(
                f"Reconstructed groups contain duplicate indices: "
                f"{duplicates.tolist()}")

    def __get_user_timeseries_by_group_name(
            self,
            group_name: str
    ):
        """Get time series by default- or user-group name.
        """
        return self._reconstruct_group_timeseries(self.groups[group_name])

    def decompose(
            self,
            n_components: int | None = None,
            **kwargs
    ) -> "SingularSpectrumAnalysis":
        """Perform Singular Value Decomposition (SVD) of the constructed matrix.

        SVD is applied on the ´SingularSpectrumAnalysis.svd_matrix´ using the
        solver selected at initialization. Decomposition enable plotting
        features for exploration prior to reconstruction. It also enables access
        to eigenvalues and eigenvectors attributes.

        Parameters
        ----------
        n_components: int | None, default=None

        Other Parameters
        ----------------
        kwargs: dict
            Any additional keyword arguments taken by the SVD decomposition
            method.

        Returns
        -------
        self: SingularSpectrumAnalysis
            The Singular Spectrum Analysis object with decomposition method
            achieved.

        See Also
        --------
        SVDHandler
            For details about available solvers and SVD algorithms.

        """
        # Input check is done by self.svd
        self.svd(self.svd_matrix, n_components, **kwargs)
        p, _, _ = self.decomposition_results

        # Extra-processing for the 'VG' algorithm
        # TODO: Explain in docstring
        if self._svd_matrix_kind == 'VG':
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
    ) -> tuple[NDArray[float], NDArray[float]] | NDArray[float]:
        raise NotImplementedError("Method get_confidence_interval is only"
                                  "available for class MonteCarloSSA")

    def reconstruct(
            self,
            groups: dict[str, int | list[int]]
    ) -> "SingularSpectrumAnalysis":
        """Reconstruct components based on eigen-triples indices.

        Define user groups for the signal reconstruction.

        Parameters
        ----------
        groups : dict[str, int | list[int]]
            Dictionary of user defined groups of eigen triples for
            reconstruction. Keys represents user defined group names as text
            strings and values are single (int) or multiple (list of int)
            indices of eigen triples to use for the reconstruction.

        Returns
        -------
        self: SingularSpectrumAnalysis
            The Singular Spectrum Analysis object with decomposition method
            achieved.

        """

        n_components = self.n_components
        if n_components is None:
            raise DecompositionError("Decomposition must be performed before "
                                     "reconstruction. Make sure to call the "
                                     "'decompose' method before the "
                                     "'reconstruct' method.")

        # Validate and set or update user groups
        self.__validate_user_groups(groups)
        self._user_groups = groups

        return self

    def test_significance(
            self,
            n_components: int | None = None,
            confidence_level: float = 0.95,
            two_tailed: bool = True
    ) -> NDArray[np.bool_]:
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
                "Cannot specify both 'include' and 'exclude' parameters.")

        group_names = list(self.groups.keys())

        if include is not None:  # TODO write test
            if any(name not in group_names for name in include):
                raise ValueError(f"Invalid group names in 'include'. "
                                 f"Valid group names are "
                                 f"{', '.join(self.groups.keys())}")
            group_names = [name for name in group_names if name in include]

        if exclude is not None:  # TODO write test
            if any(name not in group_names for name in exclude):
                logging.warning(f"Ignored unknown group names in 'exclude'. "
                                f"Valid group names are "
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

        >>> from vassal.datasets import load_sst
        >>> timeseries = load_sst()
        >>> ssa = SingularSpectrumAnalysis(timeseries)
        >>> u, s, v = ssa.decompose()
        >>> ssa.reconstruct(groups={'main': [0,1]})
        >>> ssa.groups['main']
        [0, 1]
        >>> ssa['main'][:5]
        array([-0.99356886, -1.00619557, -1.02415474, -1.04545692, -1.06560407])

        See Also
        --------

        SingularSpectrumAnalysis.reconstruct
            Method used for reconstructing components based on eigentriple
            indices.

        """

        all_names = list(self._default_groups.keys())
        all_indices = [None, None]

        if self.n_components is not None:
            all_indices += [None]

        if self._user_groups is not None:
            # Get user defined group names
            user_groups_names = list(self._user_groups.keys())
            all_names += user_groups_names

            # Get user defined grouped indexes of singular values
            user_groups_indices = list(self._user_groups.values())
            residuals_indices = list(self._residuals_indices)
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
    def svd_matrix(self) -> np.ndarray:
        """Matrix to be decomposed with SVD
        """
        svd_matrix = construct_SVD_matrix(
            self._timeseries_pp,
            window=self._w,
            kind=self._svd_matrix_kind
        )
        return svd_matrix

    @property
    def _covariance_matrix(self):
        """Return Broomhead & King or Vautard & Ghil covariance matrix.
        """
        if self._svd_matrix_kind == 'BK':
            covariance_matrix = construct_BK_covariance_matrix(
                timeseries=self._timeseries_pp,
                window=self._w
            )
        elif self._svd_matrix_kind == 'VG':
            covariance_matrix = construct_VG_covariance_matrix(
                timeseries=self._timeseries_pp,
                window=self._w
            )
        return covariance_matrix

    @property
    def _trajectory_matrix(self):
        """Return Broomhead & King trajectory matrix.
        """
        trajectory_matrix = construct_BK_trajectory_matrix(
            timeseries=self._timeseries_pp,
            window=self._w
        )
        return trajectory_matrix

    @property
    def _ssa_reconstructed(self) -> np.ndarray:
        """Return the reconstructed timeseries signal.
        """
        if self.s_ is None:
            ssa_reconstructed = None
        else:
            full_range = range(self.n_components)
            ssa_reconstructed = self._reconstruct_group_timeseries(full_range)
        return ssa_reconstructed

    @property
    def _ssa_residuals(self) -> np.ndarray:
        """Return the residual timeseries signal.
        """
        if self.s_ is None:
            ssa_residuals = None
        else:
            residuals_indices = list(self._residuals_indices)
            ssa_residuals = self._reconstruct_group_timeseries(
                residuals_indices)
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

    @property
    def _residuals_indices(self) -> set:
        """Return the set of residuals indices.
        """
        user_indices: set = self._user_indices
        residuals_indices = {i for i in range(self.n_components)
                             if i not in user_indices}
        return residuals_indices

    def wcorr(
            self,
            n_components: int
    ) -> NDArray:
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

        See Also
        --------
        vassal.math_ext.weighted_correlation_matrix
            For examples and references.

        """
        timeseries = np.array(
            [self._reconstruct_group_timeseries([i]) for i in
             range(n_components)])
        weights = correlation_weights(self._n, self._w)
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

        if self._svd_matrix_kind == 'BK':
            reconstructed_group_matrix = u_selected @ s_selected @ v_selected
        elif self._svd_matrix_kind == 'VG':
            X = self._trajectory_matrix
            S = X.T @ u_selected
            reconstructed_group_matrix = u_selected @ S.T

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
