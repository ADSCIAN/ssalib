import inspect
import logging

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from vassal.plotting import PlotSSA
from vassal.svd import SVDHandler, SVDTuple
from vassal.log_and_error import DecompositionError, ReconstructionError


class SingularSpectrumAnalysis(SVDHandler, PlotSSA):

    def __init__(
            self,
            timeseries: ArrayLike,
            window: int | None = None,
            svd_method: str = 'np_svd',
            standardize: bool = True
    ) -> None:
        """
        Initialize the SingularSpectrumAnalysis with the given timeseries data.

        Parameters
        ----------
        timeseries : ArrayLike
            The timeseries data as a one-dimensional, numeric array-like
            structure.
        window : int, optional
            The window length for the SSA algorithm. Defaults to half the series
            length if not provided.
        svd_method : str, default 'np_svd'
            The method of singular value decomposition to use.
        standardize : bool, default True
            Whether to standardize the timeseries by removing the mean and
            scaling to unit variance.

        Raises
        ------
        ValueError
            If the timeseries contains non-numeric data, NaNs, infinite values,
            or is not one-dimensional.

        Notes
        -----


        References
        ----------
        .. [1] Broomhead, D. S., & King, G. P. (1986). Extracting qualitative
            dynamics from experimental data. Physica D: Nonlinear Phenomena,
            20(2), 217–236. https://doi.org/10.1016/0167-2789(86)90031-X
        .. [2] Golyandina, N., & Zhigljavsky, A. (2020). Singular Spectrum
            Analysis for Time Series. Berlin, Heidelberg: Springer.
            https://doi.org/10.1007/978-3-662-62436-4
        .. [3] Vautard, R., Yiou, P., & Ghil, M. (1992). Singular-spectrum
            analysis: A toolkit for short, noisy chaotic signals. Physica D:
            Nonlinear Phenomena, 58(1), 95–126.
            https://doi.org/10.1016/0167-2789(92)90103-T

        """
        SVDHandler.__init__(self, svd_method)

        # Initialize timeseries

        if hasattr(timeseries, 'index'):
            self._ix = timeseries.index
        else:
            self._ix = None
        timeseries = self.__validate_timeseries(timeseries)
        self.timeseries = np.squeeze(np.array(timeseries))
        self._n = self.timeseries.shape[0]
        self._mean = np.mean(self.timeseries)
        self._std = np.std(self.timeseries)
        if standardize:
            self._timeseries_pp = (self.timeseries - self._mean) / self._std
        else:
            self._timeseries_pp = self.timeseries
        self._standardized = standardize

        # Initialize decomposition
        self.__validate_window(window)
        self._w = window if window is not None else self._n // 2
        self._k = self._n - self._w + 1

        # Initialize groups
        self._default_groups = {
            'ssa_original': 'timeseries',
            'ssa_preprocessed': '_timeseries_pp',
            'ssa_reconstructed': '_ssa_reconstructed',
            'ssa_residuals': '_ssa_residuals'
        }
        self._user_groups = None

    def __getitem__(
            self,
            item: int | slice | list[int] | str
    ) -> np.ndarray:
        if isinstance(item, str):
            if item in self._default_groups.keys():
                default_attr = self._default_groups[item]
                returned_values = getattr(self, default_attr)
            else:
                returned_values = self.__get_user_timeseries_by_group_name(item)
        else:  # TODO: document slicing int list of int behavior
            returned_values = self._reconstruct_group(item)
        return returned_values

    @staticmethod
    def __validate_timeseries(
            timeseries: ArrayLike
    ) -> np.ndarray:
        """Validates the timeseries data."""
        timeseries = np.squeeze(np.array(timeseries))
        if not np.issubdtype(timeseries.dtype, np.number):
            raise ValueError("All elements must be integers or floats.")
        if np.any(np.isinf(timeseries)) or np.any(np.isnan(timeseries)):
            raise ValueError("The array contains inf or NaN values.")
        if timeseries.ndim != 1:
            raise ValueError("Timeseries must be one-dimensional.")
        return timeseries

    def __validate_window(
            self,
            window: int | None
    ) -> None:
        if window is None:
            pass
        elif not isinstance(window, int):
            raise ValueError("Invalid window type. Parameter 'window' must be "
                             "an integer.")
        elif (window < 2) or (window > self._n - 2):
            raise ValueError(
                f'Invalid window size. Current size is {window}, but it must be'
                f' between 2 and {self._n - 2} (n-2). Recommended window size '
                f'is between 2 and {self._n // 2} (n/2).')

    def __validate_user_groups(
            self,
            groups: dict[str, int | list[int]]
    ) -> None:
        """
        Validate the user_groups dictionary.

        Parameters
        ----------
        groups : dict[str, int | list[int]]
            A dictionary where keys are strings and values are either int or
            list of int.

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
            raise ValueError("user_groups must be a dictionary.")

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

            # Validate value and collect all integers
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
        if self._user_groups is None:
            raise ReconstructionError("Cannot retrieve user indices without "
                                      "defined user groups. Define user groups "
                                      "using the 'reconstruct' method first.")
        if group_name in self._user_groups.keys():
            return self._reconstruct_group(self.groups[group_name])
        raise IndexError(f"Unknown group name '{group_name}'.")

    def decompose(
            self,
            n_components: int | None = None,
            **kwargs
    ) -> SVDTuple:
        method_name = self._method_map[self.svd_method]
        method = getattr(self, method_name)
        signature = inspect.signature(method)
        is_truncated = 'n_components' in signature.parameters
        if n_components is None and is_truncated:
            raise ValueError(f"The selected method '{self.svd_method}' requires"
                             f" 'n_components' to be specified within the "
                             f"'decompose' method. Please provide a value for "
                             f"'n_components'.")
        return self._svd(self.trajectory, n_components, **kwargs)

    def reconstruct(
            self,
            groups: dict[str, int | list[int]]
    ):
        """
        Reconstruct components based on eigen-triples indices.

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
        None

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

    def to_frame(
            self,
            include: list[str] | None = None,
            exclude: list[str] | None = None,
            recenter: bool = False,
            rescale: bool = False
    ) -> pd.DataFrame:
        """
        Return `pandas.DataFrame` with all signals unless specified otherwise
        with the include or exclude parameters. If the
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
            standard deviation.
        recenter : bool, default=False
            If True, recenter the signal relative to the original signal's mean

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

        if include is not None:
            if any(name not in group_names for name in
                   include):  # TODO write test
                raise ValueError(f"Invalid group names in 'include'. "
                                 f"Valid group are names "
                                 f"{', '.join(self.groups.keys())}")
            group_names = [name for name in group_names if name in include]

        if exclude is not None:
            if any(name not in group_names for name in
                   exclude):  # TODO write test
                logging.warning(f"Ignored unknown group names in 'exclude'. "
                                f"Valid group are names "
                                f"{', '.join(self.groups.keys())}")
            group_names = [name for name in group_names if name not in exclude]

        signals = pd.DataFrame(
            {name: self.__getitem__(name) for name in group_names})

        if self._ix is not None:
            signals.set_index(self._ix, inplace=True)

        if rescale:
            signals.apply(self._rescale)

        if recenter:
            signals.apply(self._recenter)

        return signals

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._s ** 2

    @property
    def groups(self):
        """
        Return the available group names and their corresponding eigentriple
        indices.

        Any group name registered in `groups` is a key to get the values of
        the corresponding signal using a `__getitem__` method.

        If no grouping was done using `reconstruct` method, group names
        are limited to groups defined by default:
        * 'ssa_original': the original signal passed to instantiate
        the `SingularSpectrumAnalysis` object. This group is not related to
        eigentriple indices.
        * 'ssa_preprocessed': the preprocessed signal after the instantiation
        of the `SingularSpectrumAnalysis` object. This group is not related to
        eigentriple indices.
        * 'ssa_reconstructed': the signal reconstructed from all available
        eigentriple indices. The reconstructed signal may differ from the
        original one if the singular value decomposition is truncated.

        Reconstructing the signal makes new user groups available based on their
        user-defined group names. It also adds another default group:

        * 'ssa_residuals': the residual signal key and its corresponding
        eigentriple indices

        Examples
        --------

        >>> import numpy as np
        >>> np.random.seed(42) # Enable reproducible example #TODO random doctest?
        >>> timeseries = np.random.randn(10)
        >>> ssa = SingularSpectrumAnalysis(timeseries) #TODO replace with a datasets
        >>> u, s, v = ssa.decompose()
        >>> ssa.reconstruct(groups={'main': [0,1]})
        >>> ssa.groups['main']
        [0, 1]
        >>> ssa.groups['ssa_residuals']
        [2, 3, 4]
        >>> ssa['main']
        array([ 0.19704039, -1.23717118,  0.32435365,  1.35998372, -0.95202609,
               -1.01908099,  1.50451796,  0.43365561, -1.60253083,  0.22130549])

        See Also
        --------
        SingularSpectrumAnalysis.reconstruct : Method used for reconstructing
        components based on eigen-triples indices.

        """

        n_components = self.n_components
        all_names = list(self._default_groups.keys())
        all_indices = [None, None,
                       range(n_components) if n_components else None]

        if self._user_groups:
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
    def total_variance(self) -> np.ndarray:
        # TODO Error handling + pytest
        np.linalg.norm(self.trajectory, 'fro') ** 2
        return np.sum(self.eigenvalues)

    @property
    def trajectory(self) -> np.ndarray:
        k = self._k
        w = self._w
        x = np.zeros(shape=(w, k))
        ts = self._timeseries_pp
        for i in range(k):
            x[:, i] = ts[i:i + w]
        return x

    @property
    def _ssa_reconstructed(self) -> np.ndarray:
        if self._s is None:
            ssa_reconstructed = None
        else:
            full_range = range(self.n_components)
            ssa_reconstructed = self._reconstruct_group(full_range)
        return ssa_reconstructed

    @property
    def _ssa_residuals(self) -> np.ndarray:
        if self._s is None:
            ssa_residuals = None
        else:
            residuals_indices = list(self._residuals_indices)
            ssa_residuals = self._reconstruct_group(residuals_indices)
        return ssa_residuals

    @property
    def _user_indices(self) -> set:
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
        user_indices: set = self._user_indices
        residuals_indices = {i for i in range(self.n_components)
                             if i not in user_indices}
        return residuals_indices

    def _recenter(
            self,
            timeseries: np.ndarray
    ) -> np.ndarray:
        if not np.isclose(timeseries.mean(), self._mean):
            timeseries += self._mean
        return timeseries

    def _rescale(
            self,
            timeseries: pd.Series
    ) -> pd.Series:
        if self._standardized and timeseries.name != 'ssa_original':
            timeseries *= self._std
        return timeseries

    @staticmethod
    def _antidiagonal_averaging(reconstructed_matrix: np.ndarray) -> np.ndarray:
        """Average the anti-diagonal of Hankel matrix to return 1d time series

        Parameters
        ----------
        reconstructed_matrix : np.ndarray

        Returns
        -------

        timeseries: np.ndarray

        """
        reconstructed_timeseries = [
            np.mean(reconstructed_matrix[::-1, :].diagonal(i)) for i in
            range(-reconstructed_matrix.shape[0] + 1,
                  reconstructed_matrix.shape[1])]

        return np.array(reconstructed_timeseries)

    def _reconstruct_group(
            self,
            group_indices: int | slice | range | list[int]
    ) -> np.ndarray:
        """
        Reconstructs a group of components from the trajectory matrix using the
        given indices.

        Parameters
        ----------
        group_indices : int | slice | range | list of int
            Eigentriple indices used to group and reconstruct the time series.
            Time series can be reconstructed using a slice `group_indices`.


        Returns
        -------
        np.ndarray
            The reconstructed time series.
        """
        u, s, v = self._u, self._s, self._v

        if isinstance(group_indices, int):
            group_indices = [group_indices]
        if isinstance(group_indices, slice):
            start, stop, step = group_indices.indices(self.n_components)
            group_indices = list(range(start, stop, step))

        u_selected = u[:, group_indices]
        s_selected = np.diag(s[group_indices])
        v_selected = v[group_indices, :]

        reconstructed_group = u_selected @ s_selected @ v_selected

        # Anti-diagonal averaging
        reconstructed_timeseries = self._antidiagonal_averaging(
            reconstructed_group)
        return reconstructed_timeseries


if __name__ == '__main__':
    from doctest import testmod

    testmod(verbose=True)
