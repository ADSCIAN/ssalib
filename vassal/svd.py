import numpy as np
from vassal.log_and_error import ignored_argument_warning
SVDTuple = tuple[np.ndarray, np.ndarray, np.ndarray]

class SVDHandler:
    _method_map = {
        'np_svd': "_numpy_svd",
        'sc_svd': "_scipy_svd",
        'sc_ssvd': "_scipy_sparse_svd",
        'sk_rsvd': "_sklearn_randomized_svd",
        'da_svd': "_dask_svd",
        'da_csvd': "_dask_compressed_svd"
    }

    def __init__(
            self,
            svd_method: str
    ):
        self.svd_method = svd_method
        self._u = None
        self._s = None
        self._v = None
        if svd_method not in self._method_map:
            raise ValueError("Invalid parameter 'svd_method'. Valid methods are"
                             ": " + ", ".join(self._method_map.keys()))

    def _svd(
            self,
            matrix: np.ndarray,
            n_components: int | None,
            **kwargs
    ) -> SVDTuple:
        method_name = self._method_map[self.svd_method]
        method = getattr(self, method_name)
        kwargs['n_components'] = n_components
        u, s, v = method(matrix, **kwargs)
        self._u, self._s, self._v = u, s, v
        return u, s, v

    @staticmethod
    @ignored_argument_warning('n_components')
    def _numpy_svd(
            matrix: np.ndarray,
            **kwargs
    ) -> SVDTuple:
        u, s, v = np.linalg.svd(matrix, full_matrices=True, compute_uv=True,
                                hermitian=False)
        return u, s, v

    @staticmethod
    @ignored_argument_warning('n_components')
    def _scipy_svd(
            matrix: np.ndarray,
            lapack_driver: str = 'gesdd',
            **kwargs
    ) -> SVDTuple:
        try:
            from scipy.linalg import svd as sc_svd
        except ImportError as e:
            raise ImportError(
                "Cannot import svd from scipy.linalg. "
                "Ensure that SciPy is installed.") from e
        u, s, v = sc_svd(matrix, check_finite=False, compute_uv=True,
                         lapack_driver=lapack_driver)
        return u, s, v

    @staticmethod
    def _scipy_sparse_svd(
            matrix: np.ndarray,
            n_components: int,
            **kwargs
    ) -> SVDTuple:
        try:
            from scipy.sparse.linalg import svds
        except ImportError as e:
            raise ImportError(
                "Cannot import svds from scipy.sparse.linalg. "
                "Ensure that SciPy is installed.") from e
        u, s, v = svds(matrix, k=n_components, return_singular_vectors=True,
                       **kwargs)
        # Sort the singular values and reorder the singular vectors
        sorted_indices = np.argsort(s)[::-1]
        s_sorted = s[sorted_indices]
        u_sorted = u[:, sorted_indices]
        v_sorted = v[sorted_indices, :]
        return u_sorted, s_sorted, v_sorted

    @staticmethod
    def _sklearn_randomized_svd(
            matrix: np.ndarray,
            n_components: int,
            **kwargs
    ) -> SVDTuple:
        try:
            from sklearn.utils.extmath import randomized_svd
        except ImportError as e:
            raise ImportError(
                "Cannot import randomized_svd from sklearn. "
                "Ensure that scikit-learn is installed.") from e
        u, s, v = randomized_svd(matrix, n_components, **kwargs)
        return u, s, v

    @staticmethod
    @ignored_argument_warning('n_components')
    def _dask_svd(
            matrix: np.ndarray,
            coerce_signs: bool = True,
            **kwargs
    ) -> SVDTuple:
        try:
            import dask.array as da
        except ImportError as e:
            raise ImportError(
                "Optional dependency 'dask' is not installed.") from e  # TODO harmonize
        u, s, v = da.linalg.svd(da.array(matrix), coerce_signs=coerce_signs)
        return np.array(u), np.array(s), np.array(v)

    @staticmethod
    def _dask_compressed_svd(
            matrix: np.ndarray,
            n_components: int,
            **kwargs
    ) -> SVDTuple:
        try:
            import dask.array as da
        except ImportError as e:
            raise ImportError(
                "Optional dependency 'dask' is not installed.") from e
        u, s, v = da.linalg.svd_compressed(da.array(matrix), n_components,
                                           **kwargs)
        return np.array(u), np.array(s), np.array(v)

    @property
    def n_components(self) -> int | None:
        """Returns the number of singular values."""
        if self._s is None:
            return None
        else:
            return len(self._s)

    @classmethod
    def available_methods(cls) -> list[str]:
        return list(cls._method_map.keys())
