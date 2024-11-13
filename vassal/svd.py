"""Singular Value Decomposition (SVD) Solver Handler"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd as scipy_svd
from scipy.sparse.linalg import svds as scipy_svds

from vassal.log_and_error import ignored_argument_warning

SVDTuple = tuple[NDArray, NDArray, NDArray]


class SVDHandler:
    """Singular Value Decomposition (SVD) Solver Handler

    The SVDHandler class is used to solve the singular value decomposition
    while chosing among various pre-existing svd solvers and algorithms
    implemented in Python scientific packages.

    Parameters
    ----------
    svd_solver : str
        Name of the svd solver to use. Valid names are:
        * 'np_svd': wrapper to the `numpy.linalg.svd` method.
        * 'sc_svd': wrapper to the `scipy.linalg.svd` method.
        * 'sc_svds' : wrapper to the `scipy.sparse.linalg.svds` method.
        * 'sk_rsvd' : wrapper to the `sklearn.utils.extmath.randomized_svd`
          method.
        * 'da_svd' : wrapper to the `dask.array.linalg.svd` method.
        * 'da_csvd' : wrapper to the `dask.array.linalg.svd_compressed` method.

    Attributes
    ----------
    eigentriples : SVDTuple
        Tuple of singular values (s) and left and right eigenvectors (u, vt)
        returned as (u, s, vt).
    eigenvalues : NDArray
        Squared singular values.
    n_components : int
        Number of components defined by the number singular values.
    svd_solver: str
        The name of the user-selected SVD solver.
    s_ : NDArray | None
        1D array of singular values or None prior to the call of `svd` method.
    u_ : NDArray | None
        2D array of right eigenvectors or None prior to the call of `svd`
        method.
    vt_ : NDArray | None
        2D array of left eigenvectors or None prior to the call of `svd`
        method.

    Examples
    --------

    By default, SVDHandler is used to solve the singular value decomposition
    relying on the 'np_svd' solver.

    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> svdh = SVDHandler()
    >>> u, s, vt = svdh.svd(A) # decomposition
    >>> u @ np.diag(s) @ vt[:len(s)] # reconstruction
    array([[1., 2., 3.],
           [4., 5., 6.]])

    Users may pass additional keyword arguments based on the underlying svd
    methods, in this case, `np.linalg.svd`.

    >>> u, s, vt = svdh.svd(A, full_matrices=False) # decomposition
    >>> u @ np.diag(s) @ vt # reconstruction
    array([[1., 2., 3.],
           [4., 5., 6.]])

    For truncated svd algorithms, the `svd` method accept a 'n_components'
    parameter.

    >>> svdh = SVDHandler(svd_solver='sk_rsvd') # randomized svd
    >>> u, s, vt = svdh.svd(A, n_components=1) # decomposition
    >>> u @ np.diag(s) @ vt # reconstruction
    array([[1.57454629, 2.08011388, 2.58568148],
           [3.75936076, 4.96644562, 6.17353048]])

    """
    _solver_map = {
        'np_svd': "_numpy_svd",
        'sc_svd': "_scipy_svd",
        'sc_ssvd': "_scipy_sparse_svd",
        'sk_rsvd': "_sklearn_randomized_svd",
        'da_svd': "_dask_svd",
        'da_csvd': "_dask_compressed_svd"
    }

    def __init__(
            self,
            svd_solver: str = 'np_svd'
    ):
        self.svd_solver = svd_solver
        self.u_ = None
        self.s_ = None
        self.vt_ = None
        if svd_solver not in self._solver_map:
            raise ValueError("Invalid parameter 'svd_solver'. Valid methods are"
                             ": " + ", ".join(self._solver_map.keys()))

    def __repr__(self):
        return f"SVDHandler(svd_solver={self.svd_solver})"

    def __str__(self):
        return self.__repr__()

    def svd(
            self,
            matrix: NDArray,
            n_components: int | None = None,
            **kwargs
    ) -> SVDTuple:
        """Perform Singular Value Decomposition (SVD)

        Parameters
        ----------
        matrix : NDArray
            Matrix to be decomposed.
        n_components : int | None
            Number of singular values and vectors to extract. Only used for
            truncated svd computation, e.g., scipy sparse svd, sklearn
            randomized svd, or dask compressed svd. Default is None.

        Returns
        -------
        u, s, vt : SVDTuple
            Eigenvectors and singular values.

        See Also
        --------
        SVDHandler
            The `SVDHandler` class is designed to handle SVD decomposition with
            the `svd` method as a wrapper to pre-existing SVD implementations in
            Python scientific libraries, depending on the `svd_solver`
            parameter.

        """
        method_name = self._solver_map[self.svd_solver]
        method = getattr(self, method_name)
        kwargs['n_components'] = n_components
        u, s, v = method(matrix, **kwargs)
        self.u_, self.s_, self.vt_ = u, s, v
        return u, s, v

    @staticmethod
    @ignored_argument_warning('n_components')
    def _numpy_svd(
            matrix: NDArray,
            full_matrices: bool = True,
            compute_uv: bool = True,
            hermitian: bool = False,
            **kwargs: Any
    ) -> SVDTuple:
        """numpy svd wrapper."""
        u, s, vt = np.linalg.svd(
            matrix,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            hermitian=hermitian,
            **kwargs
        )

        return u, s, vt

    @staticmethod
    @ignored_argument_warning('n_components')
    def _scipy_svd(
            matrix: NDArray,
            check_finite: bool = False,
            compute_uv: bool = True,
            lapack_driver: str = 'gesdd',
            **kwargs: Any
    ) -> SVDTuple:
        """scipy svd wrapper."""
        u, s, vt = scipy_svd(
            matrix,
            check_finite=check_finite,
            compute_uv=compute_uv,
            lapack_driver=lapack_driver,
            **kwargs
        )
        return u, s, vt

    @staticmethod
    def _scipy_sparse_svd(
            matrix: NDArray,
            n_components: int,
            return_singular_vectors: bool = True,
            **kwargs: Any
    ) -> SVDTuple:
        """scipy sparse svd wrapper."""
        u, s, vt = scipy_svds(
            matrix,
            k=n_components,
            return_singular_vectors=return_singular_vectors,
            **kwargs
        )

        # Sort the singular values and reorder the singular vectors
        sorted_indices = np.argsort(s)[::-1]
        s_sorted = s[sorted_indices]
        u_sorted = u[:, sorted_indices]
        vt_sorted = vt[sorted_indices, :]

        return u_sorted, s_sorted, vt_sorted

    @staticmethod
    def _sklearn_randomized_svd(
            matrix: NDArray,
            n_components: int,
            **kwargs: Any
    ) -> SVDTuple:
        """sklearn randomized svd wrapper."""
        try:
            from sklearn.utils.extmath import randomized_svd
        except ImportError as e:
            raise ImportError(
                "Cannot import randomized_svd from sklearn. "
                "Ensure that scikit-learn is installed.") from e

        u, s, vt = randomized_svd(
            matrix,
            n_components,
            **kwargs
        )

        return u, s, vt

    @staticmethod
    @ignored_argument_warning('n_components')
    def _dask_svd(
            matrix: NDArray,
            coerce_signs: bool = True,
            **kwargs: Any
    ) -> SVDTuple:
        """dask svd wrapper."""
        try:
            import dask.array as da
        except ImportError as e:
            raise ImportError(
                "Optional dependency 'dask' is not installed.") from e

        u, s, vt = da.linalg.svd(
            da.array(matrix),
            coerce_signs=coerce_signs,
            **kwargs
        )

        return np.array(u), np.array(s), np.array(vt)

    @staticmethod
    def _dask_compressed_svd(
            matrix: NDArray,
            n_components: int,
            **kwargs: Any
    ) -> SVDTuple:
        """dask compressed svd wrapper."""
        try:
            import dask.array as da
        except ImportError as e:
            raise ImportError(
                "Optional dependency 'dask' is not installed."
            ) from e

        u, s, vt = da.linalg.svd_compressed(
            da.array(matrix),
            n_components,
            **kwargs
        )

        return np.array(u), np.array(s), np.array(vt)

    @property
    def eigentriples(self) -> SVDTuple:
        """Singular values and eigenvectors of SVD."""
        return self.u_, self.s_, self.vt_

    @property
    def eigenvalues(self) -> NDArray | None:
        """Eigenvalues of SVD."""
        return self.s_ ** 2

    @property
    def n_components(self) -> int | None:
        """Returns the number of singular values."""
        if self.s_ is None:
            return None
        else:
            return len(self.s_)

    @classmethod
    def available_solvers(cls) -> list[str]:
        """List of available solvers."""
        return list(cls._solver_map.keys())


if __name__ == '__main__':
    import doctest

    doctest.testmod()
