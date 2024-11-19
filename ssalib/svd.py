"""Singular Value Decomposition (SVD) Solver Handler for SSA computations."""
from __future__ import annotations

import logging
from enum import Enum
from inspect import signature
from typing import Any
from functools import lru_cache

import numpy as np
import scipy
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd

logger = logging.getLogger(__name__)

class SVDSolverType(Enum):
    """Available SVD solver types.

    Enumeration of supported SVD solver implementations, mapping user-facing
    names to their corresponding method identifiers.
    """
    NUMPY_STANDARD = "numpy_standard"
    SCIPY_STANDARD = "scipy_standard"
    SCIPY_SPARSE = "scipy_sparse"
    SKLEARN_RANDOMIZED = "sklearn_randomized"

    @property
    def supports_n_components(self) -> bool:
        """Whether this solver type supports the n_components parameter."""
        return self in {
            self.SCIPY_SPARSE,
            self.SKLEARN_RANDOMIZED
        }

    @classmethod
    def available_solvers(cls) -> list[str]:
        return [solver.value for solver in cls]


# Custom type for SVD results
SVDDecomposition = tuple[NDArray[float], NDArray[float], NDArray[float]]


class SVDHandler:
    """Singular Value Decomposition (SVD) Solver Handler for SSA.

    This handler provides a unified interface to various SVD implementations,
    specifically tailored for Singular Spectrum Analysis computations.

    Parameters
    ----------
    svd_solver : SVDSolverType | str, default SVDSolverType.NUMPY_STANDARD
        Type of SVD solver to use. Can be specified either as SVDSolverType enum
        or string matching enum values.

    Attributes
    ----------
    available_solvers : list[str]
        List of available solver names.
    decomposition_results : SVDDecomposition | None
        Tuple of singular values (s) and left and right eigenvectors (u, vt)
        returned as (u, s, vt). None if svd() has not been called yet.


    Examples
    --------
    By default, SVDHandler is used to solve the singular value decomposition
    relying on the 'numpy_standard' solver.

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

    >>> svdh = SVDHandler(svd_solver='sklearn_randomized') # randomized svd
    >>> u, s, vt = svdh.svd(A, n_components=1) # decomposition
    >>> u @ np.diag(s) @ vt # reconstruction
    array([[1.57454629, 2.08011388, 2.58568148],
           [3.75936076, 4.96644562, 6.17353048]])

    """
    available_solvers: list[str] = SVDSolverType.available_solvers()
    _SOLVER_METHOD_MAPPING = {
        SVDSolverType.NUMPY_STANDARD: "_svd_numpy_standard",
        SVDSolverType.SCIPY_STANDARD: "_svd_scipy_standard",
        SVDSolverType.SCIPY_SPARSE: "_svd_scipy_sparse",
        SVDSolverType.SKLEARN_RANDOMIZED: "_svd_sklearn_randomized"
    }

    def __init__(
            self,
            svd_solver: SVDSolverType | str = SVDSolverType.NUMPY_STANDARD
    ) -> None:
        if isinstance(svd_solver, str):
            try:
                self._svd_solver = SVDSolverType(svd_solver)
            except ValueError:
                valid_solvers = SVDSolverType.available_solvers()
                raise ValueError(
                    f"Invalid svd_solver '{svd_solver}'. "
                    f"Valid solvers are: {', '.join(valid_solvers)}"
                )
        else:
            self._svd_solver = svd_solver

        self.decomposition_results: SVDDecomposition | None = None

    def __repr__(self):
        return f"SVDHandler(svd_solver={self.svd_solver})"

    def __str__(self):
        return self.__repr__()

    def __validate_solver_inputs(self, matrix, n_components, **kwargs) -> None:
        """Validate input parameters for SVD decomposition."""
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input matrix must be a numpy array")
        if matrix.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional")
        if not np.isfinite(matrix).all():
            raise ValueError("Input contains non-finite values")

        if n_components is not None:
            if not isinstance(n_components, int):
                raise TypeError("Argument n_components must be an integer")
            if n_components <= 0:
                raise ValueError("Argument n_components must be positive")
            if n_components > min(matrix.shape):
                raise ValueError(
                    f"Argument n_components cannot be larger than "
                    f"min(matrix.shape)={min(matrix.shape)}"
                )

        if self._svd_solver.supports_n_components:
            if n_components is None:
                raise ValueError(
                    f"Solver '{self._svd_solver.value}' requires "
                    f"n_components to be specified"
                )
        elif n_components is not None:
            logger.warning(
                f"Parameter n_components is not supported by "
                f"{self._svd_solver.value} solver and is ignored"
            )

        if n_components is not None and self._svd_solver.supports_n_components:
            kwargs['n_components'] = n_components
        kwargs['matrix'] = matrix

        solver_method_name = self._SOLVER_METHOD_MAPPING[self._svd_solver]
        solver_method = getattr(self, solver_method_name)
        sig = signature(solver_method)
        try:
            sig.bind(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Invalid arguments for svd decomposition with "
                f"solver '{self._svd_solver}': {str(e)}"
            )

    def svd(
            self,
            matrix: NDArray[float],
            n_components: int | None = None,
            **solver_kwargs: Any
    ) -> SVDDecomposition:
        """Perform Singular Value Decomposition (SVD)

        Parameters
        ----------
        matrix : NDArray[float]
            Two-dimensional matrix to be decomposed.
        n_components : int | None, default None
            Number of singular values and vectors to extract. Only used for
            truncated svd computation, e.g., scipy sparse svd, or sklearn
            randomized svd. Default is None.

        Other Parameters
        ----------------
        **solver_kwargs : Any
            Extra parameters to pass to the svd solver. See the specific
            solver documentation for details.

        Returns
        -------
        u, s, vt : SVDDecomposition
            Eigenvectors and singular values.
        """
        self.__validate_solver_inputs(matrix, n_components, **solver_kwargs)
        if n_components is not None and self._svd_solver.supports_n_components:
            solver_kwargs['n_components'] = n_components

        svd_method_name = self._SOLVER_METHOD_MAPPING[self._svd_solver]
        svd_method = getattr(self, svd_method_name)
        u, s, vt = svd_method(matrix, **solver_kwargs)
        self.decomposition_results: SVDDecomposition = u, s, vt
        return self.decomposition_results

    @staticmethod
    def _svd_numpy_standard(
            matrix: NDArray[float],
            full_matrices: bool = True,
            compute_uv: bool = True,
            hermitian: bool | None = None,
            **kwargs: Any
    ) -> SVDDecomposition:
        """numpy svd wrapper."""
        if hermitian is None and matrix.shape[0] == matrix.shape[1]:
            # should be True for bk_covariance and vg_covariance
            hermitian = scipy.linalg.ishermitian(matrix)
        u, s, vt = np.linalg.svd(
            matrix,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            hermitian=hermitian,
            **kwargs
        )

        return u, s, vt

    @staticmethod
    def _svd_scipy_standard(
            matrix: NDArray[float],
            check_finite: bool = False,  # disabled, already checked
            compute_uv: bool = True,
            lapack_driver: str = 'gesdd',
            **kwargs: Any
    ) -> SVDDecomposition:
        """scipy svd wrapper."""
        u, s, vt = scipy.linalg.svd(
            matrix,
            check_finite=check_finite,
            compute_uv=compute_uv,
            lapack_driver=lapack_driver,
            **kwargs
        )
        return u, s, vt

    @staticmethod
    def _svd_scipy_sparse(
            matrix: NDArray[float],
            n_components: int,
            return_singular_vectors: bool = True,
            **kwargs: Any
    ) -> SVDDecomposition:
        """scipy sparse svd wrapper."""
        u, s, vt = scipy.sparse.linalg.svds(
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
    def _svd_sklearn_randomized(
            matrix: NDArray[float],
            n_components: int,
            **kwargs: Any
    ) -> SVDDecomposition:
        """sklearn randomized svd wrapper."""
        u, s, vt = randomized_svd(
            matrix,
            n_components,
            **kwargs
        )

        return u, s, vt

    @property
    def svd_solver(self) -> 'str':
        return self._svd_solver.value

    @property
    def s_(self) -> NDArray[float] | None:
        """Singular values of SVD."""
        if self.decomposition_results is not None:
            return self.decomposition_results[1]
        return None

    @property
    def u_(self) -> NDArray[float] | None:
        """Left eigenvectors of SVD."""
        if self.decomposition_results is not None:
            return self.decomposition_results[0]
        return None

    @property
    def vt_(self) -> NDArray[float] | None:
        """Right eigenvectors of SVD."""
        if self.decomposition_results is not None:
            return self.decomposition_results[2]
        return None

    @property
    def n_components(self) -> int | None:
        """Returns the number of singular values."""
        return len(self.s_) if self.s_ is not None else None

    @property
    def eigenvalues(self) -> NDArray[float] | None:
        """Eigenvalues of SVD."""
        return self.s_ ** 2 if self.s_ is not None else None


if __name__ == '__main__':
    import doctest

    doctest.testmod()
