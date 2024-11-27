from __future__ import annotations

import logging

import numpy as np
import pytest

from ssalib.svd import SVDHandler, SVDSolverType


@pytest.mark.parametrize("svd_solver", SVDHandler.available_solvers)
def test_svd_methods(sample_matrix: np.ndarray, n_components: int,
                     svd_solver: str, full_svd_solvers: list[str]) -> None:
    handler = SVDHandler(svd_solver=svd_solver)
    nc = None if svd_solver in full_svd_solvers else n_components
    u, s, v = handler.svd(sample_matrix, n_components=nc)
    assert handler.svd_solver == svd_solver
    assert isinstance(u, np.ndarray)
    assert isinstance(s, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert np.all(
        np.diff(s) <= 0), "Singular values are not in decreasing order"


@pytest.mark.parametrize("svd_solver", SVDHandler.available_solvers)
def test_repr_str(svd_solver):
    handler = SVDHandler(svd_solver=svd_solver)
    handler.__repr__()
    handler.__str__()


@pytest.mark.parametrize("svd_solver", SVDHandler.available_solvers)
def test_solver_svd_matrix_type_error(svd_solver):
    handler = SVDHandler(svd_solver=svd_solver)
    with pytest.raises(TypeError):
        handler.svd(matrix="not a matrix")

@pytest.mark.parametrize("svd_solver", SVDHandler.available_solvers)
def test_solver_svd_matrix_value_error(svd_solver):
    handler = SVDHandler(svd_solver=svd_solver)
    with pytest.raises(ValueError):
        handler.svd(matrix=np.random.random((5,5,2)))
    with pytest.raises(ValueError):
        a = np.random.random((5,5))
        a[0,0] = np.nan
        handler.svd(matrix=a)

@pytest.mark.parametrize("svd_solver", SVDHandler.available_solvers)
def test_solver_n_components(svd_solver, sample_matrix):
    handler = SVDHandler(svd_solver=svd_solver)
    solver = SVDSolverType(svd_solver)
    if solver.supports_n_components:
        with pytest.raises(TypeError):
            handler.svd(
                matrix=sample_matrix,
                n_components="not an integer"
            )
        with pytest.raises(ValueError):
            handler.svd(
                matrix=sample_matrix,
                n_components=min(sample_matrix.shape) + 1
            )
        with pytest.raises(ValueError):
            handler.svd(sample_matrix, n_components=-1)

@pytest.mark.parametrize("svd_solver", SVDHandler.available_solvers)
def test_solver_kwargs(svd_solver, sample_matrix):
    handler = SVDHandler(svd_solver=svd_solver)
    with pytest.raises(TypeError):
        handler.svd(sample_matrix=sample_matrix, not_a_kwarg=1)

@pytest.mark.parametrize("svd_solver", SVDHandler.available_solvers)
def test_ignored_components_warning(sample_matrix, n_components, svd_solver,
                                    full_svd_solvers, caplog) -> None:
    handler = SVDHandler(svd_solver=svd_solver)
    with caplog.at_level(logging.WARNING):
        if svd_solver in full_svd_solvers:
            handler.svd(sample_matrix, n_components=n_components)
            assert "Parameter n_components is not supported" in caplog.text
        else:
            handler.svd(sample_matrix, n_components=n_components)
            assert len(caplog.records) == 0


def test_invalid_method():
    with pytest.raises(ValueError, match=r"Invalid svd_solver '\w+'\. "
                                         r"Valid solvers are:.*"):
        SVDHandler(svd_solver='invalid_method')


@pytest.mark.parametrize("svd_solver", SVDHandler.available_solvers)
def test_properties(sample_matrix: np.ndarray, svd_solver: str,
                    n_components: int, full_svd_solvers: list[str]):
    """Test all properties of SVDHandler across different solvers."""
    handler = SVDHandler(svd_solver=svd_solver)

    # Initial state checks
    assert handler.decomposition_results is None
    assert handler.s_ is None
    assert handler.u_ is None
    assert handler.vt_ is None
    assert handler.n_components is None
    assert handler.eigenvalues is None
    assert handler.svd_solver == svd_solver

    # Perform SVD
    nc = None if svd_solver in full_svd_solvers else n_components
    u, s, vt = handler.svd(sample_matrix, n_components=nc)

    # Test decomposition results storage
    assert handler.decomposition_results is not None
    assert isinstance(handler.decomposition_results, tuple)
    assert len(handler.decomposition_results) == 3

    # Test individual components
    np.testing.assert_array_equal(handler.u_, u)
    np.testing.assert_array_equal(handler.s_, s)
    np.testing.assert_array_equal(handler.vt_, vt)

    # Test derived properties
    assert handler.n_components == len(s)
    np.testing.assert_array_equal(handler.eigenvalues, s ** 2)

    # Test mathematical properties
    assert np.all(
        np.diff(s) <= 0), "Singular values are not in decreasing order"
    assert s.ndim == 1, "Singular values should be 1-dimensional"
    assert u.ndim == 2, "Left singular vectors should be 2-dimensional"
    assert vt.ndim == 2, "Right singular vectors should be 2-dimensional"

    # Test orthogonality of singular vectors
    np.testing.assert_array_almost_equal(u.T @ u, np.eye(u.shape[1]))
    np.testing.assert_array_almost_equal(vt @ vt.T, np.eye(vt.shape[0]))

    # Test reconstruction accuracy
    if nc is None:
        # For full SVD, test exact reconstruction
        reconstructed = u @ np.diag(s) @ vt
        np.testing.assert_array_almost_equal(reconstructed, sample_matrix)
    else:
        # For truncated SVD, test that reconstruction error decreases with more
        # components
        reconstruction_error = lambda k: np.linalg.norm(
            sample_matrix - u[:, :k] @ np.diag(s[:k]) @ vt[:k, :], 'fro'
        )
        errors = [reconstruction_error(k) for k in range(1, len(s) + 1)]
        assert np.all(np.diff(errors) <= 1e-10), \
            "Reconstruction error should decrease with more components"
