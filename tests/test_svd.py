import logging
from unittest import mock

import numpy as np
import pytest

from vassal.svd import SVDHandler


@pytest.fixture
def sample_matrix() -> np.ndarray:
    return np.random.random((10, 10))


@pytest.fixture
def n_components() -> int:
    return 5


@pytest.fixture
def full_svd_solvers() -> list[str]:
    return ["np_svd", "sc_svd", "da_svd"]


@pytest.mark.parametrize("svd_solver", SVDHandler.available_solvers())
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


@pytest.mark.parametrize("svd_solver", SVDHandler.available_solvers())
def test_ignored_components_warning(sample_matrix, n_components, svd_solver,
                                    full_svd_solvers, caplog) -> None:
    handler = SVDHandler(svd_solver=svd_solver)
    if svd_solver in full_svd_solvers:
        with caplog.at_level(logging.WARNING):
            handler.svd(sample_matrix, n_components=n_components)
        assert len(caplog.records) == 1
        assert "Ignored n_components" in caplog.records[0].message

def test_invalid_method():
    with pytest.raises(ValueError, match="Invalid parameter 'svd_solver'"):
        SVDHandler(svd_solver='invalid_method')


def test_properties(sample_matrix: np.ndarray):
    handler = SVDHandler(svd_solver='np_svd')
    u, s, v = handler.svd(sample_matrix, n_components=None)
    handler.u_, handler.s_, handler.vt_ = u, s, v
    assert handler.n_components == len(s)
    np.testing.assert_array_equal(handler.u_, u)
    np.testing.assert_array_equal(handler.s_, s)
    np.testing.assert_array_equal(handler.vt_, v)
    assert np.all(
        np.diff(s) <= 0), "Singular values are not in decreasing order"


@pytest.mark.parametrize("svd_solver,import_path", [
    ("sk_rsvd", "sklearn.utils.extmath.randomized_svd"),
    ("da_svd", "dask.array.linalg.svd"),
    ("da_csvd", "dask.array.linalg.svd_compressed"),
])
def test_import_errors(monkeypatch, svd_solver: str, import_path: str,
                       full_svd_solvers: list[str]) -> None:
    with mock.patch(import_path, side_effect=ImportError(
            f"Cannot import from {import_path}")):
        with pytest.raises(ImportError,
                           match=f"Cannot import from {import_path}"):
            handler = SVDHandler(svd_solver=svd_solver)
            nc = None if svd_solver in full_svd_solvers else n_components
            handler.svd(np.random.random((10, 10)), n_components=nc)
