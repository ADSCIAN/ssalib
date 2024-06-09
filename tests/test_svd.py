from unittest import mock
import logging
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
def full_svd_methods() -> list[str]:
    return ["np_svd", "sc_svd", "da_svd"]


@pytest.mark.parametrize("svd_method", SVDHandler.available_methods())
def test_svd_methods(sample_matrix: np.ndarray, n_components: int,
                     svd_method: str, full_svd_methods: list[str]) -> None:
    handler = SVDHandler(svd_method=svd_method)
    nc = None if svd_method in full_svd_methods else n_components
    u, s, v = handler._svd(sample_matrix, n_components=nc)
    assert handler.svd_method == svd_method
    assert isinstance(u, np.ndarray)
    assert isinstance(s, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert np.all(
        np.diff(s) <= 0), "Singular values are not in decreasing order"


@pytest.mark.parametrize("svd_method", SVDHandler.available_methods())
def test_ignored_components_warning(sample_matrix, n_components, svd_method,
                                    full_svd_methods, caplog) -> None:
    handler = SVDHandler(svd_method=svd_method)
    if svd_method in full_svd_methods:
        with caplog.at_level(logging.WARNING):
            handler._svd(sample_matrix, n_components=n_components)
        assert len(caplog.records) == 1
        assert "Ignored n_components" in caplog.records[0].message

def test_invalid_method():
    with pytest.raises(ValueError, match="Invalid parameter 'svd_method'"):
        SVDHandler(svd_method='invalid_method')


def test_properties(sample_matrix: np.ndarray):
    handler = SVDHandler(svd_method='np_svd')
    u, s, v = handler._svd(sample_matrix, n_components=None)
    handler._u, handler._s, handler._v = u, s, v
    assert handler.n_components == len(s)
    np.testing.assert_array_equal(handler._u, u)
    np.testing.assert_array_equal(handler._s, s)
    np.testing.assert_array_equal(handler._v, v)
    assert np.all(
        np.diff(s) <= 0), "Singular values are not in decreasing order"


@pytest.mark.parametrize("svd_method,import_path", [
    ("sc_svd", "scipy.linalg.svd"),
    ("sc_ssvd", "scipy.sparse.linalg.svds"),
    ("sk_rsvd", "sklearn.utils.extmath.randomized_svd"),
    ("da_svd", "dask.array.linalg.svd"),
    ("da_csvd", "dask.array.linalg.svd_compressed"),
])
def test_import_errors(monkeypatch, svd_method: str, import_path: str,
                       full_svd_methods: list[str]) -> None:
    with mock.patch(import_path, side_effect=ImportError(
            f"Cannot import from {import_path}")):
        with pytest.raises(ImportError,
                           match=f"Cannot import from {import_path}"):
            handler = SVDHandler(svd_method=svd_method)
            nc = None if svd_method in full_svd_methods else n_components #TODO
            handler._svd(np.random.random((10, 10)), n_components=nc)
