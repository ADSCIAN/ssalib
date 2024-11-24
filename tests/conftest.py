from __future__ import annotations
import numpy as np
import pytest

from ssalib import MonteCarloSSA
from ssalib.ssa import SingularSpectrumAnalysis
from ssalib.svd import SVDSolverType


@pytest.fixture
def sample_matrix() -> np.ndarray:
    return np.random.random((10, 10))


@pytest.fixture
def n_components() -> int:
    return 5


@pytest.fixture
def full_svd_solvers() -> list[str]:
    return [SVDSolverType.NUMPY_STANDARD.value,
            SVDSolverType.SCIPY_STANDARD.value]


@pytest.fixture
def timeseries50():
    return np.random.rand(50)

@pytest.fixture
def timeseries50_with_na():
    ts = np.random.rand(50)
    na_indices = np.random.choice(50, size=5, replace=False)
    ts[na_indices] = np.nan
    return ts

@pytest.fixture
def ar1_timeseries50():
    np.random.seed(42)
    n = 50
    b = 0.5
    ar1_series = [0]
    for _ in range(1, n):
        ar1_series.append(b * ar1_series[-1] + np.random.normal())
    return np.array(ar1_series)


@pytest.fixture
def ssa_no_decomposition(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50)
    return ssa


@pytest.fixture
def ssa_numpy_standard(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50,
                                   svd_solver=SVDSolverType.NUMPY_STANDARD)
    return ssa


@pytest.fixture
def ssa_scipy_standard(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50,
                                   svd_solver=SVDSolverType.SCIPY_STANDARD)
    return ssa


@pytest.fixture
def ssa_scipy_sparse(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50,
                                   svd_solver=SVDSolverType.SCIPY_SPARSE)
    return ssa


@pytest.fixture
def ssa_sklearn_randomized(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50,
                                   svd_solver=SVDSolverType.SKLEARN_RANDOMIZED)
    return ssa


@pytest.fixture
def ssa_sklearn_randomized_with_decomposition(timeseries50):
    ssa = SingularSpectrumAnalysis(
        timeseries50,
        svd_solver=SVDSolverType.SKLEARN_RANDOMIZED
    ).decompose(
        n_components=10
    )
    return ssa

@pytest.fixture
def ssa_with_decomposition(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50,
                                   svd_solver=SVDSolverType.NUMPY_STANDARD)
    ssa.decompose(n_components=10)
    return ssa

@pytest.fixture
def ssa_with_decomposition_fill_mean(timeseries50_with_na):
    ssa = SingularSpectrumAnalysis(timeseries50_with_na,
                                   na_strategy='fill_mean',
                                   svd_solver=SVDSolverType.SKLEARN_RANDOMIZED)
    ssa.decompose(n_components=10)
    return ssa


@pytest.fixture
def ssa_with_reconstruction(ssa_with_decomposition):
    ssa = ssa_with_decomposition
    ssa.reconstruct(groups={'group1': [0, 1, 2]})
    return ssa