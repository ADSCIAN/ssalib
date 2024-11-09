import numpy as np
import pytest

from vassal.ssa import SingularSpectrumAnalysis


@pytest.fixture
def timeseries50():
    return np.random.rand(50)


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
def ssa_np_svd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_solver='np_svd')
    return ssa


@pytest.fixture
def ssa_sc_svd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_solver='sc_svd')
    return ssa


@pytest.fixture
def ssa_sc_ssvd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_solver='sc_ssvd')
    return ssa


@pytest.fixture
def ssa_sk_rsvd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_solver='sk_rsvd')
    return ssa


@pytest.fixture
def ssa_da_svd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_solver='da_svd')
    return ssa


@pytest.fixture
def ssa_da_csvd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_solver='da_csvd')
    return ssa


@pytest.fixture
def ssa_with_decomposition(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_solver='sk_rsvd')
    ssa.decompose(n_components=10)
    return ssa


@pytest.fixture
def ssa_with_reconstruction(ssa_with_decomposition):
    ssa = ssa_with_decomposition
    ssa.reconstruct(groups={'group1': [0, 1, 2]})
    return ssa
