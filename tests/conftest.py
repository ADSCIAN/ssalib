import pytest
import numpy as np
from vassal.ssa import SingularSpectrumAnalysis

@pytest.fixture
def timeseries50():
    return np.random.rand(50)
@pytest.fixture
def ssa_no_decomposition(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50)
    return ssa

@pytest.fixture
def ssa_np_svd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_method='np_svd')
    return ssa

@pytest.fixture
def ssa_sc_svd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_method='sc_svd')
    return ssa

@pytest.fixture
def ssa_sc_ssvd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_method='sc_ssvd')
    return ssa

@pytest.fixture
def ssa_sk_rsvd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_method='sk_rsvd')
    return ssa

@pytest.fixture
def ssa_da_svd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_method='da_svd')
    return ssa

@pytest.fixture
def ssa_da_csvd(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_method='da_csvd')
    return ssa


@pytest.fixture
def ssa_with_decomposition(timeseries50):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_method='sk_rsvd')
    ssa.decompose(n_components=10)
    return ssa

@pytest.fixture
def ssa_with_reconstruction(ssa_with_decomposition):
    ssa = ssa_with_decomposition
    ssa.reconstruct(groups={'group1': [0,1,2]})
    return ssa