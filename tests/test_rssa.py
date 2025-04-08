"""Comparison with the Rssa package

This test is run locally with a configured R environment.
"""
import os
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.4.0' # Edit if necessary
import pytest
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri

from ssalib import SingularSpectrumAnalysis as SSA, SingularSpectrumAnalysis
from numpy.testing import assert_allclose
import pandas as pd

# Activate R to Pandas dataframe conversion
pandas2ri.activate()
rssa = rpackages.importr('Rssa')

@pytest.fixture
def co2():
    """Retrieve the CO2 dataset"""
    ro.r('data(co2)')
    return ro.globalenv['co2']


def test_compare_singular_values_bk(co2):
    """
    Test to compare SSA results (singular values) between Rssa and ssalib for
    the bk_trajectory matrix kind.
    """
    # rSSA
    ro.r.assign('co2', co2)
    ro.r("""
    ssa_object <- ssa(co2, L=50)
    singular_values <- ssa_object$sigma
    """)
    singular_values_rssa = ro.r['singular_values']

    # ssalib
    ssa = SSA(co2, window=50, standardize=False)
    ssa.decompose()
    singular_values_ssalib = ssa.s_

    # comparison
    assert_allclose(singular_values_rssa, singular_values_ssalib, rtol=1e-5,
                    atol=1e-8)

def test_compare_singular_values_vg(co2):
    """
    Test to compare SSA results (singular values) between Rssa and ssalib for
    the vg_covariance matrix kind.
    """
    # rSSA
    ro.r.assign('co2', co2)
    ro.r("""
    ssa_object <- ssa(co2, L=50, kind='toeplitz-ssa')
    singular_values <- ssa_object$sigma
    """)
    singular_values_rssa = ro.r['singular_values']

    # ssalib
    ssa = SSA(co2, window=50, svd_matrix_kind='vg_covariance', standardize=False)
    ssa.decompose()
    singular_values_ssalib = ssa.s_

    # comparison
    assert_allclose(singular_values_rssa, singular_values_ssalib, rtol=1e-5,
                    atol=1e-8)

def test_compare_wcorr_bk(co2):
    # rSSA
    ro.r.assign('co2', co2)
    ro.r("""
    ssa_object <- ssa(co2, L=50)
    wcor_matrix <- wcor(ssa_object)
    """)
    wcorr_rssa = ro.r['wcor_matrix']

    ssa = SingularSpectrumAnalysis(co2, window=50, standardize=False)
    ssa.decompose()
    wcorr_ssalib = ssa.wcorr(n_components=50)
    assert_allclose(wcorr_rssa, wcorr_ssalib, rtol=1e-5, atol=1e-8)

def test_compare_wcorr_vg(co2):
    # rSSA
    ro.r.assign('co2', co2)
    ro.r("""
    ssa_object <- ssa(co2, L=50, kind='toeplitz-ssa')
    wcor_matrix <- wcor(ssa_object)
    """)
    wcorr_rssa = ro.r['wcor_matrix']

    ssa = SingularSpectrumAnalysis(
        co2, window=50, standardize=False, svd_matrix_kind='vg_covariance')
    ssa.decompose()
    wcorr_ssalib = ssa.wcorr(n_components=50)
    assert_allclose(wcorr_rssa, wcorr_ssalib, rtol=1e-5, atol=1e-8)

