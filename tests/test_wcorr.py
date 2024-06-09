import pytest
import numpy as np
from vassal.wcorr import correlation_weights, weighted_correlation_matrix

def test_correlation_weights():
    weights = correlation_weights(10, 4)
    assert weights.shape[0] == 10
    assert np.isfinite(weights).all()

def test_weighted_correlation_matrix():
    series = np.random.randn(5, 10) # 5 timeseries of length 10
    weights = np.ones(10)
    wcorr = weighted_correlation_matrix(series, weights)
    assert wcorr.shape == (5, 5)
    assert np.allclose(np.diag(wcorr), 1)

    # Check symmetry
    assert np.allclose(wcorr, wcorr.T)

    # Test with different weights
    weights = np.linspace(1, 10, 10)
    wcorr = weighted_correlation_matrix(series, weights)
    assert np.allclose(wcorr, wcorr.T)
    assert (wcorr >= -1).all() and (wcorr <= 1).all()
