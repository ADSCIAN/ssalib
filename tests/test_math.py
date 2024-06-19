import numpy as np
import pytest

from vassal.math_ext import (
    correlation_weights,
    weighted_correlation_matrix,
    construct_SVD_matrix,
    construct_BK_trajectory_matrix,
    construct_VG_covariance_matrix,
    average_antidiagonals
)


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

def test_construct_SVD_matrix_BK():
    ts = np.array([1, 3, 0, -3, -2, -1])
    expected_matrix = np.array([
        [1., 3., 0.],
        [3., 0., -3.],
        [0., -3., -2.],
        [-3., -2., -1.]
    ])
    result = construct_SVD_matrix(ts, window=4, kind='BK')
    np.testing.assert_array_almost_equal(result, expected_matrix)

def test_construct_SVD_matrix_VG():
    ts = np.array([1, 3, 0, -3, -2, -1])
    expected_matrix = np.array([
        [4., 2.2, -1.5],
        [2.2, 4., 2.2],
        [-1.5, 2.2, 4.]
    ])
    result = construct_SVD_matrix(ts, window=3, kind='VG')
    np.testing.assert_array_almost_equal(result, expected_matrix)

def test_construct_BK_trajectory_matrix():
    ts = np.array([1, 3, 0, -3, -2, -1])
    expected_matrix = np.array([
        [1., 3., 0.],
        [3., 0., -3.],
        [0., -3., -2.],
        [-3., -2., -1.]
    ])
    result = construct_BK_trajectory_matrix(ts, window=4)
    np.testing.assert_array_almost_equal(result, expected_matrix)

def test_construct_VG_covariance_matrix():
    ts = np.array([1, 3, 0, -3, -2, -1])
    expected_matrix = np.array([
        [4., 2.2, -1.5],
        [2.2, 4., 2.2],
        [-1.5, 2.2, 4.]
    ])
    result = construct_VG_covariance_matrix(ts, window=3)
    np.testing.assert_array_almost_equal(result, expected_matrix)

def test_construct_SVD_matrix_invalid_kind():
    ts = np.array([1, 3, 0, -3, -2, -1])
    with pytest.raises(
            ValueError,
            match="SVD matrix 'kind' must be either 'BK_trajectory', "
                  "'BK_covariance', or 'VG_covariance'"):
        construct_SVD_matrix(ts, window=3, kind='invalid')

def test_construct_SVD_matrix_default_window():
    ts = np.array([1, 2, 3, 4, 5, 6])
    expected_matrix = np.array([
        [1., 2., 3., 4.],
        [2., 3., 4., 5.],
        [3., 4., 5., 6.]
    ])
    result = construct_SVD_matrix(ts, kind='BK')
    np.testing.assert_array_almost_equal(result, expected_matrix)

def test_average_antidiagonals():
    matrix = np.arange(12).reshape(4, 3)
    expected_timeseries = np.array([0., 2., 4., 7., 9., 11.])
    result = average_antidiagonals(matrix)
    np.testing.assert_array_almost_equal(result, expected_timeseries)

def test_average_antidiagonals_square_matrix():
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    expected_timeseries = np.array([1., 3., 5., 7., 9.])
    result = average_antidiagonals(matrix)
    np.testing.assert_array_almost_equal(result, expected_timeseries)

def test_average_antidiagonals_single_row():
    matrix = np.array([[1, 2, 3]])
    expected_timeseries = np.array([1., 2., 3.])
    result = average_antidiagonals(matrix)
    np.testing.assert_array_almost_equal(result, expected_timeseries)

def test_average_antidiagonals_single_column():
    matrix = np.array([[1], [2], [3]])
    expected_timeseries = np.array([1., 2., 3.])
    result = average_antidiagonals(matrix)
    np.testing.assert_array_almost_equal(result, expected_timeseries)

def test_average_antidiagonals_empty():
    matrix = np.array([[]])
    expected_timeseries = np.array([])
    result = average_antidiagonals(matrix)
    np.testing.assert_array_almost_equal(result, expected_timeseries)
