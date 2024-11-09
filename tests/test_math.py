import numpy as np
import pytest

from vassal.math_ext import (
    correlation_weights,
    weighted_correlation_matrix,
    construct_SVD_matrix,
    construct_BK_trajectory_matrix,
    construct_VG_covariance_matrix,
    average_antidiagonals,
    autoregressive_model_score,
    fit_autoregressive_model,
    generate_autoregressive_surrogate
)


def test_correlation_weights():
    weights = correlation_weights(10, 4)
    assert weights.shape[0] == 10
    assert np.isfinite(weights).all()


def test_weighted_correlation_matrix():
    series = np.random.randn(5, 10)  # 5 timeseries of length 10
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


def test_autoregressive_model_score_valid_ar0(ar1_timeseries50):
    timeseries = ar1_timeseries50
    order = 0  # white noise should work as well
    score = autoregressive_model_score(timeseries, order)
    assert isinstance(score, float)


def test_autoregressive_model_score_valid_bic(ar1_timeseries50):
    timeseries = ar1_timeseries50
    order = 1
    score = autoregressive_model_score(timeseries, order, criterion='bic')
    assert isinstance(score, float)


def test_autoregressive_model_score_valid_aic(ar1_timeseries50):
    timeseries = ar1_timeseries50
    order = 1
    score = autoregressive_model_score(timeseries, order, criterion='aic')
    assert isinstance(score, float)


def test_autoregressive_model_score_empty_timeseries():
    timeseries = []
    order = 1
    with pytest.raises(ValueError, match="timeseries cannot be empty"):
        autoregressive_model_score(timeseries, order)


def test_autoregressive_model_score_negative_order(ar1_timeseries50):
    timeseries = ar1_timeseries50
    order = -1
    with pytest.raises(ValueError, match="order must be non-negative"):
        autoregressive_model_score(timeseries, order)


def test_autoregressive_model_score_order_greater_than_length():
    timeseries = [1.0, 2.0, 3.0, 4.0, 5.0]
    order = 10
    with pytest.raises(ValueError,
                       match="timeseries length must be greater than order"):
        autoregressive_model_score(timeseries, order)


def test_fit_autoregressive_model_default(ar1_timeseries50):
    timeseries = ar1_timeseries50
    model = fit_autoregressive_model(timeseries)
    expected_params = np.array([0.599205, 0.827014])
    np.testing.assert_allclose(model.params, expected_params, atol=1e-5)


def test_fit_autoregressive_model_aic(ar1_timeseries50):
    timeseries = ar1_timeseries50
    model = fit_autoregressive_model(timeseries, criterion='aic')
    assert hasattr(model, 'aic')
    assert hasattr(model, 'bic')


def test_fit_autoregressive_model_specified_max_order(ar1_timeseries50):
    timeseries = ar1_timeseries50
    model = fit_autoregressive_model(timeseries, max_order=5)
    assert hasattr(model, 'aic')
    assert hasattr(model, 'bic')


def test_fit_autoregressive_model_empty_timeseries():
    timeseries = []
    with pytest.raises(ValueError,
                       match="timeseries length must be greater than max_order"):
        fit_autoregressive_model(timeseries)


def test_fit_autoregressive_model_max_order_greater_than_length():
    timeseries = [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(ValueError,
                       match="timeseries length must be greater than max_order"):
        fit_autoregressive_model(timeseries, max_order=10)


def test_fit_autoregressive_model_negative_max_order(ar1_timeseries50):
    timeseries = ar1_timeseries50
    with pytest.raises(ValueError, match="max_order must be non-negative"):
        fit_autoregressive_model(timeseries, max_order=-1)


def test_fit_autoregressive_model_non_integer_max_order(ar1_timeseries50):
    timeseries = ar1_timeseries50
    with pytest.raises(TypeError, match="max_order must be an integer"):
        fit_autoregressive_model(timeseries, max_order=1.5)


def test_fit_autoregressive_model_parallel_jobs(ar1_timeseries50):
    timeseries = ar1_timeseries50
    model = fit_autoregressive_model(timeseries, n_jobs=2)
    assert hasattr(model, 'aic')
    assert hasattr(model, 'bic')


def test_generate_ar_surrogate_valid_parameters():
    ar_coefficients = [1, -0.5]
    n_samples = 100
    scale = 1.0
    surrogate = generate_autoregressive_surrogate(ar_coefficients, n_samples,
                                                  scale)
    assert isinstance(surrogate, np.ndarray)
    assert len(surrogate) == n_samples
    model = fit_autoregressive_model(surrogate)

    np.testing.assert_allclose(-model.params[0], ar_coefficients[-1], atol=1e-1)
    np.testing.assert_allclose(model.params[-1], scale, atol=1e-1)

def test_generate_ar_surrogate_with_seed():
    ar_coefficients = [1, -0.5]
    n_samples = 10
    scale = 1.0
    seed = 42
    surrogate_1 = generate_autoregressive_surrogate(ar_coefficients, n_samples,
                                                    scale, seed)
    surrogate_2 = generate_autoregressive_surrogate(ar_coefficients, n_samples,
                                                    scale, seed)
    assert np.array_equal(surrogate_1, surrogate_2)


def test_generate_ar_surrogate_zero_samples():
    ar_coefficients = [1, -0.5]
    n_samples = 0
    scale = 1.0
    with pytest.raises(ValueError, match="n_samples must be positive"):
        generate_autoregressive_surrogate(ar_coefficients, n_samples, scale)


def test_generate_ar_surrogate_negative_samples():
    ar_coefficients = [1, -0.5]
    n_samples = -10
    scale = 1.0
    with pytest.raises(ValueError, match="n_samples must be positive"):
        generate_autoregressive_surrogate(ar_coefficients, n_samples, scale)


def test_generate_ar_surrogate_zero_scale():
    ar_coefficients = [1, -0.5]
    n_samples = 100
    scale = 0.0
    with pytest.raises(ValueError, match="scale must be positive"):
        generate_autoregressive_surrogate(ar_coefficients, n_samples, scale)


def test_generate_ar_surrogate_negative_scale():
    ar_coefficients = [1, -0.5]
    n_samples = 100
    scale = -1.0
    with pytest.raises(ValueError, match="scale must be positive"):
        generate_autoregressive_surrogate(ar_coefficients, n_samples, scale)


def test_generate_ar_surrogate_empty_ar_coefficients():
    ar_coefficients = []
    n_samples = 100
    scale = 1.0
    with pytest.raises(ValueError, match="ar_coefficients must not be empty"):
        generate_autoregressive_surrogate(ar_coefficients, n_samples, scale)


def test_generate_ar_surrogate_first_coeff_not_one():
    ar_coefficients = [0.5, -0.5]
    n_samples = 100
    scale = 1.0
    with pytest.raises(ValueError, match="First AR coefficient must be 1"):
        generate_autoregressive_surrogate(ar_coefficients, n_samples, scale)


def test_generate_ar_surrogate_burnin_effect():
    ar_coefficients = [1, -0.5]
    n_samples = 100
    scale = 1.0
    seed = 42
    surrogate_without_burnin = generate_autoregressive_surrogate(
        ar_coefficients, n_samples, scale, seed, burnin=0)
    surrogate_with_burnin = generate_autoregressive_surrogate(ar_coefficients,
                                                              n_samples, scale,
                                                              seed, burnin=100)
    assert not np.array_equal(surrogate_without_burnin, surrogate_with_burnin)
