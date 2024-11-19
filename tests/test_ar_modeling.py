import numpy as np
import pytest

from ssalib.math_ext.ar_modeling import (
    autoregressive_model_score,
    fit_autoregressive_model,
    generate_autoregressive_surrogate
)


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
                       match="Argument timeseries must have length greater "
                             "than order"):
        autoregressive_model_score(timeseries, order)


def test_autoregressive_model_score_non_integer_order(ar1_timeseries50):
    """Test that non-integer order raises TypeError."""
    with pytest.raises(TypeError, match="order must be an integer"):
        autoregressive_model_score(ar1_timeseries50, order=1.5)


def test_autoregressive_model_score_non_string_criterion(
        ar1_timeseries50):
    """Test that non-string criterion raises TypeError."""
    with pytest.raises(TypeError, match="criterion must be a string"):
        autoregressive_model_score(ar1_timeseries50, order=1,
                                   criterion=123)


def test_autoregressive_model_score_invalid_criterion(ar1_timeseries50):
    """Test that invalid criterion value raises ValueError."""
    with pytest.raises(ValueError,
                       match="criterion must be either 'aic' or 'bic'"):
        autoregressive_model_score(ar1_timeseries50, order=1,
                                   criterion='invalid')


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
                       match="Argument timeseries must have length greater "
                             "than max_order"):
        fit_autoregressive_model(timeseries)


def test_fit_autoregressive_model_max_order_greater_than_length():
    timeseries = [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(ValueError,
                       match="Argument timeseries must have length greater "
                             "than max_order"):
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


def test_fit_autoregressive_model_invalid_criterion(ar1_timeseries50):
    """Test that invalid criterion value raises ValueError in fit function."""
    with pytest.raises(ValueError,
                       match="criterion must be either 'aic' or 'bic'"):
        fit_autoregressive_model(ar1_timeseries50, criterion='invalid')


def test_fit_autoregressive_model_n_jobs_none(ar1_timeseries50):
    """Test model fitting with n_jobs=None with proper validation."""
    model = fit_autoregressive_model(ar1_timeseries50, n_jobs=None)

    # Check information criteria are valid
    assert np.isfinite(model.aic), "AIC should be a finite number"
    assert np.isfinite(model.bic), "BIC should be a finite number"
    assert model.bic > model.aic, "BIC should be greater than AIC"

    # Validate model parameters
    assert len(model.params) > 0, "Model should have parameters"
    assert all(np.isfinite(model.params)), "All parameters should be finite"

    # Test prediction capability
    predictions = model.predict(start=0, end=len(ar1_timeseries50) - 1)
    assert len(predictions) == len(ar1_timeseries50)
    assert all(np.isfinite(predictions)), "All predictions should be finite"


def test_fit_autoregressive_model_n_jobs_positive(ar1_timeseries50):
    """Test model fitting with positive n_jobs with proper validation."""
    model = fit_autoregressive_model(ar1_timeseries50, n_jobs=2)

    assert np.isfinite(model.aic), "AIC should be a finite number"
    assert np.isfinite(model.bic), "BIC should be a finite number"
    assert model.bic > model.aic, "BIC should be greater than AIC"

    predictions = model.predict(start=0, end=len(ar1_timeseries50) - 1)
    assert len(predictions) == len(ar1_timeseries50)
    assert all(np.isfinite(predictions)), "All predictions should be finite"


def test_fit_autoregressive_model_n_jobs_negative(ar1_timeseries50):
    """Test model fitting with n_jobs=-1 with proper validation."""
    model = fit_autoregressive_model(ar1_timeseries50, n_jobs=-1)

    assert np.isfinite(model.aic), "AIC should be a finite number"
    assert np.isfinite(model.bic), "BIC should be a finite number"
    assert model.bic > model.aic, "BIC should be greater than AIC"

    predictions = model.predict(start=0, end=len(ar1_timeseries50) - 1)
    assert len(predictions) == len(ar1_timeseries50)
    assert all(np.isfinite(predictions)), "All predictions should be finite"


def test_fit_autoregressive_model_results_consistent_across_n_jobs(
        ar1_timeseries50):
    """Test that results are consistent regardless of n_jobs value."""
    model_single = fit_autoregressive_model(ar1_timeseries50, n_jobs=1)
    model_multi = fit_autoregressive_model(ar1_timeseries50, n_jobs=2)
    model_all = fit_autoregressive_model(ar1_timeseries50, n_jobs=-1)

    # Compare information criteria
    for criterion in ['aic', 'bic']:
        values = [getattr(model, criterion) for model in
                  [model_single, model_multi, model_all]]
        assert all(
            np.isfinite(values)), f"All {criterion} values should be finite"
        assert np.allclose(values, values[0],
                           rtol=1e-10), f"{criterion} values should be consistent"

    # Compare model parameters
    for model in [model_multi, model_all]:
        np.testing.assert_allclose(
            model_single.params,
            model.params,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Model parameters should be consistent across different "
                    "n_jobs values"
        )

    # Compare predictions
    pred_single = model_single.predict(start=0, end=len(ar1_timeseries50) - 1)
    pred_multi = model_multi.predict(start=0, end=len(ar1_timeseries50) - 1)
    pred_all = model_all.predict(start=0, end=len(ar1_timeseries50) - 1)

    np.testing.assert_allclose(
        pred_single,
        pred_multi,
        rtol=1e-10,
        err_msg="Predictions should be consistent between single and multi-core"
    )
    np.testing.assert_allclose(
        pred_single,
        pred_all,
        rtol=1e-10,
        err_msg="Predictions should be consistent between single and all-core"
    )


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
    with pytest.raises(ValueError,
                       match="Argument ar_cofficients should have 1 as first "
                             "element"):
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


def test_generate_surrogate_non_integer_n_samples():
    """Test that non-integer n_samples raises TypeError."""
    with pytest.raises(TypeError, match="n_samples must be an integer"):
        generate_autoregressive_surrogate([1, -0.5],
                                          n_samples=10.5, scale=1.0)


def test_generate_surrogate_non_numeric_scale():
    """Test that non-numeric scale raises TypeError."""
    with pytest.raises(TypeError, match="scale must be a number"):
        generate_autoregressive_surrogate([1, -0.5], n_samples=10,
                                          scale="1.0")


def test_generate_surrogate_non_integer_seed():
    """Test that non-integer seed raises TypeError."""
    with pytest.raises(TypeError, match="seed must be None or an integer"):
        generate_autoregressive_surrogate([1, -0.5], n_samples=10,
                                          scale=1.0, seed=1.5)


def test_generate_surrogate_non_integer_burnin():
    """Test that non-integer burnin raises TypeError."""
    with pytest.raises(TypeError, match="burnin must be an integer"):
        generate_autoregressive_surrogate([1, -0.5], n_samples=10,
                                          scale=1.0, burnin=10.5)
