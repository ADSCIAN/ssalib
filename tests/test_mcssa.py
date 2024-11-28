"""Tests for class MonteCarloSSA."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

from ssalib.error import DecompositionError
from ssalib.montecarlo_ssa import MonteCarloSSA
from ssalib.ssa import SSAMatrixType
from ssalib.svd import SVDSolverType


def test_test_significance(mcssa_no_decomposition):
    mcssa_no_decomposition.decompose()
    print(mcssa_no_decomposition.test_significance())


# Test initialization errors

def test_invalid_n_surrogates():
    """Test that invalid n_surrogates raises appropriate errors."""
    with pytest.raises(TypeError, match="must be an integer"):
        MonteCarloSSA([1, 2, 3], n_surrogates=1.5)

    with pytest.raises(ValueError, match="must be positive"):
        MonteCarloSSA([1, 2, 3], n_surrogates=0)


def test_invalid_njobs():
    with pytest.raises(TypeError, match="must be an integer"):
        MonteCarloSSA([1, 2, 3], n_jobs='invalid')


def test_invalid_ar_order_max():
    """Test that invalid ar_order_max raises appropriate errors."""
    with pytest.raises(TypeError, match="must be an integer"):
        MonteCarloSSA([1, 2, 3], ar_order_max=1.5)

    with pytest.raises(ValueError, match="must be non-negative"):
        MonteCarloSSA([1, 2, 3], ar_order_max=-1)


def test_invalid_criterion():
    """Test that invalid criterion raises appropriate errors."""
    with pytest.raises(TypeError, match="must be a string"):
        MonteCarloSSA([1, 2, 3], criterion=1)

    with pytest.raises(ValueError, match="must be either 'aic' or 'bic'"):
        MonteCarloSSA([1, 2, 3], criterion='invalid')


def test_invalid_random_seed():
    """Test that invalid random_seed raises appropriate errors."""
    with pytest.raises(TypeError, match="must be None or an integer"):
        MonteCarloSSA([1, 2, 3], random_seed=1.5)


# Test initialization success cases

def test_correct_initialization(mcssa_no_decomposition):
    """Test that the SSA initializes correctly with valid inputs."""
    assert mcssa_no_decomposition._n == 50
    assert mcssa_no_decomposition._window == 25
    assert mcssa_no_decomposition._standardized is True
    assert mcssa_no_decomposition._timeseries is not None
    assert mcssa_no_decomposition._svd_solver == SVDSolverType.NUMPY_STANDARD
    assert mcssa_no_decomposition._svd_matrix_kind == SSAMatrixType.BK_TRAJECTORY
    assert mcssa_no_decomposition._na_strategy == 'raise_error'
    assert mcssa_no_decomposition.na_mask.sum() == 0
    assert mcssa_no_decomposition.groups['ssa_original'] is None
    assert mcssa_no_decomposition.groups['ssa_preprocessed'] is None
    assert mcssa_no_decomposition.n_surrogates == 10
    assert mcssa_no_decomposition.random_seed == 42
    assert mcssa_no_decomposition.ar_order_max == 1
    assert isinstance(mcssa_no_decomposition.autoregressive_model,
                      SARIMAXResultsWrapper)


def test_model_fitting_with_na(timeseries50_with_na):
    """Test model fitting with na"""
    mcssa = MonteCarloSSA(
        timeseries50_with_na,
        na_strategy='fill_mean',
        n_surrogates=1,
        random_seed=42
    )
    model1 = mcssa.autoregressive_model
    n_ar = len(model1.arparams)
    ts_zstd = timeseries50_with_na - np.nanmean(
        timeseries50_with_na) / np.nanstd(timeseries50_with_na)
    model2 = SARIMAX(
        ts_zstd,
        order=(n_ar, 0, 0),
        trend=None
    ).fit(disp=False)
    assert_array_almost_equal(model1.arparams, model2.arparams)


def test_surrogate_generation(mcssa_no_decomposition):
    """Test that surrogates are generated correctly."""
    assert mcssa_no_decomposition.surrogates.shape == (
        10, 50)  # (n_surrogates, n)
    # Test reproducibility with the same seed
    mcssa2 = MonteCarloSSA(
        mcssa_no_decomposition._timeseries,
        n_surrogates=10,
        random_seed=42
    )
    assert_array_almost_equal(mcssa_no_decomposition.surrogates,
                              mcssa2.surrogates)


# Test get_confidence_interval

def test_get_confidence_interval_invalid_inputs(mcssa_decomposed):
    """Test that get_confidence_interval raises appropriate errors"""
    with pytest.raises(TypeError, match="must be either integer or None"):
        mcssa_decomposed.get_confidence_interval(n_components=1.5)

    with pytest.raises(TypeError, match="must be a float"):
        mcssa_decomposed.get_confidence_interval(confidence_level="0.95")

    with pytest.raises(TypeError, match="must be a boolean"):
        mcssa_decomposed.get_confidence_interval(two_tailed="True")

    with pytest.raises(TypeError, match="must be a boolean"):
        mcssa_decomposed.get_confidence_interval(return_lower="True")

    with pytest.raises(ValueError, match="must be between 0 and 1"):
        mcssa_decomposed.get_confidence_interval(confidence_level=1.5)

    with pytest.raises(ValueError,
                       match="must be between 1 and the number of components"):
        mcssa_decomposed.get_confidence_interval(n_components=0)
    with pytest.raises(ValueError,
                       match="must be between 1 and the number of components"):
        n_out_of_range = mcssa_decomposed.n_components + 1
        mcssa_decomposed.get_confidence_interval(n_components=n_out_of_range)


def test_get_confidence_interval_shape(mcssa_decomposed):
    """Test that get_confidence_interval returns correct shapes."""
    # Test with return_lower=True
    lower, upper = mcssa_decomposed.get_confidence_interval(n_components=5)
    assert lower.shape == (5,)
    assert upper.shape == (5,)

    lower, upper = mcssa_decomposed.get_confidence_interval(n_components=5,
                                                            two_tailed=False)
    assert lower.shape == (5,)
    assert upper.shape == (5,)

    # Test with return_lower=False
    upper_only = mcssa_decomposed.get_confidence_interval(n_components=5,
                                                          return_lower=False)
    assert upper_only.shape == (5,)


# Test test_significance

def test_test_significance_before_decomposition(mcssa_no_decomposition):
    """Test that test_significance raises error if called before decomposition."""
    with pytest.raises(DecompositionError):
        mcssa_no_decomposition.test_significance()


def test_test_significance_output(mcssa_decomposed):
    """Test that test_significance returns expected output."""
    result = mcssa_decomposed.test_significance(n_components=5)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert result.shape == (5,)


def test_test_significance_with_different_confidence(mcssa_decomposed):
    """Test test_significance with different confidence levels."""
    result_95 = mcssa_decomposed.test_significance(n_components=5,
                                                   confidence_level=0.95)
    result_99 = mcssa_decomposed.test_significance(n_components=5,
                                                   confidence_level=0.99)
    # 99% confidence should generally lead to fewer significant components
    assert result_99.sum() <= result_95.sum()


# Test consistency between methods

def test_consistency_between_confidence_and_significance(mcssa_decomposed):
    """Test that get_confidence_interval and test_significance are consistent."""
    n_components = 5
    confidence_level = 0.95

    # Get results from both methods
    upper_limit = mcssa_decomposed.get_confidence_interval(
        n_components=n_components,
        confidence_level=confidence_level,
        return_lower=False
    )
    significance = mcssa_decomposed.test_significance(
        n_components=n_components,
        confidence_level=confidence_level
    )

    # Check consistency
    manual_significance = mcssa_decomposed.s_[:n_components] > upper_limit
    assert_array_equal(significance, manual_significance)


# Test inherited functionality

def test_inherited_decompose(mcssa_no_decomposition):
    """Test that the inherited decompose method works correctly."""
    result = mcssa_no_decomposition.decompose()
    assert result.n_components == 25
    assert hasattr(result, 's_')
    assert hasattr(result, 'u_')
    assert hasattr(result, 'vt_')


def test_inherited_reconstruct(mcssa_decomposed):
    """Test that the inherited reconstruct method works correctly."""
    result = mcssa_decomposed.reconstruct(groups={'trend': [0, 1]})
    assert 'trend' in result.groups
    assert result.groups['trend'] == [0, 1]


# Test matrix type initialization and validation

def test_matrix_type_initialization(timeseries50):
    """Test initialization with different matrix types."""
    for matrix_type in SSAMatrixType:
        mcssa = MonteCarloSSA(timeseries50, svd_matrix_kind=matrix_type.value)
        assert mcssa._svd_matrix_kind == matrix_type


def test_invalid_matrix_type(timeseries50):
    """Test that invalid matrix type raises appropriate error."""
    with pytest.raises(ValueError, match="Invalid svd_matrix_kind"):
        MonteCarloSSA(timeseries50, svd_matrix_kind='invalid_type')


# Test surrogate generation for different matrix types

def test_surrogate_generation_matrix_types(mcssa_matrix):
    """Test surrogate generation for each matrix type."""
    assert mcssa_matrix.surrogates.shape == (10, 50)
    # Verify surrogates maintain statistical properties
    orig_mean = mcssa_matrix._timeseries_pp.mean()
    surr_mean = mcssa_matrix.surrogates.mean(axis=1).mean()
    assert np.allclose(surr_mean, orig_mean, atol=0.1)


# Test decomposition results for different matrix types

def test_decomposition_matrix_types(mcssa_matrix_decomposed, matrix_kind):
    """Test decomposition results for each matrix type."""
    # Check basic properties that should hold for all matrix types
    assert mcssa_matrix_decomposed.n_components == 25
    assert mcssa_matrix_decomposed.s_.shape == (25,)
    assert mcssa_matrix_decomposed.u_.shape == (
        25, 25)  # window size × n_components

    if matrix_kind == 'bk_trajectory':
        assert mcssa_matrix_decomposed.vt_.shape == (
            26, 26)  # n_components × (n-window+1)
    elif matrix_kind in ['bk_covariance', 'vg_covariance']:
        # For covariance matrices, eigenvectors might need different checks
        assert mcssa_matrix_decomposed.s_.shape == (25,)
        assert all(s >= 0 for s in
                   mcssa_matrix_decomposed.s_)  # eigenvalues should be non-negative


# Test confidence intervals for different matrix types

def test_confidence_intervals_matrix_types(mcssa_matrix_decomposed):
    """Test confidence intervals calculation for each matrix type."""
    n_components = 5
    lower, upper = mcssa_matrix_decomposed.get_confidence_interval(
        n_components=n_components,
        confidence_level=0.95
    )

    # Basic shape and value checks
    assert lower.shape == (n_components,)
    assert upper.shape == (n_components,)
    assert all(l < u for l, u in zip(lower, upper))

    # Check if confidence intervals make sense relative to singular values
    singular_values = mcssa_matrix_decomposed.s_[:n_components]
    assert any(l <= s <= u for l, s, u in zip(lower, singular_values, upper))


# Test significance testing for different matrix types

def test_significance_matrix_types(mcssa_matrix_decomposed):
    """Test significance testing for each matrix type."""
    n_components = 5
    significance = mcssa_matrix_decomposed.test_significance(
        n_components=n_components,
        confidence_level=0.95
    )

    assert significance.shape == (n_components,)
    assert significance.dtype == bool

    # Test consistency with confidence intervals
    upper = mcssa_matrix_decomposed.get_confidence_interval(
        n_components=n_components,
        confidence_level=0.95,
        return_lower=False
    )
    manual_significance = mcssa_matrix_decomposed.s_[:n_components] > upper
    assert_array_almost_equal(significance, manual_significance)


# Test reconstruction for different matrix types

def test_reconstruction_matrix_types(mcssa_matrix_decomposed):
    """Test signal reconstruction for each matrix type."""
    # Reconstruct using first component
    reconstructed = mcssa_matrix_decomposed[0]
    assert reconstructed.shape == (50,)  # Original length

    # Reconstruct using multiple components
    group_reconstructed = mcssa_matrix_decomposed.reconstruct(
        groups={'test_group': [0, 1]}
    )['test_group']
    assert group_reconstructed.shape == (50,)


# Test process_surrogate method for different matrix types

def test_process_surrogate_matrix_types(mcssa_matrix_decomposed, matrix_kind):
    """Test surrogate processing for each matrix type."""
    # Get a single surrogate
    surrogate = mcssa_matrix_decomposed.surrogates[0]

    # Process the surrogate
    diagonal = mcssa_matrix_decomposed._process_surrogate(
        surrogate=surrogate,
        svd_matrix_kind=matrix_kind,
        window=mcssa_matrix_decomposed._window,
        u=mcssa_matrix_decomposed.u_,
        n_components=5
    )

    assert diagonal.shape == (5,)
    assert all(
        d.real >= 0 for d in diagonal)  # Should be non-negative for covariance


# Test comparison between matrix types

def test_matrix_types_comparison(timeseries50):
    """Compare results between different matrix types."""
    results = {}
    n_components = 5

    # Create and decompose MCSSA for each matrix type
    for matrix_type in SSAMatrixType:
        mcssa = MonteCarloSSA(
            timeseries50,
            svd_matrix_kind=matrix_type.value,
            n_surrogates=10,
            random_seed=42
        ).decompose()

        results[matrix_type.value] = {
            'singular_values': mcssa.s_[:n_components],
            'significance': mcssa.test_significance(n_components=n_components)
        }

    # Compare properties between matrix types
    for matrix_type in results:
        # Singular values should be positive for all types
        assert all(results[matrix_type]['singular_values'] > 0)

        # Significance patterns might differ but should be boolean
        assert results[matrix_type]['significance'].dtype == bool
