from __future__ import annotations

import inspect
import logging

import numpy as np
import pandas as pd
import pytest

from ssalib.error import DecompositionError, ReconstructionError
from ssalib.montecarlo_ssa import MonteCarloSSA
from ssalib.ssa import (
    SingularSpectrumAnalysis,
    SSAMatrixType)
from ssalib.svd import SVDSolverType


# Class test

@pytest.mark.parametrize("method",
                         ["test_significance", "get_confidence_interval"])
def test_signatures(method):
    ssa_method = getattr(SingularSpectrumAnalysis, method)
    mcssa_method = getattr(MonteCarloSSA, method)

    ssa_method_signature = inspect.signature(ssa_method)
    mcssa_method_signature = inspect.signature(mcssa_method)
    assert ssa_method_signature == mcssa_method_signature


# Test initialization

def test_correct_initialization(ssa_no_decomposition):
    """Test that the SSA initializes correctly with valid inputs."""
    assert ssa_no_decomposition._n == 50
    assert ssa_no_decomposition._window == 25  # Default window size is half of n
    assert ssa_no_decomposition._standardized is True  # Default should be True
    assert ssa_no_decomposition._timeseries is not None
    assert ssa_no_decomposition._svd_solver == SVDSolverType.NUMPY_STANDARD
    assert ssa_no_decomposition._svd_matrix_kind == SSAMatrixType.BK_TRAJECTORY
    assert ssa_no_decomposition._na_strategy == 'raise_error'
    assert ssa_no_decomposition.na_mask.sum() == 0
    assert ssa_no_decomposition.groups['ssa_original'] is None
    assert ssa_no_decomposition.groups['ssa_preprocessed'] is None
    assert ssa_no_decomposition._ssa_reconstructed is None
    assert ssa_no_decomposition._ssa_residuals is None


@pytest.mark.parametrize("invalid_data,error_msg", [
    (pd.Series(['a', 'b', 'c', 'd']),
     "All timeseries elements must be integers or floats"),
    (pd.Series([np.nan, np.nan, 1, 2]),
     "Argument timeseries cannot inf or NaN values with na_strategy set to "
     "'raise_error'"),
    (pd.Series([np.inf, np.inf, 1, 2]),
     "Argument timeseries cannot inf or NaN values with na_strategy set to "
     "'raise_error'"),
    (np.zeros(shape=(10, 2)), "Argument timeseries must be one-dimensional")
])
def test_invalid_timeseries_data(invalid_data, error_msg):
    """Test initialization with invalid timeseries data."""
    with pytest.raises(ValueError, match=error_msg):
        SingularSpectrumAnalysis(invalid_data)


def test_standardize():
    """Test the effect of the standardize parameter."""
    timeseries = pd.Series(np.array([1, 2, 3, 4, 5], dtype=float))
    ssa_standardized = SingularSpectrumAnalysis(timeseries)
    ssa_non_standardized = SingularSpectrumAnalysis(timeseries,
                                                    standardize=False)

    assert np.isclose(ssa_standardized.mean_, 3)
    assert np.isclose(ssa_standardized._timeseries_pp.std(), 1, rtol=1e-7)

    # Check that the non-standardized data does not modify the original
    # timeseries based on _mean or _std
    assert np.all(ssa_non_standardized._timeseries_pp == timeseries)
    # Check that the mean of the non-standardized data is equal to the original
    # mean
    assert np.isclose(ssa_non_standardized.mean_, timeseries.mean())


def test_window_parameter():
    """Test window parameter handling, both default and custom."""
    timeseries = pd.Series(np.random.rand(100))
    ssa_default = SingularSpectrumAnalysis(timeseries)
    ssa_custom = SingularSpectrumAnalysis(timeseries, window=20)
    assert ssa_default._window == 50
    assert ssa_custom._window == 20


@pytest.mark.parametrize("invalid_input,expected_error,error_msg", [
    ('wrong_type', TypeError, 'Argument window must be integer'),
    (np.random.rand(5), TypeError, 'Argument window must be integer'),
    (-10, ValueError, 'Invalid window size'),
    (1, ValueError, 'Invalid window size'),
    (49, ValueError, 'Invalid window size'),
])
def test_invalid_window(timeseries50, invalid_input, expected_error, error_msg):
    """Test various invalid window configurations."""
    with pytest.raises(expected_error, match=error_msg):
        SingularSpectrumAnalysis(timeseries50, window=invalid_input)


def test_na_strategy_fill_mean(timeseries50_with_na):
    ssa = SingularSpectrumAnalysis(
        timeseries50_with_na,
        na_strategy='fill_mean'
    )
    assert any(~np.isnan(ssa._timeseries_pp))
    assert any(ssa.na_mask)


def test_na_strategy_fill_mean_without_na(timeseries50):
    ssa = SingularSpectrumAnalysis(
        timeseries50,
        na_strategy='fill_mean'
    )
    assert any(~np.isnan(ssa._timeseries_pp))
    assert any(~ssa.na_mask)


def test_na_strategy_raise_error(timeseries50_with_na):
    with pytest.raises(
            ValueError,
            match="Argument timeseries cannot inf or NaN values with na_strategy "
                  "set to 'raise_error'"
    ):
        SingularSpectrumAnalysis(timeseries50_with_na,
                                 na_strategy='raise_error')


def test_na_strategy_invalid(timeseries50_with_na):
    """Test na_strategy parameter handling, both default and custom."""
    with pytest.raises(
            ValueError,
            match="Argument na_strategy should be either 'raise_error' or "
    ):
        ssa = SingularSpectrumAnalysis(
            timeseries50_with_na,
            na_strategy="invalid"
        )


@pytest.mark.parametrize("invalid_input,expected_error,error_msg", [
    ('wrong_type', ValueError, "Invalid svd_matrix_kind"),
    (np.random.rand(5), TypeError,
     "Argument svd_matrix_kind must be of type str or SSAMatrixType"),
])
def test_invalid_matrix_type(timeseries50, invalid_input, expected_error,
                             error_msg):
    """Test invalid matrix type configurations."""
    with pytest.raises(expected_error, match=error_msg):
        SingularSpectrumAnalysis(timeseries50, svd_matrix_kind=invalid_input)


@pytest.mark.parametrize("svd_matrix_kind", SSAMatrixType.available_matrices())
def test_svd_matrix_construction(svd_matrix_kind):
    timeseries = np.array([1, 3, 0, -3, -2, -1])
    ssa = SingularSpectrumAnalysis(timeseries, svd_matrix_kind=svd_matrix_kind,
                                   standardize=False)

    matrix = SSAMatrixType(svd_matrix_kind).construct_svd_matrix(
        ssa._timeseries_pp,
        ssa._window
    )
    np.testing.assert_equal(ssa.svd_matrix, matrix)


# Test decomposition

@pytest.mark.parametrize("solver", SVDSolverType.available_solvers())
def test_svd_solvers(timeseries50, solver):
    """Test different SVD solvers."""
    ssa = SingularSpectrumAnalysis(timeseries50, svd_solver=solver)
    assert ssa.svd_solver == solver
    if ssa._svd_solver.supports_n_components:
        with pytest.raises(ValueError,
                           match=f"Solver '{solver}' requires n_components"):
            ssa.decompose()
        # Test with components specified
        ssa.decompose(n_components=10)
    else:
        ssa.decompose()


@pytest.mark.parametrize("solver,n_components,rtol", [
    (SVDSolverType.SCIPY_STANDARD, None, 1e-7),
    (SVDSolverType.SCIPY_SPARSE, 10, 1e-4),
    (SVDSolverType.SKLEARN_RANDOMIZED, 10, 1e-4)
])
def test_svd_solver_results(ssa_numpy_standard, solver, n_components, rtol):
    """Test that different SVD solvers produce similar results."""
    _, s1, _ = ssa_numpy_standard.decompose().decomposition_results
    ssa = SingularSpectrumAnalysis(ssa_numpy_standard._timeseries,
                                   svd_solver=solver)

    kwargs = {'n_components': n_components} if n_components else {}

    _, s2, _ = ssa.decompose(**kwargs).decomposition_results

    if n_components:
        np.testing.assert_allclose(s1[:n_components], s2[:n_components],
                                   rtol=rtol)
    else:
        np.testing.assert_allclose(s1, s2, rtol=rtol)


def test_groups_after_decompose(ssa_with_decomposition):
    assert ssa_with_decomposition.groups['ssa_original'] is None
    assert ssa_with_decomposition.groups['ssa_preprocessed'] is None
    assert ssa_with_decomposition.groups['ssa_reconstructed'] is None


def test_repr_str(
        ssa_no_decomposition,
        ssa_with_decomposition,
        ssa_with_reconstruction
):
    ssa_no_decomposition.__repr__()
    ssa_with_decomposition.__repr__()
    ssa_with_reconstruction.__repr__()
    ssa_no_decomposition.__str__()
    ssa_with_decomposition.__str__()
    ssa_with_reconstruction.__str__()


@pytest.mark.parametrize("group_type", ['int', 'list'])
def test_valid_reconstruction_groups(ssa_with_decomposition, group_type):
    """Test valid reconstruction group configurations."""
    group_value = [1, 2, 3] if group_type == 'list' else 1
    ssa_with_decomposition.reconstruct(groups={'group1': group_value})


@pytest.mark.parametrize("invalid_group,expected_error,error_msg", [
    ("invalid_type", TypeError, "Argument groups must be a dictionary"),
    ({'group1': 1.}, ValueError,
     "Value for key 'group1' must be an int or list of int"),
    ({'group1': -1}, ValueError,
     "Values for key 'group1' must be in the range"),
    ({'group1': [1., 2, 3]}, ValueError,
     "Value for key 'group1' must be an int or list of int"),
    ({'group1': '1'}, ValueError,
     "Value for key 'group1' must be an int or list of int"),
    ({'group1': ['1', 2, 3]}, ValueError,
     "Value for key 'group1' must be an int or list of int"),
    ({123: [1, 2, 3]}, TypeError,
     "Key types in groups dictionary should be string"),
])
def test_invalid_reconstruction_groups(ssa_with_decomposition, invalid_group,
                                       expected_error, error_msg):
    """Test invalid reconstruction group configurations."""
    with pytest.raises(expected_error, match=error_msg):
        ssa_with_decomposition.reconstruct(invalid_group)


# Test __repr__ and __str__

def test_reconstruct_before_decompose(ssa_no_decomposition):
    with pytest.raises(DecompositionError,
                       match="Decomposition must be performed"):
        ssa_no_decomposition.reconstruct(groups={'group1': [1, 2, 3]})


# Test reconstruction

def test_user_ix_before_reconstruct(ssa_with_decomposition):
    with pytest.raises(ReconstructionError,
                       match="Cannot retrieve user indices"):
        ssa_with_decomposition._user_indices


def test_duplicate_reconstruction_warning(ssa_with_decomposition, caplog):
    """Test warning for duplicate indices in reconstruction groups."""
    with caplog.at_level(logging.WARNING):
        ssa_with_decomposition.reconstruct({
            'group1': [1, 2, 3],
            'group2': [3, 4, 5]
        })
    assert "Reconstructed groups contain duplicate indices: [3]" in \
           caplog.records[0].message


@pytest.mark.parametrize("default_group",
                         SingularSpectrumAnalysis._DEFAULT_GROUPS.keys())
def test_default_group_names(ssa_with_reconstruction, default_group):
    """Test that default group names cannot be used for reconstruction."""
    ssa_with_reconstruction[default_group]


@pytest.mark.parametrize("default_group",
                         SingularSpectrumAnalysis._DEFAULT_GROUPS.keys())
def test_invalid_default_group_names(ssa_with_decomposition, default_group):
    """Test that default group names cannot be used for reconstruction."""
    with pytest.raises(ValueError,
                       match=f"Group name '{default_group}' is reserved"):
        ssa_with_decomposition.reconstruct({default_group: [1, 2, 3]})


def test_getitem_by_ix(ssa_with_decomposition):
    ix = np.random.randint(0, ssa_with_decomposition.n_components)
    g1 = ssa_with_decomposition._reconstruct_group_timeseries(ix)
    g2 = ssa_with_decomposition[ix]
    g3 = ssa_with_decomposition[[ix]]
    np.testing.assert_equal(g1, g2)
    np.testing.assert_equal(g1, g3)


def test_getitem_by_slicing(ssa_with_decomposition):
    ix = np.random.randint(0, ssa_with_decomposition.n_components)
    g1 = ssa_with_decomposition._reconstruct_group_timeseries(list(range(ix)))
    g2 = ssa_with_decomposition[:ix]
    np.testing.assert_equal(g1, g2)


def test_getitem_by_groupname_reconstruction_error(ssa_with_decomposition):
    with pytest.raises(
            ReconstructionError,
            match="Cannot access user-defined key prior to group reconstruction"
    ):
        ssa_with_decomposition['group1']


def test_getitem_key_error_invalid_type(ssa_no_decomposition):
    with pytest.raises(KeyError,
                       match="Key 'nan' is not a valid key type"):
        ssa_no_decomposition[np.nan]


@pytest.mark.parametrize("key",
                         [-1, 51, [-1, 1], [1, "a"], slice(-1, 3),
                          slice(23, 26), slice(3, 2), np.nan])
def test_getitem_key_error_with_decomposition(ssa_with_decomposition, key):
    with pytest.raises(KeyError):
        ssa_with_decomposition[key]


@pytest.mark.parametrize("key", [0, [0, 1], slice(0, 4), 'key',
                                 "ssa_residuals", "ssa_reconstructed"])
def test_getitem_decomposition_error(ssa_no_decomposition, key):
    with pytest.raises(DecompositionError):
        ssa_no_decomposition[key]


# test group retrieval

@pytest.mark.parametrize("default_group",
                         SingularSpectrumAnalysis._DEFAULT_GROUPS.keys())
def test_groups_after_reconstruct(ssa_with_reconstruction, default_group):
    ssa_with_reconstruction.groups[default_group] is not None


def test_groups_unique(ssa_with_decomposition):
    groups = {'group1': 0}
    ssa_with_decomposition.reconstruct(groups)
    assert ssa_with_decomposition.groups['group1'] == 0


def test_groups_residuals(ssa_with_reconstruction):
    res_ix = [3, 4, 5, 6, 7, 8, 9]
    assert ssa_with_reconstruction.groups['ssa_residuals'] is None


def test_groups_rv_orignal(ssa_with_reconstruction):
    ts1 = ssa_with_reconstruction._timeseries
    ts2 = ssa_with_reconstruction['ssa_original']
    np.testing.assert_allclose(ts1, ts2)


def test_groups_rv_pp(ssa_with_reconstruction):
    ts1 = ssa_with_reconstruction._timeseries_pp
    ts2 = ssa_with_reconstruction['ssa_preprocessed']
    np.testing.assert_allclose(ts1, ts2)


def test_groups_rv_cpt(ssa_with_reconstruction):
    ts1 = ssa_with_reconstruction['ssa_reconstructed']
    ts2_1 = ssa_with_reconstruction['group1']
    ts2_2 = ssa_with_reconstruction['ssa_residuals']
    ts2 = ts2_1 + ts2_2
    np.testing.assert_allclose(ts1, ts2)


# test to_frame export

def test_to_frame_operations(ssa_with_reconstruction):
    """Test DataFrame conversion operations."""
    # Basic conversion
    result = ssa_with_reconstruction.to_frame()
    assert list(result.columns) == list(ssa_with_reconstruction.groups.keys())

    # Include/exclude options
    include_result = ssa_with_reconstruction.to_frame(include=['group1'])
    assert list(include_result.columns) == ['group1']

    exclude_result = ssa_with_reconstruction.to_frame(exclude=['group1'])
    assert 'group1' not in list(exclude_result.columns)

    # Test error for simultaneous include/exclude
    with pytest.raises(
            ValueError,
            match="Cannot specify both include and exclude parameters"
    ):
        ssa_with_reconstruction.to_frame(include=['group1'], exclude=['group2'])


def test_to_frame_include_exclude_invalid(ssa_with_reconstruction):
    with pytest.raises(ValueError):
        ssa_with_reconstruction.to_frame(include=['unknown_group'])
    with pytest.raises(ValueError):
        ssa_with_reconstruction.to_frame(exclude=['unknown_group'])


def test_to_frame_rescale(ssa_with_reconstruction):
    """Test DataFrame index name."""
    result = ssa_with_reconstruction.to_frame(
        include=['ssa_preprocessed'], rescale=True
    )
    assert np.isclose(np.mean(result, axis=0), ssa_with_reconstruction.mean_)
    assert np.isclose(np.std(result, axis=0), ssa_with_reconstruction.std_)


@pytest.mark.parametrize("index", [
    pd.date_range("2023-01-01", periods=10, freq='D'),
    range(10)
])
def test_to_frame_index(index):
    """Test DataFrame index"""
    series = pd.Series(index=index, data=np.arange(0, 10, 1))
    ssa = SingularSpectrumAnalysis(series)
    index_series = ssa.to_frame().index
    pd.testing.assert_index_equal(index_series, series.index)


# Class test

def test_signatures():
    import inspect
    from ssalib.ssa import SingularSpectrumAnalysis
    from ssalib.montecarlo_ssa import MonteCarloSSA

    ssa_methods = inspect.getmembers(SingularSpectrumAnalysis,
                                     predicate=inspect.isfunction)
    mc_methods = inspect.getmembers(MonteCarloSSA, predicate=inspect.isfunction)

    ssa_signatures = {name: inspect.signature(method) for name, method in
                      ssa_methods}
    mc_signatures = {name: inspect.signature(method) for name, method in
                     mc_methods}

    assert ssa_signatures['test_significance'] == mc_signatures[
        'test_significance']
    assert ssa_signatures['get_confidence_interval'] == mc_signatures[
        'get_confidence_interval']


# Not Implemented in SingularSpectrumAnalysis

def test_not_implemented_methods(ssa_no_decomposition):
    with pytest.raises(NotImplementedError):
        ssa_no_decomposition.get_confidence_interval()
    with pytest.raises(NotImplementedError):
        ssa_no_decomposition.test_significance()
