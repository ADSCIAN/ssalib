import logging

import numpy as np
import pandas as pd
import pytest

from vassal.ssa import SingularSpectrumAnalysis, DecompositionError, \
    ReconstructionError


def test_correct_initialization(ssa_no_decomposition):
    """Test that the SSA initializes correctly with valid inputs."""
    assert ssa_no_decomposition._n == 50
    assert ssa_no_decomposition._w == 25  # Default window size is half of n
    assert ssa_no_decomposition._standardized is True  # Default should be True
    assert ssa_no_decomposition.timeseries is not None
    assert ssa_no_decomposition.svd_method == 'np_svd'  # Default


def test_window_wrong_type(timeseries50):
    with pytest.raises(ValueError, match='Invalid window type.'):
        SingularSpectrumAnalysis(timeseries50, window='wrong_type')
    with pytest.raises(ValueError, match='Invalid window type.'):
        SingularSpectrumAnalysis(timeseries50, window=np.random.rand(5))


def test_window_size(timeseries50):
    with pytest.raises(ValueError, match='Invalid window size.'):
        SingularSpectrumAnalysis(timeseries50, window=-10)
    with pytest.raises(ValueError, match='Invalid window size.'):
        SingularSpectrumAnalysis(timeseries50, window=1)
    with pytest.raises(ValueError, match='Invalid window size.'):
        SingularSpectrumAnalysis(timeseries50, window=49)


def test_svd_method_attribution():
    print(SingularSpectrumAnalysis.available_methods())


def test_reject_non_numeric_data():
    """Test that the constructor rejects non-numeric data types."""
    timeseries = pd.Series(['a', 'b', 'c', 'd'])
    with pytest.raises(ValueError):
        SingularSpectrumAnalysis(timeseries)


def test_handle_nans_and_infinities():
    """Test initialization failure with NaNs or Infs."""
    timeseries_nan = pd.Series([np.nan, np.nan, 1, 2])
    timeseries_inf = pd.Series([np.inf, np.inf, 1, 2])
    with pytest.raises(ValueError,
                       match='The array contains inf or NaN values.'):
        SingularSpectrumAnalysis(timeseries_nan)
    with pytest.raises(ValueError,
                       match='The array contains inf or NaN values.'):
        SingularSpectrumAnalysis(timeseries_inf)


def test_standardization_effect():
    """Test the effect of the standardize parameter."""
    timeseries = pd.Series(np.array([1, 2, 3, 4, 5], dtype=float))
    ssa_standardized = SingularSpectrumAnalysis(timeseries)
    ssa_non_standardized = SingularSpectrumAnalysis(timeseries,
                                                    standardize=False)

    assert np.isclose(ssa_standardized._mean, 3)
    assert np.isclose(ssa_standardized._timeseries_pp.std(), 1, rtol=1e-7)

    # Check that the non-standardized data does not modify the original
    # timeseries based on _mean or _std
    assert np.all(ssa_non_standardized._timeseries_pp == timeseries)
    # Check that the mean of the non-standardized data is equal to the original
    # mean
    assert np.isclose(ssa_non_standardized._mean, timeseries.mean())


def test_window_parameter():
    """Test window parameter handling, both default and custom."""
    timeseries = pd.Series(np.random.rand(100))
    ssa_default = SingularSpectrumAnalysis(timeseries)
    ssa_custom = SingularSpectrumAnalysis(timeseries, window=20)
    assert ssa_default._w == 50
    assert ssa_custom._w == 20


@pytest.mark.parametrize("svd_method",
                         SingularSpectrumAnalysis.available_methods())
def test_svd_methods(timeseries50, svd_method: str):
    ssa = SingularSpectrumAnalysis(timeseries50, svd_method=svd_method)
    assert ssa.svd_method == svd_method


# Test decomposition
def test_np_svd(ssa_np_svd):
    ssa_np_svd.decompose()


def test_sc_svd(ssa_sc_svd):
    ssa_sc_svd.decompose()


def test_sc_ssvd(ssa_sc_ssvd):
    ssa_sc_ssvd.decompose(n_components=10)


def test_sc_ssvd_no_comp(ssa_sc_ssvd):
    with pytest.raises(ValueError, match=f"The selected method 'sc_ssvd'"):
        ssa_sc_ssvd.decompose()


def test_sk_rsvd(ssa_sk_rsvd):
    ssa_sk_rsvd.decompose(n_components=10)


def test_sk_rsvd_no_comp(ssa_sk_rsvd):
    with pytest.raises(ValueError, match=f"The selected method 'sk_rsvd'"):
        ssa_sk_rsvd.decompose()


def test_da_svd(ssa_da_svd):
    ssa_da_svd.decompose()


def test_da_csvd(ssa_da_csvd):
    ssa_da_csvd.decompose(n_components=10)


def test_da_csvd_no_comp(ssa_da_csvd):
    with pytest.raises(ValueError, match=f"The selected method 'da_csvd'"):
        ssa_da_csvd.decompose()


def test_sc_svd_close(ssa_np_svd):
    _, s1, _ = ssa_np_svd.decompose()
    ssa = SingularSpectrumAnalysis(ssa_np_svd.timeseries, svd_method='sc_svd')
    _, s2, _ = ssa.decompose()
    np.testing.assert_allclose(s1, s2)


def test_sc_ssvd_close(ssa_np_svd):
    up_to = 10
    _, s1, _ = ssa_np_svd.decompose()
    ssa = SingularSpectrumAnalysis(ssa_np_svd.timeseries, svd_method='sc_ssvd')
    _, s2, _ = ssa.decompose(n_components=up_to)
    np.testing.assert_allclose(s1[:up_to], s2[:up_to], rtol=1e-4)


def test_sk_rsvd_close(ssa_np_svd):
    up_to = 10
    _, s1, _ = ssa_np_svd.decompose()
    ssa = SingularSpectrumAnalysis(ssa_np_svd.timeseries, svd_method='sk_rsvd')
    _, s2, _ = ssa.decompose(n_components=up_to)
    np.testing.assert_allclose(s1[:up_to], s2[:up_to], rtol=1e-4)


def test_da_svd_close(ssa_np_svd):
    _, s1, _ = ssa_np_svd.decompose()
    ssa = SingularSpectrumAnalysis(ssa_np_svd.timeseries, svd_method='da_svd')
    _, s2, _ = ssa.decompose()
    np.testing.assert_allclose(s1, s2)


def test_da_csvd_close(ssa_np_svd):
    up_to = 10
    _, s1, _ = ssa_np_svd.decompose()
    ssa = SingularSpectrumAnalysis(ssa_np_svd.timeseries, svd_method='da_csvd')
    _, s2, _ = ssa.decompose(n_components=up_to, n_power_iter=4)
    np.testing.assert_allclose(s1[:up_to], s2[:up_to], rtol=1e-4)


# Test reconstruction
def test_reconstruct_before_decompose(ssa_no_decomposition):
    with pytest.raises(DecompositionError,
                       match="Decomposition must be performed"):
        ssa_no_decomposition.reconstruct(groups={'group1': [1, 2, 3]})


def test_user_ix_before_reconstruct(ssa_with_decomposition):
    with pytest.raises(ReconstructionError,
                       match="Cannot retrieve user indices"):
        ssa_with_decomposition._user_indices


def test_reconstruct_with_int(ssa_with_decomposition):
    ssa_with_decomposition.reconstruct(groups={'group1': [1]})


def test_reconstruct_with_list_of_int(ssa_with_decomposition):
    ssa_with_decomposition.reconstruct(groups={'group1': [1, 2, 3]})


def test_reconstruct_ValueError_float(ssa_with_decomposition):
    with pytest.raises(ValueError,
                       match="Value for key 'group1' must be an int or list of "
                             "int."):
        ssa_with_decomposition.reconstruct(groups={'group1': 1.})


def test_reconstruct_ValueError_neg(ssa_with_decomposition):
    with pytest.raises(ValueError,
                       match="Values for key 'group1' must be in the range"):
        ssa_with_decomposition.reconstruct(groups={'group1': -1})


def test_reconstruct_ValueError_out_of_range(ssa_with_decomposition):
    n_components = ssa_with_decomposition.n_components + 1
    with pytest.raises(ValueError,
                       match="Values for key 'group1' must be in the range"):
        ssa_with_decomposition.reconstruct(groups={'group1': n_components})


def test_reconstruct_ValueError_float_in_list(ssa_with_decomposition):
    with pytest.raises(ValueError,
                       match="Value for key 'group1' must be an int or list of "
                             "int."):
        ssa_with_decomposition.reconstruct(groups={'group1': [1., 2, 3]})


def test_reconstruct_ValueError_str(ssa_with_decomposition):
    with pytest.raises(ValueError,
                       match="Value for key 'group1' must be an int or list of "
                             "int."):
        ssa_with_decomposition.reconstruct(groups={'group1': '1'})


def test_reconstruct_ValueError_str_in_list(ssa_with_decomposition):
    with pytest.raises(ValueError,
                       match="Value for key 'group1' must be an int or list of "
                             "int."):
        ssa_with_decomposition.reconstruct(groups={'group1': ['1', 2, 3]})


def test_reconstruct_ValueError_key_int(ssa_with_decomposition):
    invalid_user_groups_key = {
        123: [1, 2, 3]
    }
    with pytest.raises(ValueError, match="Key '123' is not a string."):
        ssa_with_decomposition.reconstruct(invalid_user_groups_key)


def test_reconstruct_ValueError_key_in_default(ssa_with_decomposition):
    default_user_groups_key = {
        'ssa_original': [1, 2, 3]
    }
    with pytest.raises(ValueError, match="Unauthorized group name"):
        ssa_with_decomposition.reconstruct(default_user_groups_key)


def test_reconstruct_duplicate_integers_warning(ssa_with_decomposition, caplog):
    duplicate_values_user_groups = {
        'group1': [1, 2, 3],
        'group2': [3, 4, 5]
    }
    with caplog.at_level(logging.WARNING):
        ssa_with_decomposition.reconstruct(duplicate_values_user_groups)

    assert len(caplog.records) == 1
    assert "Reconstructed groups contain duplicate indices: [3]" in \
           caplog.records[0].message


def test_getitem_by_ix(ssa_with_decomposition):
    ix = np.random.randint(0, ssa_with_decomposition.n_components)
    g1 = ssa_with_decomposition._reconstruct_group(ix)
    g2 = ssa_with_decomposition[ix]
    g3 = ssa_with_decomposition[[ix]]
    np.testing.assert_equal(g1, g2)
    np.testing.assert_equal(g1, g3)


def test_getitem_by_slicing(ssa_with_decomposition):
    ix = np.random.randint(0, ssa_with_decomposition.n_components)
    g1 = ssa_with_decomposition._reconstruct_group(list(range(ix)))
    g2 = ssa_with_decomposition[:ix]
    np.testing.assert_equal(g1, g2)


def test_getitem_by_defaultname(ssa_with_decomposition):
    ssa_with_decomposition['ssa_original']
    with pytest.raises(ReconstructionError,
                       match="Cannot retrieve user indices"):
        ssa_with_decomposition['ssa_residuals']


def test_getitem_by_groupname_reconstruction_error(ssa_with_decomposition):
    with pytest.raises(ReconstructionError,
                       match="Cannot retrieve user indices"):
        ssa_with_decomposition['group1']


# test group retrieval

def test_groups(ssa_with_reconstruction):
    ssa_with_reconstruction.groups


def test_groups_init(ssa_no_decomposition):
    assert ssa_no_decomposition.groups['ssa_original'] is None
    assert ssa_no_decomposition.groups['ssa_preprocessed'] is None
    assert ssa_no_decomposition.groups['ssa_reconstructed'] is None


def test_groups_unique(ssa_with_decomposition):
    groups = {'group1': 0}
    ssa_with_decomposition.reconstruct(groups)
    assert ssa_with_decomposition.groups['group1'] == 0


def test_groups_residuals(ssa_with_reconstruction):
    res_ix = [3, 4, 5, 6, 7, 8, 9]
    assert ssa_with_reconstruction.groups['ssa_residuals'] == res_ix


def test_groups_rv_orignal(ssa_with_reconstruction):
    ts1 = ssa_with_reconstruction.timeseries
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

def test_to_frame(ssa_with_reconstruction):
    result = ssa_with_reconstruction.to_frame()
    assert list(result.columns) == list(ssa_with_reconstruction.groups.keys())
    np.testing.assert_array_equal(result['group1'].values,
                                  ssa_with_reconstruction['group1'])


def test_to_frame_include(ssa_with_reconstruction):
    result = ssa_with_reconstruction.to_frame(include=['group1'])
    assert list(result.columns) == ['group1']


def test_to_frame_exclude(ssa_with_reconstruction):
    result = ssa_with_reconstruction.to_frame(exclude=['group1'])
    assert 'group1' not in list(result.columns)


def test_to_frame_include_exclude_error(ssa_with_reconstruction):
    with pytest.raises(ValueError, match="Cannot specify both 'include' and "
                                         "'exclude' parameters."):
        ssa_with_reconstruction.to_frame(include=['group1'], exclude=['group2'])


def test_to_frame_ix():
    timeseries = pd.Series(np.arange(0, 10, 1))
    ssa = SingularSpectrumAnalysis(timeseries)
    index = ssa.to_frame().index
    pd.testing.assert_index_equal(ssa._ix, index)
