import pandas as pd
import pytest

from ssalib.datasets import (
    load_mortality,
    load_sst
)
from ssalib.datasets.data_loader import validate_series

load_funcs = [load_mortality, load_sst]

def test_validate_series_empty():
    """Test conversion of an empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(ValueError,
                       match="Invalid data shape. Expected a single column "
                             "DataFrame."):
        validate_series(df)


def test_validate_series_multiple_columns():
    """Test conversion of a DataFrame with more than one column."""
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    with pytest.raises(ValueError,
                       match="Invalid data shape. Expected a single column "
                             "DataFrame."):
        validate_series(df)

@pytest.mark.parametrize("load_data", load_funcs)
def test_load_data_returns_series(load_data):
    """Test if data loading function returns a pandas Series."""
    result = load_data()
    assert isinstance(result, pd.Series), \
        f"{load_data.__name__} should return a Series"


@pytest.mark.parametrize("load_data", load_funcs)
def test_load_data_index_type(load_data):
    """Test the index type of the Series returned by loading functions"""
    result = load_data()
    assert isinstance(result.index, pd.DatetimeIndex),\
        "Index should be a DatetimeIndex"
