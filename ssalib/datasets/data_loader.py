"""Data loading functions for the datasets submodule
"""

from pathlib import Path

import pandas as pd


def validate_series(data: pd.DataFrame) -> pd.Series:
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Invalid data type. Expected pd.DataFrame or "
                        "pd.Series.")
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Invalid data shape. Expected a single column "
                             "DataFrame.")
        data = data.iloc[:, 0]
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Invalid data index. Expected pd.DatetimeIndex.")
    return data


def load_mortality() -> pd.Series:
    """
    Load mortality datasets return it as a pandas Series.

    The mortality datasets contains daily counts of deaths in Belgium from
    1992-01-01 to 2023-12-31. The dataset is provided by STATBEL under an
    open data license.

    Returns
    -------
    mortality : pd.Series
        A pandas Series containing daily counts of deaths in Belgium from
        1992-01-01 to 2023-12-31, indexed using a pandas.DatetimeIndex.


    References
    ----------
    STATBEL. (2024). Number of deaths per day. Available at:
    https://statbel.fgov.be/en/open-data/number-deaths-day (Accessed: May 2024).

    Examples
    --------

    >>> from ssalib.datasets import load_mortality
    >>> mortality = load_mortality()
    >>> mortality.head()
    DATE_DEATH
    1992-01-01    329
    1992-01-02    298
    1992-01-03    321
    1992-01-04    336
    1992-01-05    319
    Name: CNT, dtype: int64

    >>> mortality.describe()
    count    11688.000000
    mean       292.826489
    std         40.832590
    min        196.000000
    25%        265.000000
    50%        287.000000
    75%        313.000000
    max        675.000000
    Name: CNT, dtype: float64

    """
    file_path = Path(__file__).parent / 'TF_DEATHS.txt'
    mortality_data = pd.read_csv(file_path, index_col=0, parse_dates=True,
                                 dayfirst=True,
                                 sep='|')
    return validate_series(mortality_data)


def load_sst() -> pd.Series:
    """
    Load Sea Surface Temperature datasets return it as a pandas Series.

    The Sea Surface Temperature (SST) datasets contains monthly mean sea
    surface temperature in °C, from 1982-01-01 to 2023-12-31 globally between
    60° North and South, provided by the Climate Change Institute of University
    of Maine under a CC-BY license.

    Returns
    -------
    sst : pd.Series
        A pandas Series containing monthly mean total sunspot number from
        1749-01 to 2023-12, indexed using a pandas.DatetimeIndex.

    References
    ----------
    Climate Reanalyzer (n.d.). Monthly Sea Surface Temperature. Climate Change
    Institute, University of Maine. Retrieved June 06, 2024,
    from https://climatereanalyzer.org/

    Examples
    --------

    >>> from ssalib.datasets import load_sst
    >>> sst = load_sst()
    >>> sst.head()
    Date
    1982-01-15    20.10
    1982-02-15    20.13
    1982-03-15    20.21
    1982-04-15    20.19
    1982-05-15    20.15
    Name: Value, dtype: float64

    >>> sst.describe()
    count    504.000000
    mean      20.318175
    std        0.262347
    min       19.730000
    25%       20.110000
    50%       20.310000
    75%       20.500000
    max       21.060000
    Name: Value, dtype: float64

    """
    import json
    file_path = Path(
        __file__).parent / 'oisst2.1_world_60s-60n_sst_monthly.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    sst = pd.DataFrame(data[0]['data'], columns=['Date', 'Value'])
    sst['Date'] = pd.to_datetime(sst['Date'], format='%Y,%m,%d')
    sst.set_index('Date', inplace=True)

    return validate_series(sst)


if __name__ == '__main__':
    from doctest import testmod

    testmod()
