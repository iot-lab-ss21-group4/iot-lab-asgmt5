from typing import List, Tuple

import numpy as np
import pandas as pd

LAG_FEATURE_TEMPLATE = "lag{}_{}"
DEFAULT_FLOAT_TYPE = np.float32
TIME_COLUMN = "t"
UNIVARIATE_DATA_COLUMN = "count"


def load_csv_data(path: str) -> Tuple[List[str], pd.DataFrame]:
    ts: pd.DataFrame = pd.read_csv(path, skip_blank_lines=False)
    ts.drop_duplicates(subset=TIME_COLUMN, inplace=True)
    ts.index = pd.to_datetime(ts[TIME_COLUMN], utc=True, unit="s")
    ts.index = ts.index.tz_convert("Europe/Berlin")
    # ts.index.freq = pd.DateOffset(minutes=1)
    ts["hour_of_day"] = ts.index.hour.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
    ts["day_of_week"] = ts.index.dayofweek.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
    ts["month_of_year"] = ts.index.month.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
    return ["hour_of_day", "day_of_week", "month_of_year"], ts


def extract_features(ts: pd.DataFrame, lag_order: int = 2) -> Tuple[List[str], pd.DataFrame, int]:
    """Extracts features from the univariate time series data.

    Args:
      ts: A pandas data frame containing 'TIME_COLUMN' and 'UNIVARIATE_DATA_COLUMN' as the column names.
      lag_order: Order of lag variables to be extracted into a new column.

    Returns:
      (lc, ts, ur) tuple where 'lc' is a list of added column names, 'ts' is the modified
      time series pd.DataFrame and 'ur' is the number of useless rows due to NaN values.
    """
    assert ts.shape[0] > lag_order, "there is not enough data for lag order {}".format(lag_order)
    cols_added: List[str] = []
    # Extract lag-k outputs
    for lag in range(1, lag_order + 1):
        lag_col = LAG_FEATURE_TEMPLATE.format(lag, UNIVARIATE_DATA_COLUMN)
        ts[lag_col] = ts[UNIVARIATE_DATA_COLUMN].shift(lag)
        cols_added.append(lag_col)
    # Extract dT and d(count)/dT.
    dT_col = "dT"
    ts[dT_col] = (ts[TIME_COLUMN] - ts[TIME_COLUMN].shift(1)).astype(DEFAULT_FLOAT_TYPE)
    cols_added.append(dT_col)

    derivative_col = "d_{}/dT".format(UNIVARIATE_DATA_COLUMN)
    ts[derivative_col] = (ts[UNIVARIATE_DATA_COLUMN].shift(1) - ts[UNIVARIATE_DATA_COLUMN].shift(2)) / (
        ts[TIME_COLUMN].shift(1) - ts[TIME_COLUMN].shift(2)
    ).astype(DEFAULT_FLOAT_TYPE)
    cols_added.append(derivative_col)
    return cols_added, ts, max(lag_order, 2)
