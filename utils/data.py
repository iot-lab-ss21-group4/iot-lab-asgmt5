from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

DEFAULT_FLOAT_TYPE = np.float32
TIME_COLUMN = "t"
UNIVARIATE_DATA_COLUMN = "count"
DT_COLUMN = "dT"
DERIVATIVE_COLUMN = "d_{}/dT".format(UNIVARIATE_DATA_COLUMN)
LAG_FEATURE_TEMPLATE = "lag{}_" + UNIVARIATE_DATA_COLUMN
# Lag order must be >= 1.
LAG_ORDER = 2


def index_from_time_column(ts: pd.DataFrame, freq: Optional[pd.Timedelta] = None) -> pd.DataFrame:
    datetimes = pd.to_datetime(ts[TIME_COLUMN], utc=True, unit="s")
    if freq is not None:
        ts.index = pd.DatetimeIndex(datetimes, freq=pd.tseries.frequencies.to_offset(freq))
    else:
        ts.index = pd.DatetimeIndex(datetimes)
    ts.index = ts.index.tz_convert("Europe/Berlin")
    return ts


def load_csv_data(path: str) -> Tuple[List[str], pd.DataFrame]:
    ts: pd.DataFrame = pd.read_csv(path, skip_blank_lines=False)
    ts.drop_duplicates(subset=TIME_COLUMN, inplace=True)
    ts = index_from_time_column(ts)
    ts["minute_of_hour"] = ts.index.minute.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
    ts["hour_of_day"] = ts.index.hour.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
    ts["minute_of_day"] = ts["minute_of_hour"] + ts["hour_of_day"] * 60
    ts["day_of_week"] = ts.index.dayofweek.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
    ts["month_of_year"] = ts.index.month.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
    return ["minute_of_hour", "hour_of_day", "minute_of_day", "day_of_week", "month_of_year"], ts


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
        lag_col = LAG_FEATURE_TEMPLATE.format(lag)
        ts[lag_col] = ts[UNIVARIATE_DATA_COLUMN].shift(lag)
        cols_added.append(lag_col)
    # Extract dT and d(count)/dT.
    DT_COLUMN = "dT"
    ts[DT_COLUMN] = (ts[TIME_COLUMN] - ts[TIME_COLUMN].shift(1)).astype(DEFAULT_FLOAT_TYPE)
    cols_added.append(DT_COLUMN)

    ts[DERIVATIVE_COLUMN] = (ts[UNIVARIATE_DATA_COLUMN].shift(1) - ts[UNIVARIATE_DATA_COLUMN].shift(2)) / (
        ts[TIME_COLUMN].shift(1) - ts[TIME_COLUMN].shift(2)
    ).astype(DEFAULT_FLOAT_TYPE)
    cols_added.append(DERIVATIVE_COLUMN)

    # Extract last count
    LAST_COUNT_COLUMN = "last_count"
    ts[LAST_COUNT_COLUMN] = ts[UNIVARIATE_DATA_COLUMN].shift(1).astype(DEFAULT_FLOAT_TYPE)
    cols_added.append(LAST_COUNT_COLUMN)

    return cols_added, ts, max(lag_order, 2)


def load_csv_data_with_features(path: str) -> Tuple[str, List[str], pd.DataFrame, int]:
    y_column, x_columns = UNIVARIATE_DATA_COLUMN, []
    cols_added, ts = load_csv_data(path)
    x_columns.extend(cols_added)
    ts[y_column] = ts[y_column].astype(DEFAULT_FLOAT_TYPE)

    cols_added, ts, useless_rows = extract_features(ts, lag_order=LAG_ORDER)
    x_columns.extend(cols_added)

    return y_column, x_columns, ts, useless_rows


def regularize_data(ts: pd.DataFrame) -> Tuple[pd.offsets.Nano, pd.DataFrame]:
    if ts.shape[0] <= 1:
        return ts
    avg_dt = (
        pd.Timestamp(ts.loc[ts.index[-1], TIME_COLUMN], unit="s") - pd.Timestamp(ts.loc[ts.index[0], TIME_COLUMN], unit="s")
    ) / ts.shape[0]
    time_scaler = MinMaxScaler().fit(ts[TIME_COLUMN].to_numpy().reshape((-1, 1)))
    regular_time_col = "regular_{}".format(TIME_COLUMN)
    ts[regular_time_col] = time_scaler.transform(ts[TIME_COLUMN].to_numpy().reshape((-1, 1)))
    univariate_f = interp1d(ts[regular_time_col].to_numpy(), ts[UNIVARIATE_DATA_COLUMN].to_numpy())
    ts[regular_time_col] = np.linspace(
        ts.loc[ts.index[0], regular_time_col], ts.loc[ts.index[-1], regular_time_col], num=ts.shape[0]
    )

    def regularize_row(row: pd.Series) -> np.ndarray:
        return univariate_f(row[regular_time_col])

    ts[UNIVARIATE_DATA_COLUMN] = ts.apply(regularize_row, axis=1).astype(DEFAULT_FLOAT_TYPE)
    ts[TIME_COLUMN] = np.round(time_scaler.inverse_transform(ts[regular_time_col].to_numpy().reshape((-1, 1)))).astype(
        np.int64
    )
    ts.drop(columns=[regular_time_col], inplace=True)
    ts = index_from_time_column(ts)
    return pd.tseries.frequencies.to_offset(avg_dt), ts
