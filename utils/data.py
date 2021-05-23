from typing import List, Tuple

import pandas as pd

LAG_FEATURE_TEMPLATE = "lag{}_{}"


def load_csv_data(path: str) -> Tuple[List[str], pd.DataFrame]:
    ts: pd.DataFrame = pd.read_csv(path, skip_blank_lines=False)
    ts.index = pd.to_datetime(ts.t, utc=False, unit="s")
    ts.index = ts.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    ts.index.freq = pd.DateOffset(minutes=1)
    ts["hour_of_day"] = ts.index.hour
    ts["day_of_week"] = ts.index.dayofweek
    ts["month_of_year"] = ts.index.month
    return ["hour_of_day", "day_of_week", "month_of_year"], ts


def extract_features(ts: pd.DataFrame, y_column: str, lag_order: int = 2) -> Tuple[List[str], pd.DataFrame]:
    assert ts.shape[0] > lag_order, "there is not enough data for lag order {}".format(lag_order)
    cols_added: List[str] = []
    # Extract lag-k outputs
    for lag in range(1, lag_order + 1):
        lag_col = LAG_FEATURE_TEMPLATE.format(lag, y_column)
        ts[lag_col] = ts[y_column].shift(lag)
        cols_added.append(lag_col)
    return cols_added, ts
