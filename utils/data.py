from typing import List, Tuple

import pandas as pd


def load_csv_data(path: str) -> Tuple[List[str], pd.DataFrame]:
    ts: pd.DataFrame = pd.read_csv(path)
    ts.index = pd.to_datetime(ts.t, utc=False, unit="s")
    ts.index = ts.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    ts.index.freq = pd.DateOffset(minutes=1)
    ts["hour_of_day"] = ts.index.hour
    ts["day_of_week"] = ts.index.dayofweek
    ts["month_of_year"] = ts.index.month
    return ["hour_of_day", "day_of_week", "month_of_year"], ts
