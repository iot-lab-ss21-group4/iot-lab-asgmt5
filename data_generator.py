# Data generation
import argparse
import math
import os
import random
from datetime import datetime

import pandas as pd

from utils.data import TIME_COLUMN, UNIVARIATE_DATA_COLUMN


# Function that models count of students in the room based on time of the day
def count_in_room(cur_val, coef, delay_val, capacity, falloff_val):
    cnt = capacity
    lb = -math.floor(0.1 * capacity)
    ub = math.ceil(0.1 * capacity)

    if cur_val <= delay_val:
        cnt = min(max(math.floor(coef * cur_val) + random.randrange(lb, ub, 1), 0), capacity)
    elif cur_val >= falloff_val:
        cnt = max(min(capacity, math.floor(capacity - coef * (cur_val - falloff_val)) + random.randrange(lb, ub, 1)), 0)
    # TODO: you can also model students going in and out of the room during the lesson

    return cnt


def main(args: argparse.Namespace):
    # Settable parameters
    timestamps_onemin = 4 * 7 * 24 * 60
    timestamp_start_s = 1561932000  # Should not overlap with change in summer/winter time!
    starting_hour = 8
    end_hour = 20
    room_capacity = 25
    lesson_duration_hours = 2
    lesson_duration_min = lesson_duration_hours * 60
    arrival_and_exit_delay_min = 8
    falloff_border = lesson_duration_min - arrival_and_exit_delay_min
    arrival_and_exit_coef_min = room_capacity / arrival_and_exit_delay_min

    data_range = range(timestamps_onemin)
    generated_ts = pd.DataFrame(index=data_range, columns=[TIME_COLUMN, UNIVARIATE_DATA_COLUMN])

    # Generating simulated students' count data -
    for i in data_range:
        cur_timestamp_s = timestamp_start_s + i * 60
        cur_date = datetime.fromtimestamp(cur_timestamp_s)
        cur_wd = cur_date.weekday()
        cur_hour = cur_date.hour
        cur_min = cur_date.minute

        cnt_in_room = 0
        if (cur_wd < 5) & (cur_hour >= starting_hour) & (cur_hour <= end_hour):
            lessons_cur_min = (cur_hour * 60 + cur_min) % lesson_duration_min
            cnt_in_room = count_in_room(
                lessons_cur_min, arrival_and_exit_coef_min, arrival_and_exit_delay_min, room_capacity, falloff_border
            )

        generated_ts.loc[i] = [cur_timestamp_s, cnt_in_room]

    generated_ts.index = pd.to_datetime(generated_ts.t, unit="s")
    generated_ts.index.freq = pd.DateOffset(minutes=1)
    generated_ts.to_csv(args.out_path, index=False)


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--out-path",
        type=str,
        default=os.path.join("datasets", "out.csv"),
        help="Path to the output '.csv' file containing generated data.",
    )
    parser.set_defaults(func=main)
