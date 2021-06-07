import argparse
import os
import pickle
import threading
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.base.wrapper import ResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

from utils import load_data, regularize_data, start_periodic_forecast
from utils.data import DEFAULT_FLOAT_TYPE, TIME_COLUMN

EXTRA_INFO_FILE_POSTFIX = ".info"


def train(args: argparse.Namespace) -> SARIMAXResultsWrapper:
    y_column, exog_columns, ts = load_data(args.training_data_path, args.is_data_csv, detailed_seasonality=False)
    ts[y_column] = ts[y_column].astype(DEFAULT_FLOAT_TYPE)
    freq, ts = regularize_data(ts, y_column)
    # Remove constant columns
    constant_columns = ~((ts != ts.iloc[0]).any())
    exog_columns_set = set(exog_columns)
    for col in exog_columns:
        if constant_columns[col]:
            ts.drop(columns=[col], inplace=True)
            exog_columns_set.remove(col)
    exog_columns = list(exog_columns_set)

    train_len = int(ts.shape[0] * 0.9)
    train_ts, test_ts = ts.iloc[:train_len], ts.iloc[train_len:]
    with open(args.model_path + EXTRA_INFO_FILE_POSTFIX, "wb") as f:
        pickle.dump(
            (
                freq,
                train_ts.loc[train_ts.index[-1], TIME_COLUMN],
                train_ts.loc[train_ts.index[-1], y_column],
                exog_columns,
            ),
            f,
        )

    trial_dict = {}
    trial_lock = threading.RLock()

    def objective(trial: optuna.Trial) -> float:
        # past_days, past_weeks = 1, 0
        # past_day_offsets = np.zeros(4 * 7, dtype=np.int)
        # past_day_offsets[np.concatenate([np.arange(past_days), 7 * (1 + np.arange(past_weeks))])] = 1
        # past_day_offsets: List = past_day_offsets.tolist()
        # while len(past_day_offsets) > 0 and past_day_offsets[-1] == 0:
        #     past_day_offsets.pop()
        # P, D, Q, S = past_day_offsets, 0, past_day_offsets, 1440
        p, d = trial.suggest_int("p", 0, 3), trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0 if p + d > 0 else 1, 3)
        with trial_lock:
            if (p, d, q) in trial_dict:
                return trial_dict[(p, d, q)]
        forecast_model = SARIMAX(train_ts[y_column], exog=train_ts[exog_columns], trend="c", order=(p, d, q))
        model_fit = forecast_model.fit(disp=False)
        test_pred = model_fit.forecast(test_ts.shape[0], exog=test_ts[exog_columns])
        loss = mean_squared_error(test_ts[y_column].to_numpy(), test_pred.to_numpy())
        with trial_lock:
            trial_dict[(p, d, q)] = loss
        return loss

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage_url,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50, n_jobs=1)
    p, d, q = study.best_params["p"], study.best_params["d"], study.best_params["q"]
    forecast_model = SARIMAX(train_ts[y_column], exog=train_ts[exog_columns], trend="c", order=(p, d, q))
    model_fit = forecast_model.fit(disp=False)
    model_fit.save(args.model_path)

    if args.plot_fitted_model:
        all_target = ts[y_column]
        all_pred = model_fit.predict(start=0, end=ts.shape[0] - 1, exog=test_ts[exog_columns])
        all_target.index.freq = None
        all_pred.index = all_target.index
        all_target.plot(legend=True)
        all_pred.plot(legend=True)
        plt.show()

    return model_fit


def prepare_pred_input(pred_time: int, pred_data_path: str):
    pred_data = pd.Series([pred_time], dtype=np.int, name=TIME_COLUMN)
    pred_data.to_csv(pred_data_path, index=False)


def predict(args: argparse.Namespace, model_fit: Optional[SARIMAXResultsWrapper] = None) -> str:
    y_column, _, ts = load_data(args.pred_data_path, detailed_seasonality=False)
    # TODO: use freq and last_t to forecast beyond the last datapoint and
    # use interpolation to predict exactly at the datapoint times.
    with open(args.model_path + EXTRA_INFO_FILE_POSTFIX, "rb") as f:
        freq, last_t, last_y, exog_columns = pickle.load(f)

    if model_fit is None:
        model_fit = ResultsWrapper.load(args.model_path)
    pred: pd.Series = model_fit.forecast(ts.shape[0], exog=ts[exog_columns])
    pred.rename(y_column).round().to_csv(args.pred_out_path, index=False)
    return args.pred_out_path


def periodic_forecast(args: argparse.Namespace):
    train_period = 86400
    forecast_period = 150
    forecast_dt = 300
    start_periodic_forecast(
        args.training_data_path,
        train,
        {
            "training_data_path": args.training_data_path,
            "is_data_csv": args.is_data_csv,
            "study_name": args.study_name,
            "storage_url": args.storage_url,
            "model_path": args.model_path,
            "plot_fitted_model": False,
        },
        prepare_pred_input,
        {"pred_data_path": args.pred_data_path},
        predict,
        {
            "pred_data_path": args.pred_data_path,
            "model_path": args.model_path,
            "pred_out_path": args.pred_out_path,
        },
        args.iot_platform_settings_path,
        train_period=train_period,
        forecast_period=forecast_period,
        forecast_dt=forecast_dt,
    )


def add_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers(title="Subcommands")
    # Add command line arguments for 'train' subcommand.
    train_parser = subparser.add_parser("train", help="Subcommand to train the model.")
    train_parser.add_argument(
        "--training-data-path",
        type=str,
        default=os.path.join("datasets", "real_counts.db"),
        help="Path to the '.db' or '.csv' file to be used for training.",
    )
    train_parser.add_argument(
        "--is-data-csv",
        action="store_true",
        help="Optional flag for telling if the data file is '.csv'. Otherwise it is assumed to be '.db' (sqlite).",
    )
    train_parser.add_argument("--study-name", type=str, help="Optional study name for the optuna study.")
    train_parser.add_argument(
        "--storage-url",
        type=str,
        help="URL to the database storage for the optuna study.",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.sarimax"),
        help="Path for the new model to be saved.",
    )
    train_parser.add_argument(
        "--plot-fitted-model", action="store_true", help="Optional flag for plotting the fitted model predictions."
    )
    train_parser.set_defaults(func=train)
    # Add command line arguments for 'pred' subcommand.
    pred_parser = subparser.add_parser("pred", help="Subcommand to predict given the saved model.")
    pred_parser.add_argument(
        "--pred-data-path",
        type=str,
        default=os.path.join("datasets", "real_pred_sarimax.csv"),
        help="Path to the '.csv' file to be used for prediction.",
    )
    pred_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.sarimax"),
        help="Path for the model to be loaded.",
    )
    pred_parser.add_argument(
        "--pred-out-path",
        type=str,
        default=os.path.join("datasets", "real_pred_sarimax_out.csv"),
        help="Path to the '.csv' file to write the prediction.",
    )
    pred_parser.set_defaults(func=predict)
    # Add command line arguments for 'periodic_forecast' subcommand.
    periodic_forecast_parser = subparser.add_parser(
        "periodic_forecast", help="Subcommand for doing periodic forecast and publishing it using MQTT."
    )
    periodic_forecast_parser.add_argument(
        "--training-data-path",
        type=str,
        default=os.path.join("datasets", "real_counts.db"),
        help="Path to the '.db' or '.csv' file to be used for training.",
    )
    periodic_forecast_parser.add_argument(
        "--is-data-csv",
        action="store_true",
        help="Optional flag for telling if the data file is '.csv'. Otherwise it is assumed to be '.db' (sqlite).",
    )
    periodic_forecast_parser.add_argument(
        "--study-name", type=str, help="Optional study name for the optuna study during hyperparameter optimization."
    )
    periodic_forecast_parser.add_argument(
        "--storage-url",
        type=str,
        help="URL to the database storage for the optuna study.",
    )
    periodic_forecast_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.sarimax"),
        help="Path for the new model to be saved.",
    )
    periodic_forecast_parser.add_argument(
        "--pred-data-path",
        type=str,
        default=os.path.join("datasets", "real_pred_sarimax.csv"),
        help="Path to the '.csv' file to be used for prediction.",
    )
    periodic_forecast_parser.add_argument(
        "--pred-out-path",
        type=str,
        default=os.path.join("datasets", "real_pred_sarimax_out.csv"),
        help="Path to the '.csv' file to write the prediction.",
    )
    periodic_forecast_parser.add_argument(
        "--iot-platform-settings-path",
        type=str,
        default=os.path.join("settings", "device_sarimax.json"),
        help="Path to the settings file for building connection to IoT Platform.",
    )
    # TODO: add more arguments for pulling data from Elasticsearch backend
    # and for publishing to the IoT platform using MQTT.
    periodic_forecast_parser.set_defaults(func=periodic_forecast)
