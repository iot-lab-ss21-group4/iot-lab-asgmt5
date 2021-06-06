import argparse
import itertools
import os
import pickle
from typing import Optional

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from sklearn.linear_model import Lasso

from utils import load_data_with_features, start_periodic_forecast
from utils.data import DERIVATIVE_COLUMN, LAG_FEATURE_TEMPLATE, LAG_ORDER, TIME_COLUMN


def train(args: argparse.Namespace) -> Lasso:
    y_column, x_columns, ts, useless_rows = load_data_with_features(
        args.training_data_path, args.is_data_csv, detailed_seasonality=False
    )
    ts = ts.iloc[useless_rows:]

    train_len = int(ts.shape[0] * 0.9)
    train_ts, test_ts = ts.iloc[:train_len], ts.iloc[train_len:]

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_uniform("alpha", 0.0, 1.0)
        forecast_model = Lasso(alpha=alpha)
        model_fit = forecast_model.fit(train_ts[x_columns].to_numpy(), train_ts[y_column].to_numpy())
        return model_fit.score(test_ts[x_columns].to_numpy(), test_ts[y_column].to_numpy())

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage_url,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50, n_jobs=1)
    alpha = study.best_params["alpha"]
    forecast_model = Lasso(alpha=alpha)
    model_fit = forecast_model.fit(train_ts[x_columns].to_numpy(), train_ts[y_column].to_numpy())
    with open(args.model_path, "wb") as f:
        pickle.dump(model_fit, f)

    if args.plot_fitted_model:
        all_target = ts[y_column]
        all_pred = pd.Series(model_fit.predict(ts[x_columns]), index=all_target.index, name="predicted_mean")
        all_target.index.freq = None
        all_pred.index.freq = None
        all_target.plot(legend=True)
        all_pred.plot(legend=True)
        plt.show()

    return model_fit


def prepare_pred_input(
    pred_time: int, forecast_period: int, pred_data_path: str, training_dataset_path: str, is_data_csv: bool
):
    y_column, x_columns, ts, useless_rows = load_data_with_features(
        training_dataset_path, is_csv=is_data_csv, detailed_seasonality=False
    )
    last_t = ts.loc[ts.index[-1], TIME_COLUMN]
    forecast_times = []
    for t in itertools.count(pred_time, -forecast_period):
        if t <= last_t:
            break
        forecast_times.append(t)
    forecast_times = list(reversed(forecast_times))
    forecast_len = len(forecast_times)
    pred_data = pd.DataFrame([[0, None]], index=range(forecast_len + useless_rows), columns=[TIME_COLUMN, y_column])
    pred_data.iloc[:useless_rows] = ts.loc[ts.index[-useless_rows:], [TIME_COLUMN, y_column]]
    pred_data.loc[pred_data.index[-forecast_len:], TIME_COLUMN] = forecast_times
    pred_data.to_csv(pred_data_path, index=False)


def predict(args: argparse.Namespace, model_fit: Optional[Lasso] = None) -> str:
    # Assumption: The first 'useless_rows' many rows were filled with count.
    y_column, x_columns, ts, useless_rows = load_data_with_features(args.pred_data_path, detailed_seasonality=False)
    if model_fit is None:
        with open(args.model_path, "rb") as f:
            model_fit: Lasso = pickle.load(f)

    for i in range(useless_rows, ts.shape[0]):
        pred_i = model_fit.predict(ts[x_columns].iloc[i : i + 1])
        ts.loc[ts.index[i], y_column] = pred_i
        for lag in range(1, LAG_ORDER + 1):
            if i + lag >= ts.shape[0]:
                break
            ts.loc[ts.index[i + lag], LAG_FEATURE_TEMPLATE.format(lag)] = pred_i
        for lag in range(1, 3):
            if i < 1 or i + lag >= ts.shape[0]:
                break
            ts.loc[ts.index[i + lag], DERIVATIVE_COLUMN] = (
                ts.loc[ts.index[i + lag - 1], y_column] - ts.loc[ts.index[i + lag - 2], y_column]
            ) / (ts.loc[ts.index[i + lag - 1], TIME_COLUMN] - ts.loc[ts.index[i + lag - 2], TIME_COLUMN])

    ts.loc[ts.index[useless_rows:], y_column].round().to_csv(args.pred_out_path, index=False)
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
        {
            "forecast_period": forecast_period,
            "pred_data_path": args.pred_data_path,
            "training_dataset_path": args.training_dataset_path,
        },
        predict,
        {
            "pred_data_path": args.pred_data_path,
            "model_path": args.model_path,
            "pred_out_path": args.pred_out_path,
            "is_data_csv": args.is_data_csv,
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
    train_parser.add_argument(
        "--study-name", type=str, help="Optional study name for the optuna study during hyperparameter optimization."
    )
    train_parser.add_argument(
        "--storage-url",
        type=str,
        help="URL to the database storage for the optuna study.",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.lr"),
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
        default=os.path.join("datasets", "real_pred_lr.csv"),
        help="Path to the '.csv' file to be used for prediction.",
    )
    pred_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.lr"),
        help="Path for the model to be loaded.",
    )
    pred_parser.add_argument(
        "--pred-out-path",
        type=str,
        default=os.path.join("datasets", "real_pred_lr_out.csv"),
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
        default=os.path.join("settings", "device_lr.json"),
        help="Path to the settings file for building connection to IoT Platform.",
    )
    # TODO: add more arguments for pulling data from Elasticsearch backend
    # and for publishing to the IoT platform using MQTT.
    periodic_forecast_parser.set_defaults(func=periodic_forecast)
