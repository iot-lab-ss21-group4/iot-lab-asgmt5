import argparse
import os
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Lasso

from utils import extract_features, load_csv_data
from utils.data import DEFAULT_FLOAT_TYPE, DERIVATIVE_COLUMN, LAG_FEATURE_TEMPLATE, TIME_COLUMN, UNIVARIATE_DATA_COLUMN

# Lag order must be >= 1.
LAG_ORDER = 2


def load_data(path: str) -> Tuple[str, List[str], pd.DataFrame, int]:
    y_column, x_columns = UNIVARIATE_DATA_COLUMN, [TIME_COLUMN]
    cols_added, ts = load_csv_data(path)
    x_columns.extend(cols_added)
    ts[y_column] = ts[y_column].astype(DEFAULT_FLOAT_TYPE)

    cols_added, ts, useless_rows = extract_features(ts, lag_order=LAG_ORDER)
    x_columns.extend(cols_added)

    return y_column, x_columns, ts, useless_rows


def train(args: argparse.Namespace):
    y_column, x_columns, ts, useless_rows = load_data(args.training_data_path)
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


def predict(args: argparse.Namespace):
    # Assumption: The first 'useless_rows' many rows were filled with count.
    y_column, x_columns, ts, useless_rows = load_data(args.pred_data_path)
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

    print(np.round(ts.loc[ts.index[LAG_ORDER:], y_column].to_numpy()).astype(int).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    train_parser = subparser.add_parser("train")
    train_parser.add_argument(
        "--training-data-path",
        type=str,
        default=os.path.join("datasets", "out.csv"),
        help="Path to the .csv file to be used for training.",
    )
    train_parser.add_argument("--study-name", type=str, default="lr_fit", help="Optional study name for the optuna study.")
    train_parser.add_argument(
        "--storage-url",
        type=str,
        default="sqlite:///hyperparams/params.db",
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

    pred_parser = subparser.add_parser("pred")
    pred_parser.add_argument(
        "--pred-data-path",
        type=str,
        default=os.path.join("datasets", "real_pred_lr.csv"),
        help="Path to the .csv file to be used for prediction.",
    )
    pred_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.lr"),
        help="Path for the model to be loaded.",
    )
    pred_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)
