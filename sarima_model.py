import argparse
import os
import threading

import matplotlib.pyplot as plt
import numpy as np
import optuna
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.base.wrapper import ResultsWrapper

from utils import load_csv_data


def train(args: argparse.Namespace):
    y_column = "count"
    exog_columns, ts = load_csv_data(args.training_data_path)
    columns_to_convert = [y_column] + exog_columns
    ts[columns_to_convert] = ts[columns_to_convert].astype(np.float32)

    training_len = int(ts.shape[0] * 0.9)
    train_ts, test_ts = ts.iloc[:training_len], ts.iloc[training_len:]

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
        forecast_model = sm.tsa.statespace.SARIMAX(train_ts[y_column], exog=train_ts[exog_columns], trend="c", order=(p, d, q))
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
    forecast_model = sm.tsa.statespace.SARIMAX(train_ts[y_column], exog=train_ts[exog_columns], trend="c", order=(p, d, q))
    model_fit = forecast_model.fit(disp=False)
    model_fit.save(args.model_path)

    if args.plot_fitted_model:
        all_target = ts[y_column]
        all_pred = model_fit.predict(start=0, end=ts.shape[0] - 1, exog=test_ts[exog_columns])
        all_target.index.freq = None
        all_pred.index.freq = None
        all_target.plot(legend=True)
        all_pred.plot(legend=True)
        plt.show()


def predict(args: argparse.Namespace):
    exog_columns, ts = load_csv_data(args.pred_data_path)
    ts[exog_columns] = ts[exog_columns].astype(np.float32)

    model_fit = ResultsWrapper.load(args.model_path)
    pred = model_fit.forecast(ts.shape[0], exog=ts[exog_columns])
    print(np.round(pred.to_numpy()).astype(int).tolist())


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
    train_parser.add_argument(
        "--study-name", type=str, default="sarimax_fit", help="Optional study name for the optuna study."
    )
    train_parser.add_argument(
        "--storage-url",
        type=str,
        default="sqlite:///hyperparams/params.db",
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

    pred_parser = subparser.add_parser("pred")
    pred_parser.add_argument(
        "--pred-data-path",
        type=str,
        default=os.path.join("datasets", "pred_sarimax.csv"),
        help="Path to the .csv file to be used for prediction.",
    )
    pred_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.sarimax"),
        help="Path for the model to be loaded.",
    )
    pred_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)
