import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from utils import load_data_with_features
from utils.data import DT_COLUMN, UNIVARIATE_DATA_COLUMN

pd.options.mode.chained_assignment = None


class TimeseriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        return (self.X[index : index + self.seq_len], self.y[index + self.seq_len - 1])


class StudentCountPredictor(pl.LightningModule):

    x_columns = ["lag1_count", DT_COLUMN, "minute_of_day", "day_of_week", "month_of_year"]
    y_column = [UNIVARIATE_DATA_COLUMN]
    useless_rows = 1

    def __init__(self, config, data_path, is_data_csv):
        super().__init__()

        self.data_path = data_path
        self.is_data_csv = is_data_csv

        self.lr = config["lr"]

        self.n_features = config["n_features"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.n_outputs = config["n_outputs"]

        self.batch_size = config["batch_size"]
        self.seq_len = config["seq_len"]

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.n_hidden,
            batch_first=True,  # batch of sequences
            num_layers=self.n_layers,  # stack LSTMs on top of each other
            # dropout = 0.2
        )
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.n_hidden, self.n_hidden)
        self.regressor = nn.Linear(self.n_hidden, self.n_outputs)
        self.criterion = nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, labels=None):
        # load to device
        x = x.to(self.device)
        # h_0 and c_0 default to zero
        # get the hidden layers
        lstm_out, (h_n, c_n) = self.lstm(x)
        # get the output of the last hidden layer
        out = lstm_out[:, -1]
        out = self.linear(out)
        out = self.relu(out)
        # apply linear layer
        return self.regressor(out)

    def general_step(self, batch, batch_idx, mode):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "test")
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        return {"val_loss": avg_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss", avg_loss)
        return {"test_loss": avg_loss}

    def fit_scaler_to_data(self, ts: pd.DataFrame):
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_features.fit(ts[self.x_columns])
        self.scaler_count = MinMaxScaler(feature_range=(0, 1))
        self.scaler_count.fit(ts[self.y_column])

    def scale_data(self, ts: pd.DataFrame):
        ts.loc[:, self.x_columns] = self.scaler_features.transform(ts.loc[:, self.x_columns])
        ts.loc[:, self.y_column] = self.scaler_count.transform(ts.loc[:, self.y_column])

    def inverse_data(self, ts: pd.DataFrame):
        ts.loc[:, self.x_columns] = self.scaler_features.inverse_transform(ts.loc[:, self.x_columns])
        ts.loc[:, self.y_column] = self.scaler_count.inverse_transform(ts.loc[:, self.y_column])

    def prepare_data(self, stage=None):

        _, _, ts, _ = load_data_with_features(self.data_path, self.is_data_csv)
        # TODO: refactor util method to customize

        ts = ts.iloc[self.useless_rows :]

        train_len, val_len = int(ts.shape[0] * 0.8), int(ts.shape[0] * 0.1)
        train_ts, validation_ts, test_ts = (
            ts.iloc[:train_len],
            ts.iloc[train_len : train_len + val_len],
            ts.iloc[train_len + val_len :],
        )

        self.fit_scaler_to_data(train_ts)
        self.scale_data(train_ts)
        self.scale_data(validation_ts)
        self.scale_data(test_ts)

        X_train = train_ts[self.x_columns].to_numpy()
        y_train = train_ts[self.y_column].to_numpy()

        X_val = validation_ts[self.x_columns].to_numpy()
        y_val = validation_ts[self.y_column].to_numpy()

        X_test = test_ts[self.x_columns].to_numpy()
        y_test = test_ts[self.y_column].to_numpy()

        y_train = y_train.reshape((-1, 1))
        y_val = y_val.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))

        training_dataset = TimeseriesDataset(X_train, y_train, seq_len=self.seq_len)
        validation_dataset = TimeseriesDataset(X_val, y_val, seq_len=self.seq_len)
        test_dataset = TimeseriesDataset(X_test, y_test, seq_len=self.seq_len)

        self.dataset = {"train": training_dataset, "val": validation_dataset, "test": test_dataset}

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.batch_size)

    def predict_single(self, sequence: torch.Tensor) -> float:
        return self(sequence).item()

    def predict(self, ts: pd.DataFrame, update_lag1_count=True):

        self.scale_data(ts)

        # first seq_len + 1 counts are still known
        for i in range(self.useless_rows, self.seq_len + 1):
            if update_lag1_count:
                ts.loc[ts.index[i], "lag1_count"] = ts.loc[ts.index[i - 1], "count"]

        # window starts at second count because first always has NaN data
        for i in range(self.useless_rows, len(ts.index) - self.seq_len):

            # update last count
            if update_lag1_count:
                last_count = ts.loc[ts.index[i + self.seq_len - 1], "count"]
                ts.loc[ts.index[i + self.seq_len], "lag1_count"] = last_count

            # get sequence and convert to tensor
            X_sequence = ts.loc[ts.index[i] : ts.index[i + self.seq_len], self.x_columns].to_numpy()
            X_sequence = torch.tensor(X_sequence).float()
            X_sequence = torch.unsqueeze(X_sequence, 0)

            # predict
            value = self.predict_single(X_sequence)
            ts.loc[ts.index[i + self.seq_len], "count"] = value

        self.inverse_data(ts)
        ts["count"] = ts["count"].round(decimals=0)
        ts["count"] = ts["count"].astype(np.int)

    def plot_after_train(self):
        _, _, ts, _ = load_data_with_features(self.data_path, self.is_data_csv)
        all_target = ts[self.y_column]
        self.predict(ts, update_lag1_count=True)
        all_pred = pd.Series(ts[self.y_column].values.reshape(-1), index=all_target.index, name="predicted_count")
        all_target.index.freq = None
        all_pred.index.freq = None
        all_target.plot(legend=True)
        all_pred.plot(legend=True, linestyle="dotted")
        plt.show()


def train_analysis(config, data_path, is_csv_data, num_epochs):
    model = StudentCountPredictor(config, data_path, is_csv_data)
    tune_callback = TuneReportCallback({"loss": "val_loss"}, on="validation_end")
    trainer = pl.Trainer(
        logger=TensorBoardLogger("tensorboard_logs", name="count-students"),
        callbacks=[tune_callback],
        max_epochs=num_epochs,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(model)


def train_with_ray_tune(args: argparse.Namespace):

    # services.get_node_ip_address = lambda: '127.0.0.1'
    # ray.init(local_mode=True, include_dashboard=False, num_cpus=1, logging_level=logging.DEBUG, num_redis_shards=0)

    analysis = tune.run(
        tune.with_parameters(train_analysis, data_path=args.training_data_path, is_data_csv=args.is_data_csv, num_epochs=10),
        metric="loss",
        mode="min",
        config={
            "n_features": 5,
            "n_layers": 1,
            "n_hidden": tune.choice([8, 16, 32]),
            "n_outputs": 1,
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([8, 16, 32]),
            "seq_len": tune.choice([2, 4, 8]),
        },
        num_samples=10,
        progress_reporter=tune.CLIReporter(
            parameter_columns=["n_hidden", "lr", "batch_size", "seq_len"], metric_columns=["loss", "training_iteration"]
        ),
        name="tune_scp",
    )
    print(analysis.best_config)


def train_standard(args: argparse.Namespace):
    print("GPU available: " + str(torch.cuda.is_available()))

    N_EPOCHS = 10

    logger = TensorBoardLogger("tensorboard_logs", name="count-students")

    config = {"n_features": 5, "n_layers": 2, "n_hidden": 32, "n_outputs": 1, "lr": 0.001, "batch_size": 4, "seq_len": 4}

    model = StudentCountPredictor(config, args.training_data_path, args.is_data_csv)

    trainer = pl.Trainer(logger=logger, max_epochs=N_EPOCHS, progress_bar_refresh_rate=30)

    trainer.fit(model)
    if args.plot_fitted_model:
        model.plot_after_train()
    trainer.test(model)

    torch.save(model, args.model_path)


def train(args: argparse.Namespace):
    if args.using_ray:
        train_with_ray_tune(args)
    else:
        train_standard(args)


def prepare_pred_input(pred_time: int, pred_data_path: str, dataset_path: str):
    # TODO: implement this.
    pass


def predict(args: argparse.Namespace, model: Optional[StudentCountPredictor] = None) -> str:
    _, _, ts, _ = load_data_with_features(args.pred_data_path)
    if model is None:
        model: StudentCountPredictor = torch.load(args.model_path)
    model.eval()

    model.predict(ts)
    predictions = ts[model.y_column].values.reshape(-1)

    all_pred = pd.Series(predictions, index=ts.index, name=UNIVARIATE_DATA_COLUMN)

    if args.plot_predicted_model:
        all_pred.index.freq = None
        all_pred.plot(legend=True, linestyle="dotted")
        plt.show()

    all_pred.to_csv(args.pred_out_path, index=False)
    return args.pred_out_path


def periodic_forecast(args: argparse.Namespace):
    # TODO: implement this.
    pass


def add_arguments(parser: argparse.ArgumentParser):
    subparser = parser.add_subparsers(title="Subcommands")
    train_parser = subparser.add_parser("train")
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
    # TODO: using raytune
    # train_parser.add_argument(
    #     "--study-name",
    #     type=str,
    #     default="lr_fit",
    #     help="Optional study name for the optuna study during hyperparameter optimization.",
    # )
    # train_parser.add_argument("--study-name", type=str, default="lr_fit", help="Optional study name for the optuna study.")
    # train_parser.add_argument(
    #     "--storage-url",
    #     type=str,
    #     default="sqlite:///hyperparams/lstm_params.db",
    #     help="URL to the database storage for the optuna study.",
    # )
    # train_parser.add_argument(
    #     "--model-name",
    #     type=str,
    #     default="lstm-model",
    #     help="Name of the new model to be saved.",
    # )
    # train_parser.add_argument(
    #     "--model-directory",
    #     type=str,
    #     default="checkpoints",
    #     help="(Relative) directory of the model that will be saved.",
    # )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.lstm"),
        help="Path for the new model to be saved.",
    )
    train_parser.add_argument(
        "--plot-fitted-model", action="store_true", help="Optional flag for plotting the fitted model predictions."
    )
    train_parser.add_argument("--using-ray", action="store_true", help="Optional flag for using training with RayTune.")
    train_parser.set_defaults(func=train)

    # Add command line arguments for 'pred' subcommand.
    pred_parser = subparser.add_parser("pred", help="Subcommand to predict given the saved model.")
    pred_parser.add_argument(
        "--pred-data-path",
        type=str,
        default=os.path.join("datasets", "real_pred_lstm.csv"),
        help="Path to the '.csv' file to be used for prediction.",
    )
    pred_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.lstm"),
        help="Path for the model to be loaded.",
    )
    pred_parser.add_argument(
        "--pred-out-path",
        type=str,
        default=os.path.join("datasets", "real_pred_lstm_out.csv"),
        help="Path to the '.csv' file to write the prediction.",
    )
    pred_parser.add_argument(
        "--plot-predicted-model", action="store_true", help="Optional flag for plotting the predicted model."
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
        default="sqlite:///hyperparams/params.db",
        help="URL to the database storage for the optuna study.",
    )
    periodic_forecast_parser.add_argument(
        "--model-name",
        type=str,
        default="lstm-model",
        help="Name of the new model to be saved.",
    )
    periodic_forecast_parser.add_argument(
        "--model-directory",
        type=str,
        default="checkpoints",
        help="(Relative) directory of the model that will be saved.",
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
    # TODO: add more arguments for pulling data from Elasticsearch backend
    # and for publishing to the IoT platform using MQTT.
    periodic_forecast_parser.set_defaults(func=periodic_forecast)
