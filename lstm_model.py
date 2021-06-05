import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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

    x_columns = ["last_count", DT_COLUMN, "minute_of_day", "day_of_week", "month_of_year"]
    y_column = [UNIVARIATE_DATA_COLUMN]
    useless_rows = 1

    def __init__(self, data_path, is_data_csv, n_features=5, n_hidden=32, n_layers=1, n_outputs=1, n_batch_size=8, seq_len=4):
        super().__init__()

        self.data_path = data_path
        self.is_data_csv = is_data_csv

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,  # batch of sequences
            num_layers=n_layers,  # stack LSTMs on top of each other
            # dropout = 0.2
        )
        self.regressor = nn.Linear(n_hidden, n_outputs)

        self.criterion = nn.MSELoss()
        self.batch_size = n_batch_size
        self.seq_len = seq_len
        self.test_results = None

    def forward(self, x, labels=None):
        # load to device
        x = x.to(self.device)
        # h_0 and c_0 default to zero
        # get the hidden layers
        lstm_out, (h_n, c_n) = self.lstm(x)
        # get the output of the last hidden layer
        out = lstm_out[:, -1]
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def fit_scaler_to_data(self, ts: pd.DataFrame):
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_features.fit(ts[self.x_columns])
        self.scaler_count = MinMaxScaler(feature_range=(0, 1))
        self.scaler_count.fit(ts[self.y_column])

    def scale_data(self, ts: pd.DataFrame):
        ts.loc[:, self.x_columns] = self.scaler_features.transform(ts.loc[:, self.x_columns])
        ts.loc[:, self.y_column] = self.scaler_count.transform(ts.loc[:, self.y_column])

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
        count = self(sequence)
        return count.item()

    def predict(self, ts: pd.DataFrame) -> numpy.ndarray:
        self.scale_data(ts)

        predictions = []

        # first seq_len + 1 counts are still known
        for i in range(self.seq_len + 1):
            predictions.append(ts[self.y_column].iloc[i].values[0])

        last_count = ts[self.y_column].iloc[self.seq_len].values[0]

        # window starts at second count because first always has NaN data
        for i in range(self.useless_rows, len(ts.index) - self.seq_len):

            # update last count
            ts["last_count"][i + self.seq_len - 1] = last_count

            # get sequence and convert to tensor
            X_sequence = ts[self.x_columns][i : i + self.seq_len].to_numpy()
            X_sequence = torch.tensor(X_sequence).float()
            X_sequence = torch.unsqueeze(X_sequence, 0)

            last_count = self.predict_single(X_sequence)
            predictions.append(last_count)

        predictions = self.scaler_count.inverse_transform(numpy.asarray(predictions).reshape(-1, 1))
        predictions = numpy.round(predictions).astype(int).reshape(-1)
        return predictions

    def plot_after_train(self):
        _, _, ts, _ = load_data_with_features(self.data_path, self.is_data_csv)
        all_target = ts[self.y_column]
        all_pred = pd.Series(self.predict(ts), index=all_target.index, name="predicted_count")
        all_target.index.freq = None
        all_pred.index.freq = None
        all_target.plot(legend=True)
        all_pred.plot(legend=True, linestyle="dotted")
        plt.show()


def train(args: argparse.Namespace) -> StudentCountPredictor:
    print("GPU available: " + str(torch.cuda.is_available()))

    N_EPOCHS = 10

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_directory, filename=args.model_name, save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )
    logger = TensorBoardLogger("tensorboard_logs", name="count-students")

    model = StudentCountPredictor(args.training_data_path, args.is_data_csv)
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback], max_epochs=N_EPOCHS, progress_bar_refresh_rate=30)
    trainer.fit(model)
    if args.plot_fitted_model:
        model.plot_after_train()
    trainer.test(model)

    end_model_path = os.path.join(args.model_directory, args.model_name)
    torch.save(model, end_model_path)

    return model


def prepare_pred_input(pred_time: int, pred_data_path: str, dataset_path: str):
    # TODO: implement this.
    pass


def predict(args: argparse.Namespace, model: Optional[StudentCountPredictor] = None) -> str:
    _, _, ts, _ = load_data_with_features(args.pred_data_path)
    if model is None:
        model: StudentCountPredictor = torch.load(args.model_path)
    model.eval()

    predictions = model.predict(ts)

    all_pred = pd.Series(predictions, index=ts.index, name=UNIVARIATE_DATA_COLUMN)
    all_pred.to_csv(args.pred_out_path, index=False)
    return args.pred_out_path


def periodic_forecast(args: argparse.Namespace):
    # TODO: implement this.
    pass


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--gpus", default=None)
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
        type=bool,
        action="store_true",
        help="Optional flag for telling if the data file is '.csv'. Otherwise it is assumed to be '.db' (sqlite).",
    )
    train_parser.add_argument(
        "--study-name",
        type=str,
        default="lr_fit",
        help="Optional study name for the optuna study during hyperparameter optimization.",
    )
    train_parser.add_argument(
        "--storage-url",
        type=str,
        default="sqlite:///hyperparams/lstm_params.db",
        help="URL to the database storage for the optuna study.",
    )
    train_parser.add_argument(
        "--model-name",
        type=str,
        default="lstm-model",
        help="Name of the new model to be saved.",
    )
    train_parser.add_argument(
        "--model-directory",
        type=str,
        default="checkpoints",
        help="(Relative) directory of the model that will be saved.",
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
        default=os.path.join("datasets", "real_pred_lstm.csv"),
        help="Path to the '.csv' file to be used for prediction.",
    )
    pred_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.ckpt"),
        help="Path for the model to be loaded.",
    )
    pred_parser.add_argument(
        "--pred-out-path",
        type=str,
        default=os.path.join("datasets", "real_pred_lstm_out.csv"),
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
        type=bool,
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
