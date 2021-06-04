import argparse
import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from utils import load_csv_data_with_features


class TimeseriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        return (self.X[index : index + self.seq_len], self.y[index + self.seq_len - 1])


class LSTM(nn.Module):
    def __init__(self, n_features=7, n_hidden=128, n_layers=2, n_outputs=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_hidden = n_hidden
        self.model = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,  # batch of sequences
            num_layers=n_layers,  # stack LSTMs on top of each other
            # dropout = 0.2
        )
        self.regressor = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, labels=None):
        # load to device
        x = x.to(self.device)
        # get the hidden layers
        lstm_out, _ = self.model(x)
        # get the output of the last hidden layer
        out = lstm_out[:, -1]
        # apply linear layer
        return self.regressor(out)


class StudentCountPredictor(pl.LightningModule):
    def __init__(self, lstm_model, n_batch_size=64, seq_len=8):
        super().__init__()

        self.model = lstm_model

        self.criterion = nn.MSELoss()
        self.batch_size = n_batch_size
        self.seq_len = seq_len

    def general_step(self, batch, batch_idx, mode):

        x, y = batch

        # forward pass
        y_hat = self.model.forward(x)

        # loss
        loss = self.criterion(y_hat, y)

        return loss

    def general_end(self, outputs, mode):
        avg_loss = torch.stack([x[mode + "_loss"] for x in outputs]).mean()
        self.test_results = avg_loss
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        # log = {'test_mrr': mrr_metric, 'train_loss': loss}
        # tensorboard_logs = {'loss': loss}
        self.logger.experiment.add_scalars("losses", {"loss": loss}, global_step=self.current_epoch)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "test")
        return {"test_loss": loss}

    def validation_end(self, outputs):
        avg_loss = self.general_end(outputs, "val")
        # tensorboard_logs = {'val_loss': avg_loss}
        log = {"val_loss": avg_loss}
        self.logger.experiment.add_scalars("losses_val", log)
        return {"val_loss": avg_loss, "log": log}

    def test_end(self, outputs):
        avg_loss = self.general_end(outputs, "test")
        # ensorboard_logs = {'test_loss': avg_loss}
        # progress_bar_metrics = tensorboard_logs
        log = {"test_loss": avg_loss}
        self.logger.experiment.add_scalars("losses_test", log)
        return {"test_loss": avg_loss, "log": log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def prepare_data(self, stage=None):

        y_column, x_columns, ts, useless_rows = load_csv_data_with_features(args.training_data_path)

        ts = ts.iloc[useless_rows:]

        train_len, val_len = int(ts.shape[0] * 0.8), int(ts.shape[0] * 0.1)
        train_ts, validation_ts, test_ts = (
            ts.iloc[:train_len],
            ts.iloc[train_len : train_len + val_len],
            ts.iloc[train_len + val_len :],
        )

        X_train = train_ts[x_columns].to_numpy()
        y_train = train_ts[y_column].to_numpy()

        X_val = validation_ts[x_columns].to_numpy()
        y_val = validation_ts[y_column].to_numpy()

        X_test = test_ts[x_columns].to_numpy()
        y_test = test_ts[y_column].to_numpy()

        preprocessing = MinMaxScaler()
        preprocessing.fit(X_train)

        X_train = preprocessing.transform(X_train)
        y_train = y_train.reshape((-1, 1))
        X_val = preprocessing.transform(X_val)
        y_val = y_val.reshape((-1, 1))
        X_test = preprocessing.transform(X_test)
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


def train(args: argparse.Namespace):
    print("GPU available: " + str(torch.cuda.is_available()))

    N_EPOCHS = 10
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", filename="model.lstm", save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )
    logger = TensorBoardLogger("tensorboard_logs", name="count-students")

    model = LSTM()
    model_fit = StudentCountPredictor(model)

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=False,
        # callbacks=[checkpoint_callback],
        max_epochs=N_EPOCHS,
        gpus=0,
        progress_bar_refresh_rate=30,
    )
    trainer.fit(model_fit)
    trainer.test(model_fit)

    with open(args.model_path, "wb") as f:
        pickle.dump(model_fit, f)

    # if args.plot_fitted_model:
    #     y_column, x_columns, ts, useless_rows = load_csv_data_with_features(args.training_data_path)
    #     ts = ts.iloc[useless_rows:]
    #     all_target = ts[y_column]
    #     all_pred = pd.Series(model_fit.predict(ts[x_columns]), index=all_target.index, name="predicted_mean")
    #     all_target.index.freq = None
    #     all_pred.index.freq = None
    #     all_target.plot(legend=True)
    #     all_pred.plot(legend=True)
    #     plt.show()


def predict(args: argparse.Namespace):
    pass


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
        default="sqlite:///hyperparams/lstm_params.db",
        help="URL to the database storage for the optuna study.",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.lstm"),
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
        default=os.path.join("datasets", "real_pred_lstm.csv"),
        help="Path to the .csv file to be used for prediction.",
    )
    pred_parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints", "model.lstm"),
        help="Path for the model to be loaded.",
    )
    pred_parser.set_defaults(func=predict)

    args = parser.parse_args()

    parser.add_argument("--gpus", default=None)
    args.func(args)
    # train(args)
