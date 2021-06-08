# Assignment 5

Try to forecast future count values using **linear regression**, **SARIMAX** and **LSTM** on both the generated data as well as the real collected count data.

## **Installing Dependencies:**
* Install Python3 either system-wide, user-wide or as a virtual environment,
* Run `pip install pip-tools` command via the `pip` command associated with the installed Python,
* Run `pip-sync` inside the project root folder.

## **Usage:**
    usage: main.py [-h] {datagen,linreg,sarimax,lstm} ...

    optional arguments:
      -h, --help            show this help message and exit

    Subcommands:
      {datagen,linreg,sarimax,lstm}
        datagen             Subcommand to generate artificial training data.
        linreg              Subcommand to use linear regression model.      
        sarimax             Subcommand to use SARIMAX model.
        lstm                Subcommand to use LSTM model.

### `datagen` subcommand usage
    usage: main.py datagen [-h] [--out-path OUT_PATH]

    optional arguments:
      -h, --help           show this help message and exit
      --out-path OUT_PATH  Path to the output '.csv' file containing generated
                           data.

### `linreg` subcommand usage
    usage: main.py linreg [-h] {train,pred,periodic_forecast} ...

    optional arguments:
      -h, --help            show this help message and exit

    Subcommands:
      {train,pred,periodic_forecast}
        train               Subcommand to train the model.
        pred                Subcommand to predict given the saved model.
        periodic_forecast   Subcommand for doing periodic forecast and publishing
                            it using MQTT.

#### `linreg train` subcommand usage
    usage: main.py linreg train [-h] [--training-data-path TRAINING_DATA_PATH]
                                [--is-data-csv] [--study-name STUDY_NAME]
                                [--storage-url STORAGE_URL]
                                [--model-path MODEL_PATH] [--plot-fitted-model]

    optional arguments:
      -h, --help            show this help message and exit
      --training-data-path TRAINING_DATA_PATH
                            Path to the '.db' or '.csv' file to be used for
                            training.
      --is-data-csv         Optional flag for telling if the data file is '.csv'.
                            Otherwise it is assumed to be '.db' (sqlite).
      --study-name STUDY_NAME
                            Optional study name for the optuna study during
                            hyperparameter optimization.
      --storage-url STORAGE_URL
                            URL to the database storage for the optuna study.
      --model-path MODEL_PATH
                            Path for the new model to be saved.
      --plot-fitted-model   Optional flag for plotting the fitted model
                            predictions.

#### `linreg pred` subcommand usage
    usage: main.py linreg pred [-h] [--pred-data-path PRED_DATA_PATH]
                               [--model-path MODEL_PATH]
                               [--pred-out-path PRED_OUT_PATH]

    optional arguments:
      -h, --help            show this help message and exit
      --pred-data-path PRED_DATA_PATH
                            Path to the '.csv' file to be used for prediction.
      --model-path MODEL_PATH
                            Path for the model to be loaded.
      --pred-out-path PRED_OUT_PATH
                            Path to the '.csv' file to write the prediction.

### `sarimax` subcommand usage
    usage: main.py sarimax [-h] {train,pred,periodic_forecast} ...

    optional arguments:
      -h, --help            show this help message and exit

    Subcommands:
      {train,pred,periodic_forecast}
        train               Subcommand to train the model.
        pred                Subcommand to predict given the saved model.
        periodic_forecast   Subcommand for doing periodic forecast and publishing
                            it using MQTT.

#### `sarimax train` subcommand usage
    usage: main.py sarimax train [-h] [--training-data-path TRAINING_DATA_PATH]
                                 [--is-data-csv] [--study-name STUDY_NAME]
                                 [--storage-url STORAGE_URL]
                                 [--model-path MODEL_PATH] [--plot-fitted-model]

    optional arguments:
      -h, --help            show this help message and exit
      --training-data-path TRAINING_DATA_PATH
                            Path to the '.db' or '.csv' file to be used for
                            training.
      --is-data-csv         Optional flag for telling if the data file is '.csv'.
                            Otherwise it is assumed to be '.db' (sqlite).
      --study-name STUDY_NAME
                            Optional study name for the optuna study.
      --storage-url STORAGE_URL
                            URL to the database storage for the optuna study.
      --model-path MODEL_PATH
                            Path for the new model to be saved.
      --plot-fitted-model   Optional flag for plotting the fitted model
                            predictions.

#### `sarimax pred` subcommand usage
    usage: main.py sarimax pred [-h] [--pred-data-path PRED_DATA_PATH]
                                [--model-path MODEL_PATH]
                                [--pred-out-path PRED_OUT_PATH]

    optional arguments:
      -h, --help            show this help message and exit
      --pred-data-path PRED_DATA_PATH
                            Path to the '.csv' file to be used for prediction.
      --model-path MODEL_PATH
                            Path for the model to be loaded.
      --pred-out-path PRED_OUT_PATH
                            Path to the '.csv' file to write the prediction.

### `lstm` subcommand usage
    usage: main.py lstm [-h] {train,pred,periodic_forecast} ...

    optional arguments:
      -h, --help            show this help message and exit

    Subcommands:
      {train,pred,periodic_forecast}
        pred                Subcommand to predict given the saved model.
        periodic_forecast   Subcommand for doing periodic forecast and publishing
                            it using MQTT.

#### `lstm train` subcommand usage
    usage: main.py lstm train [-h] [--training-data-path TRAINING_DATA_PATH]
                              [--is-data-csv] [--model-path MODEL_PATH]
                              [--plot-fitted-model] [--using-ray]

    optional arguments:
      -h, --help            show this help message and exit
      --training-data-path TRAINING_DATA_PATH
                            Path to the '.db' or '.csv' file to be used for
                            training.
      --is-data-csv         Optional flag for telling if the data file is '.csv'.
                            Otherwise it is assumed to be '.db' (sqlite).
      --model-path MODEL_PATH
                            Path for the new model to be saved.
      --plot-fitted-model   Optional flag for plotting the fitted model
                            predictions.
      --using-ray           Optional flag for using training with RayTune.

#### `lstm pred` subcommand usage
    usage: main.py lstm pred [-h] [--pred-data-path PRED_DATA_PATH]
                             [--model-path MODEL_PATH]
                             [--pred-out-path PRED_OUT_PATH]
                             [--plot-predicted-model]

    optional arguments:
      -h, --help            show this help message and exit
      --pred-data-path PRED_DATA_PATH
                            Path to the '.csv' file to be used for prediction.
      --model-path MODEL_PATH
                            Path for the model to be loaded.
      --pred-out-path PRED_OUT_PATH
                            Path to the '.csv' file to write the prediction.
      --plot-predicted-model
                            Optional flag for plotting the predicted model.
