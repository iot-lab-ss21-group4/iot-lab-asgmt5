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
    usage: main.py linreg [-h] {train,pred} ...

    optional arguments:
      -h, --help    show this help message and exit

    Subcommands:
      {train,pred}
        train       Subcommand to train the model.
        pred        Subcommand to predict given the saved model.

#### `linreg train` subcommand usage
usage: main.py linreg train [-h] [--training-data-path TRAINING_DATA_PATH]
                            [--study-name STUDY_NAME]
                            [--storage-url STORAGE_URL]
                            [--model-path MODEL_PATH] [--plot-fitted-model]

optional arguments:
  -h, --help            show this help message and exit
  --training-data-path TRAINING_DATA_PATH
                        Path to the .csv file to be used for training.
  --study-name STUDY_NAME
                        Optional study name for the optuna study.
  --storage-url STORAGE_URL
                        URL to the database storage for the optuna study.
  --model-path MODEL_PATH
                        Path for the new model to be saved.
  --plot-fitted-model   Optional flag for plotting the fitted model
                        predictions.

#### `linreg pred` subcommand usage
usage: main.py linreg pred [-h] [--pred-data-path PRED_DATA_PATH]
                           [--model-path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --pred-data-path PRED_DATA_PATH
                        Path to the .csv file to be used for prediction.
  --model-path MODEL_PATH
                        Path for the model to be loaded.

### `sarimax` subcommand usage
    usage: main.py sarimax [-h] {train,pred} ...

    optional arguments:
      -h, --help    show this help message and exit

    Subcommands:
      {train,pred}
        train       Subcommand to train the model.
        pred        Subcommand to predict given the saved model.

#### `sarimax train` subcommand usage
usage: main.py sarimax train [-h] [--training-data-path TRAINING_DATA_PATH] 
                             [--study-name STUDY_NAME]
                             [--storage-url STORAGE_URL]
                             [--model-path MODEL_PATH] [--plot-fitted-model]

optional arguments:
  -h, --help            show this help message and exit
  --training-data-path TRAINING_DATA_PATH
                        Path to the .csv file to be used for training.      
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

    optional arguments:
      -h, --help            show this help message and exit
      --pred-data-path PRED_DATA_PATH
                            Path to the .csv file to be used for prediction.
      --model-path MODEL_PATH
                            Path for the model to be loaded.

### `lstm` subcommand usage
    usage: main.py lstm [-h] [--gpus GPUS] {train,pred} ...

    optional arguments:
      -h, --help    show this help message and exit
      --gpus GPUS

    Subcommands:
      {train,pred}
        train       Subcommand to train the model.
        pred        Subcommand to predict given the saved model.

#### `lstm train` subcommand usage
    usage: main.py lstm train [-h] [--training-data-path TRAINING_DATA_PATH]
                              [--study-name STUDY_NAME]
                              [--storage-url STORAGE_URL]
                              [--model-path MODEL_PATH] [--plot-fitted-model]

    optional arguments:
      -h, --help            show this help message and exit
      --training-data-path TRAINING_DATA_PATH
                            Path to the .csv file to be used for training.
      --study-name STUDY_NAME
                            Optional study name for the optuna study.
      --storage-url STORAGE_URL
                            URL to the database storage for the optuna study.
      --model-path MODEL_PATH
                            Path for the new model to be saved.
      --plot-fitted-model   Optional flag for plotting the fitted model
                            predictions.

#### `lstm pred` subcommand usage
    usage: main.py lstm pred [-h] [--pred-data-path PRED_DATA_PATH]
                             [--model-path MODEL_PATH]

    optional arguments:
      -h, --help            show this help message and exit
      --pred-data-path PRED_DATA_PATH
                            Path to the .csv file to be used for prediction.
      --model-path MODEL_PATH
                            Path for the model to be loaded.
