# Assignment 5

Try to forecast future count values using **linear regression**, **SARIMAX** and **LSTM** on both the generated data as well as the real collected count data.

## **Installing Dependencies:**
* Install Python3 either system-wide, user-wide or as a virtual environment,
* Run `pip install pip-tools` command via the `pip` command associated with the installed Python,
* Run `pip-sync` inside the project root folder.

## **Usage:**

### `data_generator.py`
    usage: data_generator.py [-h] [--out-path OUT_PATH]

    optional arguments:
    -h, --help           show this help message and exit
    --out-path OUT_PATH  Path to the output '.csv' file containing generated
                        data.

### `sarima_model.py`
    usage: sarima_model.py [-h] {train,pred} ...

    positional arguments:
    {train,pred}

    optional arguments:
    -h, --help    show this help message and exit

### `sarima_model.py train`
    usage: sarima_model.py train [-h] [--training-data-path TRAINING_DATA_PATH]
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

### `sarima_model.py pred`
    usage: sarima_model.py pred [-h] [--pred-data-path PRED_DATA_PATH]      
                                [--model-path MODEL_PATH]

    optional arguments:
    -h, --help            show this help message and exit
    --pred-data-path PRED_DATA_PATH
                            Path to the .csv file to be used for prediction.
    --model-path MODEL_PATH
                            Path for the model to be loaded.
