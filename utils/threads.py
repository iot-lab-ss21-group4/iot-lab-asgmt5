import argparse
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from utils.publisher import Publisher, setup_publisher


class DataPullerThread(threading.Thread):
    def __init__(self, dataset_path: str, event_out_q: queue.Queue):
        super().__init__()
        self.dataset_path = dataset_path
        self.event_out_q = event_out_q

    def run(self):
        # TODO: implement data pulling from the Elasticsearch backend.
        # Then, notify the model training thread about the new data.
        while True:
            self.event_out_q.put(None)
            time.sleep(600)


class ModelTrainerThread(threading.Thread):
    CAN_TRAIN_COND = "can_train"
    NEW_DATA_AVAILABLE_COND = "new_data_available"

    def __init__(
        self,
        train_fn: Callable[[argparse.Namespace], Any],
        train_fn_kwargs: Dict[str, Any],
        event_in_q: queue.Queue,
        event_out_q: queue.Queue,
        train_period: int = 86400,
    ):
        super().__init__()
        self.train_fn = train_fn
        self.train_fn_kwargs = train_fn_kwargs
        self.event_in_q = event_in_q
        self.event_out_q = event_out_q
        self.train_period = train_period

        self._condition_q = queue.Queue()
        threading.Thread(target=self._check_data_pull_events).start()
        self._next_train_enable_time = time.time() + self.train_period
        threading.Timer(max(0.0, self._next_train_enable_time - time.time()), self._enable_training).start()

    def _check_data_pull_events(self):
        while True:
            _ = self.event_in_q.get()
            self._condition_q.put(ModelTrainerThread.NEW_DATA_AVAILABLE_COND)

    def _enable_training(self):
        self._condition_q.put(ModelTrainerThread.CAN_TRAIN_COND)
        # Using the delaying method below is not affected by time drift.
        self._next_train_enable_time += self.train_period
        threading.Timer(max(0.0, self._next_train_enable_time - time.time()), self._enable_training).start()

    def run(self):
        condition_states = {
            ModelTrainerThread.CAN_TRAIN_COND: False,
            ModelTrainerThread.NEW_DATA_AVAILABLE_COND: False,
        }
        while True:
            cond = self._condition_q.get()
            condition_states[cond] = True
            if not all(condition_states.values()):
                continue
            # Drain the queue so that we don't risk training the model twice in a succession.
            while not self._condition_q.empty():
                _ = self._condition_q.get()
            # Clear all the condition states.
            for key in condition_states.keys():
                condition_states[key] = False
            # Run the actual training of the model.
            model_fit = self.train_fn(argparse.Namespace(**self.train_fn_kwargs))
            # Notify the periodic forecaster of the updated model.
            self.event_out_q.put(model_fit)


class PeriodicForecasterThread(threading.Thread):
    def __init__(
        self,
        pred_input_prepare_fn: Callable[..., None],
        pred_input_prepare_fn_kwargs: Dict[str, Any],
        pred_fn: Callable[[argparse.Namespace, Optional[Any]], str],
        pred_fn_kwargs: Dict[str, Any],
        event_in_q: queue.Queue,
        event_out_q: queue.Queue,
        forecast_period: int = 150,
        forecast_dt: int = 300,
    ):
        super().__init__()
        self.pred_input_prepare_fn = pred_input_prepare_fn
        self.pred_input_prepare_fn_kwargs = pred_input_prepare_fn_kwargs
        self.pred_fn = pred_fn
        self.pred_fn_kwargs = pred_fn_kwargs
        self.event_in_q = event_in_q
        self.event_out_q = event_out_q
        self.forecast_period = forecast_period
        self.forecast_dt = forecast_dt
        self.model_fit: Optional[Any] = None

        self.event_in_q.put(None)
        self._next_forecast_time = time.time() + self.forecast_period
        threading.Timer(max(0.0, self._next_train_enable_time - time.time()), self._nofity_for_forecast).start()

    def _nofity_for_forecast(self):
        self.event_in_q.put(None)
        # Using the delaying method below is not affected by time drift.
        self._next_forecast_time += self.forecast_period
        threading.Timer(max(0.0, self._next_train_enable_time - time.time()), self._nofity_for_forecast).start()

    def run(self):
        while True:
            model_fit: Optional[Any] = self.event_in_q.get()
            self.model_fit = self.model_fit if model_fit is None else model_fit
            # Calculate the next prediction time.
            pred_time = int(time.time()) + self.forecast_dt
            # Prepare the prediction input '.csv' file.
            self.pred_input_prepare_fn_kwargs["pred_time"] = pred_time
            self.pred_input_prepare_fn(**self.pred_input_prepare_fn_kwargs)
            # Predict (forecast) the room count.
            pred_out_path = self.pred_fn(argparse.Namespace(**self.pred_fn_kwargs), self.model_fit)
            # Send (t, y) to the publisher thread.
            pred_y = pd.read_csv(pred_out_path)
            self.event_out_q.put((pred_time, int(pred_y.iloc[-1, 0])))


class ForecastPublisherThread(threading.Thread):
    def __init__(self, event_in_q: queue.Queue, publisher: Publisher):
        super().__init__()
        self.event_in_q = event_in_q
        self.publisher = publisher

    def run(self):
        while True:
            t, y = self.event_in_q.get()
            # msec timestamps
            t = t * 1000
            self.publisher.publish(t, y)


def start_periodic_forecast(
    dataset_path: str,
    train_fn: Callable[[argparse.Namespace], Any],
    train_fn_kwargs: Dict[str, Any],
    pred_input_prepare_fn: Callable[..., None],
    pred_input_prepare_fn_kwargs: Dict[str, Any],
    pred_fn: Callable[[argparse.Namespace, Optional[Any]], str],
    pred_fn_kwargs: Dict[str, Any],
    iot_platform_settings_path: str,
    train_period: int = 86400,
    forecast_period: int = 150,
    forecast_dt: int = 300,
):
    data_puller_out_q = queue.Queue()
    model_trainer_out_q = queue.Queue()
    periodic_forecaster_out_q = queue.Queue()
    data_puller = DataPullerThread(dataset_path, data_puller_out_q)
    model_trainer = ModelTrainerThread(train_fn, train_fn_kwargs, data_puller_out_q, model_trainer_out_q, train_period)
    periodic_forecaster = PeriodicForecasterThread(
        pred_input_prepare_fn,
        pred_input_prepare_fn_kwargs,
        pred_fn,
        pred_fn_kwargs,
        model_trainer_out_q,
        periodic_forecaster_out_q,
        forecast_period,
        forecast_dt,
    )
    publisher, mqtt_client = setup_publisher(iot_platform_settings_path)
    forecast_publisher = ForecastPublisherThread(periodic_forecaster_out_q, publisher)
    other_threads: List[threading.Thread] = [data_puller, model_trainer, periodic_forecaster, mqtt_client]
    for thread in other_threads:
        thread.start()
    forecast_publisher.run()
    for thread in other_threads:
        thread.join()
