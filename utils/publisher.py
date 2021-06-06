import argparse
import queue
import threading

import paho.mqtt.client as mqtt
from typing import Any, Dict


IOT_PLATFORM_GATEWAY_USERNAME = "JWT"
IOT_PLATFORM_GATEWAY_IP = "131.159.35.132"
IOT_PLATFORM_GATEWAY_PORT = 1883
IOT_PLATFORM_GROUP_NAME = "group4_2021_ss"
IOT_PLATFORM_SENSOR_NAME = "forecast"
IOT_PLATFORM_USER_ID = 48
IOT_PLATFORM_DEVICE_ID = 135
IOT_PLATFORM_GATEWAY_PASSWORD = ""


def iot_platform_config_loader():
    pass


class Publisher:
    def __init__(self, client: mqtt.Client, topic: str, username: str, sensor_name: str, device_id: int):
        self.client = client
        self.topic = topic
        self.username = username
        self.sensor_name = sensor_name
        self.device_id = device_id

    def publish(self, count: int, timestamp: int):
        message = "{'username':'{}','{}':{},'device_id':{},'timestamp':{}}".format(
            self.username, self.sensor_name, str(count), str(self.device_id), str(timestamp)
        )
        self.client.publish(self.topic, message)


def on_connect(client: mqtt.Client, userdata: Dict[str, Any], flags: Dict[str, int], rc: int):
    print("Connected with result code " + str(rc))
    if not userdata["is_connected"]:
        userdata["is_connected"] = True


def on_disconnect(client: mqtt.Client, userdata: Dict[str, Any], rc: int):
    print("Disconnected with result code " + str(rc))
    if userdata["is_connected"]:
        userdata["is_connected"] = False


def on_publish(client: mqtt.Client, userdata: Dict[str, Any], rc: int):
    print("Data published")


def setup_publisher() -> (Publisher, threading.Thread):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    mqtt_topic = str(IOT_PLATFORM_USER_ID) + "_" + str(IOT_PLATFORM_DEVICE_ID)

    publisher = Publisher(client, mqtt_topic, IOT_PLATFORM_GROUP_NAME, IOT_PLATFORM_SENSOR_NAME, IOT_PLATFORM_DEVICE_ID)

    user_data = {
        "is_connected": False,
    }

    client.user_data_set(user_data)
    client.username_pw_set(IOT_PLATFORM_GATEWAY_USERNAME, IOT_PLATFORM_GATEWAY_PASSWORD)
    client.connect(IOT_PLATFORM_GATEWAY_IP, port=IOT_PLATFORM_GATEWAY_PORT)

    mqtt_client = threading.Thread(target=client.loop_forever)

    return publisher, mqtt_client
