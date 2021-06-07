import json
import threading

import paho.mqtt.client as mqtt
from typing import Any, Dict


class IotPlattformSettings:
    def __init__(self, config_file):
        with open(config_file) as f:
            data = json.load(f)
        self.iot_platform_gateway_username = data["iot_platform_gateway_username"]
        self.iot_platform_gateway_ip = data["iot_platform_gateway_ip"]
        self.iot_platform_gateway_port = data["iot_platform_gateway_port"]
        self.iot_platform_group_name = data["iot_platform_group_name"]
        self.iot_platform_sensor_name = data["iot_platform_sensor_name"]
        self.iot_platform_user_id = data["iot_platform_user_id"]
        self.iot_platform_device_id = data["iot_platform_device_id"]
        self.iot_platform_gateway_password = data["iot_platform_gateway_password"]


class Publisher:
    def __init__(self, client: mqtt.Client, topic: str, username: str, sensor_name: str, device_id: int):
        self.client = client
        self.topic = topic
        self.username = username
        self.sensor_name = sensor_name
        self.device_id = device_id

    def publish(self, timestamp: int, count: int):
        message = (
            "{"
            + '"username":"{}","{}":{},"device_id":{},"timestamp":{}'.format(
                self.username, self.sensor_name, str(count), str(self.device_id), str(timestamp)
            )
            + "}"
        )
        print("Publishing '{}' on topic '{}'".format(message, self.topic))
        self.client.publish(self.topic, message)


def on_connect(client: mqtt.Client, userdata: Dict[str, Any], flags: Dict[str, int], rc: int):
    print("Connected. Result code " + str(rc))
    if not userdata["is_connected"]:
        userdata["is_connected"] = True


def on_disconnect(client: mqtt.Client, userdata: Dict[str, Any], rc: int):
    print("Disconnected. Result code " + str(rc))
    if userdata["is_connected"]:
        userdata["is_connected"] = False


def on_publish(client: mqtt.Client, userdata: Dict[str, Any], rc: int):
    print("MQTT event published. Result code: {}.".format(rc))


def setup_publisher(iot_platform_settings_path) -> (Publisher, threading.Thread):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish
    settings = IotPlattformSettings(iot_platform_settings_path)
    mqtt_topic = str(settings.iot_platform_user_id) + "_" + str(settings.iot_platform_device_id)
    publisher = Publisher(
        client,
        mqtt_topic,
        settings.iot_platform_group_name,
        settings.iot_platform_sensor_name,
        settings.iot_platform_device_id,
    )
    user_data = {
        "is_connected": False,
    }
    client.user_data_set(user_data)
    client.username_pw_set(settings.iot_platform_gateway_username, settings.iot_platform_gateway_password)
    client.connect(settings.iot_platform_gateway_ip, port=settings.iot_platform_gateway_port)
    mqtt_client = threading.Thread(target=client.loop_forever)
    return publisher, mqtt_client
