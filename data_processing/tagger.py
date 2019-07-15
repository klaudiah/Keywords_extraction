import time
from enum import Enum
from pathlib import Path

import requests

BASE_URL = "http://ws.clarin-pl.eu/nlprest2/base"
BASE_DIR = Path.cwd()
USER = "radek24_95@o2.pl"


class DataSource(Enum):
    LOCAL_UNCOMPRESSED_FILE = 0
    LOCAL_COMPRESSED_FILE = 1
    REMOTE_COMPRESSED_FILE = 2
    TEXT = 3


def download(file_id, file_path):
    url = "{}/download{}".format(BASE_URL, file_id)
    response = requests.get(url=url)
    with open(file_path, "w+", encoding="utf-8") as f:
        f.write(response.text)


def get_response(file_id):
    url = "{}/download{}".format(BASE_URL, file_id)
    response = requests.get(url=url)
    return response.text


def upload(file_path):
    url = "{}/upload/".format(BASE_URL)
    headers = {"content-type": "application/json"}
    with open(file_path, "rb", 'utf-8') as f:
        response = requests.post(url=url, data=f.read(), headers=headers)
        return response.text


def start_task(data):
    url = "{}/startTask/".format(BASE_URL)
    response = requests.post(url=url, json=data)
    task_id = response.text
    time.sleep(0.5)
    return task_id


def get_status(task_id):
    url = "{}/getStatus/{}".format(BASE_URL, task_id)
    response = requests.get(url=url)
    status = response.json()
    return status


def process(task_id):
    status = get_status(task_id)
    while status["status"] in ("QUEUE", "PROCESSING"):
        time.sleep(0.5)
        status = get_status(task_id)

    if status["status"] == "ERROR":
        raise requests.HTTPError(status["value"])

    return status["value"]


def prepare_data(text, task):
    data = {"user": USER,
            "lpmn": task,
            "text": text
            }
    return data


def tagging(text, file_path):
    data = prepare_data(text, task="any2txt|wcrft2({\"morfeusz2\":false})")
    task_id = start_task(data)
    items = process(task_id=task_id)
    for item in items:
        download(file_id=item["fileID"], file_path=file_path)
    return file_path


def lemmatization(text):
    data = prepare_data(text, task="any2txt|wcrft2({\"morfeusz2\":false})")
    task_id = start_task(data)
    items = process(task_id=task_id)
    for item in items:
        response = get_response(file_id=item["fileID"])
    return response
