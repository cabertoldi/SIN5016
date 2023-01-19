# -*- coding: utf-8 -*-
""" Downloads the LFW dataset and pairs text files """
import tarfile
from typing import List, Tuple

import requests as req
from loguru import logger

from dagster import op

RAW_DATA_PATH = "data/raw"
INTERIM_DATA_PATH = "data/interim"
LFW_ZIPPED_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
CASCADES_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
PAIRS_URLS = [
    "http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt",
    "http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt",
]
PAIRS_DOWNLOADED_FILES = [
    (f"{RAW_DATA_PATH}/pairsDevTrain.txt", 1100),
    (f"{RAW_DATA_PATH}/pairsDevTest.txt", 500),
]


@op
def download_lfw() -> str:
    logger.info("Downloading LFW dataset...")
    with req.get(LFW_ZIPPED_URL, stream=True) as rx, tarfile.open(
        name=f"{RAW_DATA_PATH}/", fileobj=rx.raw, mode="r:gz"
    ) as tarobj:
        tarobj.extractall(path=RAW_DATA_PATH)
    return f"{RAW_DATA_PATH}/lfw"


@op
def download_pairs() -> List[Tuple[str, int]]:
    logger.info("Downloading pairs files...")

    for pair_url in PAIRS_URLS:
        fname = pair_url.split("/")[-1]
        response = req.get(pair_url)
        with open(f"{RAW_DATA_PATH}/{fname}", "w") as f:
            f.write(response.text)

    return PAIRS_DOWNLOADED_FILES
