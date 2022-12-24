# -*- coding: utf-8 -*-
""" Downloads the LFW dataset and pairs text files """
import requests as req
import tarfile
from loguru import logger


RAW_DATA_PATH = "data/raw"
LFW_ZIPPED_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
PAIRS_URLS = [
    "http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt",
    "http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt",
]


def download_lfw():
    logger.info("Downloading LFW dataset...")
    with req.get(LFW_ZIPPED_URL, stream=True) as rx, tarfile.open(
        name=f"{RAW_DATA_PATH}/", fileobj=rx.raw, mode="r:gz"
    ) as tarobj:
        tarobj.extractall(path=RAW_DATA_PATH)


def download_pairs():
    logger.info("Downloading pairs files...")
    for pair_url in PAIRS_URLS:
        fname = pair_url.split("/")[-1]
        response = req.get(pair_url)
        with open(f"{RAW_DATA_PATH}/{fname}", "w") as f:
            f.write(response.text)


def main():
    download_lfw()
    download_pairs()


if __name__ == "__main__":
    main()
