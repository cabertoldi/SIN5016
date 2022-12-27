from typing import List
import json

import cv2
import numpy as np
import pandas as pd

from PIL import Image
from loguru import logger

from multiprocessing.dummy import Pool
from dagster import op

from tqdm import tqdm


HAAR_CASCADES_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DETECTED_FACES_JSON = "data/interim/detected_faces.json"
DETECTOR = cv2.CascadeClassifier(HAAR_CASCADES_PATH)
SCALE_FACTOR = 1.02
MIN_NEIGHBORS = 2


def vj_face_detector(image: np.array) -> List:
    """Recebe uma imagem (250x250) em escala de cinza
    e retorna as bounding boxes das faces detectadas.
    """

    faces = DETECTOR.detectMultiScale(
        image,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        flags=cv2.CASCADE_DO_CANNY_PRUNING,
    )

    faces = [face.tolist() for face in faces]
    return faces


def load_image(path) -> np.array:
    """Retorna um array 250x250x3"""
    img = Image.open(path)
    np_img = np.asarray(img)
    return np_img


def convert2gray(image: np.array):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


@op
def load_and_detect_faces(unique_images_df: pd.DataFrame) -> str:
    """ Extrai os bounding boxes de faces encontrados para cada imagem
    em um json
    """
    images_df = unique_images_df

    paths = images_df["img"].values

    logger.info("Loading images")
    images = [
        load_image(path)
        for path in tqdm(paths)
    ]
    
    logger.info("Converting to grayscale")
    gray_images = [
        convert2gray(im)
        for im in tqdm(images)
    ]

    logger.info("Extracting faces")
    faces = [
        vj_face_detector(gray)
        for gray in tqdm(gray_images)
    ]

    detected_faces = {path: _faces for path, _faces in zip(paths, faces)}

    with open(DETECTED_FACES_JSON, "w") as j:
        json.dump(detected_faces, j)

    return DETECTED_FACES_JSON
