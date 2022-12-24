from typing import List
import json

import cv2
import numpy as np
import pandas as pd

from PIL import Image
from loguru import logger

from multiprocessing.dummy import Pool
from dagster import op


HAAR_CASCADES_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
VJ_MINSIZE = (30, 30)
DETECTED_FACES_JSON = "data/interim/detected_faces.json"
DETECTOR = cv2.CascadeClassifier(HAAR_CASCADES_PATH)


def vj_face_detector(image: np.array) -> List:
    """Recebe uma imagem (250x250) com 3 canais (RBG)
    e retorna as bounding boxes das faces detectadas.
    """

    faces = DETECTOR.detectMultiScale(
        image,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=VJ_MINSIZE,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    faces = [face.tolist() for face in faces]
    return faces


def load_image(path) -> np.array:
    """Retorna um array 250x250x1"""
    img = Image.open(path)
    np_img = np.asarray(img)
    return np_img


def convert2gray(image: np.array):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


@op
def load_and_detect_faces(unique_images_df: pd.DataFrame) -> str:
    images_df = unique_images_df

    paths = images_df["img"].values

    with Pool(processes=15) as p:
        logger.info("Loading images")
        images = p.map(load_image, paths)

    with Pool(processes=15) as p:
        logger.info("Converting 2 grayscale")
        gray_images = p.map(convert2gray, images)

    with Pool(processes=15) as p:
        logger.info("Extracting faces")
        faces = p.map(vj_face_detector, gray_images)

    detected_faces = {path: _faces for path, _faces in zip(paths, faces)}

    with open(DETECTED_FACES_JSON, "w") as j:
        json.dump(detected_faces, j)

    return DETECTED_FACES_JSON
