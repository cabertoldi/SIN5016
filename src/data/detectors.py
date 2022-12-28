from typing import List, Dict
import json

import cv2
import numpy as np
import pandas as pd

from PIL import Image
from loguru import logger

from dagster import asset, AssetIn

from tqdm import tqdm
from loguru import logger


HAAR_CASCADES_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DETECTED_FACES_JSON = "data/interim/detected_faces.json"
FILTERED_FACES_JSON = "data/interim/filtered_faces.json"
PREPROCESSED_IMAGE_PATH = "data/preprocessed/images/{image_filename}"
PROBLEMATIC_IMAGES_JSON = "data/interim/problems.csv"
DETECTOR = cv2.CascadeClassifier(HAAR_CASCADES_PATH)
SCALE_FACTOR = 1.02
IMAGE_WIDTH = 150
IMAGE_RESIZE = (IMAGE_WIDTH, IMAGE_WIDTH)
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


def filter_faces(faces: List[List]) -> List:
    for face in faces:
        x, y, w, h = face
        in_x = x <= 125 <= (x + w)
        in_y = y <= 125 <= (y + h)
        if in_x and in_y:
            # pixel central está contido no bbox
            return face
    return list()


@asset(ins={"unique_images_df": AssetIn(key="get_unique_images")})
def load_and_detect_faces(unique_images_df: pd.DataFrame) -> Dict:
    """Extrai os bounding boxes de faces encontrados para cada imagem
    em um json
    """
    images_df = unique_images_df

    paths = images_df["img"].values

    logger.info("Loading images")
    images = [load_image(path) for path in tqdm(paths)]

    logger.info("Converting to grayscale")
    gray_images = [convert2gray(im) for im in tqdm(images)]

    logger.info("Extracting faces")
    faces = [vj_face_detector(gray) for gray in tqdm(gray_images)]

    detected_faces = {path: _faces for path, _faces in zip(paths, faces)}

    with open(DETECTED_FACES_JSON, "w") as j:
        json.dump(detected_faces, j)

    return detected_faces


@asset(ins={"detected_faces_dict": AssetIn(key="load_and_detect_faces")})
def filter_faces_list(detected_faces_dict: Dict) -> Dict:
    """Le o arquivo json com as faces detectadas para cada imagem
    e filtra apenas o bbox que contem o pixel central da imagem

    imagens que não contenham nenhum bbox com o pixel central serão
    retornadas (apenas path) em uma lista separada.
    """
    logger.info("Filtering multiple faces detection")

    filtered_faces = {
        path: filter_faces(faces) for path, faces in detected_faces_dict.items()
    }

    faces_without_bbox = [
        path for path, faces in filtered_faces.items() if len(faces) == 0
    ]

    clean_filtered_faces = {
        path: face
        for path, face in filtered_faces.items()
        if path not in faces_without_bbox
    }

    pd.DataFrame({"img": faces_without_bbox}).to_csv(
        PROBLEMATIC_IMAGES_JSON, index=False
    )

    with open(FILTERED_FACES_JSON, "w") as j:
        j.write(json.dumps(clean_filtered_faces))

    return clean_filtered_faces


@asset(ins={"filtered_faces_dict": AssetIn(key="filter_faces_list")})
def cut_faces(filtered_faces_dict: Dict) -> None:
    """Recebe o json com os caminhos das imagens e os bbox
    válidos para a realização do corte das imagens"""
    logger.info("Cutting faces")

    for path, bbox in tqdm(filtered_faces_dict.items()):
        image_filename = path.split("/")[-1]
        processed_image_path = PREPROCESSED_IMAGE_PATH.format(
            image_filename=image_filename
        )
        x, y, w, h = bbox
        img = Image.open(path)
        cutted_img = img.crop((x, y, x + w, y + h)).resize(IMAGE_RESIZE)
        cutted_img.save(processed_image_path)
