# -*- coding: utf-8 -*-
import os
from typing import List

import pandas as pd
import numpy as np

from dagster import asset, AssetIn
from PIL import Image

from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray

from src.data.detectors import PREPROCESSED_IMAGE_PATH

from tqdm import tqdm
from loguru import logger


PREPROCESSED_IMAGES_DIR = PREPROCESSED_IMAGE_PATH.format(image_filename="")


def get_images_paths() -> List[str]:
    return [
        f"{PREPROCESSED_IMAGES_DIR}{path}"
        for path in os.listdir(PREPROCESSED_IMAGES_DIR)
        if not ".gitkeep" in path
    ]


@asset(non_argument_deps={"cut_faces"})
def hog_extractor() -> pd.DataFrame:
    """Itera sobre todas as imagens do repositório de imagens
    preprocessadas e extrai para cada uma delas o seu histograma
    baseado no extrator HOG (Histogram of Oriented Gradients).
    """
    logger.info("Running hog extractor")
    images_paths = get_images_paths()

    histograms = list()
    for path in tqdm(images_paths):
        # load image
        img = np.asarray(Image.open(path))
        gray_level = rgb2gray(img)
        # extract hog
        _hog = hog(gray_level, visualize=False, feature_vector=True)
        # get histogram of hog
        hist, _ = np.histogram(_hog, bins=255)
        histograms.append(hist)

    df = pd.DataFrame([[img, hist] for img, hist in zip(images_paths, histograms)])

    df.columns = [str(c) for c in df.columns]
    df.to_parquet("data/preprocessed/extractors/hog_features.parquet")
    return df


@asset(non_argument_deps={"cut_faces"})
def lbp_extractor() -> pd.DataFrame:
    """Itera sobre todas as imagens do repositório de imagens
    preprocessadas e extrai para cada uma delas o seu histograma
    baseado no extrator LBP (Local Binary Patterns)"""
    logger.info("Running LBP extractor")
    images_paths = get_images_paths()

    histograms = list()
    for path in tqdm(images_paths):
        # load image
        img = np.asarray(Image.open(path))
        gray_level = rgb2gray(img)
        # extract hog
        _lbp = local_binary_pattern(gray_level, P=8 * 3, R=3)
        # get histogram of hog
        hist, _ = np.histogram(_lbp, bins=255)
        histograms.append(hist)

    df = pd.DataFrame([[img, hist] for img, hist in zip(images_paths, histograms)])

    df.columns = [str(c) for c in df.columns]
    df.to_parquet("data/preprocessed/extractors/lbp_features.parquet")
    return df
