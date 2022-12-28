# -*- coding: utf-8 -*-
import os
from typing import List

import pandas as pd
import numpy as np

from dagster import asset, Output, MetadataValue
from PIL import Image

from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray

from src.data.detectors import PREPROCESSED_IMAGE_PATH

from tqdm import tqdm
from loguru import logger


HOG_FEATURES_PATH = "data/preprocessed/extractors/hog_features.parquet"
LBP_FEATURES_PATH = "data/preprocessed/extractors/lbp_features.parquet"
PREPROCESSED_IMAGES_DIR = PREPROCESSED_IMAGE_PATH.format(image_filename="")
HIST_BINS = 10


def get_images_paths() -> List[str]:
    return [
        f"{PREPROCESSED_IMAGES_DIR}{path}"
        for path in os.listdir(PREPROCESSED_IMAGES_DIR)
        if not ".gitkeep" in path
    ]


@asset(non_argument_deps={"cut_faces"})
def hog_extractor():
    """Itera sobre todas as imagens do repositório de imagens
    preprocessadas e extrai para cada uma delas o seu histograma
    baseado no extrator HOG (Histogram of Oriented Gradients).
    """
    logger.info("Running hog extractor")
    images_paths = get_images_paths()

    histograms = list()
    for path in tqdm(images_paths):
        # load image
        img = np.asarray(Image.open(path).resize((62, 47)))
        gray_level = rgb2gray(img)
        # extract hog
        _hog = hog(gray_level, visualize=False, feature_vector=True)
        # get histogram of hog
        histograms.append(_hog)

    df = pd.DataFrame([[img, hist] for img, hist in zip(images_paths, histograms)])

    df.columns = [str(c) for c in df.columns]
    df.to_parquet(HOG_FEATURES_PATH)

    metadata = {
        "total_images": len(df),
        "preview": MetadataValue.md(df.head(5).to_markdown()),
    }

    return Output(value=df, metadata=metadata)


@asset(non_argument_deps={"cut_faces"})
def lbp_extractor():
    """Itera sobre todas as imagens do repositório de imagens
    preprocessadas e extrai para cada uma delas o seu histograma
    baseado no extrator LBP (Local Binary Patterns)"""
    logger.info("Running LBP extractor")
    images_paths = get_images_paths()

    histograms = list()
    for path in tqdm(images_paths):
        # load image
        img = np.asarray(Image.open(path).resize((62, 47)))
        gray_level = rgb2gray(img)
        # extract hog
        _lbp = local_binary_pattern(gray_level, P=8 * 3, R=3)
        # get histogram of hog
        histograms.append(_lbp.flatten())

    df = pd.DataFrame([[img, hist] for img, hist in zip(images_paths, histograms)])

    df.columns = [str(c) for c in df.columns]
    df.to_parquet(LBP_FEATURES_PATH)

    metadata = {
        "total_images": len(df),
        "preview": MetadataValue.md(df.head(5).to_markdown()),
    }

    return Output(value=df, metadata=metadata)
