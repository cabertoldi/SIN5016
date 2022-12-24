# -*- coding: utf-8 -*-
from typing import Tuple, List
import pandas as pd

from loguru import logger
from dagster import op

RAW_IMAGES_PATH = "data/raw/lfw"
PAIRS_WITH_LABELS_PATH = "data/interim/pairs.parquet"
UNIQUE_IMAGES_PATH = "data/interim/images_paths.parquet"


def get_image_path(person_name: str, photo_number: str) -> str:
    photo_number = str(photo_number).zfill(4)
    raw_path = f"{RAW_IMAGES_PATH}/{person_name}"
    image_filename = f"{person_name}_{photo_number}.jpg"
    return f"{raw_path}/{image_filename}"


def preprocess_match_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing match df...")
    df.columns = ["person_name", "img1_id", "img2_id"]
    df["img1"] = df.apply(
        lambda r: get_image_path(r["person_name"], r["img1_id"]), axis=1
    )
    df["img2"] = df.apply(
        lambda r: get_image_path(r["person_name"], r["img2_id"]), axis=1
    )
    df["match"] = True
    df["person_1"] = df["person_name"]
    df["person_2"] = df["person_name"]
    return df[["person_1", "person_2", "img1_id", "img2_id", "img1", "img2", "match"]]


def preprocess_nonmatch_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing non match df...")
    df.columns = ["person_1", "img1_id", "person_2", "img2_id"]
    df["img1"] = df.apply(lambda r: get_image_path(r["person_1"], r["img1_id"]), axis=1)
    df["img2"] = df.apply(lambda r: get_image_path(r["person_2"], r["img2_id"]), axis=1)
    df["match"] = False
    return df[["person_1", "person_2", "img1_id", "img2_id", "img1", "img2", "match"]]


def read_pairs_file(pairs_file: str, nrows: int) -> pd.DataFrame:
    """Returns a dataframe with this schema:
    - img1_path: str
    - img2_path: str
    - match: bool
    """
    logger.info(f"Making pairs dataframe for: {pairs_file}")
    match_df = pd.read_csv(pairs_file, sep="\t", nrows=nrows, header=None, skiprows=1)
    nonmatch_df = pd.read_csv(
        pairs_file, sep="\t", nrows=nrows, header=None, skiprows=nrows + 1
    )

    match_df = preprocess_match_df(match_df)
    nonmatch_df = preprocess_nonmatch_df(nonmatch_df)

    return pd.concat([match_df, nonmatch_df])


def get_unique_images(pairs_df: pd.DataFrame) -> pd.DataFrame:
    img1 = pairs_df[["img1"]].rename(columns={"img1": "img"})
    img2 = pairs_df[["img2"]].rename(columns={"img2": "img"})
    imgs = pd.concat([img1, img2]).drop_duplicates()
    return imgs


@op
def genereate_pairs_with_labels_df(pairs_files: List[Tuple[str, int]]) -> pd.DataFrame:
    pairs_with_labels_df = pd.concat(
        [read_pairs_file(pair_file, nrows) for pair_file, nrows in pairs_files]
    )

    pairs_with_labels_df.to_parquet(PAIRS_WITH_LABELS_PATH)
    return pairs_with_labels_df


@op
def generate_unique_images_dataset(pairs_with_labels_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating images list...")
    imgs_df = get_unique_images(pairs_with_labels_df)
    imgs_df.to_parquet(UNIQUE_IMAGES_PATH)
    return imgs_df
