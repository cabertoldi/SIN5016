# -*- coding: utf-8 -*-
from typing import Tuple, List
import pandas as pd

from loguru import logger
from dagster import asset, AssetIn, Output, MetadataValue

from src.data.download import PAIRS_DOWNLOADED_FILES

RAW_IMAGES_PATH = "data/raw/lfw"
PAIRS_WITH_LABELS_PATH = "data/interim/pairs.parquet"
UNIQUE_IMAGES_PATH = "data/interim/images_paths.parquet"


def get_image_filename(person_name: str, photo_number: str) -> str:
    photo_number = str(photo_number).zfill(4)
    return f"{person_name}_{photo_number}.jpg"


def get_raw_image_filename(person_name: str, photo_number: str) -> str:
    filename = get_image_filename(person_name, photo_number)
    return f"{RAW_IMAGES_PATH}/{person_name}/{filename}"


def preprocess_match_df(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe o dataframe de matches que contêm as colunas
    nome, imagem1, imagem2 e retorna o dataset com o completo com
    - nome pessoa 1
    - nome pessoa 2
    - id_imagem_1
    - id_imagem_2
    - caminho_imagem_1
    - caimnho_imagem_2
    - match
    """
    logger.info("Preprocessing match df...")
    df.columns = ["person_name", "img1_id", "img2_id"]
    df["match"] = True
    df["person_1"] = df["person_name"]
    df["person_2"] = df["person_name"]
    df["img1_full_id"] = df.apply(
        lambda r: get_image_filename(r["person_1"], r["img1_id"]), axis=1
    )
    df["img2_full_id"] = df.apply(
        lambda r: get_image_filename(r["person_2"], r["img2_id"]), axis=1
    )
    df["raw_img1"] = df.apply(
        lambda r: get_raw_image_filename(r["person_1"], r["img1_id"]), axis=1
    )
    df["raw_img2"] = df.apply(
        lambda r: get_raw_image_filename(r["person_2"], r["img2_id"]), axis=1
    )
    return df[
        [
            "person_1",
            "person_2",
            "img1_full_id",
            "img2_full_id",
            "raw_img1",
            "raw_img2",
            "match",
        ]
    ]


def preprocess_nonmatch_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing non match df...")
    df.columns = ["person_1", "img1_id", "person_2", "img2_id"]
    df["img1_full_id"] = df.apply(
        lambda r: get_image_filename(r["person_1"], r["img1_id"]), axis=1
    )
    df["img2_full_id"] = df.apply(
        lambda r: get_image_filename(r["person_2"], r["img2_id"]), axis=1
    )
    df["raw_img1"] = df.apply(
        lambda r: get_raw_image_filename(r["person_1"], r["img1_id"]), axis=1
    )
    df["raw_img2"] = df.apply(
        lambda r: get_raw_image_filename(r["person_2"], r["img2_id"]), axis=1
    )
    df["match"] = False
    return df[
        [
            "person_1",
            "person_2",
            "img1_full_id",
            "img2_full_id",
            "raw_img1",
            "raw_img2",
            "match",
        ]
    ]


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


@asset(ins={"pairs_df": AssetIn(key="genereate_pairs_with_labels_df")})
def get_unique_images(pairs_df):
    """Retorna um dataframe com todas as imagens que estão sendo utilizadas no conjunto de dados."""
    img1 = pairs_df[["raw_img1"]].rename(columns={"raw_img1": "img"})
    img2 = pairs_df[["raw_img2"]].rename(columns={"raw_img2": "img"})
    imgs = pd.concat([img1, img2]).drop_duplicates()

    metadata = {
        "total_images": len(imgs),
        "preview": MetadataValue.md(imgs.head(5).to_markdown()),
    }

    imgs.to_parquet(UNIQUE_IMAGES_PATH)

    return Output(value=imgs, metadata=metadata)


@asset
def genereate_pairs_with_labels_df() -> Output:
    """Gera o conjunto de dados com os pares associados e a variável alvo calculada"""

    pairs_with_labels_df = pd.concat(
        [
            read_pairs_file(pair_file, nrows)
            for pair_file, nrows in PAIRS_DOWNLOADED_FILES
        ]
    )

    pairs_with_labels_df.to_parquet(PAIRS_WITH_LABELS_PATH)

    metadata = {"preview": MetadataValue.md(pairs_with_labels_df.head(5).to_markdown())}

    return Output(value=pairs_with_labels_df, metadata=metadata)
