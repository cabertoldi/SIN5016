# -*- coding: utf-8 -*-

import pandas as pd

from src.data.make_pairs_and_labels import PAIRS_WITH_LABELS_PATH

from dagster import asset, Output, AssetIn, MetadataValue


def merge_features(
    pairs_df: pd.DataFrame, features_df: pd.DataFrame, extractor: str
) -> pd.DataFrame:
    extracor_fs_col = f"{extractor}_features"
    features_df.columns = ["img_path", extracor_fs_col]
    features_df["img_full_id"] = features_df["img_path"].str.split("/").str[-1]

    new_df = pairs_df.copy()

    new_df = new_df.merge(
        right=features_df[["img_full_id", extracor_fs_col]],
        left_on=["img1_full_id"],
        right_on=["img_full_id"],
        how="left",
        suffixes=("", "_img1"),
    )

    new_df = new_df.merge(
        right=features_df[["img_full_id", extracor_fs_col]],
        left_on=["img2_full_id"],
        right_on=["img_full_id"],
        how="left",
        suffixes=("", "_img2"),
    )

    new_df = new_df.rename(
        columns={f"{extractor}_features": f"{extractor}_features_img1"}
    )

    new_df = new_df.dropna()

    df_img1 = pd.DataFrame(new_df[f"{extractor}_features_img1"].tolist())
    n_cols = df_img1.shape[1]
    df_img1.columns = list(range(n_cols))

    df_img2 = pd.DataFrame(new_df[f"{extractor}_features_img2"].tolist())
    df_img2.columns = list(range(n_cols, n_cols * 2))

    df_final = pd.concat([df_img1, df_img2], axis=1)
    df_final["match"] = new_df["match"]
    df_final.columns = [str(c) for c in df_final.columns]

    return df_final


def make_output(df: pd.DataFrame) -> Output:
    metadata = {
        "total_images": len(df),
        "dataframe_shape": str(df.shape),
        "preview": MetadataValue.md(df.head(5).to_markdown()),
    }
    return Output(value=df, metadata=metadata)


@asset(ins={"hog_features": AssetIn(key="hog_extractor")})
def merge_hog_features_with_pairs_dataset(hog_features):
    """Monta conjunto de dados tabular final com as características
    HOG extraídas e a variável alvo.
    """
    pairs_df = pd.read_parquet(PAIRS_WITH_LABELS_PATH)
    df = merge_features(pairs_df, hog_features, extractor="hog")
    df.to_parquet("data/preprocessed/feature_matrix_hog.parquet")
    return make_output(df)


@asset(ins={"lbp_features": AssetIn(key="lbp_extractor")})
def merge_lbp_features_with_pairs_dataset(lbp_features):
    """Monta conjunto de dados tabular final com as características
    LBP extraídas e a variável alvo.
    """
    pairs_df = pd.read_parquet(PAIRS_WITH_LABELS_PATH)
    df = merge_features(pairs_df, lbp_features, extractor="lbp")
    df.to_parquet("data/preprocessed/feature_matrix_lbp.parquet")
    return make_output(df)
