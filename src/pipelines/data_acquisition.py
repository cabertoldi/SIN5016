from src.data.download import download_lfw, download_pairs
from src.data.make_pairs_and_labels import (
    genereate_pairs_with_labels_df,
    generate_unique_images_dataset,
)

from dagster import job


@job
def data_acquisition_pipeline():
    download_lfw()
    generate_unique_images_dataset(genereate_pairs_with_labels_df(download_pairs()))
