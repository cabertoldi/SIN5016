from src.data.download import download_lfw, download_pairs
from src.data.detectors import load_and_detect_faces, filter_faces_list, cut_faces
from src.data.make_pairs_and_labels import (
    genereate_pairs_with_labels_df,
    generate_unique_images_dataset,
)

from dagster import job


@job
def lfw_download():
    download_lfw()
    download_pairs()


@job
def lfw_preprocessing():
    pairs_df = genereate_pairs_with_labels_df()
    unique_images_df = generate_unique_images_dataset(pairs_df)
    filtered_faces_json = filter_faces_list(load_and_detect_faces(unique_images_df))
    cut_faces(filtered_faces_json)
