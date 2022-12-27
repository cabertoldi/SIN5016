from src.data.download import download_lfw, download_pairs

from dagster import job


@job
def lfw_download():
    download_lfw()
    download_pairs()
