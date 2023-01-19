from dagster import job
from src.data.download import download_lfw, download_pairs


@job
def lfw_download():
    download_lfw()
    download_pairs()
