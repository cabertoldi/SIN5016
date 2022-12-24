from src.pipelines.lfw_jobs import lfw_download, lfw_preprocessing

from dagster import repository


@repository
def repo():
    return [lfw_download, lfw_preprocessing]
