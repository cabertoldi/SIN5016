from src.pipelines.lfw_jobs import lfw_download, lfw_preprocessing

from dagster import repository, define_asset_job


@repository
def repo():
    return [lfw_download, lfw_preprocessing, define_asset_job("all_assets")]
