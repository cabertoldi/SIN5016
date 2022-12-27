from src.data import make_pairs_and_labels
from src.data import detectors

from src.pipelines.lfw_jobs import lfw_download

from dagster import load_assets_from_modules, repository

detectors_assets = load_assets_from_modules([make_pairs_and_labels, detectors])


@repository
def repo():
    return [detectors_assets, lfw_download]
