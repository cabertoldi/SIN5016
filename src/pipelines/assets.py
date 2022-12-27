from src.data import make_pairs_and_labels
from src.data import detectors
from src.data import extractors

from src.pipelines.lfw_jobs import lfw_download

from dagster import load_assets_from_modules, repository

# TODO: better naming
detectors_assets = load_assets_from_modules(
    modules=[make_pairs_and_labels, detectors, extractors],
    group_name="lfw_preprocessing",
)


@repository
def repo():
    return [detectors_assets, lfw_download]
