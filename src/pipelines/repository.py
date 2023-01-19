from dagster import load_assets_from_modules, repository
from src.data import detectors, extractors, make_pairs_and_labels, merge
from src.pipelines.lfw_jobs import lfw_download

assets = load_assets_from_modules(
    modules=[make_pairs_and_labels, detectors, extractors, merge],
    group_name="lfw_preprocessing",
)


@repository
def repo():
    return [assets, lfw_download]
