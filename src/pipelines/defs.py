from src.pipelines.data_acquisition import data_acquisition_pipeline

from dagster import Definitions

defs = Definitions(jobs=[data_acquisition_pipeline])
