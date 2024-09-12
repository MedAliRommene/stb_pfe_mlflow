from stb_pfe_mlflow import logger
from stb_pfe_mlflow.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from stb_pfe_mlflow.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
)
from stb_pfe_mlflow.pipeline.stage_03_data_cleaning import DataCleaningTrainingPipeline
from stb_pfe_mlflow.pipeline.stage_04_data_transformation import (
    DataTransformationTrainingPipeline,
)

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>> stage{STAGE_NAME} started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage{STAGE_NAME} completed <<<<<\n\n x===========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataValidationTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Cleaning stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started  <<<<<<")
    data_ingestion = DataCleaningTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataTransformationTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
