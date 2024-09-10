from stb_pfe_mlflow.config.configuration import ConfigurationManager
from stb_pfe_mlflow import logger
from stb_pfe_mlflow.components.data_cleaning import DataCleaning
from pathlib import Path

STAGE_NAME = "Data Cleaning stage"


class DataCleaningTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_cleaning_config = config.get_data_cleaning_config()
                data_cleaning = DataCleaning(config=data_cleaning_config)
                data_cleaning.cleaning_data()

            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started  <<<<<<")
        data_cleaning = DataCleaningTrainingPipeline()
        data_cleaning.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
