import os
import urllib.request as request
import zipfile
from stb_pfe_mlflow import logger
from stb_pfe_mlflow.utils.common import get_size
from pathlib import Path
from stb_pfe_mlflow.entity.config_entity import DataIngestionConfig
import shutil


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            shutil.copy(self.config.source_URL, self.config.local_data_file)
            logger.info(
                f"File copied from {self.config.source_URL} to {self.config.local_data_file}"
            )
        else:
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
            )

    def extract_zip_file(self):
        """
        This function is now redundant since you are working with a CSV file.
        """
        pass
