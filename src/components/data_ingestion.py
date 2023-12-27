import os
import sys

# sys.path.append("//src//components")
sys.path.append(r"E:\\machine_learning_project\\")
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTranformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingeston method or component")

        try:
            df = pd.read_csv("notebook\\data\\stud.csv")
            logging.info("Read the dataset as dataframe")
            # making the dir
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Read the dataset as a dataframe and saved as raw data")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            # splitting the data in train test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            # saving the train data

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            # saving the train data
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except FileNotFoundError as e:
            logging.info(f"File not found: {str(e)}")
            raise CustomException(e, sys)

        except Exception as e:
            logging.info(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_tranformation = DataTranformation()

    train_arr, test_arr, _ = data_tranformation.initiate_data_transformation(
        train_data, test_data
    )
    # model trainer
    model_tainer = ModelTrainer()
    print(round(model_tainer.initiate_model_trainer(train_arr, test_arr),5)*100)
