import os
import sys
import argparse
sys.path[0] = "E:/Projects/MLPROJECT"
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
import math
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self, source_data_path: str):
        self.ingestion_config = DataIngestionConfig()
        self.source_data_path = source_data_path

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(self.source_data_path)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set = df.sample(frac=0.8, random_state=42)
            test_set = df.drop(train_set.index)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", dest="data_path", required=True, help="Path to the source data file")
    args = parser.parse_args()
    data_inj_obj = DataIngestion(source_data_path=args.data_path)
    data_inj_obj.initiate_data_ingestion()