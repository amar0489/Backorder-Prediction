import os
import sys
from src.backorderproject.exception import CustomException
from src.backorderproject.logger import logging
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion phase")

        try:
            df= pd.read_csv(r'notebook\data\backorder_data.csv',low_memory=False)
            df.drop(columns=['Unnamed: 0'], inplace=True , axis=1)

            logging.info('Imported the dataset')


            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)

            logging.info("Train Test Split Initiated")

            train_set = None
            test_set = None

            
            stratified_shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in stratified_shuffle_split.split(df, df['went_on_backorder'].fillna(df['went_on_backorder'].mode()[0])):
                train_set = df.loc[train_index]
                test_set = df.loc[test_index]

            train_set.to_csv(self.ingestion_config.train_data_path,header=True,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,header=True,index=False)

            logging.info("Data Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as ex:
            raise CustomException(ex,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()