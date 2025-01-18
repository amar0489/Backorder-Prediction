from src.backorderproject.logger import logging
from src.backorderproject.exception import CustomException
import sys
from src.backorderproject.components.data_ingestion import DataIngestion
from src.backorderproject.components.data_ingestion import DataIngestionConfig
from src.backorderproject.components.data_transformation import DataTransformation
from src.backorderproject.components.data_transformation import DataTransformationConfig



if __name__== "__main__":
    obj = DataIngestion()
    train_data,test_data= obj.initiate_data_ingestion()

    data_transformation= DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data, sampling_strategy="none")

    train_arr_ovr, test_arr_ovr, preprocessor_path_ovr = data_transformation.initiate_data_transformation(train_data, test_data, sampling_strategy="oversample")

    train_arr_und, test_arr_und, preprocessor_path_und = data_transformation.initiate_data_transformation(train_data, test_data, sampling_strategy="undersample")