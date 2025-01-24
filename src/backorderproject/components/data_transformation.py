import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, LabelEncoder
from imblearn.pipeline import Pipeline as ImblearnPipeline

from src.backorderproject.exception import CustomException
from src.backorderproject.logger import logging
from src.backorderproject.utils import save_object

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

@dataclass
class DataTransformationConfig:
    base_preprocessor_path= os.path.join('artifacts')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig


    def get_data_transformer_obj(self):
        try:

            num_cols= [0,1,2,3,4,5,7,8,9]               #The index number of each column is required for transformations
            cat_cols= [6,10,11,12,13,14]

            preprocessor = ColumnTransformer(
                transformers=[
                        ('num', Pipeline(steps=[
                        ('impute', SimpleImputer(strategy='median')),
                        ('power_transform', PowerTransformer())
                            ]), num_cols),

                        ('cat', Pipeline(steps=[
                        ('impute', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(drop='if_binary'))
                            ]), cat_cols)
                        ],
                remainder='passthrough' 
            )

            logging.info("Data Transformation Started")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_data_transformer_obj_oversampling(self):

        try:
            preprocessor = self.get_data_transformer_obj()

            oversampling_pipeline = ImblearnPipeline(steps=[
            ('preprocessor', preprocessor),  
            ('oversample', RandomOverSampler(random_state=42))  
            ])

            logging.info("Data Transformation Started (With Oversampling)")
            return oversampling_pipeline

        except Exception as e:
            raise CustomException(e, sys)


    def get_data_transformer_obj_undersampling(self):

        try:
        
            preprocessor = self.get_data_transformer_obj()

            undersampling_pipeline = ImblearnPipeline(steps=[
            ('preprocessor', preprocessor),  
            ('undersample', RandomUnderSampler(random_state=42)),
            ('scaler',StandardScaler())
            ])

            logging.info("Data Transformation Started (With Undersampling & Scaling)")
            return undersampling_pipeline

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self,train_path,test_path,sampling_strategy="none"):
        try:
            train_df= pd.read_csv(train_path)
            train_df.drop(columns=['forecast_6_month', 'forecast_9_month','sales_3_month', 'sales_6_month', 'sales_9_month','perf_12_month_avg'],inplace=True)

            test_df= pd.read_csv(test_path)
            test_df.drop(columns=['forecast_6_month', 'forecast_9_month','sales_3_month', 'sales_6_month', 'sales_9_month','perf_12_month_avg'],inplace=True)

            encoder= LabelEncoder()
            target_column_name= "went_on_backorder"

            train_df[target_column_name]= train_df[target_column_name].fillna(train_df[target_column_name].mode()[0])
            test_df[target_column_name]= test_df[target_column_name].fillna(test_df[target_column_name].mode()[0])

            print(f"Unique values in target train:", set(train_df[target_column_name]))
            print(f"Unique values in target test:", set(test_df[target_column_name]))


            train_df[target_column_name]= encoder.fit_transform(train_df[target_column_name])
            test_df[target_column_name]= encoder.transform(test_df[target_column_name])

            print(f"Unique values in target train df:", set(train_df[target_column_name]))
            print(f"Unique values in target test df:", set(test_df[target_column_name]))


            input_feature_train_df= train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df= test_df[target_column_name]

            logging.info("Read the train and test data")

            logging.info("Obtaining the preprocessor object with sampling_strategy techniques")

            if sampling_strategy == "oversample":
                preprocessing_obj = self.get_data_transformer_obj_oversampling()
                file_name = "preprocessor_oversampling.pkl"

                input_feature_train_arr = preprocessing_obj.named_steps['preprocessor'].fit_transform(input_feature_train_df)
                input_feature_train_arr, target_feature_train_df= preprocessing_obj.named_steps['oversample'].fit_resample(input_feature_train_arr,target_feature_train_df)
                input_feature_test_arr = preprocessing_obj.named_steps['preprocessor'].transform(input_feature_test_df)

            elif sampling_strategy == "undersample":
                preprocessing_obj = self.get_data_transformer_obj_undersampling()
                file_name = "preprocessor_undersampling.pkl"

                input_feature_train_arr = preprocessing_obj.named_steps['preprocessor'].fit_transform(input_feature_train_df)
                input_feature_train_arr, target_feature_train_df= preprocessing_obj.named_steps['undersample'].fit_resample(input_feature_train_arr,target_feature_train_df)
                input_feature_train_arr = preprocessing_obj.named_steps['scaler'].fit_transform(input_feature_train_arr)
                input_feature_test_arr = preprocessing_obj.named_steps['preprocessor'].transform(input_feature_test_df)
                input_feature_test_arr = preprocessing_obj.named_steps['scaler'].transform(input_feature_test_arr)

            else:
                preprocessing_obj = self.get_data_transformer_obj()
                file_name = "preprocessor.pkl"

                input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)


            preprocessor_file_path = os.path.join(self.data_transformation_config.base_preprocessor_path, file_name)

            logging.info(f"Saving preprocessor object to {preprocessor_file_path}")
            save_object(
                file_path=preprocessor_file_path,
                obj=preprocessing_obj
            )

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Data Transformation Completed")

            return (
                train_arr,
                test_arr,
                preprocessor_file_path
            )

        except Exception as ex:
            raise CustomException(ex, sys)