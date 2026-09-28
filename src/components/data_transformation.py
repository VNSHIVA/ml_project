import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import traceback

try:
    from src.utils import save_object
except ImportError:
    def save_object(file_path, obj):
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        '''
        Thisfunction is responsible for data transformation
        '''

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical columns standard scaling completed')
            logging.info('Categorical columns encoding completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            print(f"Shape of train_df before transformation: {train_df.shape}")
            print(f"Shape of test_df before transformation: {test_df.shape}")

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            target_feature_train_df = train_df[target_column_name]
            input_feature_train_df = train_df.drop(columns=[target_column_name])

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            print(f"Shape of input_feature_train_df: {input_feature_train_df.shape}")
            print(f"Total Input Features: {len(input_feature_train_df.columns)}")
            print("Input Columns:", list(input_feature_train_df.columns))

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            # Get feature names from the fitted ColumnTransformer
            feature_names = preprocessing_obj.get_feature_names_out()
            print("feature_names type:",type(feature_names))

            print(f"Shape of transformed features (input_feature_train_arr): {input_feature_train_arr.shape}")
            print(f"Total Transformed Features: {len(feature_names)}\n")
            print("--- ALL 19 TRANSFORMED COLUMNS ---")
            for i, name in enumerate(feature_names, 1):
                print(f"{i:02d}. {name}")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            print(f"Shape of train_arr after transformation & target concatenation: {train_arr.shape}")
            print(f"Shape of test_arr after transformation & target concatenation: {test_arr.shape}")

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except ValueError as e:
            # format_exc() returns the full traceback as a readable string
            print("Exception : \n", traceback.format_exc())