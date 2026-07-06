import sys
import os
from src.exception import CustomException
from src.logger import logging as log
from dataclasses import dataclass
from src.utils import save_object
import pandas as pd
import numpy as np




from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder,StandardScaler

@dataclass
class DataTransformationConfig :
    preprocessor_obj_file_path= os.path.join("artifacts","preprocessor.pkl")

class DataTrasformation : 
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) :
        try :
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler(with_mean=False))
                ])
            
            cat_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("ONE Hot Encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ])

            log.info("Numerical Columns Std Scaling Completed.")
            log.info("Categorical Columns Std Scaling Completed.")

            preprocessor = ColumnTransformer([("Num Transformation",num_pipeline,numerical_columns),("Cat Transformation",cat_pipeline,categorical_columns)])

            return preprocessor
        
        except Exception as e :
         raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path) :

        try :
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            log.info("Obtaining preprocessor Object ")
            
            preprocessing_obj = self.get_data_transformer_object()

            target_col = "math_score"
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            input_feature_train_df = train_df.drop(columns=target_col,axis=1)
            target_feature_train_df = train_df[target_col]


            input_feature_test_df = test_df.drop(columns=target_col,axis=1)
            target_feature_test_df =  test_df[target_col]
            log.info("Applying preprocessing onject on training and testing Dataframe .")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_df]

            log.info("Saved preprocessing object")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj= preprocessing_obj) 

            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)




 

   


