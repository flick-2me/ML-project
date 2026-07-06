import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path,obj) :
    try :
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb")  as f: 
            dill.dump(obj,f)
    except Exception as e :
        raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models) :
    try : 
        report = {}
        for i in range(len(list(models.values()))) :
            model = list(models.values())[i]
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)

            r2 = r2_score(y_test,y_pred)

            report[list(models.keys())[i]] = r2
        
        return report
    except Exception as e :
        raise CustomException(e,sys)




