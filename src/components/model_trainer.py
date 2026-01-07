import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (mean_squared_error,r2_score,mean_absolute_error)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor)
from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import (save_object,evaluate_model)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self , train_array , test_array):
        try:
            logging.info("Split training and testing input data")

            X_train , y_train , X_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "Lasso" : Lasso(),
                "Ridge" : Ridge(),
                "K-neighbors regressor" : KNeighborsRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoostRegressor" : CatBoostRegressor(verbose=False),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "GradientRegressor" : GradientBoostingRegressor()
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            if best_model_score < 0.6 :
                raise CustomException("No best model found")
            
            best_model = models[best_model_name]

            best_model.fit(X_train,y_train)
            
            logging.info("Best found model from train and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(X_test)
            r2_sq = r2_score(y_test,predicted)

            return r2_sq
        except Exception as e:
            raise CustomException(e,sys)