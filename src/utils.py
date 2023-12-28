import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        # giving the permission to read or write
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            # all params
            # params = list(model.keys())[i]

            # hyperparmeter tunning
            # gs=GridSearchCV(model,param_grid=params,cv=2)
            # gs.fit(X_train,y_train)

            # getting the best parameter
            # model.set_params(**gs.best_params_)
            # train model
            model.fit(X_train, y_train)

            # y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            return report
    except Exception as e:
        raise CustomException(e, sys)


# saving the val
def save_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.dump(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
