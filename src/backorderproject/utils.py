import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import precision_recall_curve, auc, roc_curve, recall_score

from src.backorderproject.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)


    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:

        model_report= {}

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)


        for model_name, model in models.items():

            if model_name == "Linear SVM RUS":
                model.fit(X_train,y_train)

                y_test_pred = model.decision_function(X_test)

                precision, recall,_ = precision_recall_curve(y_test, y_test_pred)
                pr_score_test = auc(recall, precision)
                fpr, tpr, _ = roc_curve(y_test, y_test_pred)
                roc_score_test = auc(fpr, tpr)

                y_test_predict = model.predict(X_test)
                recall = recall_score(y_test, y_test_predict)

            else:
                model.fit(X_train,y_train)

                y_test_pred = model.predict_proba(X_test)[:,1]

                precision, recall,_ = precision_recall_curve(y_test, y_test_pred)
                pr_score_test = auc(recall, precision)
                fpr, tpr, _ = roc_curve(y_test, y_test_pred)
                roc_score_test = auc(fpr, tpr)

                y_test_predict= model.predict(X_test)
                recall= recall_score(y_test,y_test_predict)


            model_report[model_name] = {
                    "ROC Score Test": roc_score_test,
                    "PR Score Test": pr_score_test,
                    "Recall": recall
                }

        return model_report

    except Exception as e:
        raise CustomException(e, sys)