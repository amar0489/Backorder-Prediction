# Model Training

import os
import sys
from dataclasses import dataclass

from src.backorderproject.exception import CustomException
from src.backorderproject.logger import logging
from src.backorderproject.utils import save_object, evaluate_models

from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier      
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier



@dataclass

class ModelTrainingConfig:
    trained_model_file_path= os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainingConfig()

    
    def initiate_model_trainer(self,datasets):
        try:
            combined_model_report= {}

            for sampling_strategy, (train_array,test_array) in datasets.items():
                logging.info(f"Splitting train and test inputs {sampling_strategy}")
                X_train,y_train,X_test,y_test= (
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
                )

                print(f"Unique values:", set(y_train))
                print(f"Unique values:", set(y_test))

                if sampling_strategy == "oversample":

                    logging.info("Model Training using ROS started")

                    models = {
                        "Gradient Boosting ROS": GradientBoostingClassifier(n_estimators=10,min_samples_leaf=5,max_depth=10,random_state=42),
                        "XGBoost ROS": xgb.XGBClassifier(random_state=42)
                    }
            
                elif sampling_strategy == "undersample":

                    logging.info("Model Training with RUS started")

                    models = {
                        "Logistic Regression RUS": LogisticRegression(C= 10,solver= 'newton-cholesky',random_state=42),
                        "Linear SVM RUS": LinearSVC(loss='hinge',random_state=42),
                        "Decision Tree RUS": DecisionTreeClassifier(max_depth=7,min_samples_leaf=5,random_state=42),
                        "Random Forest RUS": RandomForestClassifier(n_estimators=10,min_samples_leaf=5,criterion='entropy',max_depth=15,random_state=42),
                        "Gradient Boosting RUS": GradientBoostingClassifier(n_estimators=10,min_samples_leaf=5,max_depth=7,random_state=42),
                        "XGBoost RUS": xgb.XGBClassifier(random_state=42),
                        "LightGBM RUS": lgb.LGBMClassifier(random_state=42),
                    }

                else:

                    logging.info("Model Training for ensemble imblearn techniques started")

                    models = {
                        "Balanced Random Forest": BalancedRandomForestClassifier(n_estimators=200,bootstrap=True,replacement=False,sampling_strategy='auto',random_state=42),
                        "Easy Ensemble": EasyEnsembleClassifier(random_state=42)
                    }

                model_report = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

                logging.info("Evaluated Models corresponding to the sampling techniques")

                combined_model_report.update(model_report)
            
            logging.info(f"Combined Model Report sucessfully")
            
            best_model_name = max(combined_model_report.keys(), key=lambda model_name: combined_model_report[model_name]["Recall"])
            best_model_metrics = combined_model_report[best_model_name]

            logging.info(f"Best Model: {best_model_name} with metrics: {best_model_metrics}")

            logging.info("Saving the best model")

            best_model= None

            for sampling_strategy, (train_array, test_array) in datasets.items():
                models = {
                    "oversample": {
                        "Gradient Boosting ROS": GradientBoostingClassifier(
                            n_estimators=10, min_samples_leaf=5, max_depth=10, random_state=42
                        ),
                        "XGBoost ROS": xgb.XGBClassifier(random_state=42),
                    },
                    "undersample": {
                        "Logistic Regression RUS": LogisticRegression(
                            C=10, solver="newton-cholesky", random_state=42
                        ),
                        "Linear SVM RUS": LinearSVC(
                            loss="hinge", random_state=42
                        ),
                        "Decision Tree RUS": DecisionTreeClassifier(
                            max_depth=7, min_samples_leaf=5, random_state=42
                        ),
                        "Random Forest RUS": RandomForestClassifier(
                            n_estimators=10,
                            min_samples_leaf=5,
                            criterion="entropy",
                            max_depth=15,
                            random_state=42,
                        ),
                        "Gradient Boosting RUS": GradientBoostingClassifier(
                            n_estimators=10, min_samples_leaf=5, max_depth=7, random_state=42
                        ),
                        "XGBoost RUS": xgb.XGBClassifier(random_state=42),
                        "LightGBM RUS": lgb.LGBMClassifier(random_state=42),
                    },
                    "none": {
                        "Balanced Random Forest": BalancedRandomForestClassifier(
                            n_estimators=200,
                            bootstrap=True,
                            replacement=False,
                            sampling_strategy="auto",
                            random_state=42,
                        ),
                        "Easy Ensemble": EasyEnsembleClassifier(random_state=42),
                    },
                }

                if best_model_name in models[sampling_strategy]:
                    best_model = models[sampling_strategy][best_model_name]
                    break

            if best_model is not None:

                best_model.fit(X_train,y_train)

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model,
                    compression=True                        # Model file compressed [Exceeding 100MB file size]
                )           

                logging.info(f"Best model saved successfully")

                return best_model_name, best_model_metrics
            
            else:
                logging.warning("Best model not found for saving.")
                return None

        except Exception as ex:
            raise CustomException(ex, sys)