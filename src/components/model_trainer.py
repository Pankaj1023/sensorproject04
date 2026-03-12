import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    artifact_folder_path = artifact_folder
    trained_model_path = os.path.join(artifact_folder_path, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join("config", "model.yaml")


class ModelTrainer:
    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

        self.utils = MainUtils()

        self.models = {
            "XGBClassifier": XGBClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(),
            "RandomForestClassifier": RandomForestClassifier(),
        }

    def evaluate_models(self, X, y, models):

        try:
            report = {}

            for model_name, model in models.items():

                model.fit(X, y)

                y_pred = model.predict(X)

                score = accuracy_score(y, y_pred)

                report[model_name] = score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(self, best_model_object, best_model_name, x_train, y_train):

        try:

            model_param_grid = self.utils.read_yaml_file(
                self.model_trainer_config.model_config_file_path
            )["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(
                best_model_object,
                param_grid=model_param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1,
            )

            grid_search.fit(x_train, y_train)

            best_params = grid_search.best_params_

            logging.info(f"Best parameters: {best_params}")

            finetuned_model = best_model_object.set_params(**best_params)

            return finetuned_model

        except Exception as e:
            raise CustomException(e, sys)

    def initiated_model_trainer(self, train_array, test_array):

        try:

            logging.info("Splitting training and testing input features")

            x_train, y_train = train_array[:, :-1], train_array[:, -1]

            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            model_report = self.evaluate_models(
                X=x_train, y=y_train, models=self.models
            )

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_object = self.models[best_model_name]

            logging.info(f"Best model found: {best_model_name}")

            best_model = self.finetune_best_model(
                best_model_object=best_model_object,
                best_model_name=best_model_name,
                x_train=x_train,
                y_train=y_train,
            )

            best_model.fit(x_train, y_train)

            y_pred = best_model.predict(x_test)

            best_model_score = accuracy_score(y_test, y_pred)

            print(f"Best model: {best_model_name}, Accuracy: {best_model_score}")

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception("No best model found with required accuracy")

            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_path}"
            )

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model,
            )

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)