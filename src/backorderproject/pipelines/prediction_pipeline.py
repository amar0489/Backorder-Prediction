# Prediction pipeline that takes the user input and gives predictions using the trained model

import sys
import os
from src.backorderproject.utils import load_object
from src.backorderproject.exception import CustomException
from src.backorderproject.logger import logging




class PredictionPipeline:
    def __init__(self, model_path, preprocessor_path):
        # Load the model and preprocessor

        model_path = os.path.join("artifacts", "model.pkl.gz")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

        self.model = load_object(file_path=model_path,compression=True)
        self.preprocessor= load_object(file_path=preprocessor_path)


    def predict(self, input_data):
        """
        Takes raw input data, processes it, and returns the prediction.
        """

        try:

            logging.info('Transforming user input data')
            
            # Applying preprocessing before passing to the model
            preprocessed_data = self.preprocessor.transform(input_data)
        
            # Get the prediction from the model
            prediction = self.model.predict(preprocessed_data)

            logging.info('Model Prediction done')
        
            # Return the prediction result
            return "Yes, product expected to go on backorder" if prediction == 1 else "No backorder situation expected"
        
        except Exception as ex:
            raise CustomException(ex, sys)

        


