''' Training Script for training and saving the model '''

''' This files contains the script for training and saving the model '''
import pandas as pd
import numpy as np
import yaml
import time as t
import pickle
from pmdarima.arima import auto_arima
from datetime import date
from sklearn.svm import OneClassSVM
from preprocess import *

# Getting the data using Pre_Process Script
data = Pre_Process().pre_processing()

class Train_and_Save:
    ''' This class helps in training and saving the model to Model Registry '''
    def training_and_saving(self,data):
        ''' Function to train and save the model '''

        print("---------- Training Started ----------")

        model = auto_arima(data['Final_Val'], seasonal=True, suppress_warnings=True)
        
        date_time = str(date.today())

        print("----- Model Training Successfull -----")

        filename = f"models//Auto_Arima_Memory_{date_time}.pkl"

        pickle.dump(model, open(filename, 'wb'))

        print("---------- Model Saved ----------")

        # Writing HyperParameters to YAML file
        file_config_name = f"model_config//Auto_Arima_Memory_{date_time}"

        with open(f'{file_config_name}.yaml', 'w') as f:
            yaml.dump(model.get_params(), f)

        print("----- Model Hyperparameter Saved -----")

        return filename


filename = Train_and_Save().training_and_saving(data)