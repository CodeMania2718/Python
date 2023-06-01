''' This file contains functions which helps with the predictions '''
from preprocess import *
import pickle
import glob
from pathlib import Path
# from training import *
import os
from pmdarima.arima import auto_arima

''' Getting the latest model name '''
list_of_files = glob.glob('./models/*') 
latest_file = max(list_of_files, key=os.path.getctime)
filename = Path(latest_file)
# print(filename)
''' Loading the model '''
model = pickle.load(open(filename,'rb'))

class Model_Output:
    ''' This class contains all the functions which helps for predictiion. '''
    def Multiple_Prediction(self,number_of_points):
        # Function to get next 'n' number of predictions from the saved model

        data_for_training = Pre_Process().pre_processing()

        predictions = model.predict(n_periods = number_of_points)

        return predictions
    
    def prediction(self,real_disk_usage):
        ''' Get Prediction for the UI (Or the input value) '''
        # # Loading latest model 
        # list_of_files = glob.glob('.//models//*') 
        # latest_file = max(list_of_files, key=os.path.getctime)
        # filename = Path(latest_file)
        # model = pickle.load(open(filename,'rb'))

        with open("data.yaml", "r") as yamlfile:
            # Load the YAML data into a Python object
            config_data = yaml.safe_load(yamlfile)
        
        predict_disk_usage = model.predict(n_periods=1)
        raw_mean = config_data["rolling_error_mean"]
        raw_std = config_data["rolling_error_std"]

        # checking model output and real output with stardard deviation
        error = real_disk_usage - predict_disk_usage

        self.prediction = predict_disk_usage
        # It checkin
        upper_3std = raw_mean+3*raw_std
        lower_3std = raw_mean-3*raw_std

        upper_1_75_std = raw_mean+1.75*raw_std
        lower_1_75_std = raw_mean-1.75*raw_std

        upper_1_5_std = raw_mean+1.5*raw_std
        lower_1_5_std = raw_mean-1.5*raw_std


        if error > upper_3std or error <lower_3std:
            return 3

        elif error > upper_1_75_std or error <lower_1_75_std:
            return  2
        elif error > upper_1_5_std or error <lower_1_5_std:
            return  1

        else:
            return -1
    
    def append_new_value(self,old_dataframe,value):
        ''' This function helps in appending new incoming values to the dataframe '''
        new_row = {"Final_Val": value}
        last_index = old_dataframe.index[-1]

        new_row_index = last_index + pd.Timedelta(minutes=5)
        new_df = pd.DataFrame(new_row, index=[new_row_index])
        new_dataframe = old_dataframe.append(new_df)

        return new_dataframe
    
    def train_model_whole_data(self,df_data):
        ''' This function helps training the model for new dataframe '''
        sarima_order = (2,0,0)
        seasonal_order = (1,0,1,23)     
        model = auto_arima(df_data['Final_Val'], seasonal=True, suppress_warnings=True)

        return model
    
    def multiplePredictionwithCompare(self,real_disk_usage,number_point = 10):
        ''' This function predicts new data points append it to data frame and retrain the model '''
        
        ''' Getting the latest model name '''
        list_of_files = glob.glob('./models/*') 
        latest_file = max(list_of_files, key=os.path.getctime)
        filename = Path(latest_file)
        ''' Loading the model '''
        model = pickle.load(open(filename,'rb'))

        window_size = number_point

        train_df = Pre_Process().pre_processing()

        train_df.index = pd.to_datetime(train_df.index) 

        new_row = {"Final_Val": real_disk_usage}
        last_index = train_df.index[-1]

        rolling_mean_n_point = train_df.iloc[-10:]["Final_Val"].mean()
        rolling_std_n_point  = train_df.iloc[-10:]["Final_Val"].std()

        # Dumping rolling mean and std in yaml file
        with open("data.yaml", "w") as yamlfile:
            # Load the YAML data into a Python object
            param_dict = {'rolling_error_mean':float(rolling_mean_n_point),'rolling_error_std':float(rolling_std_n_point),'windowSize':window_size}
            yaml.dump(param_dict,yamlfile)

        new_row_index = last_index + pd.Timedelta(minutes=5)
        new_df = pd.DataFrame(new_row, index=[new_row_index])

        train_df = train_df.append(new_df)

        model_disk_pred = np.array([])
        model_disk_pred_anomaly = np.array([])

        # print("----- Prediction Started -----")

        for window in range(window_size):
            is_anamoly = False
            disk_usage_model_pred = model.predict(n_periods=1)
            # return disk_usage_model_pred
            # 3 std devition
            upper_limit_3std = rolling_mean_n_point - 3*rolling_std_n_point
            lower_limit_3std = rolling_mean_n_point + 3*rolling_std_n_point
            # 1.75 std devition
            # upper_limit_1_75std = rolling_mean_n_point - 1.75*rolling_std_n_point
            # lower_limit_1_75std  = rolling_mean_n_point + 1.75*rolling_std_n_point
        
            if disk_usage_model_pred<lower_limit_3std or disk_usage_model_pred>upper_limit_3std:
                is_anamoly =  True


            model_disk_pred = np.append(model_disk_pred,disk_usage_model_pred)
            if is_anamoly:
                train_df = self.append_new_value(train_df,rolling_mean_n_point)
                model_disk_pred_anomaly = np.append(model_disk_pred_anomaly,1)
            else:
                train_df = self.append_new_value(train_df,disk_usage_model_pred)
                model_disk_pred_anomaly = np.append(model_disk_pred_anomaly,0)
            
            
            rolling_mean_n_point = train_df.iloc[-10:]["Final_Val"].mean()
            rolling_std_n_point  = train_df.iloc[-10:]["Final_Val"].std()
            
            # model = Model_Output().train_model_whole_data(train_df)

        return model_disk_pred, model_disk_pred_anomaly

if __name__=="__main__":
    # print(Model_Output().multiplePredictionwithCompare(34))
    print(Model_Output().prediction(34))