''' Batch Prediction using the Model instead of API ( Feeding live CPU data )'''

import psutil
import datetime
import pandas as pd
import glob
import pickle
import os
import yaml
from pathlib import Path
from action_script import *
from prediction import *
import warnings
warnings.filterwarnings('ignore')

with open('AIOps_configuration.yaml') as f:
    my_dict = yaml.load(f,Loader=yaml.SafeLoader)

''' Getting the latest model name '''
list_of_files = glob.glob('./models/*') 
latest_file = max(list_of_files, key=os.path.getctime)
filename = Path(latest_file)

''' Loading the model and predicting '''
model = pickle.load(open(filename,'rb'))


def dataframe_processing(data_df):
    ''' Function to apply conditions on the dataframe and get Anomaly State, Usage Stage, Usage Category and Action Required '''

    # Generating anomaly state
    data_df['Anomaly_State'] = ["Normal" if x == 1 else "Anomaly" for x in data_df['Anomaly_Value']]

    # Running the action script to assign stage, category and action required
    data_df['Usage_Stage'] = [metric1_assign_Stage(x) for x in data_df['Mean_Memory_Usage']]
    data_df['Usage_Category'] = [metric1_assign_category(x) for x in data_df['Usage_Stage']]

    data_df['Action_Required'] = [assign_action(x) for x in data_df['Usage_Category']]

    return data_df

    # End of dataframe_processing()

def get_action_required(df,f_df):
    ''' Function to utilize data values of last configured minutes (for eg: last 5 minutes) 
    and get a prediction about action requried '''
    # if(df.empty):
    #     df = f_df

    # now = datetime.datetime.now()
    # prev_time = now - datetime.timedelta(hours=my_dict['Metric_1_Time_To_Monitor']['last_hours'],
    # minutes=my_dict['Metric_1_Time_To_Monitor']['last_minutes'],seconds=my_dict['Metric_1_Time_To_Monitor']['last_seconds'])

    # df = df[(df['Datetime'] > prev_time) & (df['Datetime'] <= now)]

    # avg_of_usage = df['Mean_Memory_Usage'].mean()

    # predict_val = model.predict([[avg_of_usage]])

    # f_df['Anomaly_Value'] = predict_val
    avg_df = pd.DataFrame(f_df, index=[0])
    avg_df = dataframe_processing(avg_df)

    return avg_df


print("------------------------------------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------------------------------")

# Fetching last 'N' values from the csv to apply operations on
# data_to_operate_on = pd.read_csv("5_Minute_CPU_Usage_Data.csv")
# data_to_operate_on = pd.DataFrame(data_to_operate_on).tail(5)
# final df to provied the results

def batch_pred_function():
        
    final_data_df = pd.DataFrame()

    ### Generating Data ###
    for i in range(5):
        # Creating temporary dataframe
        # temp_df_col = data_to_operate_on.iloc[i]
        temp_df = pd.DataFrame()

        # Getting CPU Data and datetime
        # Memory_Data, date_time = psutil.cpu_percent(interval=1), datetime.datetime.now()
        Memory_Data,date_time = psutil.virtual_memory().percent,datetime.datetime.now()

        # Creating a data dict 
        data = {"Datetime": date_time, "Memory_Usage": Memory_Data,"Forecasted_values":0}

        # Creating dataframe from dict to run prediction and functions
        temp_df = pd.DataFrame(data, index=[0])
        temp_df.reset_index(inplace = True)
        temp_df.drop(['index'],axis=1,inplace=True)
        
        # Getting prediction from the model
        # print(temp_df['Memory_Usage'].values[0])
        predictions = Model_Output().multiplePredictionwithCompare(temp_df['Memory_Usage'].values[0])[0]
        predictions_anomaly = Model_Output().multiplePredictionwithCompare(temp_df['Memory_Usage'].values[0])[1].mean()
        if(predictions_anomaly >= 0.75):
            predictions_anomaly = -1
        else:
            predictions_anomaly = 1
        # temp_df['Forecasted_values'] = 0
        temp_df['Forecasted_values'][0] = np.array(predictions, dtype=object)
        temp_df['Mean_Memory_Usage'] = np.array(predictions).mean()
        # print(temp_df)
        # prediction_vals = model.predict(temp_df[['Memory_Usage']])
        # Appeding Data and Predictions to the Data frame
        # temp_data_df = temp_data_df.append(temp_df)
        temp_df['Anomaly_Value'] = predictions_anomaly

        # Applying function to get Usage State, Usage Category and Action Required Value
        temp_data_df = get_action_required(final_data_df,temp_df)
        pd.set_option('max_columns', 10)
        pd.set_option('display.width', 2000)
        print(temp_data_df)
        # print(temp_data_df.to_string(index=False))

        # Creating the final DataFrame
        final_data_df = final_data_df.append(temp_data_df)

    # Printing out the dataframe
    final_data_df.reset_index(inplace=True)
    final_data_df.drop(['index'],axis=1,inplace=True)

    # Getting action for last input time values
    # get_action_required(final_data_df)

    ''' Store final_data_df to datastorage '''

    print("------------------------------------------------------------------------------------------------------------------------")
    # print(final_data_df)
    print("------------------------------------------------------------------------------------------------------------------------")

    return final_data_df

if __name__ == "__main__":
    fdf = batch_pred_function()