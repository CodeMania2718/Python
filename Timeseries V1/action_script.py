''' Script to assign action to Memory Data '''

import pandas as pd 
import numpy as np
import yaml
from yaml import SafeLoader

# df =  pd.read_csv("CPU_Utilization_(Percent)-2023_01_12_11_59_00-2023_01_19_11_57_00-UTC.csv")

# df_copy = df.copy()

with open('AIOps_configuration.yaml') as f:
    my_dict = yaml.load(f,Loader=SafeLoader)

# print("--------Choose Metric--------")
# print("1 for Metric 1")
# print("2 for Metric 2")
# print("3 for Metric 3")
# ch = input("Enter Metric Value -----> ")

metric_name = "Metric_" + "2" + "_config"

METRIC1_STAGE1_LOWER_THRESHOLD = my_dict[metric_name]['Stage_1_lower_threshold']
METRIC1_STAGE1_UPPER_THRESHOLD = my_dict[metric_name]['Stage_1_Upper_threshold']
METRIC1_STAGE2_LOWER_THRESHOLD = my_dict[metric_name]['Stage_2_lower_threshold']
METRIC1_STAGE2_UPPER_THRESHOLD = my_dict[metric_name]['Stage_2_Upper_threshold']
METRIC1_STAGE3_LOWER_THRESHOLD = my_dict[metric_name]['Stage_3_lower_threshold']
METRIC1_STAGE3_UPPER_THRESHOLD = my_dict[metric_name]['Stage_3_Upper_threshold']
METRIC1_STAGE4_LOWER_THRESHOLD = my_dict[metric_name]['Stage_4_lower_threshold']
METRIC1_STAGE4_UPPER_THRESHOLD = my_dict[metric_name]['Stage_4_Upper_threshold']
METRIC1_STAGE5_LOWER_THRESHOLD = my_dict[metric_name]['Stage_5_Upper_threshold']
METRIC1_STAGE5_UPPER_THRESHOLD = my_dict[metric_name]['Stage_5_Upper_threshold']
METRIC1_STAGE6_LOWER_THRESHOLD = my_dict[metric_name]['Stage_6_lower_threshold']
METRIC1_STAGE6_UPPER_THRESHOLD = my_dict[metric_name]['Stage_6_Upper_threshold']
METRIC1_STAGE7_UPPER_THRESHOLD = my_dict[metric_name]['Stage_7_lower_threshold']
METRIC1_STAGE7_UPPER_THRESHOLD = my_dict[metric_name]['Stage_7_Upper_threshold']


''' Function to Assign Stage '''
def metric1_assign_Stage(val):
    if(val >= METRIC1_STAGE1_LOWER_THRESHOLD and val <= METRIC1_STAGE1_UPPER_THRESHOLD):
        return "Stage1"
    elif(val > METRIC1_STAGE2_LOWER_THRESHOLD and val <= METRIC1_STAGE2_UPPER_THRESHOLD):
        return "Stage2"
    elif(val > METRIC1_STAGE3_LOWER_THRESHOLD and val <= METRIC1_STAGE3_UPPER_THRESHOLD):
        return "Stage3"
    elif(val > METRIC1_STAGE4_LOWER_THRESHOLD and val <= METRIC1_STAGE4_UPPER_THRESHOLD):   
        return "Stage4"
    elif(val > METRIC1_STAGE5_LOWER_THRESHOLD and val <= METRIC1_STAGE5_UPPER_THRESHOLD):
        return "Stage5"
    elif(val > METRIC1_STAGE6_LOWER_THRESHOLD and val <= METRIC1_STAGE6_UPPER_THRESHOLD):
        return "Stage6"
    elif(val > METRIC1_STAGE7_UPPER_THRESHOLD and val <= METRIC1_STAGE7_UPPER_THRESHOLD):
        return "Stage7"

''' Function to Assign Category '''
def metric1_assign_category(val):
    if(val == "Stage1"):
        return "Very Low"
    elif(val == "Stage2"):
        return "Low"
    elif(val == "Stage3"):
        return "Normal"
    elif(val == "Stage4"):
        return "Normal"
    elif(val == "Stage5"):
        return "Normal"
    elif(val == "Stage6"):
        return "High"
    elif(val == "Stage7"):
        return "Very High"

''' Function to assign Action '''
def metric_assign_action(val):
    if(val == "Very Low"):
        return "Action 1"
    elif(val == "Low"):
        return "Action 1"
    elif(val == "Normal"):
        return "None"
    elif(val == "High"):
        return "Action 2"
    elif(val == "Very High"):
        return "Action 2"


def assign_action(val):
    ''' Function to help the dataframe_preprocessing() function in getting the action requried value ''' 

    temp = ""
    if(val == "Very Low"):
        temp =  "Action_1"
    elif(val == "Low"):
        temp =  "Action_1"
    elif(val == "Normal"):
        temp =  "None"
    elif(val == "High"):
        temp =  "Action_2"
    elif(val == "Very High"):
        temp =  "Action_2"
  
    if temp == "Action_1":
        return my_dict['Metric_2']['Action_1']
    elif temp == "Action_2":
        return my_dict['Metric_2']['Action_2']
    else:
        return "No Action Required"

    # End of assign_action() 

