''' This script contains the functions required to load and preprocess the data for further use '''
import pandas as pd
import numpy as np
import yaml
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose

class Pre_Process:
    ''' Class to load and preprocess the data '''

    def pre_processing(self):
        # Loading data from CSV
        df = pd.read_csv("new_data_preprocessed.csv", parse_dates=True, index_col='Timestamp')

        mean = df["Minimum Memory Used"].mean()
        std = df["Minimum Memory Used"].std()

        upper_bound = mean + 2 * std
        lower_bound = mean - 2 * std
        anomalies = df[(df["Minimum Memory Used"] > upper_bound) | (df["Minimum Memory Used"] < lower_bound)]

        df['Minimum Memory Used']=df['Minimum Memory Used'].replace(anomalies['Minimum Memory Used'].values, np.NaN)
        df['Minimum Memory Used']=df['Minimum Memory Used'].interpolate()
        
        df = df[df['IP Address'] == '10.16.11.252']
        df.drop(['IP Address','weekday'], axis=1, inplace=True)

        decomposition = seasonal_decompose(df, model='multiplicative', period=8)

        df1 = df.copy()
        df1['residuals'] = decomposition.resid
        df1=df1.fillna(0)
        
        df1['Final_value'] = df1['Minimum Memory Used'] - df1['residuals']

        df1.drop((['Minimum Memory Used','residuals']),axis=1,inplace=True)
        df1.rename(columns = {'Final_value':'Final_Val'}, inplace = True)
        return df1
    
# df = Pre_Process().pre_processing()
# print(df)