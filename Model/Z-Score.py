import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("5_Minute_CPU_Usage_Data.csv")
features = ['CPU_Usage']

scaler = StandardScaler()
data_norm = scaler.fit_transform(df[features])
with mlflow.start_run():
    threshold = 2
    model_pred = np.abs((data_norm - np.mean(data_norm)) / np.std(data_norm))
    model_err = np.where(model_pred > threshold, 1, 0)


    mse = mean_squared_error(df[features], model_err)
    r2 = r2_score(df[features], model_err)
    

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("df_shape", df.size)

    mlflow.log_param("threshold", threshold)
    mlflow.log_param("mean", scaler.mean_)
    mlflow.log_param("std", scaler.scale_)

    mlflow.sklearn.log_model(scaler, "StandardScaler_model")

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
