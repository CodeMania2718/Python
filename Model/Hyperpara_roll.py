import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("5_Minute_CPU_Usage_Data.csv")
features = ['CPU_Usage']

hyperparameters = {
    'window': [6, 12, 18, 24],
    'sigma': [1, 1.5, 2]
}
mlflow.set_experiment("Rolling Window")

for window in hyperparameters['window']:
    for sigma in hyperparameters['sigma']:
        rolling_window = df.rolling(window=window)
        rolling_mean = rolling_window.mean()
        rolling_std = rolling_window.std()

        anomaly = np.where((df[features] < rolling_mean - sigma*rolling_std) | (df[features] > rolling_mean + sigma*rolling_std), -1, 1)

        mse = mean_squared_error(df[features], anomaly)
        r2 = r2_score(df[features], anomaly)

        mse = mean_squared_error(df[features], anomaly)
        r2 = r2_score(df[features], anomaly)
        

        with mlflow.start_run():

            mlflow.log_metric("df_shape", df.size)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            mlflow.log_param("Window Size", window)
            mlflow.log_param("Sigma Value", sigma)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
