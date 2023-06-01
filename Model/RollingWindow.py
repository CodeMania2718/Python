import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("5_Minute_CPU_Usage_Data.csv")
features = ['CPU_Usage']

# train_size = int(0.8 * len(df))
# test_size = len(df) - train_size

# train_data = df[:train_size]
# test_data = df[-test_size:]

with mlflow.start_run():
    window = 24
    rolling_window = df.rolling(window=window)
    rolling_mean = rolling_window.mean()
    rolling_std = rolling_window.std()

    anomaly = np.where((df[features] < rolling_mean - 2*rolling_std) | (df[features] > rolling_mean + 2*rolling_std), -1, 1)
    # model = OneClassSVM(nu=0.1, kernel="linear", gamma=0.00005, degree=5)
    # model.fit(train_data[features])
    # predictions = model.predict(test_data[features])

    mse = mean_squared_error(df[features], anomaly)
    r2 = r2_score(df[features], anomaly)
    

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("df_shape", df.size)
    # mlflow.log_metric("Rolling Mean", rolling_mean)
    # mlflow.log_metric("Rolling Std", rolling_std.values)
    # mlflow.log_metric("train_data_shape", train_data.size)
    # mlflow.log_metric("test_data_shape", test_data.size)

    mlflow.log_param("Window Size", window)
    # mlflow.log_param("gamma", 0.00005)
    # mlflow.log_param("degree", 5)

    # mlflow.log_model(rolling_window, "Rolling_Window")

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
