import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import OneClassSVM
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("5_Minute_CPU_Usage_Data.csv")
features = ['CPU_Usage']

train_size = int(0.8 * len(df))
test_size = len(df) - train_size

train_data = df[:train_size]
test_data = df[-test_size:]

with mlflow.start_run():
    model = OneClassSVM(nu=0.1, kernel="linear", gamma=0.00005, degree=5)
    model.fit(train_data[features])
    predictions = model.predict(test_data[features])

    mse = mean_squared_error(test_data[features], predictions)
    r2 = r2_score(test_data[features], predictions)
    

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("df_shape", df.size)
    mlflow.log_metric("train_data_shape", train_data.size)
    mlflow.log_metric("test_data_shape", test_data.size)

    mlflow.log_param("nu", 0.1)
    mlflow.log_param("kernel", "linear")
    mlflow.log_param("gamma", 0.00005)
    mlflow.log_param("degree", 5)

    mlflow.sklearn.log_model(model, "OneClassSVM")

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
