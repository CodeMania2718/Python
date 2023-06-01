import mlflow
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.eval_measures import rmse


data = pd.read_csv("5_Minute_CPU_Usage_Data.csv")

train_size = int(0.8 * len(data))
test_size = len(data) - train_size

train_data = data[:train_size]
test_data = data[-test_size:]

model = pm.auto_arima(train_data["CPU_Usage"], seasonal=True, suppress_warnings=True)
mlflow.set_experiment("Arima model")

for m in [6, 12, 18, 24]:
    with mlflow.start_run():
        sarima_model = SARIMAX(train_data["CPU_Usage"], order = model.order, seasonal_order=(model.order[0],model.order[1],model.order[2],m))
        model1 = sarima_model.fit()
        pred = model1.predict(n_periods=len(test_data))
        aic = model1.aic
        bic = model1.bic
        mse = mean_squared_error(test_data["CPU_Usage"], pred)
        mae = mean_absolute_error(test_data["CPU_Usage"], pred)
        mape = np.mean(np.abs((test_data["CPU_Usage"] - pred) / test_data["CPU_Usage"])) * 100
        mlflow.log_metric("ARIMA MSE", mse)
        mlflow.log_metric("ARIMA MAE", mae)
        mlflow.log_metric("ARIMA MAPE", mape)
        mlflow.log_metric("ARIMA AIC", aic)
        mlflow.log_metric("ARIMA BIC", bic)
        mlflow.log_param("ARIMA Order", sarima_model.order)
        mlflow.log_param("ARIMA Seasonal order", sarima_model.seasonal_order)

with mlflow.start_run():
    pred = model.predict(n_periods=len(test_data))
    aic = model.aic()
    bic = model.bic()
    mse = mean_squared_error(test_data["CPU_Usage"], pred)
    mae = mean_absolute_error(test_data["CPU_Usage"], pred)
    mape = np.mean(np.abs((test_data["CPU_Usage"] - pred) / test_data["CPU_Usage"])) * 100
    mlflow.log_metric("ARIMA MSE", mse)
    mlflow.log_metric("ARIMA MAE", mae)
    mlflow.log_metric("ARIMA MAPE", mape)
    mlflow.log_metric("ARIMA AIC", aic)
    mlflow.log_metric("ARIMA BIC", bic)
    mlflow.log_param("ARIMA Order", model.order)
    mlflow.log_param("ARIMA Seasonal order", model.seasonal_order)
