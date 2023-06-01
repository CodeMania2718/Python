import mlflow
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.eval_measures import rmse

# Load and preprocess data
data = pd.read_csv("5_Minute_CPU_Usage_Data.csv")
# preprocess the data as required

# Split the data into training and testing sets
train_size = int(0.8 * len(data))
test_size = len(data) - train_size

train_data = data[:train_size]
test_data = data[-test_size:]

# Define function to calculate evaluation metrics
def calculate_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mse, mae, mape

# Define function to train and evaluate SARIMA models
def train_eval_sarima(m):
    model = pm.auto_arima(train_data["CPU_Usage"], seasonal=True, m=m, suppress_warnings=True, error_action="ignore")
    fitted_model = model.fit(train_data["CPU_Usage"])
    # preds = []
    # for i in range(len(test_data)):
    pred = fitted_model.predict(start=len(train_data), end=len(train_data) + len(test_data))
        # preds.append(pred[pred.iloc[0]])
        # len(train_data + i)
        # pred.iloc[0]
    aic = fitted_model.aic()
    bic = fitted_model.bic()
    mse, mae, mape = calculate_metrics(test_data["CPU_Usage"], pred)
    return mse, mae, mape, aic, bic, model.order, model.seasonal_order

# Define function to train and evaluate Auto ARIMA models
def train_eval_auto_arima():
    model = pm.auto_arima(train_data["CPU_Usage"], seasonal=True, suppress_warnings=True, error_action="ignore")
    # preds = []
    # for i in range(len(test_data)):
    pred = model.predict(start=len(train_data) + i, end=len(train_data) + i)
        # preds.append(pred[pred.iloc[0]])
    aic1 = model.aic()
    bic1 = model.bic()
    mse, mae, mape = calculate_metrics(test_data["CPU_Usage"], preds)
    return mse, mae, mape, aic1, bic1, model.order, model.seasonal_order

mlflow.set_experiment("Arima-model-iter")

# Train and evaluate SARIMA and Auto ARIMA models with automatically selected orders
with mlflow.start_run():
    # Train and evaluate Auto ARIMA model
    mse, mae, mape, aic, bic, order, order1 = train_eval_auto_arima()
    # Log the evaluation metrics and model order to MLflow
    mlflow.log_metric("ARIMA MSE", mse)
    mlflow.log_metric("ARIMA MAE", mae)
    mlflow.log_metric("ARIMA MAPE", mape)
    mlflow.log_metric("ARIMA AIC", aic)
    mlflow.log_metric("ARIMA BIC", bic)
    mlflow.log_param("ARIMA Order", order)
    mlflow.log_param("ARIMA Seasonal order", order1)

for m in [6, 12, 18, 24]:
    with mlflow.start_run():
        # Train and evaluate SARIMA model
        mse, mae, mape, aic, bic, order, order1 = train_eval_sarima(m)
        # Log the evaluation metrics and model order to MLflow
        mlflow.log_metric("ARIMA MSE", mse)
        mlflow.log_metric("ARIMA MAE", mae)
        mlflow.log_metric("ARIMA MAPE", mape)
        mlflow.log_metric("ARIMA AIC", aic)
        mlflow.log_metric("ARIMA BIC", bic)
        mlflow.log_param("ARIMA Order", order)
        mlflow.log_param("ARIMA Seasonal order", order1)

