import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
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

hyperparameters = {
    'n_estimators': [10, 30, 50, 100],
    'max_samples': [50, 100, 500, 1000],
    'contamination': ["auto", 0.01, 0.02, 0.05]
}
mlflow.set_experiment("Isolation Forest CPU")

for n_estimators in hyperparameters['n_estimators']:
    for max_samples in hyperparameters['max_samples']:
        for contamination in hyperparameters['contamination']:
            model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                    contamination=contamination, n_jobs=1, random_state=1)
            
            model.fit(train_data[features])
            predictions = model.predict(test_data[features])

            mse = mean_squared_error(test_data[features], predictions)
            r2 = r2_score(test_data[features], predictions)
            decision_function = model.decision_function(test_data[features])

            with mlflow.start_run():

                mlflow.log_metric("df_shape", df.size)
                mlflow.log_metric("train_data_shape", train_data.size)
                mlflow.log_metric("test_data_shape", test_data.size)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("decision_function", np.mean(decision_function))

                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_samples", max_samples)
                mlflow.log_param("contamination", contamination)
                mlflow.log_param("n_jobs", -1)
                mlflow.log_param("random_state", 1)

                mlflow.sklearn.log_model(model, "Isolation_Forest_model")

                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
