import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from urllib.parse import urlparse
from pathlib import Path
from stb_pfe_mlflow.utils.common import save_json
from stb_pfe_mlflow.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore", message="Setuptools is replacing distutils")


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        accuracy = accuracy_score(actual, pred)  # Compute accuracy
        return rmse, mae, r2, accuracy

    def log_into_mlflow(self):
        # Load test data
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Extract target and features
        test_y = test_data[self.config.target_column]
        test_x = test_data.drop(
            columns=["tiers_key", "Cluster"]
        )  # Drop 'tiers_key' and target column 'Cluster'

        # New encoders and scalers saved during training
        label_encoder = LabelEncoder()
        scaler = StandardScaler()

        # Encode categorical features in the same way as in training
        for col in test_x.select_dtypes(include=["object"]).columns:
            # Fit the LabelEncoder on the unique values of the current column
            label_encoder.fit(
                test_x[col].astype(str)
            )  # Fitting the encoder on the current test column
            test_x[col] = label_encoder.transform(test_x[col].astype(str))

        # Encode the target labels (ensure it matches the training labels)
        test_y_encoded = label_encoder.fit_transform(test_y)

        # Standardize the features as in the training step
        test_x_scaled = scaler.fit_transform(
            test_x
        )  # Fit the scaler on the test features

        # Set MLflow tracking
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Predict using the loaded model
            predicted_qualities = model.predict(test_x_scaled)

            # Evaluate metrics
            rmse, mae, r2, accuracy = self.eval_metrics(
                test_y_encoded, predicted_qualities
            )

            # Saving metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2, "accuracy": accuracy}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log parameters and metrics in MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("accuracy", accuracy)  # Log accuracy

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="GradientBoostingClassifier"
                )
            else:
                mlflow.sklearn.log_model(model, "model")
