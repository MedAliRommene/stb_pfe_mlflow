import pandas as pd
import os
from stb_pfe_mlflow import logger
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from stb_pfe_mlflow.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load the data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Separate features and target
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # Encode the target column
        label_encoder = LabelEncoder()
        train_y_encoded = label_encoder.fit_transform(train_y)
        test_y_encoded = label_encoder.transform(test_y)

        # Initialize and train KNN model
        knn = KNeighborsClassifier(
            n_neighbors=self.config.n_neighbors,
            weights=self.config.weights,
            algorithm=self.config.algorithm,
            p=self.config.p,
            leaf_size=self.config.leaf_size,
        )
        knn.fit(train_x, train_y_encoded)

        # Save the model
        joblib.dump(knn, os.path.join(self.config.root_dir, self.config.model_name))

        # Optionally: Save the label encoder for future use
        joblib.dump(
            label_encoder, os.path.join(self.config.root_dir, self.config.model_name)
        )

        print("Model training complete and saved.")
