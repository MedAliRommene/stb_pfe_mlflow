import pandas as pd
import os
from stb_pfe_mlflow import logger
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from stb_pfe_mlflow.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load the data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        # Remove 'tiers_key' and target 'Cluster' for X data
        train_x = train_data.drop(columns=["tiers_key", "Cluster"])
        test_x = test_data.drop(columns=["tiers_key", "Cluster"])

        # Extract the target
        train_y = train_data["Cluster"]
        test_y = test_data["Cluster"]
        # Combine train and test for consistent encoding
        combined = pd.concat([train_x, test_x], axis=0)

        # Encode categorical features
        for col in combined.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))

        # Split back into train and test sets
        train_x_encoded = combined.iloc[: len(train_x)]
        test_x_encoded = combined.iloc[len(train_x) :]

        # Encode the target labels
        label_encoder = LabelEncoder()
        train_y_encoded = label_encoder.fit_transform(train_y)
        test_y_encoded = label_encoder.transform(test_y)

        # Standardize the features
        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x_encoded)
        test_x_scaled = scaler.transform(test_x_encoded)

        gbs = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,  # Number of boosting stages
            learning_rate=self.config.learning_rate,  # Step size shrinkage
            max_depth=self.config.max_depth,  # Maximum depth of the individual trees
            min_samples_split=self.config.min_samples_split,  # Minimum number of samples required to split an internal node
            min_samples_leaf=self.config.min_samples_leaf,  # Minimum number of samples required to be at a leaf node
            subsample=self.config.subsample,  # Fraction of samples to be used for fitting the individual base learners
            random_state=self.config.random_state,
        )

        gbs.fit(train_x_scaled, train_y_encoded)

        # Save the model
        joblib.dump(gbs, os.path.join(self.config.root_dir, self.config.model_name))

        print("Model training complete and saved.")
