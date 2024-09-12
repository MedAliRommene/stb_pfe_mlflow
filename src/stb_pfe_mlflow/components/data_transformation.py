import os
from stb_pfe_mlflow import logger
from stb_pfe_mlflow.entity.config_entity import DataTransformationConfig
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transforming_data(self):
        df = pd.read_csv(self.config.data_path)
        df.drop(columns=["Date_Ouverture"], inplace=True)
        # Encodage des variables catégorielles
        categorical_cols = [
            "ENG",
            "Code_Profession",
            "Profession",
            "Code_Activite_Economique",
            "Activite_Economique",
            "Code_secteur_activite",
            "Secteur_Activite",
            "Ville",
        ]
        label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}

        encoded_data = df.copy()

        for col, le in label_encoders.items():
            encoded_data[col] = le.transform(df[col])

        # Normalisation des variables numériques
        numeric_cols = encoded_data.select_dtypes(include=["float64", "int64"]).columns
        scaler = StandardScaler()
        encoded_data[numeric_cols] = scaler.fit_transform(encoded_data[numeric_cols])

        # Clustering avec K-Means
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(encoded_data)

        # Ajouter les clusters aux données
        encoded_data["Cluster"] = clusters

        # Création de la colonne 'pouvoir_credit' basée sur les clusters
        encoded_data["pouvoir_credit"] = encoded_data["Cluster"].apply(
            lambda x: "sain" if x == 0 else "risqué"
        )
        # drop cplpnne cluster :
        encoded_data.drop(columns=["Cluster"], inplace=True)
        # Enregistrement du dataset avec les labels supervisés
        encoded_data.to_csv("data_supervised.csv", index=False)

        # Enregistrer le dataset final dans le répertoire configuré
        encoded_data.to_csv(
            os.path.join(self.config.root_dir, "transforming_data.csv"), index=False
        )

        # Log information
        logger.info("Data transformation complete")
        logger.info(f"Data shape: {df.shape}")
