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
        # 2. Séparation des colonnes numériques et catégorielles
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        # Remplir les valeurs manquantes pour les colonnes numériques et catégorielles
        df[numeric_cols] = df[numeric_cols].fillna(
            df[numeric_cols].median()
        )  # Pour les colonnes numériques
        df[categorical_cols] = df[categorical_cols].fillna(
            df[categorical_cols].mode().iloc[0]
        )  # Pour les colonnes catégorielles

        # 3. Encodage des variables catégorielles
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

        # Utilisation de OneHotEncoder pour les variables non-ordinales
        encoded_data = pd.get_dummies(df, columns=categorical_cols)

        # 4. Normalisation des variables numériques
        scaler = StandardScaler()
        encoded_data[numeric_cols] = scaler.fit_transform(encoded_data[numeric_cols])

        # 5. Détermination du nombre optimal de clusters avec la méthode Elbow
        inertia = []
        k_values = range(1, 10)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(encoded_data)
            inertia.append(kmeans.inertia_)

        # 6. Clustering avec K-Means (choix de k optimal)
        k_optimal = 3  # Ajuster selon le résultat de l'Elbow ou Silhouette Score
        kmeans = KMeans(n_clusters=k_optimal, random_state=42)
        clusters = kmeans.fit_predict(encoded_data)
        df["Cluster"] = kmeans.labels_

        # Enregistrer le dataset final dans le répertoire configuré
        df.to_csv(
            os.path.join(self.config.root_dir, "transforming_data.csv"), index=False
        )

        # Log information
        logger.info("Data transformation complete")
        logger.info(f"Data shape: {df.shape}")
