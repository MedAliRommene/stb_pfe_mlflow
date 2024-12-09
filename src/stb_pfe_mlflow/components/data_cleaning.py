import os
import urllib.request as request

from stb_pfe_mlflow import logger
from stb_pfe_mlflow.utils.common import get_size
import pandas as pd
from stb_pfe_mlflow.entity.config_entity import DataCleaningConfig


class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config

    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    # You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up

    def cleaning_data(self):
        df = pd.read_csv(self.config.data_path)
        # Renommer les colonnes qui commencent par "Var Signalitiques."
        df.rename(
            columns=lambda x: (
                x.replace("Var Signalitiques.", "")
                if x.startswith("Var Signalitiques.")
                else x
            )
        )

        # Suppression des colonnes spécifiées
        columns_to_drop = [
            "INCIDENTCHQ_R",
            "INCIDENTCHQ_N_R",
            "INCIDENTCHQ",
            "NBIMP",
            "Interdit",
            "InterditAct",
            "Interet_Non_ECHU",
            "Encaiss_Recu",
            "Code_Postal",
        ]
        df.drop(columns=columns_to_drop)

        # Remplir les valeurs manquantes dans les colonnes numériques avec la médiane ou la moyenne
        df["ca"].fillna(df["ca"].median())
        df["TOTMVTC"].fillna(df["TOTMVTC"].mean())
        df["TOTMVTD"].fillna(df["TOTMVTD"].mean())
        df["TOTMVTCnet"].fillna(df["TOTMVTCnet"].mean())
        df["TOTMVTDnet"].fillna(df["TOTMVTDnet"].mean())

        # Assurer que TOTMVTC et TOTMVTCnet soient toujours positifs
        df["TOTMVTC"] = df[
            "TOTMVTC"
        ].abs()  # Convertir en valeur absolue pour s'assurer qu'il est positif
        df["TOTMVTCnet"] = df[
            "TOTMVTCnet"
        ].abs()  # Convertir en valeur absolue pour s'assurer qu'il est positif

        # Assurer que TOTMVTD et TOTMVTDnet soient toujours négatifs
        df["TOTMVTD"] = -df[
            "TOTMVTD"
        ].abs()  # Convertir en valeur absolue et rendre négatif
        df["TOTMVTDnet"] = -df[
            "TOTMVTDnet"
        ].abs()  # Convertir en valeur absolue et rendre négatif

        # Limiter le nombre de décimales à 2 pour une meilleure lisibilité
        df["TOTMVTC"] = df["TOTMVTC"].round(2)
        df["TOTMVTCnet"] = df["TOTMVTCnet"].round(2)
        df["TOTMVTD"] = df["TOTMVTD"].round(2)
        df["TOTMVTDnet"] = df["TOTMVTDnet"].round(2)

        # Remplir les valeurs manquantes dans 'ENG' par 0 (ou "Non")
        df["ENG"].fillna(
            0
        )  # Ici, on suppose que "0" représente une catégorie manquante

        # Remplir les valeurs manquantes dans 'MontImp' par 0
        df["MontImp"].fillna(0)

        # Conversion de la colonne 'encours' en numérique (et gestion des erreurs de conversion)
        df["encours"] = pd.to_numeric(df["encours"], errors="coerce")

        # Vérification du nombre de valeurs NaN dans 'encours'
        print(df["encours"].isna().sum())

        # Remplir les valeurs manquantes dans 'encours' avec la médiane
        df["encours"].fillna(df["encours"].median())

        # Remplir les valeurs manquantes dans d'autres colonnes numériques avec la médiane
        df["Encours_Moyen_Debiteur"].fillna(df["Encours_Moyen_Debiteur"].median())
        df["Encours_Moyen_crediteur"].fillna(df["Encours_Moyen_crediteur"].median())

        # Remplir les valeurs manquantes dans 'NBECHEANCE' avec la moyenne
        df["NBECHEANCE"].fillna(df["NBECHEANCE"].mean())

        # Remplir les valeurs manquantes dans 'Code_Classe' avec la valeur la plus fréquente (mode)
        df["Code_Classe"].fillna(df["Code_Classe"].mode()[0])

        # Remplir les valeurs manquantes dans certaines colonnes catégorielles avec la valeur la plus fréquente
        categorical_columns = [
            "Profession",
            "Code_Profession",
            "Secteur_Activite",
            "Code_secteur_activite",
            "Activite_Economique",
            "Code_Activite_Economique",
            "Ville",
        ]

        for col in categorical_columns:
            if col in df.columns:  # Vérifie si la colonne existe dans le DataFrame
                df[col].fillna(df[col].mode()[0])
            else:
                print(f"Colonne '{col}' non trouvée dans le DataFrame.")

        # Assurez-vous que la colonne 'Date_Ouverture' est au format datetime
        df["Date_Ouverture"] = pd.to_datetime(df["Date_Ouverture"], errors="coerce")

        # Imputation des valeurs manquantes dans 'Date_Ouverture' avec la date médiane
        median_date = df["Date_Ouverture"].median()
        df["Date_Ouverture"].fillna(median_date)

        # Fonction pour calculer le nombre d'années écoulées depuis la date d'ouverture
        def calculate_years_since(date):
            today = pd.Timestamp.today()
            return (
                today.year
                - date.year
                - ((today.month, today.day) < (date.month, date.day))
            )

        # Appliquer la fonction pour créer une nouvelle colonne 'ancienneté' (en années)
        df["ancienneté"] = df["Date_Ouverture"].apply(calculate_years_since)

        # Afficher les premières lignes pour vérifier la transformation
        # print(df[['Date_Ouverture', 'ancienneté', 'TOTMVTC', 'TOTMVTD', 'TOTMVTCnet', 'TOTMVTDnet']].head())

        df.to_csv(os.path.join(self.config.root_dir, "clean_data.csv"), index=False)
        logger.info("Cleaning the data")
        logger.info(df.shape)
