from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from stb_pfe_mlflow.pipeline.prediction import PredictionPipeline
from stb_pfe_mlflow.entity.config_entity import (
    DataCleaningConfig,
    DataTransformationConfig,
)
from stb_pfe_mlflow.components.data_cleaning import DataCleaning
from stb_pfe_mlflow.components.data_transformation import (
    DataTransformation,
)  # Replace with the actual module name


app = Flask(__name__)  # Initializing a Flask app


@app.route("/", methods=["GET"])  # Route to display the home page
def homePage():
    return render_template("home.html")


@app.route("/index", methods=["GET"])  # Route to display the home page
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET"])  # Route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"


@app.route("/predict", methods=["POST", "GET"])  # Route to show predictions in a web UI
def index():
    if request.method == "POST":
        try:
            # Reading the inputs given by the user
            # tiers_key = int(request.form["tiers_key"])
            ca = float(request.form["ca"])
            TOTMVTC = float(request.form["TOTMVTC"])
            TOTMVTD = float(request.form["TOTMVTD"])
            TOTMVTCnet = float(request.form["TOTMVTCnet"])
            TOTMVTDnet = float(request.form["TOTMVTDnet"])
            MontImp = float(request.form["MontImp"])
            Encours_Moyen_Debiteur = float(request.form["Encours_Moyen_Debiteur"])
            Encours_Moyen_crediteur = float(request.form["Encours_Moyen_crediteur"])
            NBECHEANCE = float(request.form["NBECHEANCE"])
            Code_Classe = float(request.form["Code_Classe"])
            ancienneté = request.form["ancienneté"]

            # Categorical inputs (no float conversion for these)
            ENG = request.form["ENG"]
            encours = request.form["encours"]
            Code_Profession = request.form["Code_Profession"]
            Profession = request.form["Profession"]
            Code_Activite_Economique = request.form["Code_Activite_Economique"]
            Activite_Economique = request.form["Activite_Economique"]
            Code_secteur_activite = request.form["Code_secteur_activite"]
            Secteur_Activite = request.form["Secteur_Activite"]
            Ville = request.form["Ville"]

            # Construct data array
            data = {
                # "tiers_key": tiers_key,
                "ca": ca,
                "TOTMVTC": TOTMVTC,
                "TOTMVTD": TOTMVTD,
                "TOTMVTCnet": TOTMVTCnet,
                "TOTMVTDnet": TOTMVTDnet,
                "ENG": ENG,
                "MontImp": MontImp,
                "encours": encours,
                "Encours_Moyen_Debiteur": Encours_Moyen_Debiteur,
                "Encours_Moyen_crediteur": Encours_Moyen_crediteur,
                "NBECHEANCE": NBECHEANCE,
                "Code_Classe": Code_Classe,
                "Code_Profession": Code_Profession,
                "Profession": Profession,
                "Code_Activite_Economique": Code_Activite_Economique,
                "Activite_Economique": Activite_Economique,
                "Code_secteur_activite": Code_secteur_activite,
                "Secteur_Activite": Secteur_Activite,
                "Ville": Ville,
                "ancienneté": ancienneté,
            }

            input_df = pd.DataFrame([data])

            # Categorical columns for label encoding
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

            # Label encode categorical columns
            label_encoders = {
                col: LabelEncoder().fit(input_df[col]) for col in categorical_cols
            }
            for col, le in label_encoders.items():
                input_df[col] = le.transform(input_df[col])

            # Normalize numerical columns
            numeric_cols = [
                "ca",
                "TOTMVTC",
                "TOTMVTD",
                "TOTMVTCnet",
                "TOTMVTDnet",
                "MontImp",
                "Encours_Moyen_Debiteur",
                "Encours_Moyen_crediteur",
                "NBECHEANCE",
                "Code_Classe",
            ]

            scaler = StandardScaler()
            input_df[numeric_cols] = scaler.fit_transform(input_df[numeric_cols])

            # Pass transformed data to the prediction pipeline
            obj = PredictionPipeline()
            predict = obj.predict(input_df)
            prediction = int(predict[0])

            return render_template("results.html", prediction=prediction)

        except Exception as e:
            print("The Exception message is:", e)
            return "Something is wrong"

    else:
        return render_template("index.html")


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=8080, debug=True)
    app.run(host="0.0.0.0", port=8080)

## try another test
