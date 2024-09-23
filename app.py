from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from stb_pfe_mlflow.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a flask app


@app.route("/", methods=["GET"])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route("/train", methods=["GET"])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"


@app.route("/predict", methods=["POST", "GET"])  # route to show predictions in a web UI
def index():
    if request.method == "POST":
        try:
            #  reading the inputs given by the user (Updated for your new columns)
            tiers_key = int(request.form["tiers_key"])
            ca = float(request.form["ca"])
            TOTMVTC = float(request.form["TOTMVTC"])
            TOTMVTD = float(request.form["TOTMVTD"])
            TOTMVTCnet = float(request.form["TOTMVTCnet"])
            TOTMVTDnet = float(request.form["TOTMVTDnet"])
            ENG = request.form["ENG"]
            MontImp = float(request.form["MontImp"])
            encours = request.form["encours"]
            Encours_Moyen_Debiteur = float(request.form["Encours_Moyen_Debiteur"])
            Encours_Moyen_crediteur = float(request.form["Encours_Moyen_crediteur"])
            INCIDENTCHQ = float(request.form["INCIDENTCHQ"])
            INCIDENTCHQ_R = float(request.form["INCIDENTCHQ_R"])
            INCIDENTCHQ_N_R = float(request.form["INCIDENTCHQ_N_R"])
            Interdit = request.form["Interdit"]
            InterditAct = request.form["InterditAct"]
            NBIMP = float(request.form["NBIMP"])
            Interet_Non_ECHU = float(request.form["Interet_Non_ECHU"])
            Encaiss_Recu = float(request.form["Encaiss_Recu"])
            NBECHEANCE = float(request.form["NBECHEANCE"])
            Code_Classe = float(request.form["Code_Classe"])
            Code_Profession = request.form["Code_Profession"]
            Profession = request.form["Profession"]
            Code_Activite_Economique = request.form["Code_Activite_Economique"]
            Activite_Economique = request.form["Activite_Economique"]
            Code_secteur_activite = request.form["Code_secteur_activite"]
            Secteur_Activite = request.form["Secteur_Activite"]
            Ville = request.form["Ville"]
            Code_Postal = float(request.form["Code_Postal"])
            Date_Ouverture = request.form["Date_Ouverture"]

            # Construct data array (modify this to match the expected format of your pipeline)
            data = [
                tiers_key,
                ca,
                TOTMVTC,
                TOTMVTD,
                TOTMVTCnet,
                TOTMVTDnet,
                ENG,
                MontImp,
                encours,
                Encours_Moyen_Debiteur,
                Encours_Moyen_crediteur,
                INCIDENTCHQ,
                INCIDENTCHQ_R,
                INCIDENTCHQ_N_R,
                Interdit,
                InterditAct,
                NBIMP,
                Interet_Non_ECHU,
                Encaiss_Recu,
                NBECHEANCE,
                Code_Classe,
                Code_Profession,
                Profession,
                Code_Activite_Economique,
                Activite_Economique,
                Code_secteur_activite,
                Secteur_Activite,
                Ville,
                Code_Postal,
                Date_Ouverture,
            ]
            data = np.array(data).reshape(1, -1)  # Adjust the shape if needed

            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template("results.html", prediction=str(predict))

        except Exception as e:
            print("The Exception message is: ", e)
            return "something is wrong"

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
