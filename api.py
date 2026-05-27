from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI(
    title="API de prédiction électorale",
    description="API MSPR Electio-Analytics pour prédire les voix estimées d’un candidat dans une commune",
    version="1.0"
)

model = joblib.load("best_model.pkl")
model_columns = joblib.load("model_columns.pkl")


class PredictionInput(BaseModel):
    nom_complet: str
    population_totale_2022: float
    population_15_29_ans: float
    population_45_59_ans: float
    population_60_74_ans: float
    revenu_median_menages_2021: float
    nombre_total_etablissements_2023: float
    entreprises_par_habitant: float
    taux_chomage_global: float
    taux_pauvrete_total_2021: float
    taux_hlm: float
    associations_par_habitant: float
    participation_reelle: float
    taux_logements_vacants: float
    taux_cambriolage: float


@app.get("/")
def accueil():
    return {
        "message": "Bienvenue sur l'API de prédiction électorale Electio-Analytics"
    }


@app.get("/health")
def health():
    return {
        "status": "OK",
        "message": "API opérationnelle"
    }


@app.post("/predict")
def predict(data: PredictionInput):
    input_df = pd.DataFrame([data.dict()])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    log_voix_predites = model.predict(input_df)[0]

    voix_predites = np.expm1(log_voix_predites)

    return {
        "candidat": data.nom_complet,
        "log_voix_predites": round(float(log_voix_predites), 3),
        "voix_predites_estimees": round(float(voix_predites), 2),
        #"message": "Prédiction réalisée avec succès"
    }

CANDIDATS = [
    "ARTHAUD Nathalie",
    "MÉLENCHON Jean-Luc",
    "ROUSSEL Fabien",
    "HIDALGO Anne",
    "LASSALLE Jean",
    "DUPONT-AIGNAN Nicolas",
    "LE PEN Marine",
    "PÉCRESSE Valérie",
    "ZEMMOUR Éric",
    "JADOT Yannick",
    "POUTOU Philippe",
    "MACRON Emmanuel"
]

FAMILLES_POLITIQUES = {
    "MACRON Emmanuel": "Centre",
    "LE PEN Marine": "Extrême droite",
    "MÉLENCHON Jean-Luc": "Gauche",
    "ZEMMOUR Éric": "Extrême droite",
    "PÉCRESSE Valérie": "Droite",
    "JADOT Yannick": "Gauche",
    "ROUSSEL Fabien": "Gauche",
    "HIDALGO Anne": "Gauche",
    "LASSALLE Jean": "Centre",
    "DUPONT-AIGNAN Nicolas": "Droite",
    "ARTHAUD Nathalie": "Extrême gauche",
    "POUTOU Philippe": "Extrême gauche"
}

class CommuneScenarioInput(BaseModel):
    commune: str
    population_totale_2022: float
    population_15_29_ans: float
    population_45_59_ans: float
    population_60_74_ans: float
    revenu_median_menages_2021: float
    nombre_total_etablissements_2023: float
    entreprises_par_habitant: float
    taux_chomage_global: float
    taux_pauvrete_total_2021: float
    taux_hlm: float
    associations_par_habitant: float
    participation_reelle: float
    taux_logements_vacants: float
    taux_cambriolage: float


@app.post("/predict_winner")
def predict_winner(data: CommuneScenarioInput):

    results = []

    for candidat in CANDIDATS:
        row = data.dict()
        row["nom_complet"] = candidat
        commune = row.pop("commune")

        input_df = pd.DataFrame([row])

        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        log_pred = model.predict(input_df)[0]
        voix_predites = np.expm1(log_pred)

        results.append({
            "commune": commune,
            "candidat": candidat,
            "famille_politique": FAMILLES_POLITIQUES.get(candidat, "Non défini"),
            "log_voix_predites": round(float(log_pred), 3),
            "voix_predites_estimees": round(float(voix_predites), 2)
        })

    classement = sorted(
        results,
        key=lambda x: x["voix_predites_estimees"],
        reverse=True
    )

    gagnant = classement[0]

    return {
        "commune": data.commune,
        "gagnant_predit": gagnant["candidat"],
        "famille_politique_gagnant": gagnant["famille_politique"],
        "score_estime": gagnant["voix_predites_estimees"],
        "classement_complet": classement,
        #"message": "Prédiction du gagnant"
    }