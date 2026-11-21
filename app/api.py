"""
API estimating the price per m² of a property from its address

L'API charge le modèle de production depuis le registre de modèles de MLflow
(modèle `dvf`, alias `production`). Pour la lancer en local, la variable
d'environnement `MLFLOW_TRACKING_URI` doit être définie (ainsi que
`MLFLOW_TRACKING_USERNAME` et `MLFLOW_TRACKING_PASSWORD` si le serveur demande
une authentification), et un modèle `dvf` avec l'alias `production` doit exister.
Lancer l'API depuis le dossier du projet :

    uv run uvicorn app.api:app
"""

import logging
import os

import duckdb
import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.models.predict import build_inference_features, predict_prix_m2

load_dotenv()
JETON_API = os.environ.get("JETON_API", "")

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

URL_RAW = "https://minio.lab.sspcloud.fr/lgaliana/ensae-reproductibilite/data/raw"
URL_FILOSOFI = f"{URL_RAW}/filosofi_carreaux_200m_2021.parquet"

MODEL_NAME = "dvf"
MODEL_ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

con = duckdb.connect(database=":memory:")

# MLflow reads the tracking server address from the MLFLOW_TRACKING_URI
# environment variable. Without it, MLflow silently falls back to an empty
# local database where the production model does not exist: better stop here
# with a clear message.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    raise RuntimeError(
        "La variable d'environnement MLFLOW_TRACKING_URI n'est pas définie : "
        "impossible de joindre le serveur MLFlow"
    )
logger.info(f"Connexion au serveur MLFlow {MLFLOW_TRACKING_URI}")

# The types trusted by skops were saved with the model at training time,
# so no trusted list is needed here.
logger.info(f"Chargement du modèle {MODEL_URI}")
model = mlflow.sklearn.load_model(MODEL_URI)
version_production = (
    mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
)
logger.info(f"Version {version_production} du modèle {MODEL_NAME} chargée")

app = FastAPI(
    title="Démonstration du modèle de prix au m² en petite couronne",
    description=(
        "<b>Estimation du prix au m² des logements</b> à Paris et dans les "
        "départements 92, 93 et 94, à partir des transactions DVF enrichies "
        "par des données socio-économiques Filosofi 🏠<br>"
        "Une version par API pour faciliter la réutilisation du modèle 🚀"
    ),
)


@app.get("/")
def accueil():
    """Présentation de l'API"""
    return {
        "Message": "Bienvenue sur l'API d'estimation du prix au m²",
        "Model_name": "Prix au m² DVF (LightGBM)",
        "Model_version": "0.3",
        "Model_registry": f"{MODEL_NAME} (alias {MODEL_ALIAS}, version {version_production})",
    }


@app.get("/predict")
def predire(
    adresse: str = "20 avenue de Ségur 75007 Paris",
    type_local: str = "Appartement",
    surface: float = 45.0,
    nombre_pieces: int = 2,
):
    """Estimer le prix d'un logement à partir de son adresse

    - `type_local` : `Appartement` ou `Maison`
    - `surface` : surface réelle du bâti, en m²
    - `nombre_pieces` : nombre de pièces principales

    Seules les adresses de Paris et des départements 92, 93 et 94 sont acceptées.
    """
    try:
        features = build_inference_features(
            adresse=adresse,
            type_local=type_local,
            surface_reelle_bati=surface,
            nombre_pieces_principales=nombre_pieces,
            connection=con,
            url_filosofi=URL_FILOSOFI,
            jeton_api=JETON_API,
        )
    except ValueError as erreur:
        logger.warning(f"Adresse refusée : {erreur}")
        raise HTTPException(status_code=400, detail=str(erreur))

    prix_m2 = predict_prix_m2(model, features)
    logger.info(f"Prédiction de {round(prix_m2)} €/m² pour {features.loc[0, 'label']}")
    return {
        "adresse": features.loc[0, "label"],
        "code_commune": features.loc[0, "code_commune"],
        "prix_m2": round(prix_m2),
        "prix_estime": round(prix_m2 * surface),
    }
