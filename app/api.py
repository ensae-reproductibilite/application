"""A simple API to expose our trained RandomForest model for Tutanic survival."""
from fastapi import FastAPI
import mlflow

import pandas as pd

# Preload model -------------------

model_name = "production"
model_version = "latest"

# Load the model from the Model Registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

# Define app -------------------------

app = FastAPI(
    title="Prédiction de survie sur le Titanic",
    description=
    "Application de prédiction de survie sur le Titanic 🚢 <br>Une version par API pour faciliter la réutilisation du modèle 🚀" +\
        "<br><br><img src=\"https://media.vogue.fr/photos/5faac06d39c5194ff9752ec9/1:1/w_2404,h_2404,c_limit/076_CHL_126884.jpg\" width=\"200\">"
    )


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """

    return {
        "Message": "API de prédiction de survie sur le Titanic",
        "Model_name": 'Titanic ML',
        "Model_version": "0.3",
    }


@app.get("/predict", tags=["Predict"])
async def predict(
    sex: str = "female",
    age: float = 29.0,
    fare: float = 16.5,
    embarked: str = "S"
) -> str:
    """
    """

    df = pd.DataFrame(
        {
            "Sex": [sex],
            "Age": [age],
            "Fare": [fare],
            "Embarked": [embarked],
        }
    )

    prediction = "Survived 🎉" if int(model.predict(df)) == 1 else "Dead ⚰️"

    return prediction
