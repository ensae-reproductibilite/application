"""
Prediction de la survie d'un individu sur le Titanic
"""

import os
from dotenv import load_dotenv
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline.build_pipeline import create_pipeline
from models.train_evaluate import evaluate_model

# ENVIRONMENT CONFIGURATION ---------------------------

load_dotenv()

parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres"
)
args = parser.parse_args()

n_trees = args.n_trees
jeton_api = os.environ.get("JETON_API", "")
data_path = os.environ.get("data_path", "data/raw/data.csv")
data_train_path = os.environ.get("train_path", "data/derived/train.csv")
data_test_path = os.environ.get("test_path", "data/derived/test.csv")
MAX_DEPTH = None
MAX_FEATURES = "sqrt"

if jeton_api.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")


# IMPORT ET STRUCTURATION DONNEES --------------------------------

TrainingData = pd.read_csv("data.csv")

y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1
)
pd.concat([X_train, y_train], axis = 1).to_csv("train.csv", index=False)
pd.concat([X_test, y_test], axis = 1).to_csv("test.csv", index=False)


# PIPELINE ----------------------------

pipe = create_pipeline(
    n_trees, max_depth=MAX_DEPTH, max_features=MAX_FEATURES
)


# ESTIMATION ET EVALUATION ----------------------

pipe.fit(X_train, y_train)
score, matrix = evaluate_model(pipe, X_test, y_test)

print(f"{score:.1%} de bonnes réponses sur les données de test pour validation")
print(20 * "-")
print("matrice de confusion")
print(matrix)
