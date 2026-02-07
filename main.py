"""
Prediction de la survie d'un individu sur le Titanic
"""

import os
import argparse
from dotenv import load_dotenv

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import pandas as pd
import duckdb

from check import (
    check_name_formatting,
    check_missing_values,
    check_data_leakage
)


load_dotenv()
con = duckdb.connect(database=":memory:")


# PARAMETERS ---------------------------------------


N_TREES = 20
MAX_DEPTH = None
MAX_FEATURES = "sqrt"
NUMERIC_FEATURES = ["Age", "Fare"]
CATEGORICAL_FEATURES = ["Embarked", "Sex"]

jeton_api = os.environ["JETON_API"]


# ENVIRONMENT CONFIGURATION ---------------------------

parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument("--n_trees", type=int, default=20, help="Nombre d'arbres")
args = parser.parse_args()

n_trees = args.n_trees

print(f"Valeur de l'argument n_trees: {n_trees}")


# QUALITY DIAGNOSTICS  ---------------------------------------

titanic = pd.read_csv("data.csv")

column_names = con.sql("SELECT column_name FROM (DESCRIBE titanic)").to_df()[
    "column_name"
]  # DuckDB ici, sinon titanic.columns serait OK

check_name_formatting(connection=con, df=titanic)

for var in column_names:
    check_missing_values(connection=con, df=titanic, variable=var)


# FEATURE ENGINEERING    -----------------------------------------

y = titanic["Survived"]
X = titanic.drop("Survived", axis="columns")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

for string_var in CATEGORICAL_FEATURES:
    check_data_leakage(X_train, X_test, string_var)


# MODEL DEFINITION -----------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, NUMERIC_FEATURES),
        (
            "Preprocessing categorical",
            categorical_transformer,
            CATEGORICAL_FEATURES,
        ),
    ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=N_TREES, max_depth=MAX_DEPTH, max_features=MAX_FEATURES
            ),
        ),
    ]
)

# TRAINING AND EVALUATION --------------------------------------------

pipe.fit(X_train, y_train)
rdmf_score = pipe.score(X_test, y_test)
rdmf_score_tr = pipe.score(X_train, y_train)

print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")

print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_test)))
