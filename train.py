"""
Prediction de la survie d'un individu sur le Titanic
(avec GridSearchCV + cross-validation)
"""

import os
import argparse
import logging
from dotenv import load_dotenv

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import skops.io as sio
import mlflow

import duckdb
import pandas as pd

from src.validation.check import (
    check_name_formatting,
    check_missing_values,
    check_data_leakage,
)

load_dotenv()
con = duckdb.connect(database=":memory:")

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
    handlers=[logging.FileHandler("recording.log"), logging.StreamHandler()],
)

# PARAMETERS ---------------------------------------

NUMERIC_FEATURES = ["Age", "Fare"]
CATEGORICAL_FEATURES = ["Embarked", "Sex"]
URL_RAW = "https://minio.lab.sspcloud.fr/lgaliana/ensae-reproductibilite/data/raw/data.parquet"

jeton_api = os.environ["JETON_API"]

# ENVIRONMENT CONFIGURATION ---------------------------

parser = argparse.ArgumentParser(
    description="Paramètres du random forest + grid search"
)
parser.add_argument(
    "--experiment_name", type=str, default="titanicml", help="MLFlow experiment name"
)
parser.add_argument(
    "--n_trees",
    type=int,
    default=20,
    help="Valeur par défaut pour n_estimators dans la grille",
)
parser.add_argument(
    "--cv", type=int, default=5, help="Nombre de folds pour la cross-validation"
)
args = parser.parse_args()

n_trees_default = args.n_trees
cv_folds = args.cv

logging.debug(f"Valeur de l'argument n_trees: {n_trees_default}")
logging.debug(f"Valeur de l'argument cv: {cv_folds}")


# LOGGING IN MLFLOW -----------------

mlflow_server = os.getenv("MLFLOW_TRACKING_URI")

logging.debug(f"Saving experiment in {mlflow_server}")

mlflow.set_tracking_uri(mlflow_server)
mlflow.set_experiment(args.experiment_name)


# QUALITY DIAGNOSTICS  ---------------------------------------

logging.debug(f"\n{80 * '-'}\nStarting data validation step\n{80 * '-'}")

query_definition = (
    f"CREATE TEMP TABLE titanic AS (SELECT * FROM read_parquet('{URL_RAW}'))"
)
con.sql(query_definition)

column_names = con.sql("SELECT column_name FROM (DESCRIBE titanic)").to_df()[
    "column_name"
]

check_name_formatting(connection=con)

for var in column_names:
    check_missing_values(connection=con, variable=var)

# FEATURE ENGINEERING    -----------------------------------------

logging.debug(f"\n{80 * '-'}\nStarting feature engineering phase\n{80 * '-'}")

titanic = con.sql(
    f"SELECT Survived, {', '.join(CATEGORICAL_FEATURES + NUMERIC_FEATURES)} FROM titanic"
).to_df()

y = titanic["Survived"]
X = titanic.drop("Survived", axis="columns")

# Stratify: utile sur Titanic car classes pas parfaitement équilibrées
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)


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
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, NUMERIC_FEATURES),
        ("Preprocessing categorical", categorical_transformer, CATEGORICAL_FEATURES),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# GRID SEARCH + CV --------------------------------------------

# Grille d'hyperparamètres (tu peux l'élargir/réduire selon le temps de calcul)
param_grid = {
    "classifier__n_estimators": [
        n_trees_default,
    ],
    "classifier__max_depth": [3, 5],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__min_samples_split": [2, 5]
}


train_data = pd.concat([X_train, y_train], axis=1)

with mlflow.start_run():

    logging.debug(f"\n{80 * '-'}\nLogging input in MLFlow\n{80 * '-'}")

    mlflow.log_input(
        mlflow.data.from_pandas(titanic),
        context="raw",
    )

    mlflow.log_input(
        mlflow.data.from_pandas(train_data),
        context="raw",
    )

    mlflow.log_param("n_trees", n_trees_default)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        refit=True,
    )

# TRAINING AND EVALUATION --------------------------------------------

    logging.debug(f"\n{80 * '-'}\nStarting grid search fitting phase\n{80 * '-'}")

    search.fit(X_train, y_train)

    logging.info(f"Best CV score: {search.best_score_:.3f}")
    logging.info(f"Best params: {search.best_params_}")

    best_model = search.best_estimator_

    best_params = search.best_params_

    for param, value in best_params.items():
        mlflow.log_param(param, value)

    # Sauvegarde du meilleur pipeline complet
    sio.dump(best_model, "model.skops")

    test_score = best_model.score(X_test, y_test)
    train_score = best_model.score(X_train, y_train)

    logging.info(
        f"{test_score:.1%} de bonnes réponses sur les données de test (best model)"
    )
    logging.info(
        f"{train_score:.1%} de bonnes réponses sur les données de train (best model)"
    )

    # Log metrics
    mlflow.log_metric("accuracy", test_score)

    matrix = confusion_matrix(y_test, best_model.predict(X_test))

    logging.info("Matrice de confusion (test):")
    logging.info(matrix)

    logging.debug(f"\n{80 * '-'}\nFILE ENDED SUCCESSFULLY!\n{80 * '-'}")

    # Log confusion matrix as an artifact
    matrix_path = "confusion_matrix.txt"
    with open(matrix_path, "w") as f:
        f.write(str(matrix))
    mlflow.log_artifact(matrix_path)

    # Log model
    mlflow.sklearn.log_model(best_model, "model")
