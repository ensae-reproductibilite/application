"""
Predict the price per m² of homes sold in Paris and its inner suburbs
from DVF transactions enriched with Filosofi socio-economic data
"""

import argparse
import os
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from check import (
    check_code_commune_formatting,
    check_missing_values,
    check_unseen_categories,
)
from evaluation import evaluate_with_skore, rmse_by_commune
from features import build_features
from import_data import (
    add_filosofi_carreaux,
    add_filosofi_communes,
    download_missing_sources,
)
from model import create_pipeline

load_dotenv()
con = duckdb.connect(database=":memory:")

LEARNING_RATE = 0.03
NUM_LEAVES = 31
NUMERIC_FEATURES = [
    "log_surface",
    "log_distance_eiffel",
    "nombre_pieces_principales",
    "nombre_lots",
    "log_surface_terrain",
    "ratio_carrez",
    "nbre_menages",
    "tx_pauvrete",
    "nv_vie_moyen",
    "part_proprietaires",
    "part_logements_sociaux",
    "part_logements_avant_1945",
    "surface_moyenne_logements",
    "NIVVIE_MEDIAN",
    "TAUX_PAUVRETE",
]
CATEGORICAL_FEATURES = ["code_commune", "type_local"]
DEPARTEMENTS_PARIS = ["75", "92", "93", "94"]

jeton_api = os.environ["JETON_API"]

DATA_LOCATION = Path("data")
SOURCES = {
    "dvf.parquet": (
        "https://minio.lab.sspcloud.fr/"
        "projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet"
    ),
    "filosofi_carreaux_200m_2021.parquet": (
        "https://www.data.gouv.fr/api/1/datasets/r/55432374-a91d-43d0-923d-4514dc3eb951"
    ),
}


# ENVIRONMENT CONFIGURATION ---------------------------------------

parser = argparse.ArgumentParser(description="Paramètres du modèle LightGBM")
parser.add_argument("--n_trees", type=int, default=1500, help="Nombre d'arbres")
args = parser.parse_args()

n_trees = args.n_trees

print(f"Valeur de l'argument n_trees: {n_trees}")


# DATA IMPORT ---------------------------------------

print("Lecture des données ------------------------")

download_missing_sources(SOURCES, DATA_LOCATION)
dvf = gpd.read_parquet(DATA_LOCATION / "dvf.parquet")


# QUALITY DIAGNOSTICS ---------------------------------------

# DuckDB cannot read the geometry column as Pandas formatted it
dvf_attributs = dvf.drop(columns="geometry")

# dvf_attributs.columns would work too: DuckDB anticipates its wider use later
column_names = con.sql("SELECT column_name FROM (DESCRIBE dvf_attributs)").to_df()[
    "column_name"
]

check_code_commune_formatting(connection=con, df=dvf_attributs)

for var in column_names:
    check_missing_values(connection=con, df=dvf_attributs, variable=var)


# SCOPE RESTRICTION ---------------------------------------

print("Restriction aux départements de la petite couronne")

dvf = dvf.loc[dvf["code_departement"].isin(DEPARTEMENTS_PARIS)]
dvf = dvf.loc[dvf["valeur_fonciere"] < 1e6]


# ADDITIONAL DATA ---------------------------------------

dvf = add_filosofi_carreaux(dvf, DATA_LOCATION / "filosofi_carreaux_200m_2021.parquet")
dvf = add_filosofi_communes(dvf, jeton_api)

print(dvf.head(5))


# DESCRIPTIVE STATISTICS ---------------------------------------

print("Statistiques descriptives ---------------------")

sns.histplot(data=dvf, x="valeur_fonciere")
sns.histplot(data=dvf, x="valeur_fonciere", log_scale=True)

grid = sns.scatterplot(data=dvf, x="surface_reelle_bati", y="valeur_fonciere")
grid.set(xscale="log", yscale="log")


# FEATURE ENGINEERING ---------------------------------------

print("Feature engineering")

dvf = build_features(dvf)

# Missing features are imputed in the pipeline: only rows without a target
# are dropped.
dvf_start_data = dvf.dropna(subset=["prix_m2"])


# PRICE PER M² BY MUNICIPALITY ---------------------------------------

prix_m2_commune = (
    dvf.groupby(["code_commune", "nom_commune"])["prix_m2"].mean().sort_values()
)

print("Bottom 5 communes (prix au m2 le plus bas) :")
print(prix_m2_commune.head(5))
print("Top 5 communes (prix au m2 le plus élevé) :")
print(prix_m2_commune.tail(5))


# TRAIN/TEST SPLIT ---------------------------------------

features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
X = dvf_start_data[features]
y = dvf_start_data["prix_m2"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

pd.concat([X_train, y_train], axis=1).to_csv("train.csv")
pd.concat([X_test, y_test], axis=1).to_csv("test.csv")


# SPLIT VALIDATION ---------------------------------------

for string_var in CATEGORICAL_FEATURES:
    check_unseen_categories(X_train, X_test, string_var)


# TRAINING AND EVALUATION ---------------------------------------

pipe = create_pipeline(
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    n_trees=n_trees,
    learning_rate=LEARNING_RATE,
    num_leaves=NUM_LEAVES,
)
pipe.fit(X_train, y_train)

rmse_train = root_mean_squared_error(y_train, pipe.predict(X_train))
rmse_test = root_mean_squared_error(y_test, pipe.predict(X_test))
print(f"RMSE sur l'apprentissage : {rmse_train:,.0f} €/m²")
print(f"RMSE sur le test : {rmse_test:,.0f} €/m²")

cog = dvf_start_data[["code_commune", "nom_commune"]].drop_duplicates()
rmse_commune = rmse_by_commune(pipe, X_test, y_test, cog)

print("Bottom 5 communes (RMSE la plus faible) :")
print(rmse_commune.head(5))
print("Top 5 communes (RMSE la plus élevée) :")
print(rmse_commune.tail(5))


# EVALUATION WITH SKORE ---------------------------------------

print("Évaluation skore ---------------------")

evaluate_with_skore(pipe, X_train, y_train, X_test, y_test)

print("Fin du script !")
