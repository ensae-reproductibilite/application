"""
Predict the price per m² of homes sold in Paris and its inner suburbs
from DVF transactions enriched with Filosofi socio-economic data
"""

import argparse
import logging
import os
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.data.build_features import build_features
from src.data.check import (
    check_code_commune_formatting,
    check_missing_values,
    check_unseen_categories,
)
from src.data.import_data import add_filosofi_carreaux, add_filosofi_communes
from src.models.evaluation import evaluate_with_skore, rmse_by_commune
from src.models.model import create_pipeline

load_dotenv()
con = duckdb.connect(database=":memory:")

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
    handlers=[logging.FileHandler("recording.log"), logging.StreamHandler()],
)

# DEBUG only for our own messages: globally, libraries such as matplotlib
# would flood the log
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

URL_RAW = "https://minio.lab.sspcloud.fr/lgaliana/ensae-reproductibilite/data/raw"
URL_DVF = f"{URL_RAW}/dvf.parquet"
URL_FILOSOFI = f"{URL_RAW}/filosofi_carreaux_200m_2021.parquet"
DERIVED_DATA_LOCATION = Path("data/derived")


# ENVIRONMENT CONFIGURATION ---------------------------------------

parser = argparse.ArgumentParser(description="Paramètres du modèle LightGBM")
parser.add_argument("--n_trees", type=int, default=1500, help="Nombre d'arbres")
args = parser.parse_args()

n_trees = args.n_trees

logger.debug(f"Valeur de l'argument n_trees: {n_trees}")


# DATA IMPORT ---------------------------------------

logger.debug(f"\n{80 * '-'}\nLecture des données\n{80 * '-'}")

# Filtering in the query avoids loading the whole of France. The geometry is
# rebuilt later from longitude and latitude.
departements = ", ".join(f"'{dep}'" for dep in DEPARTEMENTS_PARIS)
query_definition = f"""
    CREATE TEMP TABLE dvf AS
    SELECT * EXCLUDE (geometry, __index_level_0__)
    FROM read_parquet('{URL_DVF}')
    WHERE code_departement IN ({departements})
"""
con.sql(query_definition)


# QUALITY DIAGNOSTICS ---------------------------------------

logger.debug(f"\n{80 * '-'}\nValidation des données\n{80 * '-'}")

column_names = con.sql("SELECT column_name FROM (DESCRIBE dvf)").to_df()["column_name"]

check_code_commune_formatting(connection=con)

for var in column_names:
    check_missing_values(connection=con, variable=var)


# SCOPE RESTRICTION ---------------------------------------

logger.info("Restriction aux transactions de moins d'un million d'euros")

dvf = con.sql("SELECT * FROM dvf WHERE valeur_fonciere < 1e6").to_df()
dvf = gpd.GeoDataFrame(
    dvf,
    geometry=gpd.points_from_xy(dvf["longitude"], dvf["latitude"]),
    crs="EPSG:4326",
)


# ADDITIONAL DATA ---------------------------------------

dvf = add_filosofi_carreaux(dvf, connection=con, url_filosofi=URL_FILOSOFI)
dvf = add_filosofi_communes(dvf, jeton_api)

logger.debug(f"Aperçu des données enrichies:\n{dvf.head(5)}")


# DESCRIPTIVE STATISTICS ---------------------------------------

logger.debug(f"\n{80 * '-'}\nStatistiques descriptives\n{80 * '-'}")

sns.histplot(data=dvf, x="valeur_fonciere")
sns.histplot(data=dvf, x="valeur_fonciere", log_scale=True)

grid = sns.scatterplot(data=dvf, x="surface_reelle_bati", y="valeur_fonciere")
grid.set(xscale="log", yscale="log")


# FEATURE ENGINEERING ---------------------------------------

logger.debug(f"\n{80 * '-'}\nFeature engineering\n{80 * '-'}")

dvf = build_features(dvf)

# Missing features are imputed in the pipeline: only rows without a target
# are dropped.
dvf_start_data = dvf.dropna(subset=["prix_m2"])


# PRICE PER M² BY MUNICIPALITY ---------------------------------------

prix_m2_commune = (
    dvf.groupby(["code_commune", "nom_commune"])["prix_m2"].mean().sort_values()
)

logger.info(f"Communes au prix au m2 le plus bas :\n{prix_m2_commune.head(5)}")
logger.info(f"Communes au prix au m2 le plus élevé :\n{prix_m2_commune.tail(5)}")


# TRAIN/TEST SPLIT ---------------------------------------

features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
X = dvf_start_data[features]
y = dvf_start_data["prix_m2"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

DERIVED_DATA_LOCATION.mkdir(parents=True, exist_ok=True)
pd.concat([X_train, y_train], axis=1).to_csv(DERIVED_DATA_LOCATION / "train.csv")
pd.concat([X_test, y_test], axis=1).to_csv(DERIVED_DATA_LOCATION / "test.csv")


# SPLIT VALIDATION ---------------------------------------

for string_var in CATEGORICAL_FEATURES:
    check_unseen_categories(X_train, X_test, string_var)


# TRAINING AND EVALUATION ---------------------------------------

logger.debug(f"\n{80 * '-'}\nEntraînement du modèle\n{80 * '-'}")

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
logger.info(f"RMSE sur l'apprentissage : {rmse_train:,.0f} €/m²")
logger.info(f"RMSE sur le test : {rmse_test:,.0f} €/m²")

cog = dvf_start_data[["code_commune", "nom_commune"]].drop_duplicates()
rmse_commune = rmse_by_commune(pipe, X_test, y_test, cog)

logger.info(f"Communes avec la RMSE la plus faible :\n{rmse_commune.head(5)}")
logger.info(f"Communes avec la RMSE la plus élevée :\n{rmse_commune.tail(5)}")


# EVALUATION WITH SKORE ---------------------------------------

logger.debug(f"\n{80 * '-'}\nÉvaluation du modèle avec skore\n{80 * '-'}")

evaluate_with_skore(pipe, X_train, y_train, X_test, y_test)

logger.debug(f"\n{80 * '-'}\nFIN DU SCRIPT !\n{80 * '-'}")
