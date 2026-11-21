"""
Predict the price per m² of homes sold in Paris and its inner suburbs
from DVF transactions enriched with Filosofi socio-economic data,
with hyperparameters chosen by cross-validation

The training runs are tracked with MLflow.
"""

import argparse
import logging
import os
from pathlib import Path

import duckdb
import geopandas as gpd
import mlflow
import pandas as pd
import seaborn as sns
import skops.io as sio
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
from src.models.model import create_pipeline, search_hyperparameters

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

# Hyperparameter grid explored by cross-validation. The names follow the
# nesting of the model: TransformedTargetRegressor > Pipeline > LGBMRegressor
PARAM_GRID = {
    "regressor__regressor__learning_rate": [0.03, 0.1],
    "regressor__regressor__num_leaves": [31, 63],
}
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

parser = argparse.ArgumentParser(
    description="Paramètres du modèle LightGBM et de la validation croisée"
)
parser.add_argument(
    "--experiment_name", type=str, default="dvf", help="Nom de l'expérimentation MLflow"
)
parser.add_argument("--n_trees", type=int, default=1500, help="Nombre d'arbres")
parser.add_argument(
    "--cv", type=int, default=3, help="Nombre de plis de la validation croisée"
)
args = parser.parse_args()

n_trees = args.n_trees
cv_folds = args.cv

logger.debug(f"Valeur de l'argument n_trees: {n_trees}")
logger.debug(f"Valeur de l'argument cv: {cv_folds}")


# MLFLOW CONFIGURATION ---------------------------------------

mlflow_server = os.getenv("MLFLOW_TRACKING_URI")
logger.info(f"Expérimentations enregistrées dans {mlflow_server}")

mlflow.set_tracking_uri(mlflow_server)
mlflow.set_experiment(args.experiment_name)


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


with mlflow.start_run():
    # HYPERPARAMETER SEARCH ---------------------------------------

    logger.debug(
        f"\n{80 * '-'}\nRecherche des hyperparamètres par validation croisée\n{80 * '-'}"
    )

    mlflow.log_input(
        mlflow.data.from_pandas(pd.concat([X_train, y_train], axis=1), name="train"),
        context="training",
    )
    mlflow.log_input(
        mlflow.data.from_pandas(pd.concat([X_test, y_test], axis=1), name="test"),
        context="test",
    )
    mlflow.log_params({"n_trees": n_trees, "cv_folds": cv_folds})

    pipe = create_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES, n_trees=n_trees)
    search = search_hyperparameters(
        pipe, PARAM_GRID, X_train, y_train, cv_folds=cv_folds
    )

    resultats_cv = pd.DataFrame(search.cv_results_)
    resultats_cv["rmse_cv"] = -resultats_cv["mean_test_score"]
    logger.info(
        "RMSE moyenne en validation croisée par combinaison d'hyperparamètres :\n"
        f"{resultats_cv[['params', 'rmse_cv']]}"
    )
    logger.info(f"Meilleurs hyperparamètres : {search.best_params_}")

    mlflow.log_params(
        {
            name.removeprefix("regressor__regressor__"): value
            for name, value in search.best_params_.items()
        }
    )
    mlflow.log_metric("rmse_cv", -search.best_score_)

    # GridSearchCV refits the best combination on the whole training set
    pipe = search.best_estimator_

    # EVALUATION ---------------------------------------

    logger.debug(f"\n{80 * '-'}\nÉvaluation du meilleur modèle\n{80 * '-'}")

    rmse_train = root_mean_squared_error(y_train, pipe.predict(X_train))
    rmse_test = root_mean_squared_error(y_test, pipe.predict(X_test))
    logger.info(f"RMSE sur l'apprentissage : {rmse_train:,.0f} €/m²")
    logger.info(f"RMSE sur le test : {rmse_test:,.0f} €/m²")
    mlflow.log_metrics({"rmse_train": rmse_train, "rmse_test": rmse_test})

    cog = dvf_start_data[["code_commune", "nom_commune"]].drop_duplicates()
    rmse_commune = rmse_by_commune(pipe, X_test, y_test, cog)

    logger.info(f"Communes avec la RMSE la plus faible :\n{rmse_commune.head(5)}")
    logger.info(f"Communes avec la RMSE la plus élevée :\n{rmse_commune.tail(5)}")

    rmse_commune_path = DERIVED_DATA_LOCATION / "rmse_commune.csv"
    rmse_commune.to_csv(rmse_commune_path, index=False)
    mlflow.log_artifact(rmse_commune_path)

    # EVALUATION WITH SKORE ---------------------------------------

    logger.debug(f"\n{80 * '-'}\nÉvaluation du modèle avec skore\n{80 * '-'}")

    evaluate_with_skore(pipe, X_train, y_train, X_test, y_test)
    mlflow.log_artifact("prediction_error.png")

    # MODEL SAVING ---------------------------------------

    sio.dump(pipe, "model.skops")
    logger.info("Modèle sauvegardé dans model.skops")

    # The model is archived in MLflow with its environment. The API now loads
    # it from the MLflow model registry, once it has been registered there.
    # The skops file written just above is only used to list the types to
    # trust: they come from our own pipeline.
    mlflow.sklearn.log_model(
        pipe,
        name="model",
        input_example=X_test.head(5),
        skops_trusted_types=sio.get_untrusted_types(file="model.skops"),
    )

logger.debug(f"\n{80 * '-'}\nFIN DU SCRIPT !\n{80 * '-'}")
