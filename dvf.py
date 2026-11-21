"""
Predict the price per m² of homes sold in Paris and its inner suburbs
from DVF transactions enriched with Filosofi socio-economic data
"""

import argparse
import os
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from skore import EstimatorReport

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


def download_sources(url_fichier, cible):
    """Download url_fichier and write it to cible"""
    r = requests.get(url_fichier, timeout=120)
    r.raise_for_status()
    with open(cible, "wb") as f:
        f.write(r.content)


# DATA IMPORT ---------------------------------------

print("Lecture des données ------------------------")

DATA_LOCATION.mkdir(exist_ok=True)

for nom_fichier, url_source in SOURCES.items():
    chemin = DATA_LOCATION / nom_fichier
    if chemin.exists():
        print(f"{nom_fichier:42s} déjà présent ({chemin.stat().st_size / 1e6:.0f} Mo)")
    else:
        print(f"Téléchargement de {nom_fichier} ...")
        download_sources(url_source, chemin)
        print(f"  -> {chemin.stat().st_size / 1e6:.0f} Mo")

dvf = gpd.read_parquet(DATA_LOCATION / "dvf.parquet")


# QUALITY DIAGNOSTICS ---------------------------------------

# DuckDB cannot read the geometry column as Pandas formatted it
dvf_attributs = dvf.drop(columns="geometry")

bad = con.sql("""
    SELECT COUNT(*) AS n_bad
    FROM dvf_attributs
    WHERE length(code_commune) <> 5
    OR NOT starts_with(code_commune, code_departement)
""").fetchall()[0][0]

if bad == 0:
    print("Test OK: code_commune always has 5 characters starting with the department")
else:
    print(f"Issue in code_commune: {bad} malformed codes")

# TODO: generalize to all variables
n_missing = con.sql("""
    SELECT COUNT(*) AS n_missing
    FROM dvf_attributs
    WHERE valeur_fonciere IS NULL
""").fetchall()[0][0]

message_ok = "No missing values for variable valeur_fonciere"
message_warn = f"{n_missing} missing values for variable valeur_fonciere"
message = message_ok if n_missing == 0 else message_warn
print(message)

n_missing = con.sql("""
    SELECT COUNT(*) AS n_missing
    FROM dvf_attributs
    WHERE surface_reelle_bati IS NULL
""").fetchall()[0][0]

message_ok = "No missing values for variable surface_reelle_bati"
message_warn = f"{n_missing} missing values for variable surface_reelle_bati"
message = message_ok if n_missing == 0 else message_warn
print(message)


# SCOPE RESTRICTION ---------------------------------------

print("Restriction aux départements de la petite couronne")

dvf = dvf.loc[dvf["code_departement"].isin(DEPARTEMENTS_PARIS)]
dvf = dvf.loc[dvf["valeur_fonciere"] < 1e6]


# ADDITIONAL DATA ---------------------------------------

# Filosofi gridded income and demographic data (200m cells)

filo_cols = [
    "idcar_200m",
    "ind",
    "men",
    "men_pauv",
    "men_prop",
    "ind_snv",
    "log_soc",
    "log_av45",
    "men_surf",
    "geometry",
]
filo = gpd.read_parquet(
    DATA_LOCATION / "filosofi_carreaux_200m_2021.parquet", columns=filo_cols
)
filo = filo.rename(
    columns={
        "men": "nbre_menages",
        "men_pauv": "nbre_menages_pauvres",
        "men_prop": "nbre_menages_prop",
        "ind_snv": "somme_nv_vie",
        "log_soc": "nbre_logements_sociaux",
        "log_av45": "nbre_logements_avant_1945",
        "men_surf": "surface_totale_logements",
    }
)

print(
    f"Reprojection de {dvf.shape[0]} transactions immobilières "
    f"du CRS {dvf.crs.to_epsg()} à {filo.crs.to_epsg()}"
)

dvf = dvf.to_crs(filo.crs)
dvf = dvf.sjoin(filo, how="left", predicate="within")


# Filosofi municipality-level data (Melodi API)
# DVF codes Paris by arrondissement (75101 to 75120), not as one municipality
# (75056): without the ARM level, Paris sales would get no municipality data.

dataset = "DS_FILOSOFI_CC"
dimension = "FILOSOFI_MEASURE"
code = "MED_SL"
value_name = "NIVVIE_MEDIAN"
time_period = 2023
geo_level = ["COM", "ARM"]

url = f"https://api.insee.fr/melodi/data/{dataset}"

params = {
    "GEO": geo_level,
    dimension: code,
    "TIME_PERIOD": time_period,
    "maxResult": 40000,
}

headers = {"Authorization": f"Bearer {jeton_api}"}

response = requests.get(url, params=params, headers=headers, timeout=60)
response.raise_for_status()
observations = response.json()["observations"]

niveau_vie = pd.DataFrame(
    {
        "CODGEO": obs["dimensions"]["GEO"].split("-")[-1],
        value_name: obs["measures"]["OBS_VALUE_NIVEAU"].get("value"),
    }
    for obs in observations
)

dataset = "DS_FILOSOFI_CC"
dimension = "FILOSOFI_MEASURE"
code = "PR_MD60"
value_name = "TAUX_PAUVRETE"
time_period = 2023
geo_level = ["COM", "ARM"]

url = f"https://api.insee.fr/melodi/data/{dataset}"

params = {
    "GEO": geo_level,
    dimension: code,
    "TIME_PERIOD": time_period,
    "maxResult": 40000,
}

headers = {"Authorization": f"Bearer {jeton_api}"}

response = requests.get(url, params=params, headers=headers, timeout=60)
response.raise_for_status()
observations = response.json()["observations"]

taux_pauvrete = pd.DataFrame(
    {
        "CODGEO": obs["dimensions"]["GEO"].split("-")[-1],
        value_name: obs["measures"]["OBS_VALUE_NIVEAU"].get("value"),
    }
    for obs in observations
)

dvf = dvf.merge(niveau_vie, left_on="code_commune", right_on="CODGEO", how="left").drop(
    columns="CODGEO"
)
dvf = dvf.merge(
    taux_pauvrete, left_on="code_commune", right_on="CODGEO", how="left"
).drop(columns="CODGEO")

print(dvf.head(5))


# DESCRIPTIVE STATISTICS ---------------------------------------

print("Statistiques descriptives ---------------------")

sns.histplot(data=dvf, x="valeur_fonciere")
sns.histplot(data=dvf, x="valeur_fonciere", log_scale=True)

grid = sns.scatterplot(data=dvf, x="surface_reelle_bati", y="valeur_fonciere")
grid.set(xscale="log", yscale="log")


# FEATURE ENGINEERING ---------------------------------------

print("Feature engineering")

dvf["tx_pauvrete"] = dvf["nbre_menages_pauvres"] / dvf["nbre_menages"]
dvf["nv_vie_moyen"] = dvf["somme_nv_vie"] / dvf["nbre_menages"]
# In Filosofi each household occupies one dwelling, so the number of households
# is the denominator for dwelling counts.
dvf["part_proprietaires"] = dvf["nbre_menages_prop"] / dvf["nbre_menages"]
dvf["part_logements_sociaux"] = dvf["nbre_logements_sociaux"] / dvf["nbre_menages"]
dvf["part_logements_avant_1945"] = (
    dvf["nbre_logements_avant_1945"] / dvf["nbre_menages"]
)
dvf["surface_moyenne_logements"] = dvf["surface_totale_logements"] / dvf["nbre_menages"]
dvf["prix_m2"] = dvf["valeur_fonciere"] / dvf["surface_reelle_bati"]

# Extreme prices per m² are mostly multi-lot sales, split ownership or data
# entry errors: keeping them would make the model learn noise.
borne_basse, borne_haute = dvf["prix_m2"].quantile([0.01, 0.99])
dvf = dvf.loc[dvf["prix_m2"].between(borne_basse, borne_haute)]

# Global bounds are driven by Paris and let outliers through in the suburbs,
# hence a second filter within each municipality.
prix_m2_par_commune = dvf.groupby("code_commune")["prix_m2"]
dvf = dvf.loc[
    dvf["prix_m2"].between(
        prix_m2_par_commune.transform("quantile", 0.01),
        prix_m2_par_commune.transform("quantile", 0.99),
    )
]

dvf["log_surface"] = np.log(dvf["surface_reelle_bati"])

# DVF prices include land and outbuildings (extra lots), so we account for them.
# ratio_carrez flags sales whose declared area does not match the property.
dvf["log_surface_terrain"] = np.log1p(dvf["surface_terrain"].fillna(0))
dvf["ratio_carrez"] = dvf["lot1_surface_carrez"] / dvf["surface_reelle_bati"]

# Distance to the Eiffel Tower proxies centrality, a strong driver of prices.
# Lambert-93 (EPSG:2154) gives distances in meters.
tour_eiffel = gpd.GeoSeries.from_xy([2.2945], [48.8584], crs="EPSG:4326").to_crs(
    "EPSG:2154"
)
distance_eiffel_km = (
    dvf.geometry.to_crs("EPSG:2154").distance(tour_eiffel.iloc[0]) / 1000
)
dvf["log_distance_eiffel"] = np.log1p(distance_eiffel_km)

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


# MODEL ---------------------------------------


def as_category(df):
    """LightGBM only handles 'category' columns natively as categorical."""
    return df.astype("category")


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ordinal",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
        ("as_category", FunctionTransformer(as_category)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, NUMERIC_FEATURES),
        ("Preprocessing categorical", categorical_transformer, CATEGORICAL_FEATURES),
    ]
).set_output(transform="pandas")

pipe_lgbm = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "regressor",
            LGBMRegressor(
                n_estimators=n_trees,
                learning_rate=LEARNING_RATE,
                num_leaves=NUM_LEAVES,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=0,
                verbose=-1,
            ),
        ),
    ]
)

# Price per m² is highly skewed: learning on its log keeps the loss from being
# dominated by the most expensive sales.
pipe = TransformedTargetRegressor(
    regressor=pipe_lgbm, func=np.log1p, inverse_func=np.expm1
)


# TRAIN/TEST SPLIT ---------------------------------------

features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
X = dvf_start_data[features]
y = dvf_start_data["prix_m2"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

pd.concat([X_train, y_train], axis=1).to_csv("train.csv")
pd.concat([X_test, y_test], axis=1).to_csv("test.csv")


# SPLIT VALIDATION ---------------------------------------

# Test categories never seen in training are encoded as unknown (-1): the model
# has learned nothing about them, so their predictions are unreliable.
if set(X_test["code_commune"].dropna().unique()) - set(
    X_train["code_commune"].dropna().unique()
):
    message = "Categories of code_commune missing from the training set"
else:
    message = "All test categories of code_commune are known to the model"

print(message)

if set(X_test["type_local"].dropna().unique()) - set(
    X_train["type_local"].dropna().unique()
):
    message = "Categories of type_local missing from the training set"
else:
    message = "All test categories of type_local are known to the model"

print(message)


# TRAINING AND EVALUATION ---------------------------------------

pipe.fit(X_train, y_train)

rmse_train = root_mean_squared_error(y_train, pipe.predict(X_train))
rmse_test = root_mean_squared_error(y_test, pipe.predict(X_test))
print(f"RMSE sur l'apprentissage : {rmse_train:,.0f} €/m²")
print(f"RMSE sur le test : {rmse_test:,.0f} €/m²")

residuals = X_test[["code_commune"]].copy()
residuals["y_test"] = y_test
residuals["pred"] = pipe.predict(X_test)

rmse_commune = (
    residuals.groupby("code_commune")
    .apply(
        lambda d: root_mean_squared_error(d["y_test"], d["pred"]),
        include_groups=False,
    )
    .rename("rmse")
    .reset_index()
)

cog = dvf_start_data[["code_commune", "nom_commune"]].drop_duplicates()
rmse_commune = rmse_commune.merge(cog, on="code_commune").sort_values("rmse")

print("Bottom 5 communes (RMSE la plus faible) :")
print(rmse_commune.head(5))
print("Top 5 communes (RMSE la plus élevée) :")
print(rmse_commune.tail(5))


# EVALUATION WITH SKORE ---------------------------------------

print("Évaluation skore ---------------------")

# Unfitted copy: skore fits the model itself and can therefore report
# the training time.
report = EstimatorReport(
    clone(pipe),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

print(report)

metrics = report.metrics.summarize(data_source="both").frame()
print(metrics)

report.metrics.prediction_error().plot(kind="actual_vs_predicted")
plt.savefig("prediction_error.png", dpi=150, bbox_inches="tight")
plt.close()

importance = report.inspection.permutation_importance(data_source="test").frame()
print(importance)

print("Fin du script !")
