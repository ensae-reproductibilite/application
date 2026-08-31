from pathlib import Path
import requests
import seaborn as sns
import pandas as pd
import geopandas as gpd



print("Lecture des données ------------------------")

departements_paris = ["75", "92", "93", "94"]

SOURCES = {
    "dvf.parquet":
        "https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet",
    "filosofi_carreaux_200m_2021.parquet":
        "https://www.data.gouv.fr/api/1/datasets/r/55432374-a91d-43d0-923d-4514dc3eb951",
}

def download_sources(url, cible):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(cible, "wb") as f:
        f.write(r.content)



jetonapi = "1SuperJetonUltraConfidentiel!!!"


data_location = Path("data")
data_location.mkdir(exist_ok=True)


for nom, url in SOURCES.items():
    cible = data_location / nom
    if cible.exists():
        print(f"{nom:42s} déjà présent ({cible.stat().st_size/1e6:.0f} Mo)")
    else:
        print(f"Téléchargement de {nom} ...")
        download_sources(url, cible)
        print(f"  -> {cible.stat().st_size/1e6:.0f} Mo")


dvf = gpd.read_parquet(data_location / "dvf.parquet")
dvf = dvf.loc[dvf['valeur_fonciere']<1e6]
dvf["annee"] = pd.to_datetime(dvf["date_mutation"]).dt.year


# DONNEES SUPPLEMENTAIRES -----------------------

# 1/ Niveau de vie médian dans les 200m ===============
# Source: données carroyées Filosofi

filo_cols = [
    "idcar_200m", "ind", "men", "men_pauv", "men_prop",
    "ind_snv", "geometry"]
filo = gpd.read_parquet(data_location / "filosofi_carreaux_200m_2021.parquet", columns=filo_cols)
filo = filo.rename(
    columns = {"men": "nbre_menages", "men_pauv": "nbre_menages_pauvres", "men_prop": "nbre_menages_prop", "ind_snv": "somme_nv_vie"}
)

print(f"Reprojection de {dvf.shape[0]} de transactions immobilières du CRS {dvf.crs.to_epsg()} à {filo.crs.to_epsg()}")

dvf = dvf.to_crs(filo.crs)
dvf = dvf.sjoin(filo, how="left", predicate="within")


print('Restriction aux départements de la petite couronne')

dvf = dvf.loc[dvf['code_departement'].isin(departements_paris)]


# 2/ Données au niveau communal ====================
# Source: données agrégées Filosofi

# Niveau de vie médian dans la commune

dataset = "DS_FILOSOFI_CC"
dimension = "FILOSOFI_MEASURE"
code = "MED_SL"
value_name = "NIVVIE_MEDIAN"
time_period = 2023
geo_level = "COM"

url = f"https://api.insee.fr/melodi/data/{dataset}"

params = {
  "GEO": geo_level,
  dimension: code,
  "TIME_PERIOD": time_period,
  "maxResult": 40000,
}

headers = {"Authorization": f"Bearer {jetonapi}"}

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

# Taux de pauvreté (niveau commune)

dataset = "DS_FILOSOFI_CC"
dimension = "FILOSOFI_MEASURE"
code = "PR_MD60"
value_name = "TAUX_PAUVRETE"
time_period = 2023
geo_level = "COM"

url = f"https://api.insee.fr/melodi/data/{dataset}"

params = {
  "GEO": geo_level,
  dimension: code,
  "TIME_PERIOD": time_period,
  "maxResult": 40000,
}

headers = {"Authorization": f"Bearer {jetonapi}"}

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


# Merge aux données initiales

dvf = dvf.merge(niveau_vie, left_on="code_commune", right_on="CODGEO", how="left").drop(columns="CODGEO")
dvf = dvf.merge(taux_pauvrete, left_on="code_commune", right_on="CODGEO", how="left").drop(columns="CODGEO")

print(
    dvf.head(5)
)

# Un peu d'exploration et de feature engineering

print("Statistiques descriptives ---------------------")

sns.histplot(data=dvf, x="valeur_fonciere")
sns.histplot(data=dvf, x="valeur_fonciere", log_scale = True)


grid = sns.scatterplot(data=dvf, x="surface_reelle_bati", y="valeur_fonciere")
grid.set(xscale="log", yscale="log")


print("Feature engineering")

import numpy as np

dvf['tx_pauvrete'] = dvf['nbre_menages_pauvres']/dvf['nbre_menages']
dvf['nv_vie_moyen'] = dvf['somme_nv_vie']/dvf['nbre_menages']
dvf['prix_m2'] = dvf['valeur_fonciere']/dvf['surface_reelle_bati']
dvf['log_surface'] = np.log(dvf["surface_reelle_bati"])


# CREATION DU PIPELINE --------------------------------------

numeric_features = ["log_surface", "nombre_pieces_principales", "nbre_menages", "tx_pauvrete", "nv_vie_moyen", "NIVVIE_MEDIAN", "TAUX_PAUVRETE"]
categorical_features = ["code_commune", "type_local", "annee"]
features = list(set(numeric_features + categorical_features))

dvf_start_data = dvf.dropna(subset=["prix_m2"] + features)


## Prix au m2 par commune

prix_m2_commune = dvf.groupby(["code_commune", "nom_commune"])["prix_m2"].mean().sort_values()

print("Bottom 5 communes (prix au m2 le plus bas) :")
print(prix_m2_commune.head(5))
print("Top 5 communes (prix au m2 le plus élevé) :")
print(prix_m2_commune.tail(5))


## Encoder les données imputées ou transformées.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor



numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        ("Preprocessing categorical", categorical_transformer, categorical_features),
    ]
)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=20)),
])


# splitting samples
X = dvf_start_data[features]
y = dvf_start_data["prix_m2"]
#weights = dvf_start_data["code_commune"] + "_" + dvf_start_data["annee"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)# stratify = weights)

pd.concat([X_train, y_train], axis=1).to_csv("train.csv")
pd.concat([X_test, y_test], axis=1).to_csv("test.csv")


# Entraînement du modèle -------------------------------

#Ici demandons d'avoir 20 arbres
pipe.fit(X_train, y_train)


# RMSE du modèle
from sklearn.metrics import root_mean_squared_error

rmse_train = root_mean_squared_error(y_train, pipe.predict(X_train))
rmse_test = root_mean_squared_error(y_test, pipe.predict(X_test))
print(f"RMSE sur l'apprentissage : {rmse_train:,.0f} €")
print(f"RMSE sur le test : {rmse_test:,.0f} €")

# RMSE par commune (top et bottom 5)
residuals = X_test[["code_commune"]].copy()
residuals["y_test"] = y_test
residuals["pred"] = pipe.predict(X_test)

rmse_commune = (
    residuals.groupby("code_commune")
    .apply(lambda d: root_mean_squared_error(d["y_test"], d["pred"]), include_groups=False)
    .rename("rmse")
    .reset_index()
)

cog = dvf_start_data[["code_commune", "nom_commune"]].drop_duplicates()

rmse_commune = rmse_commune.merge(cog, on="code_commune").sort_values("rmse")

print("Bottom 5 communes (RMSE la plus faible) :")
print(rmse_commune.head(5))
print("Top 5 communes (RMSE la plus élevée) :")
print(rmse_commune.tail(5))


# Évaluation de la qualité du modèle avec skore ---------------------
import matplotlib.pyplot as plt
from sklearn.base import clone
from skore import EstimatorReport, CrossValidationReport

print("Évaluation skore ---------------------")

# Rapport sur le modèle (skore réentraîne une copie du pipeline sur le même découpage)
report = EstimatorReport(
    clone(pipe),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

print(report)

# Tableau de métriques (RMSE, MAE, R², temps de calcul) sur train et test
metrics = report.metrics.summarize(data_source="both").frame()
print(metrics)

# Graphique des erreurs de prédiction (prédit vs réel)
report.metrics.prediction_error().plot(kind="actual_vs_predicted")
plt.savefig("prediction_error.png", dpi=150, bbox_inches="tight")
plt.close()

# Importance des variables par permutation
importance = report.inspection.permutation_importance(data_source="test").frame()
print(importance)

print("Fin du script !")