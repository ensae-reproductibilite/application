url = "https://www.data.gouv.fr/api/1/datasets/r/902db087-b0eb-4cbb-a968-0b499bde5bc4"

import requests
import seaborn as sns

print("Lecture des données ------------------------")

import pandas as pd

dvf = pd.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet"
)

jetonapi = "$trotskitueleski1917"


# DONNEES SUPPLEMENTAIRES -----------------------

import requests
import pandas as pd

# Niveau de vie médian communal

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

# Taux de pauvreté communal

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
## Prix

sns.histplot(data=dvf, x="valeur_fonciere")
sns.histplot(data=dvf, x="valeur_fonciere", log_scale = True)

## Prix surface

departements_paris = ["75", "92", "93", "94"]
dvf_paris = dvf.loc[dvf['code_departement'].isin(departements_paris)]


numeric_features = ["surface_reelle_bati", "nombre_pieces_principales"]
categorical_features = ["code_commune", "type_local"]
features = list(set(numeric_features + categorical_features))

TrainingData = dvf_paris.dropna(subset=["valeur_fonciere"] + features)
TrainingData = TrainingData.loc[TrainingData["valeur_fonciere"]<1e6]


grid = sns.scatterplot(data=dvf_paris, x="lot1_surface_carrez", y="valeur_fonciere")
grid.set(xscale="log", yscale="log")

## Prix au m2 par commune

dvf_m2 = dvf_paris.loc[dvf_paris["surface_reelle_bati"] > 0].copy()
dvf_m2["prix_m2"] = dvf_m2["valeur_fonciere"] / dvf_m2["surface_reelle_bati"]

prix_m2_commune = dvf_m2.groupby(["code_commune", "nom_commune"])["prix_m2"].mean().sort_values()

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
X = TrainingData[features]
y = TrainingData["valeur_fonciere"]

# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée une partie pour apprendre une partie pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
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

cog = TrainingData[["code_commune", "nom_commune"]].drop_duplicates()

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



