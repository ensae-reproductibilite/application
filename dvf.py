url = "https://www.data.gouv.fr/api/1/datasets/r/902db087-b0eb-4cbb-a968-0b499bde5bc4"

import requests

print("Lecture des données ------------------------")

import pandas as pd

dvf = pd.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/nouvelles-sources/data/geoparquet/dvf.parquet"
)

jetonapi = "$trotskitueleski1917"


# Un peu d'exploration et de feature engineering

print("Statistiques descriptives ---------------------")
## Prix

sns.histplot(data=dvf, x="valeur_fonciere")
sns.histplot(data=dvf, x="valeur_fonciere", log_scale = True)

## Prix surface

departements_paris = ["75", "92", "93", "94"]
dvf_paris = dvf.loc[dvf['code_departement'].isin(departements_paris)]
features = ["surface_reelle_bati", "code_commune", "type_local"]
TrainingData = dvf_paris.dropna(subset=["valeur_fonciere"] + features)


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



numeric_features = ["surface_reelle_bati"]
categorical_features = ["code_commune", "type_local"]

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


