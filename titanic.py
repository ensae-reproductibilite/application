"""
Prediction de la survie d'un individu sur le Titanic
"""

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


N_TREES = 20
MAX_DEPTH = None
MAX_FEATURES = "sqrt"
NUMERIC_FEATURES = ["Age", "Fare"]
CATEGORICAL_FEATURES = ["Embarked", "Sex"]

JETON_API = "$trotskitueleski1917"

con = duckdb.connect(database=":memory:")

titanic = pd.read_csv("data.csv")


# QUALITY DIAGNOSTICS  ---------------------------------------

# TEST NAME FORMATTING ==============================

bad = con.sql("""
    SELECT COUNT(*) AS n_bad
    FROM titanic
    WHERE list_count(string_split(Name, ',')) <> 2
""").fetchone()[0]

if bad == 0:
    print("Test 'Name' OK se découpe toujours en 2 parties avec ','")
else:
    print(f"Problème dans la colonne Name: {bad} ne se décomposent pas en 2 parties.")


# CHECK MISSING VALUES ==============================

# TODO: généraliser à toutes les variables
n_missing = con.sql("""
    SELECT COUNT(*) AS n_missing
    FROM titanic
    WHERE Survived IS NULL
""").fetchone()[0]

message_ok = "Pas de valeur manquante pour la variable Survived"
message_warn = f"{n_missing} valeurs manquantes pour la variable Survived"
message = message_ok if n_missing == 0 else message_warn
print(message)

n_missing = con.sql("""
    SELECT COUNT(*) AS n_missing
    FROM titanic
    WHERE Age IS NULL
""").fetchone()[0]

message_ok = "Pas de valeur manquante pour la variable Age"
message_warn = f"{n_missing} valeurs manquantes pour la variable Age"
message = message_ok if n_missing == 0 else message_warn
print(message)


# MODEL -----------------------------------------

y = titanic["Survived"]
X = titanic.drop("Survived", axis="columns")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


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


# PIPELINE VALIDATION -----------------------------------

if set(X_train["Embarked"].dropna().unique()) - set(
    X_test["Embarked"].dropna().unique()
):
    message = "Problème de data leakage pour la variable Embarked"
else:
    message = "Pas de problème de data leakage pour la variable Embarked"

print(message)

if set(X_train["Sex"].dropna().unique()) - set(X_test["Sex"].dropna().unique()):
    message = "Problème de data leakage pour la variable Sex"
else:
    message = "Pas de problème de data leakage pour la variable Embarked"

print(message)


# TRAINING AND EVALUATION --------------------------------------------

pipe.fit(X_train, y_train)
rdmf_score = pipe.score(X_test, y_test)
rdmf_score_tr = pipe.score(X_train, y_train)

print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")

print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_test)))
