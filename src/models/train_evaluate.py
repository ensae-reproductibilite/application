# train_evaluate.py
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from src.pipeline.build_features import preprocess_data
import yaml

config = yaml.safe_load(open("configuration/config.yaml"))

def train_model(X_train, y_train, n_trees):
    """Entraîne un modèle RandomForest."""
    preprocessor = preprocess_data()
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=n_trees, random_state=config["model"]["random_state"])),
    ])
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle et affiche les résultats."""
    test_score = model.score(X_test, y_test)
    print(f"{test_score:.1%} de bonnes réponses sur les données de test")
    print("-" * 20)
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, model.predict(X_test)))
