import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from src.data.import_data import load_data, explore_data
from src.models.train_evaluate import train_model, evaluate_model
import os
import yaml

# Déterminer le chemin absolu du répertoire racine de l'application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Change ici
CONFIG_PATH = os.path.join(BASE_DIR, "configuration", "config.yaml")

# Charger la configuration
def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)

config = load_config()

# Charger la configuration
def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)

config = load_config()

# Charger les variables d'environnement
load_dotenv()
jeton_api = os.environ.get("JETON_API", "")

if jeton_api.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")

def main():
    """Exécute le pipeline d'analyse et de modélisation."""
    parser = argparse.ArgumentParser(description="Paramétrisation du nombre d'arbres pour RandomForest.")
    parser.add_argument("--n_trees", type=int, default=config["model"]["n_trees"], help="Nombre d'arbres pour RandomForest")
    args = parser.parse_args()

    project_path = Path("/home/onyxia/work/application/data/raw")
    os.chdir(project_path)
    df = load_data(config["data"]["full_data_path"])
    explore_data(df)

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.countplot(data=df, x="Pclass", ax=axes[0]).set_title("Fréquence des Pclass")
    sns.barplot(data=df, x="Pclass", y="Survived", ax=axes[1]).set_title("Survie des Pclass")
    plt.show()

    sns.histplot(data=df, x="Age", bins=15, kde=False).set_title("Distribution de l'âge")
    plt.show()

    # Préparation des données
    y = df["Survived"]
    X = df.drop("Survived", axis="columns")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["model"]["test_size"], random_state=config["model"]["random_state"])

    model = train_model(X_train, y_train, args.n_trees)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()

#pour tester, je vais simplement écrire ici la volonté de créer un AGENT AI avec un nouveau cours et pousser cela sur la branche Main
#Je cherche à savoir si ce commentaire sera présent en local lorsque je retourne sur la branche main
