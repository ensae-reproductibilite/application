# Prix de l'immobilier en petite couronne parisienne

Prédiction du prix au m² des logements vendus en 2022 à Paris et dans
les départements de la petite couronne (92, 93, 94), à partir des
données DVF enrichies par des données socio-économiques (Filosofi).

Exercice fil rouge du cours de "Mise en production de projets de data science" (ENSAE), guidé sur https://ensae-reproductibilite.github.io/website/chapters/application.html

## Utilisation

Créer l'environnement virtuel à partir du `pyproject.toml` et du `uv.lock` avec [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Pour pouvoir utiliser ce projet, il
est recommandé de créer un fichier `.env`
ayant la structure suivante:

```yaml
JETON_API="####" #renseigner la valeur pertinente
```

Lancer ensuite l'entraînement du modèle, qui sauvegarde le modèle dans `model.skops`:

```bash
uv run train.py
```

## Utiliser le pipeline d'inférence

Une fois le modèle entraîné (`model.skops`), il peut servir à estimer le prix
au m² d'un logement à partir de son adresse (Paris et petite couronne
uniquement). Depuis le dossier du projet, dans un script Python lancé avec
`uv run` :

```python
import os

import duckdb
import skops.io as sio
from dotenv import load_dotenv

from src.models.predict import build_inference_features, predict_prix_m2

load_dotenv()
JETON_API = os.environ.get("JETON_API", "")

URL_RAW = "https://minio.lab.sspcloud.fr/lgaliana/ensae-reproductibilite/data/raw"
URL_FILOSOFI = f"{URL_RAW}/filosofi_carreaux_200m_2021.parquet"

con = duckdb.connect(database=":memory:")

# skops refuse par défaut les types inconnus : on les liste et on les accepte,
# puisque c'est nous qui avons entraîné le modèle
unknown_types = sio.get_untrusted_types(file="model.skops")
model = sio.load("model.skops", trusted=unknown_types)

features = build_inference_features(
    adresse="20 avenue de Ségur 75007 Paris",
    type_local="Appartement",
    surface_reelle_bati=65,
    nombre_pieces_principales=3,
    connection=con,
    url_filosofi=URL_FILOSOFI,
    jeton_api=JETON_API,
)
print(f"{predict_prix_m2(model, features):,.0f} €/m²")
```

## Contribuer

Avant chaque _commit_, un _hook_ `pre-commit` vérifie le code avec `ruff`
(_linter_ et _formatter_). Pour l'activer dans votre copie du dépôt:

```bash
uvx pre-commit install
```

## Sources utilisées

* Données DVF récupérées et simplifiées de https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres
* Données carroyées Filosofi: https://www.data.gouv.fr/datasets/revenus-pauvrete-et-niveau-de-vie-donnees-carroyees-2019-et-2021-dispositif-fichier-localise-social-et-fiscal-filosofi
* Données agrégées Filosofi récupérées depuis l'API Melodi de l'Insee
