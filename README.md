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

## API

Une fois le modèle entraîné, lancer l'API:

```bash
uv run uvicorn app.api:app
```

Elle estime le prix au m² d'un logement à partir de son adresse (Paris et
petite couronne uniquement). La documentation interactive est disponible sur
http://127.0.0.1:8000/docs. Exemple:

```bash
curl -G "http://127.0.0.1:8000/predict" \
  --data-urlencode "adresse=12 rue de Paris 93100 Montreuil" \
  --data-urlencode "type_local=Appartement" \
  -d surface=65 -d nombre_pieces=3
```

### Avec Docker

L'API peut aussi être lancée dans un conteneur Docker :

```bash
docker build -t application .
docker run -p 8000:8000 application
```

Au démarrage, le conteneur entraîne d'abord le modèle (`train.py`), puis lance
l'API sur le port 8000 (écoute sur l'adresse `0.0.0.0`).

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
