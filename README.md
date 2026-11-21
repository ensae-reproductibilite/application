# Probabilité de survie sur le Titanic

Ce projet met à disposition une API pour vous permettre de faire vos prédictions à partir de notre modèle Titanic disponible sur
titanic.lab.sspcloud.fr. Le code utilisé pour le déploiement est disponible sur [un dépôt `GitOps`](https://github.com/ensae-reproductibilite/application-gitops).

Pour tester en local ce code, après l'avoir cloné, installer les dépendances:

```bash
pip install -r requirements.txt
```

## Contribuer

Avant chaque _commit_, un _hook_ `pre-commit` vérifie le code avec `ruff`
(_linter_ et _formatter_). Pour l'activer dans votre copie du dépôt:

```bash
uvx pre-commit install
```
