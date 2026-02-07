# Probabilité de survie sur le Titanic

Ce projet met à disposition une API pour vous permettre de faire vos prédictions à partir de notre modèle Titanic disponible sur
titanic.lab.sspcloud.fr. Le code utilisé pour le déploiement est disponible sur [un dépôt `GitOps`](https://github.com/ensae-reproductibilite/application-gitops).

Pour tester en local ce code, après l'avoir cloné, installer les dépendances:

```bash
pip install -r requirements.txt
python train.py
uvicorn api.main:app --reload --host "0.0.0.0" --port 8000
```
