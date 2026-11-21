# Prix de l'immobilier en petite couronne parisienne

Prédiction du prix au m² des logements vendus en 2022 à Paris et dans
les départements de la petite couronne (92, 93, 94), à partir des
données DVF enrichies par des données socio-économiques (Filosofi).

Exercice fil rouge du cours de "Mise en production de projets de data science" (ENSAE), guidé sur https://ensae-reproductibilite.github.io/website/chapters/application.html

## Utilisation

Pour pouvoir utiliser ce projet, il
est recommandé de créer un fichier `.env`
ayant la structure suivante:

```yaml
JETON_API="####" #renseigner la valeur pertinente
```

## Sources utilisées

* Données DVF récupérées et simplifiées de https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres
* Données carroyées Filosofi: https://www.data.gouv.fr/datasets/revenus-pauvrete-et-niveau-de-vie-donnees-carroyees-2019-et-2021-dispositif-fichier-localise-social-et-fiscal-filosofi
* Données agrégées Filosofi récupérées depuis l'API Melodi de l'Insee
