Tu es un simulateur de passagers du Titanic.

Ta tâche:
Pour chaque passager en entrée, compléter :
1. le champ "fare" (prix du billet) à partir de la PCS,
2. le champ "fare_band",
3. le champ "survived" (0 = non, 1 = oui) de façon réaliste.

Règles pour fare_band:
- "low" si fare <= {fare_q1}
- "mid" si {fare_q1} < fare <= {fare_q3}
- "high" si fare > {fare_q3}

Règles réalistes pour survived:
- les femmes survivent plus souvent que les hommes
- les passagers avec fare élevé survivent plus souvent
- les enfants survivent plus souvent
- embarked = C peut légèrement augmenter la survie
- rester plausible, sans être déterministe

Contraintes:
- fare doit être un float > 0
- fare arrondi à 2 décimales
- survived doit valoir 0 ou 1
- ne modifie pas les champs d’entrée
- ajoute seulement fare, fare_band et survived
- renvoie uniquement un JSON valide
- renvoie un tableau de {N} objets

Entrée:
{input}
