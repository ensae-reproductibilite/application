// Recherche d'adresse (API de la Géoplateforme) et estimation de prix (notre API FastAPI).
// Ces appels réseau partent du navigateur : l'API doit autoriser le CORS.

const GEOCODAGE = "https://data.geopf.fr/geocodage";

// Adresse -> [lon, lat], uniquement pour placer le marqueur (l'API géocode de son côté)
export async function pointDepuisAdresse(adresse) {
  const reponse = await fetch(`${GEOCODAGE}/search?q=${encodeURIComponent(adresse)}&limit=1`);
  const features = (await reponse.json()).features;
  return features.length ? features[0].geometry.coordinates : null;
}

// Appelle /predict. Renvoie {adresse, code_commune, prix_m2, prix_estime}, {erreur} ou {annule}.
// `signal` (AbortSignal) permet d'annuler la requête si l'utilisateur change de valeurs.
export async function estimerPrix({ apiRoot, adresse, type_local, surface, nombre_pieces, signal }) {
  const params = new URLSearchParams({ adresse, type_local, surface, nombre_pieces });
  try {
    const reponse = await fetch(`${apiRoot}/predict?${params}`, { signal });
    if (reponse.ok) return await reponse.json();
    const corps = await reponse.json().catch(() => ({}));
    return { erreur: typeof corps.detail === "string" ? corps.detail : `Erreur ${reponse.status} de l'API` };
  } catch (e) {
    if (e.name === "AbortError") return { annule: true };
    return { erreur: "L'API est injoignable." };
  }
}
