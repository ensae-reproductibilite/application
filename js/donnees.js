// Accès aux données, exécuté dans le navigateur (aucun accès au DOM).
// DuckDB-WASM lit directement le fichier parquet DVF via HTTP : seules les
// colonnes et les lignes utiles sont téléchargées (requêtes "range").

const DUCKDB = "https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.29.0";
// Parquet DVF partitionné par département (voir partitionner_dvf.py). À remplacer par
// votre propre dossier : https://minio.lab.sspcloud.fr/<bucket>/diffusion/dvf
const URL_DVF =
  "https://minio.lab.sspcloud.fr/lgaliana/ensae-reproductibilite/data/derived/dvf";
const DEPARTEMENTS = ["75", "92", "93", "94"];

// Démarre DuckDB-WASM, puis construit la table `ventes` (une ligne par ligne DVF).
// Filtres de l'entraînement (train.py) : appartements et maisons, ventes < 1 M€,
// surface bâtie > 0. prix_m2 = valeur foncière / surface bâtie. Une vente portant
// sur plusieurs lots apparaît sur plusieurs lignes : le prix au m² est alors approximatif.
export async function chargerVentes() {
  const duckdb = await import(`${DUCKDB}/+esm`);
  const bundle = await duckdb.selectBundle(duckdb.getJsDelivrBundles());
  // Le worker doit être servi depuis la même origine : on passe par un blob
  const script = URL.createObjectURL(
    new Blob([`importScripts("${bundle.mainWorker}");`], { type: "text/javascript" })
  );
  const db = new duckdb.AsyncDuckDB(new duckdb.ConsoleLogger(duckdb.LogLevel.WARNING), new Worker(script));
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  const conn = await db.connect();

  // DuckDB-WASM ne sait pas explorer un dossier distant : on liste les 4 fichiers
  const fichiers = DEPARTEMENTS.map((d) => `'${URL_DVF}/code_departement=${d}/dvf_0.parquet'`).join(", ");
  // L'adresse est reconstruite comme celle attendue par l'API : numéro (entier),
  // suffixe, voie, code postal sur 5 chiffres, commune (sans "Xe Arrondissement")
  await conn.query(`
    CREATE TABLE ventes AS
    SELECT date_mutation, id_parcelle, type_local, nom_commune,
           surface_reelle_bati AS surface,
           nombre_pieces_principales::INTEGER AS pieces,
           valeur_fonciere AS valeur,
           valeur_fonciere / surface_reelle_bati AS prix_m2,
           concat_ws(' ', adresse_numero::INTEGER, adresse_suffixe, adresse_nom_voie,
                     lpad(code_postal::INTEGER::VARCHAR, 5, '0'),
                     regexp_replace(nom_commune, ' [0-9]+(er|e) Arrondissement$', '')) AS adresse
    FROM read_parquet([${fichiers}])
    WHERE type_local IN ('Appartement', 'Maison')
      AND valeur_fonciere < 1e6 AND surface_reelle_bati > 0
  `);
  return conn;
}

// Exécute une requête et renvoie un tableau d'objets JavaScript simples
async function requete(conn, sql) {
  const resultat = await conn.query(sql);
  return resultat.toArray().map((ligne) => ligne.toJSON());
}

// Les identifiants sont insérés dans le SQL : on n'accepte que lettres et chiffres
function sur(identifiant) {
  if (!/^[0-9A-Za-z]+$/.test(identifiant)) throw new Error("Identifiant invalide");
  return identifiant;
}

export async function nombreVentes(conn) {
  return (await requete(conn, "SELECT COUNT(*)::INTEGER AS n FROM ventes"))[0].n;
}

// Prix médian au m² et nombre de ventes de chaque section : [{id_section, mediane, n}]
export function statistiquesParSection(conn) {
  return requete(conn, `
    SELECT left(id_parcelle, 10) AS id_section, median(prix_m2) AS mediane, COUNT(*)::INTEGER AS n
    FROM ventes GROUP BY id_section`);
}

// Prix médian au m² des parcelles d'une section : [{id_parcelle, mediane, n}]
// (les 10 premiers caractères d'un id de parcelle forment l'id de sa section)
export function medianeParParcelleDeSection(conn, idSection) {
  return requete(conn, `
    SELECT id_parcelle, median(prix_m2) AS mediane, COUNT(*)::INTEGER AS n
    FROM ventes WHERE id_parcelle LIKE '${sur(idSection)}%' GROUP BY id_parcelle`);
}

// Ventes d'une parcelle, de la plus récente à la plus ancienne
export function ventesDeParcelle(conn, idParcelle) {
  return requete(conn, `
    SELECT date_mutation, type_local, surface, pieces, valeur, prix_m2, adresse, nom_commune
    FROM ventes WHERE id_parcelle = '${sur(idParcelle)}' ORDER BY date_mutation DESC`);
}

// Prix au m² des ventes dont l'id de parcelle commence par `prefixe` :
// 5 caractères pour une commune, 10 pour une section (pour les histogrammes)
export async function prixParPrefixe(conn, prefixe) {
  const lignes = await requete(conn, `
    SELECT prix_m2 FROM ventes WHERE id_parcelle LIKE '${sur(prefixe)}%'`);
  return lignes.map((l) => l.prix_m2);
}
