"""
Data retrieval functions
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests


def download_sources(url_fichier: str, cible: Path) -> None:
    """Download url_fichier and write it to cible"""
    r = requests.get(url_fichier, timeout=120)
    r.raise_for_status()
    with open(cible, "wb") as f:
        f.write(r.content)


def get_melodi_indicator(
    code: str,
    value_name: str,
    jeton_api: str,
    dataset: str = "DS_FILOSOFI_CC",
    dimension: str = "FILOSOFI_MEASURE",
    time_period: int = 2023,
    geo_level: tuple[str, ...] = ("COM", "ARM"),
) -> pd.DataFrame:
    """Fetch a municipality-level indicator from the Insee Melodi API

    DVF codes Paris by arrondissement (75101 to 75120), not as one municipality
    (75056): without the ARM level, Paris sales would get no municipality data.
    """

    url = f"https://api.insee.fr/melodi/data/{dataset}"

    params = {
        "GEO": list(geo_level),
        dimension: code,
        "TIME_PERIOD": time_period,
        "maxResult": 40000,
    }

    headers = {"Authorization": f"Bearer {jeton_api}"}

    response = requests.get(url, params=params, headers=headers, timeout=60)
    response.raise_for_status()
    observations = response.json()["observations"]

    return pd.DataFrame(
        {
            "CODGEO": obs["dimensions"]["GEO"].split("-")[-1],
            value_name: obs["measures"]["OBS_VALUE_NIVEAU"].get("value"),
        }
        for obs in observations
    )


def download_missing_sources(sources: dict[str, str], data_location: Path) -> None:
    """Download each source file that is not already in data_location"""
    data_location.mkdir(exist_ok=True)

    for nom_fichier, url_source in sources.items():
        chemin = data_location / nom_fichier
        if chemin.exists():
            taille = chemin.stat().st_size / 1e6
            print(f"{nom_fichier:42s} déjà présent ({taille:.0f} Mo)")
        else:
            print(f"Téléchargement de {nom_fichier} ...")
            download_sources(url_source, chemin)
            print(f"  -> {chemin.stat().st_size / 1e6:.0f} Mo")


def add_filosofi_carreaux(
    dvf: gpd.GeoDataFrame, chemin_filosofi: Path
) -> gpd.GeoDataFrame:
    """Attach to each transaction the Filosofi 200m cell it falls into"""
    filo_cols = [
        "idcar_200m",
        "ind",
        "men",
        "men_pauv",
        "men_prop",
        "ind_snv",
        "log_soc",
        "log_av45",
        "men_surf",
        "geometry",
    ]
    filo = gpd.read_parquet(chemin_filosofi, columns=filo_cols)
    filo = filo.rename(
        columns={
            "men": "nbre_menages",
            "men_pauv": "nbre_menages_pauvres",
            "men_prop": "nbre_menages_prop",
            "ind_snv": "somme_nv_vie",
            "log_soc": "nbre_logements_sociaux",
            "log_av45": "nbre_logements_avant_1945",
            "men_surf": "surface_totale_logements",
        }
    )

    print(
        f"Reprojection de {dvf.shape[0]} transactions immobilières "
        f"du CRS {dvf.crs.to_epsg()} à {filo.crs.to_epsg()}"
    )

    dvf = dvf.to_crs(filo.crs)
    return dvf.sjoin(filo, how="left", predicate="within")


def add_filosofi_communes(dvf: gpd.GeoDataFrame, jeton_api: str) -> gpd.GeoDataFrame:
    """Attach the median income and poverty rate of each municipality"""
    indicateurs = [
        get_melodi_indicator(
            code="MED_SL", value_name="NIVVIE_MEDIAN", jeton_api=jeton_api
        ),
        get_melodi_indicator(
            code="PR_MD60", value_name="TAUX_PAUVRETE", jeton_api=jeton_api
        ),
    ]

    for indicateur in indicateurs:
        dvf = dvf.merge(
            indicateur, left_on="code_commune", right_on="CODGEO", how="left"
        ).drop(columns="CODGEO")

    return dvf
