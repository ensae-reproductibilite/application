"""
Data retrieval functions
"""

import logging

import duckdb
import geopandas as gpd
import pandas as pd
import requests

logger = logging.getLogger(__name__)


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


def add_filosofi_carreaux(
    dvf: gpd.GeoDataFrame, connection: duckdb.DuckDBPyConnection, url_filosofi: str
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
    # The GeoParquet bbox column (EPSG:3035) lets us read only the cells that
    # overlap the transactions instead of the whole of France
    xmin, ymin, xmax, ymax = dvf.to_crs("EPSG:3035").total_bounds
    filo = connection.sql(f"""
        SELECT {", ".join(filo_cols)}
        FROM read_parquet('{url_filosofi}')
        WHERE bbox.xmin <= {xmax} AND bbox.xmax >= {xmin}
        AND bbox.ymin <= {ymax} AND bbox.ymax >= {ymin}
    """).to_df()
    # DuckDB returns the geometry as WKB bytearrays
    filo = gpd.GeoDataFrame(
        filo,
        geometry=gpd.GeoSeries.from_wkb(filo["geometry"].apply(bytes)),
        crs="EPSG:3035",
    )
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

    logger.info(
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
