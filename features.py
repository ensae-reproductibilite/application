"""
Feature engineering functions
"""

import geopandas as gpd
import numpy as np


def add_filosofi_ratios(dvf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Turn Filosofi cell counts into rates per household"""
    print("Computing Filosofi ratios per household")
    dvf = dvf.copy()
    dvf["tx_pauvrete"] = dvf["nbre_menages_pauvres"] / dvf["nbre_menages"]
    dvf["nv_vie_moyen"] = dvf["somme_nv_vie"] / dvf["nbre_menages"]
    # In Filosofi each household occupies one dwelling, so the number of households
    # is the denominator for dwelling counts.
    dvf["part_proprietaires"] = dvf["nbre_menages_prop"] / dvf["nbre_menages"]
    dvf["part_logements_sociaux"] = dvf["nbre_logements_sociaux"] / dvf["nbre_menages"]
    dvf["part_logements_avant_1945"] = (
        dvf["nbre_logements_avant_1945"] / dvf["nbre_menages"]
    )
    dvf["surface_moyenne_logements"] = (
        dvf["surface_totale_logements"] / dvf["nbre_menages"]
    )
    return dvf


def filter_prix_m2_outliers(
    dvf: gpd.GeoDataFrame, quantile_bas: float = 0.01, quantile_haut: float = 0.99
) -> gpd.GeoDataFrame:
    """Drop the most extreme prices per m², globally then within each municipality

    Extreme prices per m² are mostly multi-lot sales, split ownership or data
    entry errors: keeping them would make the model learn noise. Global bounds are
    driven by Paris and let outliers through in the suburbs, hence the second filter.
    """
    n_avant = dvf.shape[0]
    borne_basse, borne_haute = dvf["prix_m2"].quantile([quantile_bas, quantile_haut])
    dvf = dvf.loc[dvf["prix_m2"].between(borne_basse, borne_haute)]

    prix_m2_par_commune = dvf.groupby("code_commune")["prix_m2"]
    dvf = dvf.loc[
        dvf["prix_m2"].between(
            prix_m2_par_commune.transform("quantile", quantile_bas),
            prix_m2_par_commune.transform("quantile", quantile_haut),
        )
    ]

    print(
        f"Filtering extreme prices per m²: {n_avant - dvf.shape[0]} transactions "
        f"dropped out of {n_avant}"
    )
    return dvf


def add_surface_features(dvf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Describe the property sold, beyond the dwelling itself

    DVF prices include land and outbuildings (extra lots), so we account for them.
    ratio_carrez flags sales whose declared area does not match the property.
    """
    print("Computing surface features (building, land, Carrez ratio)")
    dvf = dvf.copy()
    dvf["log_surface"] = np.log(dvf["surface_reelle_bati"])
    dvf["log_surface_terrain"] = np.log1p(dvf["surface_terrain"].fillna(0))
    dvf["ratio_carrez"] = dvf["lot1_surface_carrez"] / dvf["surface_reelle_bati"]
    return dvf


def add_distance_eiffel(dvf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add the log distance to the Eiffel Tower, a proxy for centrality

    Lambert-93 (EPSG:2154) gives distances in meters.
    """
    print("Computing distance to the Eiffel Tower")
    dvf = dvf.copy()
    tour_eiffel = gpd.GeoSeries.from_xy([2.2945], [48.8584], crs="EPSG:4326").to_crs(
        "EPSG:2154"
    )
    distance_eiffel_km = (
        dvf.geometry.to_crs("EPSG:2154").distance(tour_eiffel.iloc[0]) / 1000
    )
    dvf["log_distance_eiffel"] = np.log1p(distance_eiffel_km)
    return dvf


def build_features(dvf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute the target prix_m2 and the model features"""
    dvf = add_filosofi_ratios(dvf)
    dvf["prix_m2"] = dvf["valeur_fonciere"] / dvf["surface_reelle_bati"]
    dvf = filter_prix_m2_outliers(dvf)
    dvf = add_surface_features(dvf)
    return add_distance_eiffel(dvf)
