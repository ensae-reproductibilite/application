"""
Réécrit le fichier parquet DVF dans une organisation adaptée au navigateur.

À lancer une seule fois (le résultat est réutilisable par tout le monde) :

    uv run partitionner_dvf.py s3://<bucket>/diffusion/dvf
    uv run partitionner_dvf.py dvf_part          # ou vers un dossier local

Attention : sur le SSP Cloud, les fichiers du dossier `diffusion/` sont publics.

Pourquoi ? Le fichier d'origine contient très peu de gros groupes de lignes
(row groups) et aucune statistique utile sur le département : pour lire les
ventes de Paris, un navigateur doit tout télécharger. Après réécriture, un
fichier par département, des lignes triées par commune puis parcelle et de petits
groupes de lignes : DuckDB-WASM ne télécharge que ce dont il a besoin.
"""

import argparse
import logging
import os

import duckdb

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

URL_RAW = "https://minio.lab.sspcloud.fr/lgaliana/ensae-reproductibilite/data/raw"
URL_DVF = f"{URL_RAW}/dvf.parquet"

parser = argparse.ArgumentParser(
    description="Partitionne le parquet DVF par département"
)
parser.add_argument(
    "destination", help="dossier local ou chemin s3://<bucket>/diffusion/dvf"
)
destination = parser.parse_args().destination

con = duckdb.connect(database=":memory:")

if destination.startswith("s3://"):
    # Identifiants fournis par les variables d'environnement d'Onyxia
    logger.info("Création du secret S3 à partir des variables d'environnement")
    con.sql(f"""
        CREATE SECRET (
            TYPE s3,
            KEY_ID '{os.environ["AWS_ACCESS_KEY_ID"]}',
            SECRET '{os.environ["AWS_SECRET_ACCESS_KEY"]}',
            SESSION_TOKEN '{os.environ.get("AWS_SESSION_TOKEN", "")}',
            ENDPOINT '{os.environ["AWS_S3_ENDPOINT"]}',
            URL_STYLE 'path',
            REGION '{os.environ.get("AWS_DEFAULT_REGION", "us-east-1")}'
        )
    """)

logger.info(f"Réécriture de {URL_DVF} vers {destination}")
# Le tri permet aux statistiques (min/max) de chaque groupe de lignes d'écarter
# les groupes sans rapport avec la commune ou la parcelle demandée.
# On garde tous les départements : seuls ceux demandés par le site sont lus.
con.sql(f"""
    COPY (
        SELECT * EXCLUDE (geometry, __index_level_0__)
        FROM read_parquet('{URL_DVF}')
        ORDER BY code_departement, code_commune, id_parcelle
    ) TO '{destination}' (
        FORMAT parquet,
        COMPRESSION zstd,
        PARTITION_BY (code_departement),
        ROW_GROUP_SIZE 20000,
        FILENAME_PATTERN 'dvf_{{i}}',
        OVERWRITE_OR_IGNORE
    )
""")
logger.info("Terminé")
