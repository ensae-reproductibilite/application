import duckdb


OBSERVED_DATA = (
    "https://minio.lab.sspcloud.fr/projet-formation/"
    "bonnes-pratiques/data/RPindividus/REGION=84/part-0.parquet"
)
TITANIC_HISTORICAL_DATA = (
    "https://minio.lab.sspcloud.fr/"
    "lgaliana/ensae-reproductibilite/data/raw/data.parquet"
)

def pick_observed_data(
    con: duckdb.DuckDBPyConnection,
    path: str,
    n: int = 5
):

    query = f"""
    CREATE TEMP TABLE titanic AS
    SELECT
        AGED AS age,
        'C' AS embarked,
        CS1 AS pcs,
        SEXE AS sex
    FROM '{path}'
    LIMIT 1000
    """
    # Assume they all embarked from Cherbourg ('C')

    con.sql(query)

    sample_query = f"""
    SELECT age, embarked, pcs, sex
    FROM titanic
    ORDER BY random()
    LIMIT {n}
    """

    passengers = (
        con.execute(sample_query)
        .fetchdf()
        .to_dict(orient="records")
    )

    return passengers


def compute_observed_distribution_fare(
    con: duckdb.DuckDBPyConnection,
    titanic_historical_data: str
):

    stat_agg_query = f"""
    SELECT
        min(fare) AS fare_min,
        quantile(fare, 0.25) AS fare_q1,
        quantile(fare, 0.50) AS fare_q2,
        quantile(fare, 0.75) AS fare_q3,
        max(fare) AS fare_max
    FROM read_parquet('{titanic_historical_data}');
    """

    fare_stats = (
        con.sql(stat_agg_query)
        .to_df().to_dict()
    )

    return fare_stats