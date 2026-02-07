"""
Fonctions de validation de données
"""

import pandas as pd
import duckdb


def check_name_formatting(
    connection: duckdb.DuckDBPyConnection,
    df: pd.DataFrame
):

    query = (
        "SELECT COUNT(*) AS n_bad "
        "FROM df "
        "WHERE list_count(string_split(Name, ',')) <> 2"
    )

    bad = connection.sql(query).fetchone()[0]

    if bad == 0:
        print("Test 'Name' OK se découpe toujours en 2 parties avec ','")
    else:
        print(
            f"Problème dans la colonne Name: {bad} ne se décomposent pas en 2 parties."
        )


def check_missing_values(
    connection: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    variable: str = "Survived",
):

    query = f"SELECT COUNT(*) AS n_missing FROM df WHERE {variable} IS NULL"

    n_missing = connection.sql(query).fetchone()[0]

    message_ok = f"Pas de valeur manquante pour la variable {variable}"
    message_warn = f"{n_missing} valeurs manquantes pour la variable {variable}"
    message = message_ok if n_missing == 0 else message_warn
    print(message)


def check_data_leakage(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    variable: str,
):

    if set(train_dataset[variable].dropna().unique()) - set(
        test_dataset[variable].dropna().unique()
    ):
        message = f"Problème de data leakage pour la variable {variable}"
    else:
        message = f"Pas de problème de data leakage pour la variable {variable}"

    print(message)
