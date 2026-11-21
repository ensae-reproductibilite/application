"""
Data validation functions
"""

import duckdb
import pandas as pd


def check_code_commune_formatting(
    connection: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
):
    """Check that code_commune has 5 characters and starts with the department"""

    query = (
        "SELECT COUNT(*) AS n_bad "
        "FROM df "
        "WHERE length(code_commune) <> 5 "
        "OR NOT starts_with(code_commune, code_departement)"
    )

    bad = connection.sql(query).fetchall()[0][0]

    if bad == 0:
        print(
            "Test OK: code_commune always has 5 characters starting with the department"
        )
    else:
        print(f"Issue in code_commune: {bad} malformed codes")


def check_missing_values(
    connection: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    variable: str = "valeur_fonciere",
):
    """Count the missing values of a variable"""

    query = f"SELECT COUNT(*) AS n_missing FROM df WHERE {variable} IS NULL"

    n_missing = connection.sql(query).fetchall()[0][0]

    message_ok = f"No missing values for variable {variable}"
    message_warn = f"{n_missing} missing values for variable {variable}"
    message = message_ok if n_missing == 0 else message_warn
    print(message)


def check_unseen_categories(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    variable: str,
):
    """Check that every test category also appears in the training set

    Unseen categories are encoded as unknown (-1): the model has learned nothing
    about them, so their predictions are unreliable.
    """

    if set(test_dataset[variable].dropna().unique()) - set(
        train_dataset[variable].dropna().unique()
    ):
        message = f"Categories of {variable} missing from the training set"
    else:
        message = f"All test categories of {variable} are known to the model"

    print(message)
