"""
Data validation functions
"""

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def check_code_commune_formatting(
    connection: duckdb.DuckDBPyConnection,
    table: str = "dvf",
):
    """Check that code_commune has 5 characters and starts with the department"""

    query = (
        "SELECT COUNT(*) AS n_bad "
        f"FROM {table} "
        "WHERE length(code_commune) <> 5 "
        "OR NOT starts_with(code_commune, code_departement)"
    )

    bad = connection.sql(query).fetchall()[0][0]

    if bad == 0:
        logger.info(
            "Test OK: code_commune always has 5 characters starting with the department"
        )
    else:
        logger.warning(f"Issue in code_commune: {bad} malformed codes")


def check_missing_values(
    connection: duckdb.DuckDBPyConnection,
    variable: str = "valeur_fonciere",
    table: str = "dvf",
):
    """Count the missing values of a variable"""

    query = f"SELECT COUNT(*) AS n_missing FROM {table} WHERE {variable} IS NULL"

    n_missing = connection.sql(query).fetchall()[0][0]

    message_ok = f"No missing values for variable {variable}"
    message_warn = f"{n_missing} missing values for variable {variable}"
    if n_missing == 0:
        logger.info(message_ok)
    else:
        logger.warning(message_warn)


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
        logger.warning(f"Categories of {variable} missing from the training set")
    else:
        logger.info(f"All test categories of {variable} are known to the model")
