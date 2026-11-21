"""
Model evaluation functions
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import root_mean_squared_error
from skore import EstimatorReport

logger = logging.getLogger(__name__)


def rmse_by_commune(
    pipe: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cog: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the test RMSE within each municipality, sorted in ascending order"""
    residuals = X_test[["code_commune"]].copy()
    residuals["y_test"] = y_test
    residuals["pred"] = pipe.predict(X_test)

    rmse_commune = (
        residuals.groupby("code_commune")
        .apply(
            lambda d: root_mean_squared_error(d["y_test"], d["pred"]),
            include_groups=False,
        )
        .rename("rmse")
        .reset_index()
    )

    return rmse_commune.merge(cog, on="code_commune").sort_values("rmse")


def evaluate_with_skore(
    pipe: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    chemin_graphique: str = "prediction_error.png",
) -> EstimatorReport:
    """Print skore metrics and variable importance, save the prediction error plot"""
    # Unfitted copy: skore fits the model itself and can therefore report
    # the training time.
    report = EstimatorReport(
        clone(pipe),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    logger.info(report)

    metrics = report.metrics.summarize(data_source="both").frame()
    logger.info(f"Métriques de performance:\n{metrics}")

    report.metrics.prediction_error().plot(kind="actual_vs_predicted")
    plt.savefig(chemin_graphique, dpi=150, bbox_inches="tight")
    plt.close()

    importance = report.inspection.permutation_importance(data_source="test").frame()
    logger.info(f"Importance des variables:\n{importance}")

    return report
