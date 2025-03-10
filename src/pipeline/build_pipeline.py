from loguru import logger

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import pandas as pd

@logger.catch
def split_train_test(data, test_size, train_path="train.csv", test_path="test.csv"):
    """
    Split the data into training and testing sets based on the specified test size.
    Optionally, save the split datasets to CSV files.

    Args:
        data (pandas.DataFrame): The input data to split.
        test_size (float): The proportion of the dataset to include in the test split.
        train_path (str, optional): The file path to save the training dataset.
            Defaults to "train.csv".
        test_path (str, optional): The file path to save the testing dataset.
            Defaults to "test.csv".

    Returns:
        tuple: A tuple containing the training and testing datasets.
    """
    y = data["Survived"]
    X = data.drop("Survived", axis="columns")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    if train_path:
        pd.concat([X_train, y_train], axis = 1).to_csv(train_path)
    if test_path:
        pd.concat([X_test, y_test], axis = 1).to_csv(test_path)

    return X_train, X_test, y_train, y_test


def create_pipeline(
    n_trees,
    numeric_features=["Age", "Fare"],
    categorical_features=["Embarked", "Sex"],
    max_depth=None,
    max_features="sqrt",
):
    """
    Create a pipeline for preprocessing and model definition.

    Args:
        n_trees (int): The number of trees in the random forest.
        numeric_features (list, optional): The numeric features to be included in the pipeline.
            Defaults to ["Age", "Fare"].
        categorical_features (list, optional): The categorical features to be included
            in the pipeline.
            Defaults to ["Embarked", "Sex"].
        max_depth (int, optional): The maximum depth of the random forest. Defaults to None.
        max_features (str, optional): The maximum number of features to consider
            when looking for the best split.
            Defaults to "sqrt".

    Returns:
        sklearn.pipeline.Pipeline: The pipeline object.
    """
    # Variables numériques
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # Variables catégorielles
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
        ]
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("Preprocessing numerical", numeric_transformer, numeric_features),
            (
                "Preprocessing categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    # Pipeline
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_trees, max_depth=max_depth, max_features=max_features
                ),
            ),
        ]
    )

    return pipe
