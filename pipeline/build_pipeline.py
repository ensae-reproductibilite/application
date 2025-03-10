from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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
