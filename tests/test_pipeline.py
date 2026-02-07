import unittest
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from src.pipeline.build_pipeline import create_pipeline


class TestCreatePipeline(unittest.TestCase):
    def test_pipeline_structure_and_params(self):
        pipe = create_pipeline(n_trees=123, max_depth=7, max_features="log2")

        self.assertIsInstance(pipe, Pipeline)
        self.assertIn("preprocessor", pipe.named_steps)
        self.assertIn("classifier", pipe.named_steps)

        preprocessor = pipe.named_steps["preprocessor"]
        self.assertIsInstance(preprocessor, ColumnTransformer)

        clf = pipe.named_steps["classifier"]
        self.assertIsInstance(clf, RandomForestClassifier)
        self.assertEqual(clf.n_estimators, 123)
        self.assertEqual(clf.max_depth, 7)
        self.assertEqual(clf.max_features, "log2")

        # ✅ Fix ici
        transformers = {name: (trans, cols) for name, trans, cols in preprocessor.transformers}

        self.assertIn("Preprocessing numerical", transformers)
        self.assertIn("Preprocessing categorical", transformers)

        num_pipe, num_cols = transformers["Preprocessing numerical"]
        cat_pipe, cat_cols = transformers["Preprocessing categorical"]

        self.assertEqual(list(num_cols), ["Age", "Fare"])
        self.assertEqual(list(cat_cols), ["Embarked", "Sex"])

        self.assertIsInstance(num_pipe, Pipeline)
        self.assertIsInstance(num_pipe.named_steps["imputer"], SimpleImputer)
        self.assertEqual(num_pipe.named_steps["imputer"].strategy, "median")
        self.assertIsInstance(num_pipe.named_steps["scaler"], MinMaxScaler)

        self.assertIsInstance(cat_pipe, Pipeline)
        self.assertIsInstance(cat_pipe.named_steps["imputer"], SimpleImputer)
        self.assertEqual(cat_pipe.named_steps["imputer"].strategy, "most_frequent")
        self.assertIsInstance(cat_pipe.named_steps["onehot"], OneHotEncoder)

    def test_pipeline_fit_predict_runs(self):
        pipe = create_pipeline(n_trees=10)

        # Mini dataset avec valeurs manquantes (pour tester les imputers)
        X = pd.DataFrame(
            {
                "Age": [22, None, 35, 28],
                "Fare": [7.25, 71.83, None, 8.05],
                "Embarked": ["S", "C", None, "S"],
                "Sex": ["male", "female", "female", None],
            }
        )
        y = [0, 1, 1, 0]

        pipe.fit(X, y)
        preds = pipe.predict(X)

        self.assertEqual(len(preds), len(X))
        # RandomForest -> prédictions dans les classes {0,1}
        self.assertTrue(set(preds).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
