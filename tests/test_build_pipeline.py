import unittest
import pandas as pd
from src.pipeline.build_pipeline import split_train_test

class TestBuildPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.data = pd.DataFrame({
            'Survived': [1, 0, 1, 0, 1],
            'Age': [22, 38, 26, 35, 35],
            'Sex': ['male', 'female', 'female', 'male', 'female']
        })

    def test_split_train_test(self):
        """Test splitting data into train and test sets."""
        test_size = 0.2
        train_path = "train.csv"
        test_path = "test.csv"

        X_train, X_test, y_train, y_test = split_train_test(self.data, test_size, train_path, test_path)

        # Assert that the train and test sets have the correct shapes
        self.assertEqual(X_train.shape[0], int((1 - test_size) * self.data.shape[0]))
        self.assertEqual(X_test.shape[0], int(test_size * self.data.shape[0]))

        # Assert that the train and test sets have the correct columns
        self.assertCountEqual(X_train.columns, self.data.drop("Survived", axis="columns").columns)
        self.assertCountEqual(X_test.columns, self.data.drop("Survived", axis="columns").columns)

        # Assert that the train and test sets have the correct target variable
        self.assertCountEqual(y_train, self.data["Survived"].iloc[X_train.index])
        self.assertCountEqual(y_test, self.data["Survived"].iloc[X_test.index])


if __name__ == '__main__':
    unittest.main()