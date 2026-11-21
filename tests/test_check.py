import unittest

import pandas as pd

from src.data.check import check_unseen_categories


class TestCheckUnseenCategories(unittest.TestCase):
    def test_warns_when_test_has_unseen_category(self):
        # 75101 only appears in the test set: the model has never seen it
        train = pd.DataFrame({"code_commune": ["92012", "93066"]})
        test = pd.DataFrame({"code_commune": ["92012", "75101"]})

        with self.assertLogs("src.data.check", level="WARNING") as cm:
            check_unseen_categories(train, test, variable="code_commune")

        self.assertIn(
            "Categories of code_commune missing from the training set", cm.output[0]
        )

    def test_info_when_test_categories_are_known(self):
        # Categories only present in train are not an issue, missing values are ignored
        train = pd.DataFrame({"type_local": ["Appartement", "Maison", None]})
        test = pd.DataFrame({"type_local": ["Appartement", None]})

        with self.assertLogs("src.data.check", level="INFO") as cm:
            check_unseen_categories(train, test, variable="type_local")

        self.assertEqual(len(cm.output), 1)
        self.assertIn(
            "All test categories of type_local are known to the model", cm.output[0]
        )


if __name__ == "__main__":
    unittest.main()
