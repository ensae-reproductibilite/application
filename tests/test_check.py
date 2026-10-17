import unittest
import pandas as pd

from src.validation.check import check_data_leakage


class TestCheckDataLeakage(unittest.TestCase):

    def test_logs_error_when_train_has_unseen_values(self):
        # Train contient une modalité "X" absente du test => data leakage
        train = pd.DataFrame({"Sex": ["male", "female", "X", None]})
        test = pd.DataFrame({"Sex": ["male", "female", None]})

        # assertLogs capture les logs du logger root (logging.*)
        with self.assertLogs(level="ERROR") as cm:
            check_data_leakage(train, test, variable="Sex")

        # cm.output est une liste de strings du style:
        # ["ERROR:root:Problème de data leakage pour la variable Sex"]
        self.assertTrue(
            any("Problème de data leakage pour la variable Sex" in msg for msg in cm.output)
        )

    def test_logs_info_when_values_are_covered(self):
        # Toutes les modalités de train sont couvertes dans test (hors NaN) => OK
        train = pd.DataFrame({"Embarked": ["S", "C", None, "Q"]})
        test = pd.DataFrame({"Embarked": ["S", "C", "Q", None]})

        with self.assertLogs(level="INFO") as cm:
            check_data_leakage(train, test, variable="Embarked")

        self.assertTrue(
            any(
                "Pas de problème de data leakage pour la variable Embarked" in msg
                for msg in cm.output
            )
        )


if __name__ == "__main__":
    unittest.main()
