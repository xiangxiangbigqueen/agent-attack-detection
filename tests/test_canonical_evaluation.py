import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.canonical_evaluation import DATA, evaluate


class CanonicalEvaluationTests(unittest.TestCase):
    def test_evaluation_is_deterministic_and_session_isolated(self):
        first = evaluate(DATA)
        second = evaluate(DATA)

        self.assertEqual(first["dataset"]["sha256"], second["dataset"]["sha256"])
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["counts"]["benign_train"], 15)
        self.assertEqual(first["counts"]["benign_test"], 10)
        self.assertEqual(first["counts"]["attack_test"], 12)
        self.assertEqual(first["protocol"]["cross_session_ready"], False)
        self.assertIn("do not claim cross-session", first["protocol"]["warning"])


if __name__ == "__main__":
    unittest.main()
