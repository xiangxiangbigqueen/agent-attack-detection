"""Completeness checks for the frozen paper-evaluation protocol."""

from collections import Counter
import unittest

from experiments.canonical_protocol import ADAPTATION_LEVELS, all_episodes, attack_episodes, benign_episodes


class CanonicalProtocolTest(unittest.TestCase):
    def test_benign_splits_and_attack_matrix_are_complete_and_unique(self):
        benign = benign_episodes()
        attacks = attack_episodes()
        self.assertEqual(Counter(item.split for item in benign), {"train": 100, "validation": 50, "test": 100})
        self.assertEqual(len(attacks), 6 * 10 * len(ADAPTATION_LEVELS))
        self.assertEqual(Counter(item.adaptation_level for item in attacks), {level: 60 for level in ADAPTATION_LEVELS})
        self.assertEqual(len({item.group_id for item in all_episodes()}), len(all_episodes()))
        self.assertTrue(all(item.sessions for item in all_episodes()))


if __name__ == "__main__":
    unittest.main()
