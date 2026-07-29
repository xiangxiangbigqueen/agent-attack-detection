import unittest

from experiments.confirmation_protocol import PROTOCOL_VERSION, all_episodes, attack_episodes, benign_episodes, long_benign_task


class ConfirmationProtocolTests(unittest.TestCase):
    def test_protocol_is_complete_and_distinct(self):
        episodes = all_episodes()
        self.assertEqual(PROTOCOL_VERSION, "2026-07-30.confirmation-v1")
        self.assertEqual(len(benign_episodes()), 250)
        self.assertEqual(len(attack_episodes()), 240)
        self.assertEqual(len(episodes), 490)
        self.assertEqual(len({episode.group_id for episode in episodes}), 490)
        self.assertEqual(sum(len(episode.sessions) for episode in episodes), 570)

    def test_confirmation_operands_are_held_out_from_development_protocol(self):
        text = " ".join(task for episode in attack_episodes() for task in episode.sessions)
        self.assertIn("@offsite.example", text)
        self.assertNotIn("@external.example", text)
        self.assertTrue(long_benign_task(1, 5).startswith("Please complete this ordinary account review"))


if __name__ == "__main__":
    unittest.main()
