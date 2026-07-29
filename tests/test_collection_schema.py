"""Regression tests for the raw-trajectory collector's local guarantees."""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.collect_canonical_api import collect_smoke, failed_group_ids, make_record, write_record


class CollectionSchemaTest(unittest.TestCase):
    def test_smoke_collection_is_atomic_and_resumable_without_live_dependencies(self):
        """A local smoke run must produce one indexed, valid record."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "collector"
            collect_smoke(root)

            records = list((root / "records").glob("*.json"))
            self.assertEqual(len(records), 1)
            self.assertTrue((root / "index.jsonl").is_file())

            record = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertEqual(record["collector_status"], "ok")
            self.assertIsNone(record["error_type"])
            self.assertEqual(record["n_calls"], 1)
            self.assertEqual(record["tools"][0]["name"], "get_balance")

    def test_recovery_selection_only_targets_recorded_api_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            failed = make_record(split="test", label="attack", attack_type="test",
                                 group_id="failed-group", session_index=0, task="x", calls=[],
                                 collector_status="api_error", error_type="APIConnectionError")
            successful = make_record(split="test", label="attack", attack_type="test",
                                     group_id="successful-group", session_index=0, task="x", calls=[])
            self.assertTrue(write_record(root, failed))
            self.assertTrue(write_record(root, successful))
            self.assertEqual(failed_group_ids(root), {"failed-group"})


if __name__ == "__main__":
    unittest.main()
