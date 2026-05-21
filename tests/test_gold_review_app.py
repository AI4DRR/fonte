from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "eval" / "gold_review_app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location("gold_review_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GoldReviewAppCsvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = load_app_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "gold.csv"

    def test_roundtrip_preserves_csv_content(self) -> None:
        rows = [
            {
                "asset_key": "abc",
                "sample_bucket": "disagreement",
                "hazard_family": "flood",
                "title": "A title, with comma",
                "content_url": "https://example.org/doc",
                "language": "English",
                "runs_hazard_summary": "run1=Flood; run2=Storm",
                "is_real_event": "T",
                "true_hazard": "Flood",
                "true_country": "Countryland",
                "true_locations": "Town A; Town B",
                "true_date_start": "2024-01-01",
                "true_date_end": "2024-01-03",
                "notes": "first line\nsecond line",
            }
        ]

        self.app.write_gold_csv_atomic(self.path, self.app.REQUIRED_COLUMNS, rows)
        fieldnames, loaded = self.app.read_gold_csv(self.path)

        self.assertEqual(list(self.app.REQUIRED_COLUMNS), fieldnames)
        self.assertEqual(rows, loaded)

    def test_create_backup_is_unique(self) -> None:
        rows = [{c: "" for c in self.app.REQUIRED_COLUMNS}]
        self.app.write_gold_csv_atomic(self.path, self.app.REQUIRED_COLUMNS, rows)

        first = self.app.create_backup(self.path, timestamp="20260101-000000")
        second = self.app.create_backup(self.path, timestamp="20260101-000000")

        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        self.assertNotEqual(first, second)

    def test_missing_required_columns_raise(self) -> None:
        self.app.write_gold_csv_atomic(self.path, ["asset_key"], [{"asset_key": "abc"}])

        with self.assertRaises(self.app.GoldCsvError):
            self.app.read_gold_csv(self.path)

    def test_resume_and_date_helpers(self) -> None:
        rows = [
            {"is_real_event": "T"},
            {"is_real_event": ""},
            {"is_real_event": "F"},
        ]

        self.assertEqual(1, self.app.first_blank_index(rows))
        self.assertEqual(1, self.app.remaining_blank_count(rows))
        self.assertTrue(self.app.is_valid_date_or_blank(""))
        self.assertTrue(self.app.is_valid_date_or_blank("2024-02-29"))
        self.assertFalse(self.app.is_valid_date_or_blank("2024-02-30"))


if __name__ == "__main__":
    unittest.main()
