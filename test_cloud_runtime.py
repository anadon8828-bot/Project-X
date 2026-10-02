import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import project_x_auth
from daytrade_mode import jp_detail_symbols
from refresh_watchlist import acquire_lock, release_lock


class CloudRuntimeTest(unittest.TestCase):
    def test_price_coverage_rejects_tiny_partial_scan(self):
        from refresh_watchlist import price_coverage

        self.assertLess(price_coverage(3700, 3429), 0.08)
        self.assertGreater(price_coverage(3700, 23), 0.99)

    def test_incomplete_current_day_bar_is_removed(self):
        from refresh_watchlist import completed_daily_bars

        frame = pd.DataFrame(
            {"Close": [100.0, 101.0], "Volume": [1000, 200]},
            index=pd.to_datetime(["2026-10-01", "2026-10-02"]),
        )
        completed = completed_daily_bars(frame, "2026-10-01")
        self.assertEqual(len(completed), 1)
        self.assertEqual(str(completed.index[-1].date()), "2026-10-01")

    def test_portable_lock_excludes_second_process(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.lock"
            descriptor = acquire_lock(path)
            self.assertIsNotNone(descriptor)
            self.assertIsNone(acquire_lock(path))
            release_lock(path, descriptor)
            self.assertFalse(path.exists())

    def test_jp_shortlist_uses_liquid_all_tse_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                [
                    {"コード": "1111", "平均売買代金(百万円)": 10, "株価基準日": "2026-09-28"},
                    {"コード": "2222", "平均売買代金(百万円)": 50, "株価基準日": "2026-09-28"},
                ]
            ).to_csv(root / "watchlist_fresh_candidates.csv", index=False, encoding="utf-8-sig")
            symbols, source = jp_detail_symbols(root, {"1111": "A", "2222": "B"}, "2026-09-28", limit=1)
            self.assertEqual(symbols, ["2222"])
            self.assertEqual(source, "当日全東証スキャン")

    def test_public_research_setting_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "access.json"
            with patch.object(project_x_auth, "ACCESS_FILE", path):
                self.assertTrue(project_x_auth.save_public_research(True))
                self.assertTrue(project_x_auth.public_research_enabled())
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"public_research": True})


if __name__ == "__main__":
    unittest.main()
