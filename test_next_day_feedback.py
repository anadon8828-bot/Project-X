import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from next_day_feedback import DailyFeedback, HISTORY_NAME, _next_session, _read_frame, _write_frame


class NextDayFeedbackTests(unittest.TestCase):
    def _history(self, root: Path):
        _write_frame(root / HISTORY_NAME, pd.DataFrame([{
            "prediction_id": "forecast-1",
            "prediction_date": "2026-10-02",
            "target_date": "2026-10-05",
            "code": "7203",
            "close": 100.0,
            "raw_probability": 0.6,
            "raw_expected_pct": 1.0,
        }]))

    @staticmethod
    def _bars(include_target=True):
        dates = list(pd.bdate_range("2026-06-01", "2026-10-06"))
        if not include_target:
            dates.remove(pd.Timestamp("2026-10-05"))
        close = np.linspace(80, 102, len(dates))
        return pd.DataFrame({
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": np.full(len(dates), 100_000),
        }, index=pd.DatetimeIndex(dates))

    def test_next_session_skips_weekend(self):
        self.assertEqual(_next_session("2026-10-02"), "2026-10-05")

    def test_settlement_uses_exact_target_session(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._history(root)
            feedback = DailyFeedback(root)
            feedback.observe("7203", "輸送用機器", self._bars())
            self.assertEqual(feedback.new_results[0]["settlement_status"], "SETTLED")
            self.assertEqual(feedback.new_results[0]["actual_date"], "2026-10-05")

    def test_missing_target_is_not_replaced_by_later_bar(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._history(root)
            feedback = DailyFeedback(root)
            feedback.observe("7203", "輸送用機器", self._bars(include_target=False))
            result = feedback.new_results[0]
            self.assertEqual(result["settlement_status"], "MISSING_TARGET_BAR")
            self.assertTrue(pd.isna(result["actual_close"]))

    def test_empty_checkpoint_is_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "empty.csv.gz"
            _write_frame(path, pd.DataFrame())
            self.assertTrue(_read_frame(path).empty)


if __name__ == "__main__":
    unittest.main()
