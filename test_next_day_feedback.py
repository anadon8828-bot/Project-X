import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from next_day_feedback import DailyFeedback, HISTORY_NAME, RESULTS_NAME, _next_session, _read_frame, _write_frame, live_feedback_status, live_rule_metrics


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

    def test_live_feedback_status_uses_settled_rows_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_frame(root / HISTORY_NAME, pd.DataFrame([
                {"prediction_id": "a", "prediction_date": "2026-10-01", "code": "1"},
                {"prediction_id": "b", "prediction_date": "2026-10-01", "code": "2"},
            ]))
            _write_frame(root / RESULTS_NAME, pd.DataFrame([
                {"prediction_id": "a", "settlement_status": "SETTLED", "direction_correct": 1, "actual_return_pct": 2.0, "raw_expected_pct": 1.0, "raw_probability": .7, "actual_up": 1},
                {"prediction_id": "b", "settlement_status": "MISSING_TARGET_BAR", "direction_correct": np.nan, "actual_return_pct": np.nan, "raw_expected_pct": 1.0, "raw_probability": .7, "actual_up": np.nan},
            ]))
            status = live_feedback_status(root)
            self.assertEqual(status["latest_predictions"], 2)
            self.assertEqual(status["settled"], 1)
            self.assertEqual(status["missing_target_bars"], 1)
            self.assertEqual(status["direction_accuracy_pct"], 100.0)

    def test_live_rule_uses_top_three_and_transaction_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for index in range(4):
                rows.append({
                    "prediction_id": str(index), "prediction_date": "2026-10-01", "code": str(index),
                    "settlement_status": "SETTLED", "raw_probability": .60,
                    "raw_expected_pct": float(4 - index), "actual_return_pct": 1.0,
                })
            _write_frame(root / RESULTS_NAME, pd.DataFrame(rows))
            metrics = live_rule_metrics(root)
            self.assertEqual(metrics["trades"], 3)
            self.assertAlmostEqual(metrics["avg_net_return_pct"], .85)
            self.assertEqual(metrics["status"], "COLLECTING")


if __name__ == "__main__":
    unittest.main()
