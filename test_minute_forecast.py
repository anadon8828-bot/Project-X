import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from minute_forecast import completed_bars, feature_frame, live_correction, settle_history, update_forecast


def sample_bars(count=500, end="2026-10-08 01:00:00+00:00"):
    index = pd.date_range(end=end, periods=count, freq="min")
    trend = np.linspace(100, 102, count)
    noise = np.sin(np.arange(count) / 9) * .05
    close = trend + noise
    return pd.DataFrame({
        "Open": close - .01, "High": close + .04, "Low": close - .04,
        "Close": close, "Volume": 1000 + (np.arange(count) % 20) * 10,
    }, index=index)


class MinuteForecastTests(unittest.TestCase):
    def test_incomplete_bar_is_removed(self):
        frame = sample_bars(3, "2026-10-08 01:02:00+00:00")
        result = completed_bars(frame, pd.Timestamp("2026-10-08 01:02:30+00:00"))
        self.assertEqual(result.index[-1], pd.Timestamp("2026-10-08 01:01:00+00:00"))

    def test_target_is_strictly_next_bar(self):
        frame = sample_bars(50)
        features = feature_frame(frame)
        expected = frame.Close.iloc[-1] / frame.Close.iloc[-2] - 1
        self.assertAlmostEqual(features.target_return.iloc[-2], expected)
        self.assertTrue(pd.isna(features.target_return.iloc[-1]))

    def test_settlement_requires_exact_target_timestamp(self):
        frame = sample_bars(2)
        target = frame.index[-1]
        row = {"ticker": "7203.T", "target_time": target.isoformat(), "base_close": 100.0,
               "predicted_close": 101.0, "predicted_return": .01, "actual_close": None}
        settled = settle_history([row], "7203.T", frame)
        self.assertEqual(settled[0]["actual_close"], float(frame.Close.iloc[-1]))

    def test_correction_waits_for_500_settlements(self):
        rows = [{"ticker": "7203.T", "actual_return": .01, "raw_predicted_return": 0.0} for _ in range(499)]
        correction, status = live_correction(rows, "7203.T")
        self.assertEqual(correction, 0.0)
        self.assertIn("499/500", status)

    def test_forecast_is_recorded(self):
        frame = sample_bars(600, "2026-10-08 01:00:00+00:00")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "minute.json"
            now = pd.Timestamp("2026-10-08 01:01:20+00:00")
            result = update_forecast("7203.T", frame, now=now, path=path)
            self.assertGreater(result["training_count"], 250)
            self.assertTrue(path.exists())

    def test_more_than_twenty_minutes_stale_fails_closed(self):
        frame = sample_bars(600, "2026-10-08 01:00:00+00:00")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "予測を停止"):
                update_forecast("7203.T", frame, now=pd.Timestamp("2026-10-08 01:22:00+00:00"), path=Path(directory) / "minute.json")


if __name__ == "__main__":
    unittest.main()
