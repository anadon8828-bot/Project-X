"""Daily all-TSE forecast ledger and conservative live calibration.

The ledger is point-in-time: a forecast row is immutable and an outcome is
written only after a later trading session exists.  Calibration is promoted
only when a chronological validation tail improves over the raw model.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from persistent_store import read_bytes, write_bytes
FEATURES = [
    "MA25", "MA75", "RSI", "MACD", "Volume", "Return_1D", "Return_5D",
    "Return_20D", "MA25_Distance", "MA75_Distance", "Volume_Change",
    "Volatility_20D", "MACD_Change", "RSI_Change",
]


HISTORY_NAME = "all_tse_next_day_predictions.csv.gz"
RESULTS_NAME = "all_tse_next_day_results.csv.gz"
CALIBRATION_NAME = "next_day_live_calibration.json"
MIN_SETTLED = 500


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[(frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)].sort_values("date").copy()
    close = data["close"]
    data["MA25"] = close.rolling(25).mean()
    data["MA75"] = close.rolling(75).mean()
    data["Volume"] = data["volume"]
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    data["RSI"] = 100 - 100 / (1 + gains / losses)
    data["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    data["Return_1D"] = close.pct_change() * 100
    data["Return_5D"] = close.pct_change(5) * 100
    data["Return_20D"] = close.pct_change(20) * 100
    data["MA25_Distance"] = (close / data["MA25"] - 1) * 100
    data["MA75_Distance"] = (close / data["MA75"] - 1) * 100
    data["Volume_Change"] = data["volume"].pct_change() * 100
    data["Volatility_20D"] = data["Return_1D"].rolling(20).std()
    data["MACD_Change"] = data["MACD"].diff()
    data["RSI_Change"] = data["RSI"].diff()
    return data.replace([np.inf, -np.inf], np.nan)


def _read_frame(path: Path) -> pd.DataFrame:
    payload = read_bytes(path)
    if not payload:
        return pd.DataFrame()
    return pd.read_csv(BytesIO(payload), compression="gzip", dtype={"code": str})


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    buffer = BytesIO()
    frame.to_csv(buffer, index=False, compression="gzip")
    write_bytes(path, buffer.getvalue())


def _model_hash(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fit_calibration(results: pd.DataFrame) -> dict:
    """Use a chronological holdout; never promote an in-sample improvement."""
    base = {"status": "COLLECTING", "settled": int(len(results)), "min_settled": MIN_SETTLED}
    if len(results) < MIN_SETTLED:
        return base
    ordered = results.sort_values(["actual_date", "code"]).tail(10_000).copy()
    split = max(int(len(ordered) * .7), MIN_SETTLED // 2)
    train, test = ordered.iloc[:split], ordered.iloc[split:]
    if len(test) < 100:
        return base

    # Reliability correction: affine probability calibration, clipped to
    # avoid pretending that a small live sample supports extreme confidence.
    x = train["raw_probability"].astype(float).to_numpy()
    y = train["actual_up"].astype(float).to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    raw = test["raw_probability"].astype(float).to_numpy()
    calibrated = np.clip(intercept + slope * raw, .05, .95)
    truth = test["actual_up"].astype(float).to_numpy()
    raw_brier = float(np.mean((raw - truth) ** 2))
    calibrated_brier = float(np.mean((calibrated - truth) ** 2))

    bias = float((train["actual_return_pct"] - train["raw_expected_pct"]).median())
    raw_mae = float(np.mean(np.abs(test["actual_return_pct"] - test["raw_expected_pct"])))
    calibrated_mae = float(np.mean(np.abs(test["actual_return_pct"] - (test["raw_expected_pct"] + bias))))
    approved_probability = calibrated_brier + .001 < raw_brier
    approved_return = calibrated_mae + .01 < raw_mae
    return {
        "status": "APPROVED" if approved_probability or approved_return else "REJECTED",
        "settled": int(len(ordered)),
        "validation_rows": int(len(test)),
        "probability_approved": bool(approved_probability),
        "probability_slope": float(slope),
        "probability_intercept": float(intercept),
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "return_approved": bool(approved_return),
        "return_bias_pct": bias,
        "raw_mae_pct": raw_mae,
        "calibrated_mae_pct": calibrated_mae,
        "evaluated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def load_calibration(root: Path) -> dict:
    payload = read_bytes(root / CALIBRATION_NAME)
    if not payload:
        return {"status": "COLLECTING", "settled": 0}
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"status": "INVALID", "settled": 0}


def apply_calibration(root: Path, probability: float, expected_pct: float) -> tuple[float, float, dict]:
    calibration = load_calibration(root)
    if calibration.get("status") == "APPROVED" and calibration.get("probability_approved"):
        probability = float(np.clip(
            calibration["probability_intercept"] + calibration["probability_slope"] * probability,
            .05, .95,
        ))
    if calibration.get("status") == "APPROVED" and calibration.get("return_approved"):
        expected_pct += float(calibration["return_bias_pct"])
    return probability, expected_pct, calibration


def latest_saved_forecast(root: Path, code: str, prediction_date: str) -> dict | None:
    """Return the immutable cloud-generated forecast for the requested close."""
    history = _read_frame(root / HISTORY_NAME)
    if history.empty:
        return None
    match = history[
        (history["code"].astype(str).str.upper() == str(code).upper())
        & (history["prediction_date"].astype(str) == str(prediction_date))
    ]
    if match.empty:
        return None
    row = match.iloc[-1]
    return {
        "probability": float(row["probability"]),
        "expected_pct": float(row["expected_pct"]),
        "raw_probability": float(row["raw_probability"]),
        "raw_expected_pct": float(row["raw_expected_pct"]),
        "model_hash": str(row["model_hash"]),
    }


class DailyFeedback:
    def __init__(self, root: Path):
        self.root = root
        self.history_path = root / HISTORY_NAME
        self.results_path = root / RESULTS_NAME
        self.calibration_path = root / CALIBRATION_NAME
        self.history = _read_frame(self.history_path)
        self.results = _read_frame(self.results_path)
        self.features: list[dict] = []
        settled_ids = set(self.results.get("prediction_id", pd.Series(dtype=str)).astype(str))
        if self.history.empty:
            self.pending_by_code = {}
        else:
            pending = self.history[~self.history["prediction_id"].astype(str).isin(settled_ids)]
            self.pending_by_code = {str(code): rows.to_dict("records") for code, rows in pending.groupby("code")}
        self.new_results: list[dict] = []

    def observe(self, code: str, sector: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        bars = frame.copy()
        bars.index = pd.to_datetime(bars.index).tz_localize(None)
        for old in self.pending_by_code.get(str(code), []):
            future = bars[bars.index.normalize() > pd.Timestamp(old["prediction_date"]).normalize()]
            if future.empty:
                continue
            actual_close = float(future["Close"].iloc[0])
            actual_return = (actual_close / float(old["close"]) - 1) * 100
            self.new_results.append({
                **old,
                "actual_date": future.index[0].strftime("%Y-%m-%d"),
                "actual_close": actual_close,
                "actual_return_pct": actual_return,
                "actual_up": int(actual_return > 0),
                "direction_correct": int((actual_return > 0) == (float(old["raw_probability"]) >= .5)),
            })

        raw = frame.reset_index().rename(columns={
            frame.index.name or "index": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        featured = add_features(raw)
        if featured.empty:
            return
        latest = featured.iloc[-1]
        if latest[FEATURES].isna().any():
            return
        self.features.append({"code": str(code), "sector": str(sector), **latest[FEATURES + ["date", "close"]].to_dict()})

    def finalize(self, direction_model: Path, return_model: Path) -> dict:
        if self.new_results:
            combined = pd.concat([self.results, pd.DataFrame(self.new_results)], ignore_index=True)
            combined = combined.drop_duplicates("prediction_id", keep="last")
        else:
            combined = self.results
        if not combined.empty:
            _write_frame(self.results_path, combined)
        calibration = _fit_calibration(combined)
        write_bytes(self.calibration_path, json.dumps(calibration, ensure_ascii=False).encode("utf-8"))

        current = pd.DataFrame(self.features).replace([np.inf, -np.inf], np.nan).dropna()
        if current.empty:
            raise RuntimeError("全東証の翌日予測用特徴量を作成できませんでした。")
        current["market_today"] = current["Return_1D"].median()
        current["sector_today"] = current.groupby("sector")["Return_1D"].transform("median")
        current["relative_sector_today"] = current["Return_1D"] - current["sector_today"]
        columns = FEATURES + ["market_today", "sector_today", "relative_sector_today"]
        classifier, regressor = joblib.load(direction_model), joblib.load(return_model)
        current["raw_probability"] = classifier.predict_proba(current[columns])[:, 1]
        current["raw_expected_pct"] = regressor.predict(current[columns]) * 100
        current["probability"] = current["raw_probability"]
        current["expected_pct"] = current["raw_expected_pct"]
        if calibration.get("status") == "APPROVED" and calibration.get("probability_approved"):
            current["probability"] = np.clip(
                calibration["probability_intercept"] + calibration["probability_slope"] * current["raw_probability"], .05, .95
            )
        if calibration.get("status") == "APPROVED" and calibration.get("return_approved"):
            current["expected_pct"] += calibration["return_bias_pct"]
        day = pd.to_datetime(current["date"]).max().strftime("%Y-%m-%d")
        current = current[pd.to_datetime(current["date"]).dt.strftime("%Y-%m-%d") == day].copy()
        model_hash = _model_hash(direction_model, return_model)
        current["prediction_date"] = day
        current["created_at"] = pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
        current["model_hash"] = model_hash
        current["prediction_id"] = [hashlib.sha256(f"{day}|{code}|{model_hash}".encode()).hexdigest() for code in current["code"]]
        columns_out = ["prediction_id", "created_at", "prediction_date", "code", "sector", "close", "model_hash", "raw_probability", "raw_expected_pct", "probability", "expected_pct"]
        additions = current[columns_out]
        history = pd.concat([self.history, additions], ignore_index=True) if not self.history.empty else additions
        history = history.drop_duplicates("prediction_id", keep="first")
        _write_frame(self.history_path, history)
        return {"predictions": int(len(additions)), "settled_total": int(len(combined)), "newly_settled": int(len(self.new_results)), "calibration": calibration.get("status")}
