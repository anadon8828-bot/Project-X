"""One-minute-ahead research forecast with immutable live feedback.

The forecast targets the close of the next completed one-minute bar.  It is a
research display, not a real-time quote or an execution signal.  Yahoo bars can
be delayed, so stale inputs fail closed instead of being labelled live.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from persistent_store import read_bytes, write_bytes


HISTORY_PATH = Path(__file__).resolve().parent / "minute_forecast_history.json"
FEATURES = ["ret1", "ret2", "ret5", "range", "body", "vwap_gap", "volume_z", "vol10", "minute_sin", "minute_cos"]


def _flat(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)
    return result


def completed_bars(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove the still-forming one-minute candle."""
    now = now or pd.Timestamp.now(tz="UTC")
    data = _flat(frame).dropna(subset=["Open", "High", "Low", "Close", "Volume"]).sort_index()
    index = pd.DatetimeIndex(data.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    data.index = index
    return data.loc[data.index + pd.Timedelta(minutes=1) <= now].copy()


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["Close"].astype(float)
    volume = data["Volume"].astype(float)
    typical = (data["High"] + data["Low"] + data["Close"]) / 3
    session = pd.Series(data.index.tz_convert("UTC").date, index=data.index)
    pv = typical * volume
    data["VWAP"] = pv.groupby(session).cumsum() / volume.groupby(session).cumsum().replace(0, np.nan)
    data["ret1"] = close.pct_change(1)
    data["ret2"] = close.pct_change(2)
    data["ret5"] = close.pct_change(5)
    data["range"] = (data["High"] - data["Low"]) / close.replace(0, np.nan)
    data["body"] = (data["Close"] - data["Open"]) / data["Open"].replace(0, np.nan)
    data["vwap_gap"] = close / data["VWAP"] - 1
    mean_volume = volume.rolling(20).mean()
    std_volume = volume.rolling(20).std().replace(0, np.nan)
    data["volume_z"] = (volume - mean_volume) / std_volume
    data["vol10"] = data["ret1"].rolling(10).std()
    minute = data.index.hour * 60 + data.index.minute
    data["minute_sin"] = np.sin(2 * np.pi * minute / 1440)
    data["minute_cos"] = np.cos(2 * np.pi * minute / 1440)
    data["target_return"] = close.shift(-1) / close - 1
    return data


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, row: np.ndarray) -> float:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    x = (train_x - mean) / scale
    latest = (row - mean) / scale
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * 3.0
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return float(np.r_[1.0, latest] @ coefficients)


def model_forecast(frame: pd.DataFrame) -> dict:
    features = feature_frame(frame)
    train = features.dropna(subset=FEATURES + ["target_return"]).tail(1800)
    latest = features.dropna(subset=FEATURES).tail(1)
    if len(train) < 250 or latest.empty:
        raise ValueError("1分足の学習データが不足しています（最低250本必要）。")
    split = max(200, int(len(train) * .8))
    fit, test = train.iloc[:split], train.iloc[split:]
    validation_predictions = np.array([
        _ridge_predict(fit[FEATURES].to_numpy(float), fit["target_return"].to_numpy(float), row)
        for row in test[FEATURES].to_numpy(float)
    ]) if len(test) else np.array([])
    expected = _ridge_predict(train[FEATURES].to_numpy(float), train["target_return"].to_numpy(float), latest[FEATURES].iloc[0].to_numpy(float))
    residuals = test["target_return"].to_numpy(float) - validation_predictions if len(test) else train["target_return"].to_numpy(float)
    sigma = max(float(np.std(residuals)), 1e-6)
    mae = max(float(np.mean(np.abs(residuals))), 1e-6)
    probability = .5 * (1 + math.erf(expected / (sigma * math.sqrt(2))))
    direction_accuracy = float(np.mean(np.sign(validation_predictions) == np.sign(test["target_return"].to_numpy(float)))) if len(test) else np.nan
    return {
        "raw_return": expected,
        "probability": probability,
        "mae": mae,
        "validation_count": int(len(test)),
        "validation_direction_accuracy": direction_accuracy,
        "training_count": int(len(train)),
    }


def _read_history(path: Path = HISTORY_PATH) -> list[dict]:
    payload = read_bytes(path)
    if not payload:
        return []
    try:
        value = json.loads(payload.decode("utf-8"))
        return value if isinstance(value, list) else []
    except (ValueError, UnicodeDecodeError):
        return []


def _write_history(rows: list[dict], path: Path = HISTORY_PATH) -> None:
    write_bytes(path, json.dumps(rows[-50_000:], ensure_ascii=False, allow_nan=False).encode("utf-8"))


def settle_history(rows: list[dict], ticker: str, bars: pd.DataFrame) -> list[dict]:
    prices = {timestamp.isoformat(): float(value) for timestamp, value in bars["Close"].items()}
    for row in rows:
        if row.get("ticker") != ticker or row.get("actual_close") is not None:
            continue
        actual = prices.get(row.get("target_time"))
        if actual is None:
            continue
        row["actual_close"] = actual
        row["actual_return"] = actual / float(row["base_close"]) - 1
        row["error"] = actual - float(row["predicted_close"])
        row["direction_correct"] = bool((row["actual_return"] >= 0) == (row["predicted_return"] >= 0))
    return rows


def live_correction(rows: list[dict], ticker: str) -> tuple[float, str]:
    settled = [row for row in rows if row.get("ticker") == ticker and row.get("actual_return") is not None]
    if len(settled) < 500:
        return 0.0, f"補正待ち（答え合わせ {len(settled)}/500件）"
    values = np.array([float(row["actual_return"]) - float(row["raw_predicted_return"]) for row in settled[-1000:]])
    split = int(len(values) * .8)
    bias = float(values[:split].mean())
    validation = settled[-(len(values) - split):]
    raw = np.array([abs(float(row["actual_return"]) - float(row["raw_predicted_return"])) for row in validation])
    adjusted = np.array([abs(float(row["actual_return"]) - (float(row["raw_predicted_return"]) + bias)) for row in validation])
    if adjusted.mean() >= raw.mean():
        return 0.0, "補正不採用（時系列検証で改善なし）"
    return bias, f"補正稼働中（{len(settled)}件で検証）"


def update_forecast(ticker: str, frame: pd.DataFrame, now: pd.Timestamp | None = None, path: Path = HISTORY_PATH) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    bars = completed_bars(frame, now)
    if bars.empty:
        raise ValueError("確定済みの1分足がありません。")
    age = now - bars.index[-1] - pd.Timedelta(minutes=1)
    if age > pd.Timedelta(minutes=20):
        raise ValueError(f"最新の確定1分足が{int(age.total_seconds() // 60)}分前です。市場休場中または配信遅延のため予測を停止しました。")
    model = model_forecast(bars)
    rows = settle_history(_read_history(path), ticker, bars)
    correction, correction_status = live_correction(rows, ticker)
    predicted_return = model["raw_return"] + correction
    base_time = bars.index[-1]
    target_time = base_time + pd.Timedelta(minutes=1)
    base_close = float(bars["Close"].iloc[-1])
    predicted_close = base_close * (1 + predicted_return)
    key = (ticker, base_time.isoformat())
    if not any((row.get("ticker"), row.get("base_time")) == key for row in rows):
        rows.append({
            "ticker": ticker, "created_at": now.isoformat(), "base_time": base_time.isoformat(),
            "target_time": target_time.isoformat(), "base_close": base_close,
            "raw_predicted_return": model["raw_return"], "predicted_return": predicted_return,
            "predicted_close": predicted_close, "actual_close": None,
        })
    _write_history(rows, path)
    settled = [row for row in rows if row.get("ticker") == ticker and row.get("actual_return") is not None]
    live_mae = float(np.mean([abs(float(row["error"])) for row in settled])) if settled else None
    live_accuracy = float(np.mean([bool(row["direction_correct"]) for row in settled])) if settled else None
    return {
        **model, "predicted_return": predicted_return, "base_close": base_close,
        "predicted_close": predicted_close, "low": predicted_close - base_close * model["mae"],
        "high": predicted_close + base_close * model["mae"], "base_time": base_time,
        "target_time": target_time, "correction_status": correction_status,
        "data_delay_minutes": max(0, int(age.total_seconds() // 60)),
        "settled_count": len(settled), "live_mae_yen": live_mae, "live_direction_accuracy": live_accuracy,
    }


def fetch_minute_bars(ticker: str) -> pd.DataFrame:
    return yf.download(ticker, period="7d", interval="1m", auto_adjust=True, prepost=False, progress=False, timeout=15)


def render_minute_forecast(ticker: str, currency: str = "円") -> None:
    """Render an auto-refreshing research card while this page is open."""
    import streamlit as st

    st.subheader("1分後予測")

    @st.fragment(run_every="60s")
    def live_panel() -> None:
        try:
            frame = fetch_minute_bars(ticker)
            result = update_forecast(ticker, frame)
            c1, c2, c3 = st.columns(3)
            symbol = "¥" if currency == "円" else "$"
            c1.metric("次の1分足・予測終値", f"{symbol}{result['predicted_close']:,.2f}", f"{result['predicted_return'] * 100:+.3f}%")
            c2.metric("モデル上の上昇確率", f"{result['probability'] * 100:.1f}%")
            c3.metric("誤差目安（検証MAE）", f"±{symbol}{result['base_close'] * result['mae']:,.2f}")
            market_tz = "Asia/Tokyo" if ticker.endswith(".T") else "America/New_York"
            base_time = result["base_time"].tz_convert(market_tz).strftime("%Y-%m-%d %H:%M")
            target_time = result["target_time"].tz_convert(market_tz).strftime("%Y-%m-%d %H:%M")
            st.caption(f"{base_time} 時点 → {target_time} の予測（市場現地時刻・1分ごとに自動更新）")
            if result["data_delay_minutes"] > 2:
                st.error(f"価格データは約{result['data_delay_minutes']}分遅延しています。現在時刻基準の予測ではありません。")
            st.caption("参考予測です。注文前に証券会社の現在値・板・スプレッドを確認してください。")
        except Exception as exc:
            st.info(f"1分後予測を停止中：{exc}")

    live_panel()
