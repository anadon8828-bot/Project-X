"""Project X V3 stock analysis dashboard. Run: streamlit run app.py"""

from pathlib import Path
import json
import re
import sqlite3

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from risk_engine import make_trade_plan
from trade_journal import add_plan, close_plan, journal_metrics, journal_state, load_journal, open_plan, position_status
from project_x_auth import password_is_configured, verify_password
from portfolio_store import add_holding, load_portfolio, save_portfolio
from settings_store import load_settings, save_settings
from home_research import render_home_research

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model_v3.pkl"
MODEL_2PCT_PATH = APP_DIR / "model_v3_2pct.pkl"
OOS_REFERENCE_PATH = APP_DIR / "train_v3_oos_predictions.csv"
WALK_FORWARD_SUMMARY_PATH = APP_DIR / "walk_forward_v3_summary.csv"
WALK_FORWARD_V4_SUMMARY_PATH = APP_DIR / "walk_forward_v4_summary.csv"
WALK_FORWARD_CONDITION_PATH = APP_DIR / "walk_forward_v3_condition_analysis.csv"
RANKING_HISTORY_PATH = APP_DIR / "project_x_ranking_history.csv"
RANKING_PERFORMANCE_PATH = APP_DIR / "project_x_ranking_performance.csv"
RANKING_PERFORMANCE_SUMMARY_PATH = APP_DIR / "project_x_ranking_performance_summary.csv"
TSE_SCAN_PATH = APP_DIR / "tse_all_scan_candidates.csv"
TSE_SCAN_PROGRESS_PATH = APP_DIR / "tse_all_scan_progress.csv"
TSE_SCAN_PROCESSED_PATH = APP_DIR / "tse_all_scan_processed.csv"
TSE_REFINED_PATH = APP_DIR / "tse_refined_top10.csv"
TSE_V3_OOS_SUMMARY_PATH = APP_DIR / "tse_v3_oos_summary.csv"
TSE_V3_STATUS_PATH = APP_DIR / "tse_v3_model_status.txt"
RELATIVE_STRENGTH_AUDIT_PATH = APP_DIR / "relative_strength_oos_audit.json"
SECTOR_RELATIVE_AUDIT_PATH = APP_DIR / "sector_relative_oos_audit.json"
MARKET_DATABASE_PATH = APP_DIR / "project_x_market_data.sqlite"
MARKET_DATA_AUDIT_PATH = APP_DIR / "market_data_audit.json"
SHORT_SUPPLY_OOS_PATH = APP_DIR / "short_supply_oos_summary.csv"
FIXED_RULES_OOS_PATH = APP_DIR / "fixed_rules_oos_summary.csv"
NEXT_DAY_OOS_PATH = APP_DIR / "next_day_oos_summary.csv"
NEXT_DAY_ALL_TSE_SUMMARY_PATH = APP_DIR / "next_day_all_tse_oos_summary.csv"
NEXT_DAY_DIRECTION_PATH = APP_DIR / "model_next_day_all_tse_direction.pkl"
NEXT_DAY_RETURN_PATH = APP_DIR / "model_next_day_all_tse_return.pkl"
FEATURES = ["MA25", "MA75", "RSI", "MACD", "Volume", "Return_1D", "Return_5D", "Return_20D", "MA25_Distance", "MA75_Distance", "Volume_Change", "Volatility_20D", "MACD_Change", "RSI_Change"]
PERIODS = {"6か月": "6mo", "1年": "1y", "3年": "3y", "5年": "5y"}
CHART_TIMEFRAMES = {
    "1分足": ("1m", "5d", 260),
    "5分足": ("5m", "60d", 300),
    "15分足": ("15m", "60d", 300),
    "日足": ("1d", "1y", 180),
    "週足": ("1wk", "5y", 180),
    "月足": ("1mo", "max", 180),
}


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the same 14 features used by train_model_v3.py."""
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["MA25"] = df["Close"].rolling(25).mean()
    df["MA75"] = df["Close"].rolling(75).mean()
    delta = df["Close"].diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + rs)
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Return_1D"] = df["Close"].pct_change(1)
    df["Return_5D"] = df["Close"].pct_change(5)
    df["Return_20D"] = df["Close"].pct_change(20)
    df["MA25_Distance"] = df["Close"] / df["MA25"] - 1
    df["MA75_Distance"] = df["Close"] / df["MA75"] - 1
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volatility_20D"] = df["Return_1D"].rolling(20).std()
    df["MACD_Change"] = df["MACD"].diff()
    df["RSI_Change"] = df["RSI"].diff()
    df["MA200"] = df["Close"].rolling(200).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"], df["BB_Lower"] = df["MA25"] + 2 * std20, df["MA25"] - 2 * std20
    tr = pd.concat([(df["High"] - df["Low"]), (df["High"] - df["Close"].shift()).abs(), (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    up, down = df["High"].diff(), -df["Low"].diff()
    plus_dm, minus_dm = up.where((up > down) & (up > 0), 0.0), down.where((down > up) & (down > 0), 0.0)
    plus_di, minus_di = 100 * plus_dm.rolling(14).sum() / tr.rolling(14).sum(), 100 * minus_dm.rolling(14).sum() / tr.rolling(14).sum()
    df["ADX"] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).rolling(14).mean()
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["High_20"], df["Low_20"] = df["High"].rolling(20).max(), df["Low"].rolling(20).min()
    return df


@st.cache_resource
def load_models():
    if not MODEL_PATH.exists() or not MODEL_2PCT_PATH.exists():
        raise FileNotFoundError("model_v3.pkl または model_v3_2pct.pkl がありません。")
    return joblib.load(MODEL_PATH), joblib.load(MODEL_2PCT_PATH)


@st.cache_data(ttl=900, show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("株価データを取得できませんでした。銘柄コードを確認してください。")
    return add_features(df)


@st.cache_data(ttl=300, show_spinner=False)
def load_chart_data(ticker: str, timeframe: str) -> pd.DataFrame:
    """Load price bars for viewing only; predictive models always use daily bars."""
    interval, period, _ = CHART_TIMEFRAMES[timeframe]
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"{timeframe}の株価データを取得できませんでした。配信元の提供期間を確認してください。")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for window in (25, 75):
        df[f"MA{window}"] = df["Close"].rolling(window).mean()
    delta = df["Close"].diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gains / losses.replace(0, np.nan))
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def probability(model, X: pd.DataFrame) -> float:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("モデルにクラス1がありません。")
    return float(model.predict_proba(X)[:, classes.index(1)][0])


def predict(model, model_2pct, data: pd.DataFrame) -> tuple[float, float]:
    f1 = list(getattr(model, "feature_names_in_", FEATURES))
    f2 = list(getattr(model_2pct, "feature_names_in_", FEATURES))
    row = data.iloc[[-1]]
    if row[f1 + f2].isna().any(axis=None):
        raise ValueError("分析期間が短すぎます。6か月以上を選んでください。")
    return probability(model, row[f1]), probability(model_2pct, row[f2])


def extension_predictions(data: pd.DataFrame) -> dict:
    paths = {"+5%確率": APP_DIR / "model_v4_5pct.pkl", "+10%確率": APP_DIR / "model_v4_10pct.pkl", "5日後上昇幅": APP_DIR / "model_v4_return_5d.pkl"}
    if not all(path.exists() for path in paths.values()):
        return {}
    row, result = data.iloc[[-1]], {}
    for label, path in paths.items():
        model = joblib.load(path)
        names = list(getattr(model, "feature_names_in_", FEATURES))
        if label == "5日後上昇幅":
            result[label] = float(model.predict(row[names])[0])
        else:
            result[label] = probability(model, row[names])
    return result


def sakata_patterns(data: pd.DataFrame) -> list[dict]:
    """Detect selected Sakata Gohō candle patterns in the latest bars."""
    df = data.tail(6)
    o, h, l, c = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
    body, spread = np.abs(c - o), np.maximum(h - l, 1e-9)
    patterns = []
    if c[-2] < o[-2] and c[-1] > o[-1] and o[-1] <= c[-2] and c[-1] >= o[-2]:
        patterns.append({"パターン": "陽の包み足", "方向": "強気", "根拠": "陰線を陽線が包み込む反転候補", "score": 1})
    if c[-2] > o[-2] and c[-1] < o[-1] and o[-1] >= c[-2] and c[-1] <= o[-2]:
        patterns.append({"パターン": "陰の包み足", "方向": "弱気", "根拠": "陽線を陰線が包み込む反転候補", "score": -1})
    lower = min(o[-1], c[-1]) - l[-1]
    upper = h[-1] - max(o[-1], c[-1])
    if lower > body[-1] * 2 and upper < spread[-1] * .25:
        patterns.append({"パターン": "たくり線", "方向": "強気", "根拠": "長い下ヒゲによる反発候補", "score": 1})
    if upper > body[-1] * 2 and lower < spread[-1] * .25:
        patterns.append({"パターン": "流れ星", "方向": "弱気", "根拠": "長い上ヒゲによる反落候補", "score": -1})
    if np.all(c[-3:] > o[-3:]) and c[-3] < c[-2] < c[-1]:
        patterns.append({"パターン": "赤三兵", "方向": "強気", "根拠": "3本連続の陽線", "score": 2})
    if np.all(c[-3:] < o[-3:]) and c[-3] > c[-2] > c[-1]:
        patterns.append({"パターン": "黒三兵", "方向": "弱気", "根拠": "3本連続の陰線", "score": -2})
    return patterns


def swings(data: pd.DataFrame, window: int = 3) -> list[tuple]:
    prices = data["Close"].tail(120)
    points = []
    for i in range(window, len(prices) - window):
        value, part = float(prices.iloc[i]), prices.iloc[i - window:i + window + 1]
        kind = "高値" if value == float(part.max()) else "安値" if value == float(part.min()) else None
        if not kind:
            continue
        if points and points[-1][2] == kind:
            extreme = value > points[-1][1] if kind == "高値" else value < points[-1][1]
            if extreme:
                points[-1] = (prices.index[i], value, kind)
        else:
            points.append((prices.index[i], value, kind))
    return points


def elliott(data: pd.DataFrame) -> dict:
    """Conservative Elliott-style estimate based on local swing structure."""
    points = swings(data)
    if len(points) < 4:
        return {"label": "判定データ不足", "bias": "中立", "score": 0, "points": points}
    recent = points[-6:]
    values = [p[1] for p in recent]
    kinds = [p[2] for p in recent]
    close, ma = float(data["Close"].iloc[-1]), float(data["MA25"].iloc[-1])
    higher = values[-1] > values[-3] and values[-2] > values[-4]
    lower = values[-1] < values[-3] and values[-2] < values[-4]
    alternating = all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1))
    if alternating and higher and close >= ma:
        return {"label": "上昇インパルス構造の候補", "bias": "強気", "score": 1, "points": recent}
    if alternating and lower and close < ma:
        return {"label": "下降インパルス構造の候補", "bias": "弱気", "score": -1, "points": recent}
    return {"label": "上向きの調整／推進波を観察中" if close >= ma else "下向きの調整／推進波を観察中", "bias": "やや強気" if close >= ma else "やや弱気", "score": 0, "points": recent}


def judge(p: float, p2: float, wave: dict, patterns: list[dict]) -> tuple[str, str, int]:
    pattern_score = sum(x["score"] for x in patterns)
    score = int(p >= .60) + int(p2 >= .60) + wave["score"] + int(pattern_score > 0) - int(pattern_score < 0)
    if score >= 3:
        return "強気候補", "AI予測とチャート形状が同方向です。価格・出来高を確認して判断してください。", score
    if score <= 0:
        return "慎重", "AI予測またはチャート形状に弱気要因があります。", score
    return "様子見", "材料が混在しています。確度が上がるまで待つ局面です。", score


MARKET_SYMBOLS = {
    "日経平均": "^N225", "TOPIX": "^TOPX", "NASDAQ": "^IXIC", "S&P500": "^GSPC",
    "VIX": "^VIX", "ドル円": "JPY=X", "米10年金利": "^TNX", "原油": "CL=F",
}


@st.cache_data(ttl=900, show_spinner=False)
def market_snapshot() -> pd.DataFrame:
    """Fetch a compact, shared market snapshot for the score and the display."""
    rows = []
    for name, symbol in MARKET_SYMBOLS.items():
        try:
            close = yf.Ticker(symbol).history(period="1mo").Close.dropna()
            if len(close) < 6:
                continue
            last, previous, ma5 = float(close.iloc[-1]), float(close.iloc[-2]), float(close.rolling(5).mean().iloc[-1])
            rows.append({"指標": name, "終値": last, "前日比(%)": (last / previous - 1) * 100, "5日方向": "上向き" if last >= ma5 else "下向き", "MA5比": last / ma5 - 1})
        except Exception:
            rows.append({"指標": name, "終値": np.nan, "前日比(%)": np.nan, "5日方向": "取得不可", "MA5比": np.nan})
    return pd.DataFrame(rows)


def market_regime() -> tuple[int, str]:
    """Transparent broad-market contribution for the integrated score."""
    snapshot = market_snapshot().set_index("指標")
    required = {"日経平均", "TOPIX", "NASDAQ", "S&P500", "VIX", "米10年金利"}
    if not required.issubset(snapshot.index):
        return 0, "市場環境: 一部取得不可"
    try:
        score = 0.0
        score += 1 if snapshot.loc["日経平均", "MA5比"] >= 0 else -1
        score += 1 if snapshot.loc["TOPIX", "MA5比"] >= 0 else -1
        score += .5 if snapshot.loc["NASDAQ", "MA5比"] >= 0 else -.5
        score += .5 if snapshot.loc["S&P500", "MA5比"] >= 0 else -.5
        score += 1 if snapshot.loc["VIX", "MA5比"] <= 0 else -1
        score += .5 if snapshot.loc["米10年金利", "MA5比"] <= 0 else -.5
        regime = 1 if score >= 1.5 else -1 if score <= -1.5 else 0
        label = "市場環境: 良好" if regime > 0 else "市場環境: 注意" if regime < 0 else "市場環境: 中立"
        return regime, f"{label}（市場スコア {score:+.1f}）"
    except (KeyError, TypeError, ValueError):
        return 0, "市場環境: 取得不可"


def news_material_score(ticker: str) -> tuple[int, list[str]]:
    """Only issuer-matched official disclosures; no unvalidated sentiment points."""
    from research_rules import valid_materials
    try:
        snapshot = json.loads((APP_DIR / 'verified_materials.json').read_text(encoding='utf-8'))
        items = [m for m in valid_materials(snapshot) if m['code']==ticker.removesuffix('.T')]
        return 0, [f"{m['category']}: {m['title']}（{m['published']}）" for m in items]
    except (OSError,ValueError):
        return 0, []


def score_breakdown(p: float, p2: float, technical: int, wave: dict, patterns: list[dict], market: int, news: int) -> pd.DataFrame:
    """Show exactly how the Project X score is assembled."""
    sakata = max(-2, min(2, sum(item["score"] for item in patterns)))
    rows = [
        {"要素": "5日上昇確率", "状態": f"{p * 100:.1f}%", "加点": p * 40},
        {"要素": "5営業日後の終値が+2%以上となる確率", "状態": f"{p2 * 100:.1f}%", "加点": p2 * 20},
        {"要素": "テクニカル", "状態": f"{technical}/4", "加点": technical * 5},
        {"要素": "エリオット波動", "状態": wave["label"], "加点": wave["score"] * 5},
        {"要素": "酒田五法", "状態": f"シグナル {sakata:+d}", "加点": sakata * 2.5},
        {"要素": "市場環境", "状態": "良好" if market > 0 else "注意" if market < 0 else "中立", "加点": market * 5},
        {"要素": "ニュース・材料", "状態": f"シグナル {news:+d}", "加点": news * 2.5},
    ]
    return pd.DataFrame(rows)


def integrated_score(p: float, p2: float, technical: int, wave: dict, patterns: list[dict], market: int, news: int) -> tuple[int, str]:
    details = score_breakdown(p, p2, technical, wave, patterns, market, news)
    raw = details["加点"].sum()
    score = int(max(0, min(100, round(raw))))
    label = "買い候補" if score >= 75 else "保有・監視" if score >= 60 else "見送り"
    return score, label


def chart(data: pd.DataFrame, wave: dict | None, timeframe: str) -> go.Figure:
    """Render a clean, timeframe-specific price chart for desktop and phone."""
    from vwap_chart import with_vwap
    data = with_vwap(data, timeframe in {"1分足", "5分足", "15分足"})
    _, _, bars = CHART_TIMEFRAMES[timeframe]
    df = data.tail(bars)
    intraday = timeframe in {"1分足", "5分足", "15分足"}
    date_format = "%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.035, row_heights=[.64, .16, .20])
    candle_text = [f"始値: ¥{o:,.0f}<br>高値: ¥{h:,.0f}<br>安値: ¥{l:,.0f}<br>終値: ¥{c:,.0f}" for o, h, l, c in zip(df.Open, df.High, df.Low, df.Close)]
    fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="ローソク足", hovertext=candle_text, hoverinfo="x+text"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.MA25, name="MA25", line=dict(color="#f7c86c"), hovertemplate=f"日時: %{{x|{date_format}}}<br>MA25: ¥%{{y:,.0f}}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.MA75, name="MA75", line=dict(color="#63b9ed"), hovertemplate=f"日時: %{{x|{date_format}}}<br>MA75: ¥%{{y:,.0f}}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.VWAP, name="日別VWAP（近似）" if intraday else "取得期間起点VWAP（近似）", line=dict(color="#ff7fbf",width=2)),row=1,col=1)
    pts = [p for p in (wave or {}).get("points", []) if p[0] in df.index]
    if pts:
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], name="波動の節目（推定）", mode="lines+markers", line=dict(color="#b083f5", width=2), hovertemplate="日付: %{x|%Y-%m-%d}<br>推定節目: ¥%{y:,.0f}<extra></extra>"), row=1, col=1)
    volume_colors = np.where(df["Close"] >= df["Open"], "#32b98b", "#e67b88")
    fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="出来高", marker_color=volume_colors, opacity=.78, hovertemplate=f"日時: %{{x|{date_format}}}<br>出来高: %{{y:,.0f}}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI", line=dict(color="#b083f5"), hovertemplate=f"日時: %{{x|{date_format}}}<br>RSI: %{{y:.1f}}<extra></extra>"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#22c55e", row=3, col=1)
    fig.update_layout(height=630, margin=dict(l=8, r=8, t=25, b=8), xaxis_rangeslider_visible=False, legend_orientation="h", hovermode="x unified", dragmode=False, paper_bgcolor="#111c2e", plot_bgcolor="#111c2e", font=dict(color="#dbe7f6"), bargap=.08)
    fig.update_xaxes(showgrid=False, tickformat="%m/%d\n%H:%M" if intraday else "%Y/%m", rangeslider_visible=False)
    fig.update_yaxes(gridcolor="rgba(151, 174, 207, .12)", zerolinecolor="rgba(151, 174, 207, .12)")
    fig.update_yaxes(title_text="株価（円）", row=1, col=1)
    fig.update_yaxes(title_text="出来高", row=2, col=1)
    fig.update_yaxes(title_text="RSI（相対力指数）", range=[0, 100], row=3, col=1)
    return fig


def chart_patterns(data: pd.DataFrame) -> list[dict]:
    """Rule-based chart-pattern candidates; they are signals, not certainties."""
    pts = swings(data, 4)
    out = []
    if len(pts) >= 3:
        a, b, c = pts[-3:]
        tolerance = .03
        if a[2] == c[2] == "高値" and abs(a[1] / c[1] - 1) <= tolerance:
            out.append({"候補": "ダブルトップ", "方向": "弱気", "根拠": "直近2高値が近い水準"})
        if a[2] == c[2] == "安値" and abs(a[1] / c[1] - 1) <= tolerance:
            out.append({"候補": "ダブルボトム", "方向": "強気", "根拠": "直近2安値が近い水準"})
    recent = data.tail(20)
    if recent["Close"].iloc[-1] >= recent["High"].iloc[:-1].max():
        out.append({"候補": "レンジ上放れ", "方向": "強気", "根拠": "20日高値を更新"})
    if recent["Close"].iloc[-1] <= recent["Low"].iloc[:-1].min():
        out.append({"候補": "レンジ下放れ", "方向": "弱気", "根拠": "20日安値を更新"})
    return out


def fundamental_table(ticker: str) -> pd.DataFrame:
    info = yf.Ticker(ticker).info
    keys = {"PER": "trailingPE", "PBR": "priceToBook", "ROE": "returnOnEquity", "EPS": "trailingEps", "売上高": "totalRevenue", "営業利益率": "operatingMargins", "売上成長率": "revenueGrowth", "配当利回り": "dividendYield", "時価総額": "marketCap"}
    rows = []
    for label, key in keys.items():
        value = info.get(key)
        if value is not None:
            if label in {"ROE", "配当利回り", "営業利益率", "売上成長率"}:
                value = f"{value * 100:.2f}%"
            elif label in {"時価総額", "売上高"}:
                value = f"¥{value / 1e8:,.0f}億"
            else:
                value = f"{value:,.2f}"
        rows.append({"項目": label, "値": value if value is not None else "取得不可"})
    return pd.DataFrame(rows)


def market_table() -> pd.DataFrame:
    table = market_snapshot().copy()
    if table.empty:
        return pd.DataFrame(columns=["指標", "終値", "前日比(%)", "5日方向", "総合スコアへの扱い"])
    score_indicators = {"日経平均", "TOPIX", "NASDAQ", "S&P500", "VIX", "米10年金利"}
    table["総合スコアへの扱い"] = table["指標"].map(lambda name: "反映" if name in score_indicators else "参考")
    table["終値"] = table["終値"].map(lambda value: round(value, 2) if pd.notna(value) else "取得不可")
    table["前日比(%)"] = table["前日比(%)"].map(lambda value: round(value, 2) if pd.notna(value) else "-")
    return table[["指標", "終値", "前日比(%)", "5日方向", "総合スコアへの扱い"]]


def walk_forward_summary() -> pd.DataFrame:
    if not WALK_FORWARD_SUMMARY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(WALK_FORWARD_SUMMARY_PATH)


def walk_forward_v4_summary() -> pd.DataFrame:
    if not WALK_FORWARD_V4_SUMMARY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(WALK_FORWARD_V4_SUMMARY_PATH)


def v4_return_model_approved() -> bool:
    """Allow return magnitude in trade rules only after acceptable OOS evidence."""
    summary = walk_forward_v4_summary()
    if summary.empty or "モデル" not in summary.columns:
        return False
    row = summary[summary["モデル"] == "5日後上昇幅"]
    if row.empty:
        return False
    direction = pd.to_numeric(row.iloc[0].get("方向一致率(%)"), errors="coerce")
    mae = pd.to_numeric(row.iloc[0].get("MAE(%)"), errors="coerce")
    return bool(pd.notna(direction) and pd.notna(mae) and direction >= 55 and mae <= 3)


def tse_scan_candidates() -> pd.DataFrame:
    if not TSE_SCAN_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(TSE_SCAN_PATH, encoding="utf-8-sig", dtype={"コード": "string"})


@st.cache_data(ttl=900, show_spinner=False)
def sector_trend(code: str) -> tuple[dict, pd.DataFrame]:
    """Measure recent momentum among liquid peers in the same JPX 33-sector."""
    universe = tse_scan_candidates()
    if universe.empty or "業種" not in universe.columns:
        return {}, pd.DataFrame()
    row = universe[universe["コード"].astype(str).str.upper() == code.upper()]
    if row.empty:
        return {}, pd.DataFrame()
    sector = row.iloc[0]["業種"]
    peers = universe[universe["業種"] == sector].drop_duplicates("コード").head(30).copy()
    tickers = [f"{item}.T" for item in peers["コード"].astype(str)]
    try:
        raw = yf.download(tickers, period="1mo", auto_adjust=True, group_by="ticker", threads=True, progress=False)
    except Exception:
        return {}, pd.DataFrame()
    returns = []
    for _, peer in peers.iterrows():
        ticker = f"{str(peer['コード'])}.T"
        try:
            frame = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0) else raw
            close = frame["Close"].dropna()
            if len(close) >= 6:
                returns.append({"コード": str(peer["コード"]), "銘柄名": peer.get("銘柄名", ""), "5日騰落率(%)": round((close.iloc[-1] / close.iloc[-6] - 1) * 100, 2)})
        except Exception:
            continue
    table = pd.DataFrame(returns)
    if table.empty:
        return {}, table
    average = float(table["5日騰落率(%)"].mean())
    rising = float((table["5日騰落率(%)"] > 0).mean() * 100)
    label = "上昇傾向" if average > .5 and rising >= 55 else "下落傾向" if average < -.5 and rising <= 45 else "方向感なし"
    return {"業種": sector, "平均5日騰落率(%)": average, "上昇銘柄比率(%)": rising, "判定": label, "対象数": len(table)}, table.sort_values("5日騰落率(%)", ascending=False)


def investor_psychology(data: pd.DataFrame, sector: dict) -> dict:
    """Price/volume proxy for selling pressure; it is not order-book data."""
    last = data.iloc[-1]
    volume_ratio = float(last.Volume / data.Volume.tail(21).iloc[:-1].mean()) if len(data) >= 21 else np.nan
    spread = max(float(last.High - last.Low), 1e-9)
    close_position = float((last.Close - last.Low) / spread)
    body = abs(float(last.Close - last.Open))
    lower_wick = float(min(last.Open, last.Close) - last.Low)
    sector_down = sector.get("判定") == "下落傾向"
    sharp_selloff = float(last.Return_1D) <= -.02 and volume_ratio >= 1.5
    if sharp_selloff and close_position <= .30 and last.Close < last.MA25 and sector_down:
        return {"判定": "売り連鎖を警戒", "理由": "大幅安・出来高急増・安値圏での引け・業種弱含みが重なっています", "出来高倍率": volume_ratio, "引け位置": close_position, "色": "warning"}
    if sharp_selloff and close_position >= .65 and lower_wick >= body and not sector_down:
        return {"判定": "売り吸収・反発候補", "理由": "売り急増後に下ヒゲを付けて高い位置で引けています。ただし反発確定ではありません", "出来高倍率": volume_ratio, "引け位置": close_position, "色": "success"}
    return {"判定": "売り圧力は中立", "理由": "売りの連鎖・吸収を示す強い価格出来高の組み合わせは確認できません", "出来高倍率": volume_ratio, "引け位置": close_position, "色": "info"}


def tse_scan_progress_count() -> int:
    """Return saved all-TSE scan progress without making the page depend on it."""
    if not TSE_SCAN_PROCESSED_PATH.exists():
        return 0
    try:
        return len(pd.read_csv(TSE_SCAN_PROCESSED_PATH, encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return 0


def tse_v3_oos_summary() -> pd.DataFrame:
    if not TSE_V3_OOS_SUMMARY_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(TSE_V3_OOS_SUMMARY_PATH, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def relative_strength_audit() -> dict:
    if not RELATIVE_STRENGTH_AUDIT_PATH.exists():
        return {}
    try:
        return json.loads(RELATIVE_STRENGTH_AUDIT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def sector_relative_audit() -> dict:
    if not SECTOR_RELATIVE_AUDIT_PATH.exists():
        return {}
    try:
        return json.loads(SECTOR_RELATIVE_AUDIT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def short_supply_oos_summary() -> pd.DataFrame:
    if not SHORT_SUPPLY_OOS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(SHORT_SUPPLY_OOS_PATH, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def fixed_rules_oos_summary() -> pd.DataFrame:
    if not FIXED_RULES_OOS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(FIXED_RULES_OOS_PATH, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def next_day_oos_summary() -> pd.DataFrame:
    if not NEXT_DAY_OOS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(NEXT_DAY_OOS_PATH, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def next_day_forecast(code: str, data: pd.DataFrame) -> dict | None:
    """Approved next-day estimate using only end-of-day market/sector context."""
    if not all(path.exists() for path in (NEXT_DAY_ALL_TSE_SUMMARY_PATH, NEXT_DAY_DIRECTION_PATH, NEXT_DAY_RETURN_PATH)):
        return None
    summary = pd.read_csv(NEXT_DAY_ALL_TSE_SUMMARY_PATH, encoding="utf-8-sig").iloc[0]
    if summary.get("status") != "APPROVED":
        return None
    try:
        from tse_universe import load_universe
        universe = load_universe()[["コード", "業種"]].rename(columns={"コード": "code", "業種": "sector"})
        sector = universe.loc[universe["code"].astype(str).str.upper() == code, "sector"].iloc[0]
        as_of = pd.Timestamp(data.index[-1]).tz_localize(None).normalize()
        with sqlite3.connect(MARKET_DATABASE_PATH) as connection:
            prices = pd.read_sql_query("SELECT code,date,close FROM price_history WHERE date BETWEEN ? AND ?", connection, params=[str((as_of - pd.Timedelta(days=14)).date()), str(as_of.date())], parse_dates=["date"])
        prices = prices.sort_values(["code", "date"]); prices["r"] = prices.groupby("code")["close"].pct_change() * 100
        latest = prices[prices["date"] == as_of].merge(universe, on="code", how="left")
        if latest.empty:
            return None
        market = float(latest["r"].median()); sector_return = float(latest.loc[latest["sector"] == sector, "r"].median())
        if not np.isfinite([market, sector_return]).all():
            return None
        row = data.iloc[[-1]][FEATURES].copy()
        for name in ("Return_1D", "Return_5D", "Return_20D", "MA25_Distance", "MA75_Distance", "Volume_Change", "Volatility_20D"):
            row[name] *= 100
        row["market_today"], row["sector_today"], row["relative_sector_today"] = market, sector_return, float(row["Return_1D"].iloc[0] - sector_return)
        cols = FEATURES + ["market_today", "sector_today", "relative_sector_today"]
        probability = float(joblib.load(NEXT_DAY_DIRECTION_PATH).predict_proba(row[cols])[:, 1][0])
        expected = float(joblib.load(NEXT_DAY_RETURN_PATH).predict(row[cols])[0])
        mae = float(summary["mae_pct"]) / 100
        close = float(data["Close"].iloc[-1])
        return {"probability": probability, "expected_pct": expected * 100, "low": close * (1 + expected - mae), "high": close * (1 + expected + mae)}
    except Exception:
        return None


def research_status_summary() -> pd.DataFrame:
    """Show only factual research readiness; never infer a trade recommendation."""
    rows = []
    relative = relative_strength_audit()
    rows.append({"項目": "相対強度モデル", "状態": relative.get("status", "未検証"), "内容": relative.get("reason", "完全OOS監査待ち")})
    sector_relative = sector_relative_audit()
    rows.append({"項目": "市場・業種相対モデル", "状態": sector_relative.get("status", "未検証"), "内容": sector_relative.get("reason", "完全OOS監査待ち")})
    fixed = fixed_rules_oos_summary()
    fixed_status = "不採用" if not fixed.empty and (pd.to_numeric(fixed["net_avg_excess_return_5d_pct"], errors="coerce") <= 0).all() else "検証待ち"
    rows.append({"項目": "固定テクニカルルール", "状態": fixed_status, "内容": "コスト後の完全OOS結果を検証画面で確認"})
    short_oos = short_supply_oos_summary()
    rows.append({"項目": "JPX空売り需給", "状態": "研究中" if not short_oos.empty else "データ蓄積中", "内容": "単独の売買シグナルには使用しない"})
    rows.append({"項目": "ペーパートレード", "状態": "利用可能", "内容": "実資金の発注なしで成績を蓄積"})
    return pd.DataFrame(rows)


def official_supply_demand(code: str) -> pd.DataFrame:
    """Latest imported point-in-time JPX observations for one code."""
    if not MARKET_DATABASE_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(MARKET_DATABASE_PATH) as connection:
            rows = pd.read_sql_query(
                "SELECT observation_date, metric, value, published_at FROM code_market_observations "
                "WHERE code = ? ORDER BY observation_date DESC, metric",
                connection, params=[code],
            )
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()
    if rows.empty:
        return rows
    latest_dates = rows.groupby("metric")["observation_date"].transform("max")
    latest = rows[rows["observation_date"] == latest_dates].copy()
    labels = {
        "short_balance_ratio_pct": "空売り残高比率(%)",
        "short_reported_balance_ratio_pct": "公表空売り残高合計(%)",
        "margin_buy_balance": "信用買い残",
        "margin_sell_balance": "信用売り残",
    }
    latest["指標"] = latest["metric"].map(labels).fillna(latest["metric"])
    return latest[["observation_date", "指標", "value", "published_at"]].rename(
        columns={"observation_date": "対象日", "value": "値", "published_at": "公表日時"}
    )


def tse_v3_status() -> str:
    try:
        return TSE_V3_STATUS_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return "PENDING"


def production_gate() -> tuple[bool, str]:
    """Keep research signals out of live plans until the fixed rule is validated."""
    try:
        audit = json.loads(MARKET_DATA_AUDIT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False, "市場データの監査結果がありません"
    if not audit.get("pass_validation_window", False):
        return False, "市場データの品質監査が未通過です"
    summary = walk_forward_summary()
    if summary.empty:
        return False, "最終売買ルールの検証結果がありません"
    rule = summary[(summary.get("strategy") == "最終ルール（AI+テクニカル・上位3）") & (summary.get("threshold").astype(str) == "0.6")]
    if rule.empty:
        return False, "最終売買ルールの検証結果が不足しています"
    item = rule.iloc[0]
    if float(item.get("trades", 0)) < 30:
        return False, "最終ルールの検証件数が30件未満です"
    if float(item.get("profit_factor", 0)) < 1.1 or float(item.get("avg_return", 0)) <= 0:
        return False, "最終ルールがPF 1.1・平均騰落率プラスの基準を満たしていません"
    return True, "最終売買ルールの検証基準を通過しています"


def condition_analysis() -> pd.DataFrame:
    if not WALK_FORWARD_CONDITION_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(WALK_FORWARD_CONDITION_PATH, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def tse_refined_top10() -> pd.DataFrame:
    if not TSE_REFINED_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(TSE_REFINED_PATH, encoding="utf-8-sig", dtype={"コード": "string"})


def require_login() -> bool:
    st.markdown("""<style>
    .stApp { background: radial-gradient(circle at 70% -15%, #203c61 0, transparent 36%), #09111f; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stHeader"] button { color: #cbd8ec; }
    .block-container { max-width: 560px; padding-top: min(17vh, 9rem); }
    [data-testid="stSidebar"] { display: none; }
    .px-login-card { padding: 2.2rem; border-radius: 22px; border: 1px solid rgba(151, 174, 207, .18); background: linear-gradient(135deg, rgba(22, 36, 58, .96), rgba(12, 22, 38, .97)); box-shadow: 0 28px 75px rgba(0, 0, 0, .34); }
    .px-login-kicker { color: #4bd6a6; font-size: .72rem; letter-spacing: .18em; font-weight: 700; }
    .px-login-title { margin: .45rem 0 .45rem; color: #f8fbff; font-size: 2.25rem; line-height: 1; letter-spacing: -.05em; font-weight: 750; }
    .px-login-copy { margin: 0 0 1.65rem; color: #9aaac0; font-size: .92rem; }
    [data-baseweb="input"] > div { background: rgba(9, 17, 31, .86); border-color: rgba(151, 174, 207, .2); border-radius: 11px; }
    [data-baseweb="input"] input { color: #edf4ff; }
    .stButton > button { width: 100%; border: 0; border-radius: 11px; background: linear-gradient(135deg, #20a77c, #2fbea0); color: #061710; font-weight: 700; }
    [data-testid="stCaptionContainer"] { color: #8292a9; }
    @media (max-width: 700px) { .block-container { padding: 18vh 1rem 2rem; } .px-login-card { padding: 1.45rem; border-radius: 18px; } .px-login-title { font-size: 1.9rem; } }
    </style>""", unsafe_allow_html=True)
    if not password_is_configured():
        st.warning("外部アクセス保護が未設定です。PC上で python set_project_x_password.py を実行してください。")
        return False
    if st.session_state.get("project_x_authenticated"):
        return True
    st.markdown('<div class="px-login-card"><div class="px-login-kicker">株式分析・個人専用</div><div class="px-login-title">Project X</div><p class="px-login-copy">パスワードを入力してログインしてください。</p>', unsafe_allow_html=True)
    with st.form("project_x_login"):
        password = st.text_input("パスワード", type="password", placeholder="パスワードを入力")
        submitted = st.form_submit_button("ログイン", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if submitted:
        if verify_password(password):
            st.session_state.project_x_authenticated = True
            st.rerun()
        else:
            st.error("パスワードが正しくありません。")
    return False


def v3_backtest(model, model2, data: pd.DataFrame, threshold: float = .60) -> tuple[pd.DataFrame, dict]:
    """Historical simulation using only the V3 out-of-sample period."""
    f1, f2 = list(getattr(model, "feature_names_in_", FEATURES)), list(getattr(model2, "feature_names_in_", FEATURES))
    df = data.dropna(subset=f1 + f2).copy()
    df["future_return"] = df.Close.shift(-5) / df.Close - 1
    df = df.dropna(subset=["future_return"])
    if OOS_REFERENCE_PATH.exists():
        oos_dates = pd.to_datetime(pd.read_csv(OOS_REFERENCE_PATH, usecols=["Date"])["Date"], errors="coerce").dropna()
        if not oos_dates.empty:
            df = df.loc[df.index >= oos_dates.min()]
    if df.empty:
        return pd.DataFrame(), {}
    p1 = model.predict_proba(df[f1])[:, list(model.classes_).index(1)]
    p2 = model2.predict_proba(df[f2])[:, list(model2.classes_).index(1)]
    result = pd.DataFrame({"date": df.index, "AI確率": p1, "+2%確率": p2, "5日後リターン": df.future_return.to_numpy()})
    result = result[result["AI確率"] >= threshold].copy()
    if result.empty:
        return result, {}
    gains, losses = result.loc[result["5日後リターン"] > 0, "5日後リターン"], result.loc[result["5日後リターン"] <= 0, "5日後リターン"]
    equity = (1 + result["5日後リターン"].clip(lower=-.99)).cumprod()
    drawdown = equity / equity.cummax() - 1
    metrics = {"勝率": (result["5日後リターン"] > 0).mean() * 100, "平均利益": gains.mean() * 100 if not gains.empty else 0, "平均損失": losses.mean() * 100 if not losses.empty else 0, "期待値": result["5日後リターン"].mean() * 100, "Profit Factor": gains.sum() / abs(losses.sum()) if not losses.empty and losses.sum() else np.inf, "最大DD": drawdown.min() * 100, "最大連敗": int((result["5日後リターン"] <= 0).astype(int).groupby((result["5日後リターン"] > 0).cumsum()).sum().max())}
    return result, metrics


def rank_stocks(model, model2) -> pd.DataFrame:
    codes = ["7203", "6758", "9984", "8306", "9432", "7011", "6857", "6146", "8035", "4063", "6501", "6098", "8058", "8001", "8766", "8411", "4502", "5108", "6367", "9433"]
    rows = []
    market_score, _ = market_regime()
    return_model_approved = v4_return_model_approved()
    for code in codes:
        try:
            d = load_data(f"{code}.T", "1y")
            p, p2 = predict(model, model2, d)
            last = d.iloc[-1]
            extended = extension_predictions(d)
            expected_return = extended.get("5日後上昇幅")
            technical = int(last.Close > last.MA25) + int(last.MA25 > last.MA75) + int(last.MACD > 0) + int(last.Volume_Change > 0)
            patterns, wave = sakata_patterns(d), elliott(d)
            material_score, material_matches = news_material_score(f"{code}.T")
            total_score, label = integrated_score(p, p2, technical, wave, patterns, market_score, material_score)
            sector, _ = sector_trend(code)
            sector_bonus = 0
            sector_label = "比較データなし"
            if sector:
                relative = float(last.Return_5D * 100) - float(sector["平均5日騰落率(%)"])
                sector_label = f"{sector['業種']}：{sector['判定']}（業種比 {relative:+.2f}%）"
                if sector["判定"] == "上昇傾向" and relative > 0:
                    sector_bonus = 4
                elif sector["判定"] == "下落傾向" and relative < 0:
                    sector_bonus = -4
            total_score = int(max(0, min(100, total_score + sector_bonus)))
            label = "買い候補" if total_score >= 75 else "保有・監視" if total_score >= 60 else "見送り"
            downside_risk = max(float(last.ATR * 1.5 / last.Close), float(last.Volatility_20D))
            # Do not treat an unvalidated return-magnitude model as a trade edge.
            if return_model_approved and expected_return is not None:
                expected_value = p * max(float(expected_return), 0) - (1 - p) * downside_risk - .0015
                reward_risk = float(expected_return) / downside_risk if downside_risk else 0
            else:
                expected_value, reward_risk = np.nan, np.nan
            material = material_matches[0].replace("好材料: ", "").replace("注意材料: ", "") if material_matches else "テクニカル・市場環境"
            rows.append({"コード": code, "総合判断": label, "総合スコア": total_score, "業種加点": sector_bonus, "業種状況": sector_label, "AI上昇確率(%)": round(p * 100, 1), "+2%確率(%)": round(p2 * 100, 1), "予測上昇幅(%)": round(float(expected_return) * 100, 2) if expected_return is not None else np.nan, "損失リスク(%)": round(downside_risk * 100, 2), "期待値(%)": round(expected_value * 100, 2), "リスクリワード": round(reward_risk, 2), "上昇幅モデル": "検証済み" if return_model_approved else "参考値", "材料": material})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values(["総合スコア", "+2%確率(%)", "AI上昇確率(%)"], ascending=False) if rows else pd.DataFrame()


def archive_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    saved = ranking.copy()
    saved.insert(0, "分析日時", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    saved.insert(1, "順位", range(1, len(saved) + 1))
    write_header = not RANKING_HISTORY_PATH.exists()
    saved.to_csv(RANKING_HISTORY_PATH, mode="a", header=write_header, index=False, encoding="utf-8-sig")
    return saved


def portfolio_snapshot(holdings: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows, total = [], 0.0
    for item in holdings.to_dict("records"):
        try:
            code = str(item["コード"]).upper().replace(".0", "")
            price = float(yf.Ticker(f"{code}.T").history(period="5d").Close.iloc[-1])
            quantity, buy = float(item["株数"]), float(item["取得単価"])
            trade_type = str(item.get("取引区分", "現物"))
            direction = -1 if trade_type == "信用売り" else 1
            profit = (price - buy) * quantity * direction
            total += profit
            market_value = price * quantity
            stop = float(item.get("損切り価格", 0) or 0)
            stop_loss = (stop - price) * quantity * direction if stop > 0 else np.nan
            rows.append({**item, "現在値": round(price, 1), "時価評価額": round(market_value), "損益": round(profit), "損益率": round((price / buy - 1) * 100 * direction, 2), "損切りまでの想定損益": round(stop_loss) if pd.notna(stop_loss) else "未設定"})
        except Exception:
            rows.append({**item, "現在値": "取得不可", "時価評価額": "-", "損益": "-", "損益率": "-", "損切りまでの想定損益": "確認不可"})
    return pd.DataFrame(rows), total


def portfolio_risk_summary(snapshot: pd.DataFrame, capital_yen: float) -> dict:
    """Summarize cash-account concentration and stop-loss exposure."""
    if snapshot.empty:
        return {"時価評価額": 0.0, "投資比率": 0.0, "最大銘柄比率": 0.0, "損切り時の合計想定損失": 0.0}
    usable = snapshot.copy()
    usable["時価評価額"] = pd.to_numeric(usable["時価評価額"], errors="coerce")
    usable["損切りまでの想定損益"] = pd.to_numeric(usable["損切りまでの想定損益"], errors="coerce")
    market_value = float(usable["時価評価額"].sum(skipna=True))
    worst_stop_loss = float(usable["損切りまでの想定損益"].clip(upper=0).sum(skipna=True))
    largest_weight = float(usable["時価評価額"].max() / capital_yen * 100) if market_value and capital_yen else 0.0
    return {"時価評価額": market_value, "投資比率": market_value / capital_yen * 100 if capital_yen else 0.0, "最大銘柄比率": largest_weight, "損切り時の合計想定損失": worst_stop_loss}


def credit_expiry_advice(holdings: pd.DataFrame) -> pd.DataFrame:
    """Apply an AI-based credit-expiry check only after its return model is validated."""
    credit = holdings[holdings["取引区分"] != "現物"].copy()
    if credit.empty:
        return pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    try:
        model, model2 = load_models()
    except Exception as exc:
        return pd.DataFrame([{"コード": "-", "判定": "AI読み込み不可", "理由": str(exc)}])
    return_model_approved = v4_return_model_approved()
    rows = []
    for item in credit.to_dict("records"):
        expiry = pd.to_datetime(item.get("信用期限"), errors="coerce")
        if pd.isna(expiry):
            rows.append({"コード": item["コード"], "取引区分": item["取引区分"], "信用期限": "未設定", "残り日数": "-", "5日予測上昇幅(%)": "-", "判定": "期限を登録", "理由": "信用取引は期限の登録が必須です"})
            continue
        remaining = int((expiry.normalize() - today).days)
        try:
            data = load_data(f"{str(item['コード']).upper()}.T", "1y")
            # The V4 return model currently remains reference-only until its
            # walk-forward validation clears the predefined quality gate.
            expected = extension_predictions(data).get("5日後上昇幅") if return_model_approved else None
            predicted = float(expected) * 100 if expected is not None else None
            if remaining < 0:
                action, reason = "期限超過を確認", "信用期限を過ぎています。証券会社の建玉状況を確認してください"
            elif remaining > 30:
                action, reason = "監視", "期限まで31日以上あります"
            elif predicted is None:
                action, reason = "期限優先で確認", "上昇幅モデルは検証基準未達のため、返済判断には使いません。価格・期限・証券会社の建玉状況を確認してください"
            elif item["取引区分"] == "信用買い" and predicted <= 0:
                action, reason = "返済売りを検討", "期限まで30日以内で、5日後の予測上昇幅がプラスではありません"
            elif item["取引区分"] == "信用売り" and predicted >= 0:
                action, reason = "買い戻しを検討", "期限まで30日以内で、5日後の予測上昇幅がプラスです"
            else:
                action, reason = "監視", "期限が近いため、毎日AI予測と建玉状況を確認してください"
            rows.append({"コード": item["コード"], "取引区分": item["取引区分"], "信用期限": expiry.strftime("%Y-%m-%d"), "残り日数": remaining, "5日予測上昇幅(%)": round(predicted, 2) if predicted is not None else "-", "判定": action, "理由": reason})
        except Exception as exc:
            rows.append({"コード": item["コード"], "取引区分": item["取引区分"], "信用期限": expiry.strftime("%Y-%m-%d"), "残り日数": remaining, "5日予測上昇幅(%)": "-", "判定": "取得不可", "理由": str(exc)})
    return pd.DataFrame(rows)


def daily_action_items(holdings: pd.DataFrame, journal: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    """Create a local, no-network checklist for the top screen."""
    rows = []
    today = pd.Timestamp.today().normalize()
    for item in holdings.to_dict("records"):
        code, trade_type = str(item["コード"]), str(item.get("取引区分", "現物"))
        stop = pd.to_numeric(pd.Series([item.get("損切り価格", 0)]), errors="coerce").iloc[0]
        if pd.isna(stop) or stop <= 0:
            rows.append({"優先度": 2, "重要度": "注意", "分類": "損切り", "コード": code, "確認事項": "損切り価格を設定", "詳細": "想定損失を管理するため、保有ごとに損切り価格を入力してください。"})
        if trade_type != "現物":
            expiry = pd.to_datetime(item.get("信用期限"), errors="coerce")
            remaining = int((expiry.normalize() - today).days) if pd.notna(expiry) else None
            if remaining is None:
                rows.append({"優先度": 1, "重要度": "重要", "分類": "信用期限", "コード": code, "確認事項": "信用期限を登録", "詳細": "信用建玉には期限の登録が必要です。"})
            elif remaining < 0:
                rows.append({"優先度": 0, "重要度": "緊急", "分類": "信用期限", "コード": code, "確認事項": "期限超過を確認", "詳細": "信用期限を過ぎています。証券会社の建玉状況を確認してください。"})
            elif remaining <= 7:
                rows.append({"優先度": 0, "重要度": "緊急", "分類": "信用期限", "コード": code, "確認事項": "期限が7日以内", "詳細": f"信用期限まで残り{remaining}日です。返済計画とAI判定を確認してください。"})
            elif remaining <= 30:
                rows.append({"優先度": 1, "重要度": "重要", "分類": "信用期限", "コード": code, "確認事項": "期限が30日以内", "詳細": f"信用期限まで残り{remaining}日です。信用期限リスクを確認してください。"})
    open_positions = journal[journal.status == "OPEN"] if not journal.empty else pd.DataFrame()
    planned = journal[journal.status == "PLANNED"] if not journal.empty else pd.DataFrame()
    if len(open_positions) >= max_positions:
        rows.append({"優先度": 1, "重要度": "重要", "分類": "資金管理", "コード": "-", "確認事項": "最大保有数に到達", "詳細": f"保有中 {len(open_positions)} 銘柄。新規候補より既存ポジションを優先して確認してください。"})
    for item in open_positions.to_dict("records"):
        rows.append({"優先度": 2, "重要度": "注意", "分類": "保有中", "コード": item["code"], "確認事項": "保有状況を更新", "詳細": "現在値と損切り・利確ラインを確認してください。"})
    for item in planned.to_dict("records"):
        rows.append({"優先度": 3, "重要度": "確認", "分類": "買い候補", "コード": item["code"], "確認事項": "買い候補を再確認", "詳細": "記録時点から価格・市場環境が変わっていないか確認してください。"})
    if not rows:
        rows.append({"優先度": 4, "重要度": "正常", "分類": "日次確認", "コード": "-", "確認事項": "緊急の確認事項なし", "詳細": "ランキング更新と保有状況の更新を必要に応じて実行してください。"})
    return pd.DataFrame(rows).sort_values(["優先度", "コード"]).drop(columns="優先度")


def render_home_portfolio(capital_yen: float, journal: pd.DataFrame, max_positions: int) -> None:
    """Show holdings on the top screen, before a stock analysis is requested."""
    saved_portfolio = load_portfolio()
    st.subheader("今日の確認事項")
    st.dataframe(daily_action_items(saved_portfolio, journal, max_positions), hide_index=True, use_container_width=True)
    st.subheader("保有銘柄")
    if saved_portfolio.empty:
        st.info("保存済みの保有銘柄はありません。ここから最初の保有銘柄を登録できます。")
    else:
        home_holdings = saved_portfolio.copy()
        home_holdings["取得金額"] = (home_holdings["株数"] * home_holdings["取得単価"]).map(lambda value: f"¥{value:,.0f}")
        home_holdings["損切り価格"] = home_holdings["損切り価格"].replace(0, np.nan).map(lambda value: f"¥{value:,.0f}" if pd.notna(value) else "未設定")
        home_holdings["信用期限"] = home_holdings["信用期限"].replace("", "-")
        st.dataframe(home_holdings[["コード", "銘柄名", "取引区分", "信用期限", "株数", "取得金額", "損切り価格"]], hide_index=True, use_container_width=True)
    with st.expander("保有銘柄を追加", expanded=saved_portfolio.empty):
        with st.form("home_portfolio_registration", clear_on_submit=True):
            h1, h2, h3 = st.columns(3)
            home_code = h1.text_input("銘柄コード", placeholder="例: 7203 または 485A")
            home_type = h2.selectbox("取引区分", ["現物", "信用買い", "信用売り"])
            home_name = h3.text_input("銘柄名（任意）", placeholder="例: トヨタ")
            h4, h5, h6, h7 = st.columns(4)
            home_shares = h4.number_input("株数", min_value=1, value=100, step=1)
            home_entry = h5.number_input("取得単価", min_value=0.01, value=1000.0, step=1.0)
            home_stop = h6.number_input("損切り価格（任意）", min_value=0.0, value=0.0, step=1.0)
            home_expiry = h7.date_input("信用期限（信用は必須）", value=pd.Timestamp.today().date())
            if st.form_submit_button("保有銘柄を登録", type="primary", use_container_width=True):
                try:
                    expiry_value = home_expiry.isoformat() if home_type != "現物" else ""
                    first_holding = pd.DataFrame([{"コード": home_code, "銘柄名": home_name, "取引区分": home_type, "信用期限": expiry_value, "株数": home_shares, "取得単価": home_entry, "損切り価格": home_stop}])
                    st.session_state.holdings = add_holding(first_holding)
                    st.session_state.pop("home_portfolio_snapshot", None)
                    st.session_state.pop("credit_expiry_advice", None)
                    st.success("保有銘柄を登録しました。")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    if (saved_portfolio["取引区分"] != "現物").any():
        st.warning("信用取引の保有があります。Project Xの売買提案は現物・レバレッジなしを前提に計算しています。信用取引は別枠で慎重に管理してください。")
    if st.button("保有状況を更新", use_container_width=True):
        with st.spinner("保有銘柄の現在値を取得しています…"):
            snapshot, total_profit = portfolio_snapshot(saved_portfolio)
        st.session_state.home_portfolio_snapshot = (snapshot, total_profit)
    current_snapshot = st.session_state.get("home_portfolio_snapshot")
    if current_snapshot:
        snapshot, total_profit = current_snapshot
        overview = portfolio_risk_summary(snapshot, capital_yen)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("時価評価額", f"¥{overview['時価評価額']:,.0f}")
        p2.metric("含み損益", f"¥{total_profit:,.0f}")
        p3.metric("投資比率", f"{overview['投資比率']:.1f}%")
        p4.metric("最大銘柄比率", f"{overview['最大銘柄比率']:.1f}%")
        st.dataframe(snapshot, hide_index=True, use_container_width=True)
        if overview["最大銘柄比率"] > 40:
            st.warning("1銘柄が資金の40%を超えています。投資額の縮小または分散を検討してください。")
        if overview["損切り時の合計想定損失"]:
            st.caption(f"設定済み損切り価格に到達した場合の合計想定損失: ¥{overview['損切り時の合計想定損失']:,.0f}")
    if (saved_portfolio["取引区分"] != "現物").any():
        st.subheader("信用期限とAI判定")
        if st.button("信用期限リスクを確認", use_container_width=True):
            with st.spinner("信用建玉の期限とAI予測を確認しています…"):
                st.session_state.credit_expiry_advice = credit_expiry_advice(saved_portfolio)
        advice = st.session_state.get("credit_expiry_advice")
        if advice is not None and not advice.empty:
            st.dataframe(advice, hide_index=True, use_container_width=True)
            st.caption("この判定は5日後の予測上昇幅を使う保守的な注意喚起です。1か月先を予測するものではありません。期限・保証金・建玉状況は証券会社の画面で必ず確認してください。")


    render_home_research()


def company_display_name(code: str) -> str:
    """Resolve the Japanese company name from the locally saved JPX master."""
    path = APP_DIR / "tse_domestic_common_stocks.csv"
    try:
        names = pd.read_csv(path, encoding="utf-8-sig", dtype={"コード": "string"})
        matches = names.loc[names["コード"].str.strip().str.upper() == code.strip().upper(), "銘柄名"].dropna()
        if not matches.empty and str(matches.iloc[0]).strip():
            return str(matches.iloc[0]).strip()
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        pass
    return "会社名未取得"


def main() -> None:
    st.set_page_config(page_title="Project X | 株価予測", page_icon="📈", layout="wide")
    if not require_login():
        st.stop()
    st.markdown("""<style>
    :root { --px-bg: #09111f; --px-surface: #111c2e; --px-surface-2: #16243a; --px-line: rgba(151, 174, 207, .16); --px-text: #edf4ff; --px-muted: #9aaac0; --px-green: #4bd6a6; --px-amber: #f7c86c; --px-red: #f48491; }
    .stApp { background: radial-gradient(circle at 72% -20%, #1c3554 0, transparent 34%), var(--px-bg); color: var(--px-text); }
    [data-testid="stHeader"] { background: rgba(9, 17, 31, .72); border-bottom: 1px solid var(--px-line); backdrop-filter: blur(16px); }
    .block-container { max-width: 1480px; padding: 1.5rem 2rem 3.5rem; }
    [data-testid="stSidebar"] { display: block; background: linear-gradient(180deg, #0e192b 0%, #0a1321 100%); border-right: 1px solid var(--px-line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
    [data-testid="stSidebar"] h2 { color: var(--px-text); font-size: 1.02rem; letter-spacing: .04em; }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: var(--px-text); }
    [data-testid="stCaptionContainer"] { color: var(--px-muted); }
    [data-testid="stMetric"] { background: linear-gradient(135deg, rgba(22, 36, 58, .92), rgba(14, 25, 43, .92)); border: 1px solid var(--px-line); border-radius: 16px; padding: 1rem 1.05rem; min-height: 112px; box-shadow: 0 12px 30px rgba(0, 0, 0, .13); }
    [data-testid="stMetricLabel"] { color: var(--px-muted); font-size: .82rem; }
    [data-testid="stMetricValue"] { color: var(--px-text); font-size: 1.55rem; letter-spacing: -.025em; }
    [data-testid="stMetricDelta"] svg { display: none; }
    [data-testid="stMetricDelta"] { color: var(--px-green); font-size: .78rem; }
    [data-testid="stDataFrame"], [data-testid="stExpander"], [data-testid="stPlotlyChart"] { border: 1px solid var(--px-line); border-radius: 16px; overflow: hidden; background: rgba(17, 28, 46, .72); }
    [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--px-line); }
    [data-baseweb="tab"] { color: var(--px-muted); background: transparent; border-radius: 9px 9px 0 0; padding: .7rem .8rem; }
    [data-baseweb="tab"][aria-selected="true"] { color: var(--px-text); background: rgba(67, 116, 180, .18); }
    [data-baseweb="tab-highlight"] { background: var(--px-green); height: 2px; }
    .stButton > button, [data-testid="stFormSubmitButton"] > button { border-radius: 11px; border: 1px solid rgba(125, 190, 229, .24); background: #1a3150; color: var(--px-text); font-weight: 600; transition: all .18s ease; }
    .stButton > button:hover { border-color: var(--px-green); color: #fff; background: #214164; transform: translateY(-1px); }
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, #20a77c, #2fbea0); color: #061710; border-color: transparent; }
    [data-baseweb="input"] > div, [data-baseweb="select"] > div { background: rgba(18, 31, 51, .92); border-color: var(--px-line); color: var(--px-text); border-radius: 10px; }
    [data-testid="stAlert"] { border-radius: 13px; border: 1px solid var(--px-line); }
    .px-brand { margin: .15rem 0 1.4rem; padding: 1.45rem 1.55rem; border: 1px solid var(--px-line); border-radius: 20px; background: linear-gradient(110deg, rgba(29, 55, 89, .78), rgba(13, 23, 39, .7) 60%, rgba(31, 123, 108, .18)); box-shadow: 0 22px 55px rgba(0, 0, 0, .18); }
    .px-brand__kicker { color: var(--px-green); font-size: .72rem; letter-spacing: .16em; font-weight: 700; }
    .px-brand__title { margin: .35rem 0 .25rem; color: #f8fbff; font-size: clamp(1.8rem, 3vw, 2.65rem); font-weight: 700; letter-spacing: -.04em; }
    .px-brand__sub { color: var(--px-muted); font-size: .92rem; }
    .px-sidebar-mark { padding: .15rem 0 .8rem; font-size: 1.3rem; font-weight: 700; letter-spacing: -.03em; color: #f5f9ff; }
    .px-sidebar-mark span { color: var(--px-green); }
    @media (max-width: 700px) {
        .block-container { padding: .8rem .6rem 2.5rem; }
        .px-brand { padding: 1.1rem; margin-bottom: 1rem; border-radius: 16px; }
        .px-brand__title { font-size: 1.65rem; }
        [data-testid="stHorizontalBlock"] { gap: .45rem; }
        [data-testid="stMetric"] { padding: .65rem .7rem; min-height: 88px; border-radius: 13px; }
        [data-testid="stMetricValue"] { font-size: 1.15rem; }
        [data-baseweb="tab-list"] { overflow-x: auto; white-space: nowrap; }
        [data-baseweb="tab"] { padding: .6rem .62rem; font-size: .81rem; }
        [data-testid="stPlotlyChart"] { border-radius: 13px; }
    }
    </style>""", unsafe_allow_html=True)
    st.markdown("""<div class="px-brand"><div class="px-brand__kicker">JAPAN EQUITY RESEARCH · PRIVATE</div><div class="px-brand__title">Project X</div><div class="px-brand__sub">AI予測・市場環境・需給・テクニカルを、検証結果とともに確認するリサーチ環境</div></div>""", unsafe_allow_html=True)
    market = st.radio("対象市場", ["日本株", "米国株"], horizontal=True, key="selected_market")
    mode = st.radio("分析モード", ["通常分析", "デイトレ"], horizontal=True, key='analysis_mode')
    if mode == 'デイトレ':
        from daytrade_mode import render_daytrade
        render_daytrade('JP' if market == '日本株' else 'US', APP_DIR)
        return
    if market == "米国株":
        from us_equities import render_us_equities
        render_us_equities(load_data, load_chart_data, sakata_patterns, elliott, chart_patterns)
        return
    journal = load_journal()
    open_positions, recorded_loss_streak = journal_state(journal)
    production_ready, production_reason = production_gate()
    if production_ready:
        st.success(f"実戦投入ゲート: 通過 — {production_reason}")
    else:
        st.warning(f"実戦投入ゲート: 未通過 — {production_reason}")
    with st.expander("Project Xの検証・運用状況", expanded=False):
        st.dataframe(research_status_summary(), hide_index=True, use_container_width=True)
    saved_settings = load_settings()
    if "capital_yen" not in st.session_state:
        st.session_state.capital_yen = saved_settings["capital_yen"]
    if "max_positions" not in st.session_state:
        st.session_state.max_positions = saved_settings["max_positions"]
    if "risk_per_trade" not in st.session_state:
        st.session_state.risk_per_trade = saved_settings["risk_per_trade"]
    with st.sidebar:
        st.markdown('<div class="px-sidebar-mark">✦ Project <span>X</span></div>', unsafe_allow_html=True)
        st.header("分析条件")
        code = st.text_input("東証銘柄コード", "7203", max_chars=4).strip()
        period_name = st.selectbox("表示期間", list(PERIODS), index=1)
        capital_yen = st.number_input("運用資金（円）", min_value=100_000, step=100_000, key="capital_yen")
        max_positions = st.slider("最大保有数", 1, 10, key="max_positions")
        risk_per_trade = st.select_slider("1取引の損失上限", options=[.005, .01, .015, .02], format_func=lambda value: f"{value * 100:.1f}%", key="risk_per_trade")
        loss_streak = st.number_input("直近の連敗数", min_value=0, max_value=20, value=int(recorded_loss_streak), step=1)
        if st.button("運用設定を保存", use_container_width=True):
            try:
                save_settings(capital_yen, max_positions, risk_per_trade)
                st.success("運用設定を保存しました。")
            except ValueError as exc:
                st.error(str(exc))
        run = st.button("分析する", type="primary", use_container_width=True)
        go_home = st.button("ホームを表示", use_container_width=True)
        st.caption("例: 7203（トヨタ）、6758（ソニーG）、485A")
        with st.expander("保有銘柄を追加", expanded=False):
            with st.form("sidebar_portfolio_registration", clear_on_submit=True):
                side_code = st.text_input("銘柄コード", placeholder="例: 7203 / 485A", key="side_holding_code")
                side_type = st.selectbox("取引区分", ["現物", "信用買い", "信用売り"], key="side_holding_type")
                side_name = st.text_input("銘柄名（任意）", key="side_holding_name")
                side_shares = st.number_input("株数", min_value=1, value=100, step=1, key="side_holding_shares")
                side_entry = st.number_input("取得単価", min_value=0.01, value=1000.0, step=1.0, key="side_holding_entry")
                side_stop = st.number_input("損切り価格（任意）", min_value=0.0, value=0.0, step=1.0, key="side_holding_stop")
                side_expiry = st.date_input("信用期限（信用は必須）", value=pd.Timestamp.today().date(), key="side_holding_expiry")
                if st.form_submit_button("保有銘柄を登録", type="primary", use_container_width=True):
                    try:
                        expiry_value = side_expiry.isoformat() if side_type != "現物" else ""
                        holding = pd.DataFrame([{"コード": side_code, "銘柄名": side_name, "取引区分": side_type, "信用期限": expiry_value, "株数": side_shares, "取得単価": side_entry, "損切り価格": side_stop}])
                        add_holding(holding)
                        st.session_state.pop("home_portfolio_snapshot", None)
                        st.success("保有銘柄を登録しました。")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        legacy_portfolio = load_portfolio()
        legacy_credit = legacy_portfolio[(legacy_portfolio["取引区分"] != "現物") & (legacy_portfolio["信用期限"].fillna("").astype(str).str.strip() == "")]
        if not legacy_credit.empty:
            st.warning("登録済みの信用建玉に期限未登録があります。期限を入れるまで新規保有の追加は保存されません。")
            with st.form("repair_credit_expiry"):
                repair_code = st.selectbox("期限を登録する銘柄", legacy_credit["コード"].astype(str).tolist())
                repair_expiry = st.date_input("信用期限", value=pd.Timestamp.today().date(), key="repair_credit_expiry_date")
                if st.form_submit_button("信用期限を保存", use_container_width=True):
                    try:
                        repaired = legacy_portfolio.copy()
                        repaired.loc[repaired["コード"].astype(str) == repair_code, "信用期限"] = repair_expiry.isoformat()
                        save_portfolio(repaired)
                        st.success("信用期限を保存しました。続けて保有銘柄を追加できます。")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        editable_portfolio = load_portfolio()
        if not editable_portfolio.empty:
            with st.expander("保有銘柄を修正・削除", expanded=False):
                edit_options = {f"{row['コード']}（ロット {number + 1}）": index for number, (index, row) in enumerate(editable_portfolio.iterrows())}
                edit_label = st.selectbox("修正する銘柄", list(edit_options), key="edit_holding_code")
                edit_index = edit_options[edit_label]
                current = editable_portfolio.loc[edit_index]
                with st.form("edit_portfolio_holding"):
                    edit_name = st.text_input("銘柄名", value="" if pd.isna(current["銘柄名"]) else str(current["銘柄名"]))
                    edit_type = st.selectbox("取引区分", ["現物", "信用買い", "信用売り"], index=["現物", "信用買い", "信用売り"].index(str(current["取引区分"])))
                    edit_shares = st.number_input("株数", min_value=1, value=int(current["株数"]), step=1)
                    edit_entry = st.number_input("取得単価", min_value=0.01, value=float(current["取得単価"]), step=1.0)
                    edit_stop = st.number_input("損切り価格（任意）", min_value=0.0, value=float(current["損切り価格"]), step=1.0)
                    parsed_expiry = pd.to_datetime(current["信用期限"], errors="coerce")
                    edit_expiry = st.date_input("信用期限（信用は必須）", value=parsed_expiry.date() if pd.notna(parsed_expiry) else pd.Timestamp.today().date())
                    if st.form_submit_button("修正を保存", type="primary", use_container_width=True):
                        try:
                            revised = editable_portfolio.copy()
                            revised.loc[edit_index, ["銘柄名", "取引区分", "株数", "取得単価", "損切り価格"]] = [edit_name, edit_type, edit_shares, edit_entry, edit_stop]
                            revised.loc[edit_index, "信用期限"] = edit_expiry.isoformat() if edit_type != "現物" else ""
                            save_portfolio(revised)
                            st.success("保有銘柄を修正しました。")
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
                if st.button(f"{edit_label} を保有一覧から削除", use_container_width=True, key="delete_holding"):
                    try:
                        save_portfolio(editable_portfolio.drop(index=edit_index))
                        st.success("保有銘柄を削除しました。")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
    if go_home:
        st.session_state.pop("active_analysis", None)
        st.rerun()
    if run:
        selected_code = code.upper()
        if not re.fullmatch(r"(?:\d{4}|\d{3}[A-Z])", selected_code):
            st.error("東証コードを入力してください（例: 7203、485A）。")
            return
        st.session_state.active_analysis = {"code": selected_code, "period_name": period_name}
    active_analysis = st.session_state.get("active_analysis")
    if not active_analysis:
        render_home_portfolio(capital_yen, journal, max_positions)
        st.info("左の「分析する」を押すと、最新データで分析します。")
        return
    code = active_analysis["code"]
    period_name = active_analysis["period_name"]
    try:
        with st.spinner("株価データとAIモデルを読み込んでいます…"):
            model, model2 = load_models()
            data = load_data(f"{code}.T", PERIODS[period_name])
            p, p2 = predict(model, model2, data)
    except Exception as exc:
        st.error(f"分析を実行できませんでした: {exc}")
        return
    latest, patterns, wave = data.iloc[-1], sakata_patterns(data), elliott(data)
    next_day = next_day_forecast(code, data)
    judgement, message, _ = judge(p, p2, wave, patterns)
    extended = extension_predictions(data)
    return_model_approved = v4_return_model_approved()
    technical_score = int(latest.Close > latest.MA25) + int(latest.MA25 > latest.MA75) + int(latest.MACD > 0) + int(latest.Volume_Change > 0)
    market_score, market_label = market_regime()
    material_score, material_matches = news_material_score(f"{code}.T")
    score, integrated_label = integrated_score(p, p2, technical_score, wave, patterns, market_score, material_score)
    score_details = score_breakdown(p, p2, technical_score, wave, patterns, market_score, material_score)
    plan_expected_return = extended.get("5日後上昇幅") if return_model_approved else None
    plan = make_trade_plan(float(latest.Close), float(latest.ATR), p, p2, plan_expected_return, technical_score, capital_yen, open_positions, int(loss_streak), max_positions, risk_per_trade)
    st.subheader(f"{code}　{company_display_name(code)}")
    st.caption(f"分析結果（{latest.name:%Y-%m-%d} 時点）")
    a, b, c, d = st.columns(4)
    a.metric("終値", f"¥{latest.Close:,.0f}", f"{latest.Return_1D * 100:+.2f}%")
    b.metric("5営業日後の終値が上昇する確率", f"{p * 100:.1f}%")
    c.metric("5営業日後の終値が+2%以上となる確率", f"{p2 * 100:.1f}%")
    st.caption("学習対象：基準日の終値に対する5営業日後の終値。途中で+2%に到達する確率ではありません。表示はモデル推定値で、確率の校正・実戦有効性は保証されません。")
    d.metric("参考ルール判定（未検証）", integrated_label, f"参考スコア {score}")
    st.caption("参考ルール判定は買い推奨ではありません。ホームの候補採用とは別で、実売買の条件成立を保証しません。")
    from candidate_watchlist import render_candidate_detail
    render_candidate_detail(APP_DIR, code)
    from research_extras import render_risk_check, render_events
    render_risk_check(code, float(latest.Close), capital_yen, load_portfolio(), APP_DIR)
    render_events(f"{code}.T")
    st.caption(f"{message}／{market_label}")
    if next_day:
        n1, n2, n3 = st.columns(3)
        n1.metric("翌営業日の上昇確率", f"{next_day['probability'] * 100:.1f}%")
        n2.metric("翌営業日の予測騰落率", f"{next_day['expected_pct']:+.2f}%")
        n3.metric("翌営業日の予測レンジ", f"¥{next_day['low']:,.0f}〜¥{next_day['high']:,.0f}")
        st.warning("翌日モデルは検証方法の不備により再検証中です。表示値は研究記録用で、採用基準通過とは扱いません。価格の幅は予測値±過去の平均絶対誤差で、一定確率を保証する予測区間ではありません。")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["総合チャート", "酒田五法・パターン", "エリオット波動", "AI・テクニカル", "企業・ニュース", "相場・ランキング", "売買シミュレーション"])
    with tab1:
        timeframe = st.radio("時間足", list(CHART_TIMEFRAMES), horizontal=True, key="chart_timeframe")
        try:
            with st.spinner(f"{timeframe}を読み込んでいます…"):
                chart_data = load_chart_data(f"{code}.T", timeframe)
            chart_wave = wave if timeframe == "日足" else None
            st.plotly_chart(chart(chart_data, chart_wave, timeframe), use_container_width=True, config={"displayModeBar": False, "scrollZoom": False, "displaylogo": False})
            st.caption('VWAPは各足の高値・安値・終値の平均×出来高から計算した近似です。分足は日本時間の日ごとにリセット、日足以上は取得期間の先頭起点で、当日VWAPとは異なります。')
            if timeframe in {"1分足", "5分足", "15分足"}:
                st.caption("分足は配信元の提供可能期間内で表示します。AI予測・総合スコアは日足を基準に計算しています。")
            else:
                st.caption("日足以上ではMA25・MA75・RSIを表示します。エリオットの節目は日足のみの参考表示です。")
        except Exception as exc:
            st.warning(f"{timeframe}を表示できませんでした: {exc}")
    with tab2:
        st.subheader("直近ローソク足の酒田五法パターン")
        if patterns:
            st.dataframe(pd.DataFrame(patterns).drop(columns="score"), hide_index=True, use_container_width=True)
        else:
            st.info("対象にした酒田五法パターンは検出されませんでした。")
        st.caption("形状の自動検出です。相場の位置・出来高・ニュースと併せて確認してください。")
        st.subheader("チャートパターン")
        cp = chart_patterns(data)
        if cp:
            st.dataframe(pd.DataFrame(cp), hide_index=True, use_container_width=True)
        else:
            st.info("明確なパターン候補はありません。")
        st.caption("サポート: ¥{:,.0f} / レジスタンス: ¥{:,.0f}".format(data.Low.tail(20).min(), data.High.tail(20).max()))
    with tab3:
        st.subheader("エリオット波動の推定")
        st.metric("現在の見立て", wave["label"])
        st.write(f"方向感: **{wave['bias']}**")
        if wave["points"]:
            table = pd.DataFrame(wave["points"], columns=["日付", "価格", "種別"])
            table["日付"] = table["日付"].dt.strftime("%Y-%m-%d")
            st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption("波動カウントには複数の解釈があり得ます。ここでは局所的な高値・安値による機械的な推定です。")
    with tab4:
        table = pd.DataFrame([{"MA25乖離率": f"{latest.MA25_Distance * 100:+.2f}%", "MA75乖離率": f"{latest.MA75_Distance * 100:+.2f}%", "MA200": f"¥{latest.MA200:,.0f}", "RSI": f"{latest.RSI:.1f}", "MACD": f"{latest.MACD:.2f}", "BB": f"¥{latest.BB_Lower:,.0f}〜¥{latest.BB_Upper:,.0f}", "ATR": f"{latest.ATR:,.1f}", "ADX": f"{latest.ADX:.1f}", "VWAP": f"¥{latest.VWAP:,.0f}", "出来高": f"{latest.Volume:,.0f}"}])
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.subheader("同業種の強さ（流動性のある同業種銘柄）")
        sector, peers = sector_trend(code)
        if sector:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("業種", sector["業種"])
            s2.metric("業種平均・5日", f"{sector['平均5日騰落率(%)']:+.2f}%")
            s3.metric("上昇銘柄比率", f"{sector['上昇銘柄比率(%)']:.0f}%")
            s4.metric("業種判断", sector["判定"])
            own_5d = float(latest.Return_5D * 100)
            relative = own_5d - float(sector["平均5日騰落率(%)"])
            peer_row = peers[peers["コード"].astype(str).str.upper() == code]
            peer_rank = int(peers.index.get_loc(peer_row.index[0]) + 1) if not peer_row.empty else None
            r1, r2 = st.columns(2)
            r1.metric("この銘柄・5日", f"{own_5d:+.2f}%", f"業種平均との差 {relative:+.2f}%")
            r2.metric("業種内の強さ", f"{peer_rank} 位 / {len(peers)}" if peer_rank else "比較対象外")
            st.caption(f"同業種 {sector['対象数']} 銘柄を対象。個別銘柄だけでなく、業種全体の資金流入・流出も確認してください。")
            st.dataframe(peers.head(10), hide_index=True, use_container_width=True)
        else:
            st.info("同業種の比較データは、全東証一次スキャン結果の作成後に表示されます。")
        st.subheader("投資家心理の推定（価格・出来高ベース）")
        psychology = investor_psychology(data, sector)
        p1, p2, p3 = st.columns(3)
        p1.metric("心理シグナル", psychology["判定"])
        p2.metric("出来高倍率", f"{psychology['出来高倍率']:.2f}倍" if np.isfinite(psychology["出来高倍率"]) else "算出不可")
        p3.metric("当日レンジ内の引け位置", f"{psychology['引け位置'] * 100:.0f}%")
        getattr(st, psychology["色"])(psychology["理由"])
        st.caption("板・注文・個々の投資家の感情を直接取得したものではなく、価格・出来高・ローソク足・業種の動きからの推定です。")
        st.subheader("JPX公式の需給データ")
        official_demand = official_supply_demand(code)
        if official_demand.empty:
            st.info("この銘柄のJPX公式需給データはまだ取り込まれていません。空売り残高・信用残は公表対象と時点に制約があるため、未取得をゼロとして扱いません。")
        else:
            st.dataframe(official_demand, hide_index=True, use_container_width=True)
            st.caption("空売り残高・信用残は公表日時点の公式データです。需給の注意材料として表示し、単独で売買判断には使いません。")
        if extended:
            e1, e2, e3 = st.columns(3)
            e1.metric("5日以内に+5%", f"{extended['+5%確率'] * 100:.1f}%")
            e2.metric("5日以内に+10%", f"{extended['+10%確率'] * 100:.1f}%")
            e3.metric("5日後の予測上昇幅", f"{extended['5日後上昇幅'] * 100:+.2f}%")
            if not return_model_approved:
                st.warning("上昇幅モデルはWalk-forwardで誤差が大きいため、現在は参考表示です。売買判定・期待値・提案株数には使いません。")
        else:
            st.info("+5%・+10%・上昇幅モデルは未学習です。train_model_v4_extensions.py を実行すると自動表示されます。")
        st.subheader("+5%・+10%・上昇幅モデルの検証")
        v4_summary = walk_forward_v4_summary()
        if v4_summary.empty:
            st.info("Walk-forward検証結果はまだありません。walk_forward_v4_extensions.py の完了後に自動表示されます。")
        else:
            st.dataframe(v4_summary, hide_index=True, use_container_width=True)
            st.caption("+5%・+10%はAUC・Brier・確率帯ごとの実績で、上昇幅はMAE・方向一致率で確認します。少ない件数の確率帯は売買判断に使いません。")
        st.subheader("翌営業日予測モデルの検証")
        next_day = next_day_oos_summary()
        if next_day.empty:
            st.info("翌営業日予測モデルは検証待ちです。")
        else:
            st.warning("翌営業日予測モデルは最終OOS基準未達のため、予測値・予測レンジを表示していません。")
            st.dataframe(next_day, hide_index=True, use_container_width=True)
        st.subheader("売買ルール・資金管理")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("判定", plan.action)
        r2.metric("損切り目安", f"¥{plan.stop:,.0f}")
        r3.metric("利確目安", f"¥{plan.target:,.0f}")
        r4.metric("提案株数", f"{plan.suggested_shares} 株")
        st.caption(f"{plan.reason}／リスクリワード {plan.risk_reward:.2f}／最大許容損失 ¥{plan.maximum_loss_yen:,.0f}。前提: 資金¥{capital_yen:,.0f}・最大{max_positions}銘柄・1取引の損失上限{risk_per_trade * 100:.1f}%・信用取引なし。")
        st.caption(f"取引記録: 保有中 {open_positions} 銘柄、直近連敗 {recorded_loss_streak} 回")
        if not production_ready:
            st.warning(f"実戦投入ゲート: 未通過 — {production_reason}。AI・業種・心理シグナルは研究・監視用途として表示し、新規売買計画は記録できません。")
        else:
            st.success(f"実戦投入ゲート: 通過 — {production_reason}")
        st.subheader("Project X 総合スコアの内訳")
        display_score_details = score_details.copy()
        display_score_details["加点"] = display_score_details["加点"].round(1)
        st.dataframe(display_score_details, hide_index=True, use_container_width=True)
        if st.button("ペーパートレード計画を記録", use_container_width=True):
            add_plan(code, plan, mode="PAPER")
            st.success("ペーパートレードとして取引記録に保存しました。実資金の発注は行いません。")
        if not production_ready:
            st.caption("実資金の売買計画は、実戦投入ゲートを通過するまで記録・提案しません。")
        st.caption("AIモデルには学習時に保存された特徴量名・順番で入力しています。")
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({"特徴量": getattr(model, "feature_names_in_", FEATURES), "重要度": model.feature_importances_}).sort_values("重要度", ascending=False)
            st.bar_chart(importance.set_index("特徴量"))
    with tab5:
        st.subheader("ファンダメンタル")
        try:
            st.dataframe(fundamental_table(f"{code}.T"), hide_index=True, use_container_width=True)
        except Exception as exc:
            st.warning(f"ファンダメンタルを取得できませんでした: {exc}")
        st.subheader("ニュース・材料")
        st.caption("銘柄コード・公表日時・出典を照合した公式開示のみ。見出し分類を利益予測や総合スコアには加点しません。")
        try:
            from research_rules import valid_materials
            snapshot = json.loads((APP_DIR / 'verified_materials.json').read_text(encoding='utf-8'))
            news = [m for m in valid_materials(snapshot) if m['code']==code]
            if news:
                for item in news:
                    st.link_button(f"{item['category']}：{item['title']}（{item['published']}／TDnet原文）",item['url'])
            else:
                st.info("確認可能な直近72時間の公式開示はありません。一般ニュースの不在を意味しません。")
        except Exception as exc:
            st.warning(f"ニュースを取得できませんでした: {exc}")
    with tab6:
        st.subheader("相場全体")
        st.dataframe(market_table(), hide_index=True, use_container_width=True)
        from candidate_watchlist import render_watchlist
        from home_research import read_saved, open_ranked_stock
        st.session_state.pop("latest_ranking", None)
        render_watchlist(read_saved, APP_DIR, open_ranked_stock)
        st.caption("ホームと同じ最新性チェック・同じ選定元を使用しています。旧20銘柄ランキングと古い詳細評価TOP10は表示しません。")
        st.subheader("ポートフォリオ・資金管理")
        if "holdings" not in st.session_state:
            st.session_state.holdings = load_portfolio()
        edited_holdings = st.data_editor(st.session_state.holdings, num_rows="dynamic", use_container_width=True, key="portfolio_editor", column_config={"コード": st.column_config.TextColumn(help="4桁コード、または485Aのような4文字"), "取引区分": st.column_config.SelectboxColumn(options=["現物", "信用買い", "信用売り"], required=True), "信用期限": st.column_config.TextColumn(help="信用取引はYYYY-MM-DD形式で必須。現物は空欄で構いません。"), "株数": st.column_config.NumberColumn(min_value=1, step=1), "取得単価": st.column_config.NumberColumn(min_value=0.01), "損切り価格": st.column_config.NumberColumn(min_value=0.0, help="未設定の場合は0")})
        save_col, evaluate_col = st.columns(2)
        with save_col:
            if st.button("ポートフォリオを保存", use_container_width=True):
                try:
                    st.session_state.holdings = save_portfolio(edited_holdings)
                    st.session_state.pop("home_portfolio_snapshot", None)
                    st.session_state.pop("credit_expiry_advice", None)
                    st.success("ポートフォリオを保存しました。")
                except ValueError as exc:
                    st.error(str(exc))
        with evaluate_col:
            if st.button("ポートフォリオを評価", use_container_width=True):
                try:
                    saved_holdings = save_portfolio(edited_holdings)
                    st.session_state.holdings = saved_holdings
                    st.session_state.pop("home_portfolio_snapshot", None)
                    st.session_state.pop("credit_expiry_advice", None)
                    snapshot, total_profit = portfolio_snapshot(saved_holdings)
                    overview = portfolio_risk_summary(snapshot, capital_yen)
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("時価評価額", f"¥{overview['時価評価額']:,.0f}")
                    p2.metric("合計損益", f"¥{total_profit:,.0f}")
                    p3.metric("資金に対する投資比率", f"{overview['投資比率']:.1f}%")
                    p4.metric("最大銘柄比率", f"{overview['最大銘柄比率']:.1f}%")
                    st.dataframe(snapshot, hide_index=True, use_container_width=True)
                    if overview["最大銘柄比率"] > 40:
                        st.warning("1銘柄が資金の40%を超えています。分散または投資額の縮小を検討してください。")
                    if overview["損切り時の合計想定損失"]:
                        st.caption(f"設定済み損切り価格に到達した場合の合計想定損失: ¥{overview['損切り時の合計想定損失']:,.0f}")
                    else:
                        st.info("損切り価格が未設定です。保有ごとに損切り価格を設定すると、最大損失を集計できます。")
                except ValueError as exc:
                    st.error(str(exc))
    with tab7:
        st.subheader("Walk-forward検証（最優先）")
        tse_oos = tse_v3_oos_summary()
        tse_status = tse_v3_status()
        if tse_status == "HOLD":
            st.warning("全東証版AIは完全OOS検証で採用基準未達です。現在のランキングには使用していません。")
        elif tse_status == "APPROVED":
            st.success("全東証版AIは検証基準を通過し、採用候補です。")
        if tse_oos.empty:
            st.info("全東証版AIは、履歴データ収集と完全OOS検証の完了後にここへ表示されます。完了前のモデルは採用しません。")
        else:
            st.subheader("全東証版AI・完全OOS結果")
            st.dataframe(tse_oos, hide_index=True, use_container_width=True)
            st.caption("全東証版は既存V3との比較・期間別検証を通過するまで、ランキングの採用モデルには切り替えません。")
        relative_audit = relative_strength_audit()
        if relative_audit:
            st.subheader("相対強度モデル・完全OOS監査")
            if relative_audit.get("status") == "REJECTED":
                st.warning("相対強度モデルは完全OOS監査で不採用です。ランキングや売買計画には使用していません。")
            audit_columns = {
                "取引数": relative_audit.get("trades"),
                "OOS期間": relative_audit.get("oos_periods"),
                "コスト後平均超過リターン(%)": relative_audit.get("net_average_excess_return_pct_after_0_15pct_cost"),
                "コスト後PF": relative_audit.get("profit_factor_after_cost"),
                "最大DD(%)": relative_audit.get("monthly_portfolio_max_drawdown_pct"),
            }
            st.dataframe(pd.DataFrame([audit_columns]), hide_index=True, use_container_width=True)
            st.caption(relative_audit.get("reason", ""))
        sector_audit = sector_relative_audit()
        if sector_audit:
            st.subheader("市場・業種相対モデル・完全OOS監査")
            st.warning("市場・業種相対モデルはコスト後の成績が基準未達のため不採用です。ランキングや売買計画には使用していません。")
            sector_columns = {
                "取引数": sector_audit.get("trades"),
                "OOS期間": sector_audit.get("oos_periods"),
                "コスト後市場超過リターン(%)": sector_audit.get("net_avg_market_excess_pct"),
                "コスト後PF": sector_audit.get("profit_factor_after_cost"),
                "最大DD(%)": sector_audit.get("monthly_max_drawdown_pct"),
            }
            st.dataframe(pd.DataFrame([sector_columns]), hide_index=True, use_container_width=True)
            st.caption(sector_audit.get("reason", ""))
        st.subheader("JPX空売り需給シグナル・OOSイベント検証")
        short_oos = short_supply_oos_summary()
        if short_oos.empty:
            st.info("JPX空売り残高の蓄積後に、翌営業日からの5日間で検証結果を表示します。")
        else:
            st.dataframe(short_oos, hide_index=True, use_container_width=True)
            st.caption("公表対象のみを検証しています。未公表をゼロとして扱わず、成績改善が明確になるまで総合スコア・売買判断には加えません。")
        st.subheader("事前固定テクニカルルール・完全OOS検証")
        fixed_oos = fixed_rules_oos_summary()
        if fixed_oos.empty:
            st.info("事前固定ルールの完全OOS結果がありません。")
        else:
            st.warning("表示中の固定ルールは採用基準未達です。ランキング・売買計画には使用していません。")
            st.dataframe(fixed_oos, hide_index=True, use_container_width=True)
        wf_summary = walk_forward_summary()
        if wf_summary.empty:
            st.warning("Walk-forward結果がありません。python walk_forward_v3.py を実行してください。")
        else:
            st.dataframe(wf_summary, hide_index=True, use_container_width=True)
            st.caption("AI選別と同日ベースラインを比較してください。AI選別が一貫してPF・DD・平均リターンで上回るまでは、買い候補は研究用途です。")
        st.subheader("AI確率・テクニカル条件別のOOS成績")
        conditions = condition_analysis()
        if conditions.empty:
            st.info("条件別成績はまだありません。analyze_rule_conditions.py 実行後に表示されます。")
        else:
            st.dataframe(conditions, hide_index=True, use_container_width=True)
            st.caption("件数30未満の組み合わせは偶然の可能性が高いため、実戦投入ゲートには使いません。")
        st.divider()
        st.subheader("個別銘柄の参考シミュレーション")
        threshold = st.slider("AI確率の条件", .50, .90, .60, .05)
        if st.button("この銘柄を検証", use_container_width=True):
            result, metrics = v3_backtest(model, model2, data, threshold)
            if not metrics:
                st.info("条件に一致するシグナルがありません。")
            else:
                st.dataframe(pd.DataFrame([metrics]), hide_index=True, use_container_width=True)
                result["年"] = result.date.dt.year
                st.dataframe(result.groupby("年")["5日後リターン"].agg(["count", "mean"]).rename(columns={"count": "件数", "mean": "平均リターン"}), use_container_width=True)
                st.dataframe(result.tail(100), hide_index=True, use_container_width=True)
        st.caption("V3学習時の未学習期間だけで、勝率・平均利益／損失・期待値・Profit Factor・最大DD・最大連敗を検証します。上段のWalk-forward検証を最終判断の基準にします。")
        st.divider()
        st.subheader("TOP10ランキングの5営業日後実績")
        if RANKING_PERFORMANCE_SUMMARY_PATH.exists():
            try:
                ranking_summary = pd.read_csv(RANKING_PERFORMANCE_SUMMARY_PATH, encoding="utf-8-sig")
                st.dataframe(ranking_summary, hide_index=True, use_container_width=True)
                if RANKING_PERFORMANCE_PATH.exists():
                    ranking_performance = pd.read_csv(RANKING_PERFORMANCE_PATH, encoding="utf-8-sig")
                    completed = ranking_performance[ranking_performance["判定"] == "確定"]
                    st.caption(f"確定済み {len(completed)} 件。直近5営業日未満のランキングは「保留」として次回の照合で確定します。")
                    if not completed.empty:
                        st.dataframe(completed.sort_values(["分析日", "順位"], ascending=[False, True]).head(100), hide_index=True, use_container_width=True)
            except Exception as exc:
                st.warning(f"ランキング実績を読み込めませんでした: {exc}")
        else:
            st.info("ランキング実績はまだありません。TOP10を保存後、python evaluate_ranking_history.py を実行すると5営業日後の結果がここに表示されます。")
        st.subheader("取引記録")
        if journal.empty:
            st.info("まだ記録はありません。AI・テクニカル画面で売買計画を記録できます。")
        else:
            performance, monthly_performance, stock_performance = journal_metrics(journal)
            if performance:
                st.caption("実際に決済した取引だけを集計しています。計画・保有中の取引は含みません。")
                j1, j2, j3, j4, j5 = st.columns(5)
                j1.metric("実現損益", f"¥{performance['実現損益']:,.0f}")
                j2.metric("勝率", f"{performance['勝率']:.1f}%")
                j3.metric("期待値", f"{performance['期待値']:+.2f}%")
                j4.metric("Profit Factor", f"{performance['Profit Factor']:.2f}" if performance["Profit Factor"] is not None else "-" )
                j5.metric("最大連敗", f"{performance['最大連敗']} 回")
                with st.expander("実運用成績の詳細"):
                    detail = pd.DataFrame([performance]).rename(columns={"平均利益": "平均利益(%)", "平均損失": "平均損失(%)", "期待値": "期待値(%)", "最大DD": "最大DD(%)"})
                    st.dataframe(detail, hide_index=True, use_container_width=True)
                    if not monthly_performance.empty:
                        st.dataframe(monthly_performance, hide_index=True, use_container_width=True)
                    if not stock_performance.empty:
                        st.dataframe(stock_performance, hide_index=True, use_container_width=True)
            st.dataframe(journal.sort_values("date", ascending=False), hide_index=True, use_container_width=True)
            planned_journal = journal[journal.status == "PLANNED"]
            if not planned_journal.empty:
                choices = {f"{idx}: {row['code']}（{row['action']}）": idx for idx, row in planned_journal.iterrows()}
                selected_plan = st.selectbox("保有開始する計画", list(choices))
                if st.button("ペーパートレードを開始", use_container_width=True):
                    try:
                        open_plan(int(choices[selected_plan]))
                        st.success("ペーパートレードの保有開始として記録しました。実資金の発注は行いません。")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
            open_journal = journal[journal.status == "OPEN"]
            if not open_journal.empty:
                if st.button("保有中ポジションを更新", use_container_width=True):
                    monitored = []
                    for idx, row in open_journal.iterrows():
                        try:
                            current = float(yf.Ticker(f"{str(row['code']).upper()}.T").history(period="5d").Close.iloc[-1])
                            monitored.append({"コード": row["code"], "現在値": round(current, 1), "損切り": row["stop"], "利確": row["target"], "状態": position_status(current, float(row["stop"]), float(row["target"]))})
                        except Exception:
                            monitored.append({"コード": row["code"], "現在値": "取得不可", "損切り": row["stop"], "利確": row["target"], "状態": "確認不可"})
                    st.dataframe(pd.DataFrame(monitored), hide_index=True, use_container_width=True)
                st.caption("決済を記録すると、保有数と連敗数が次回の資金管理へ反映されます。")
                choices = {f"{idx}: {row['code']}（建値 ¥{float(row['entry']):,.0f}）": idx for idx, row in open_journal.iterrows()}
                close_label = st.selectbox("決済する記録", list(choices))
                close_price = st.number_input("決済価格", min_value=1.0, value=float(open_journal.loc[choices[close_label], "entry"]), step=1.0)
                if st.button("ペーパートレードの決済を記録", use_container_width=True):
                    try:
                        close_plan(int(choices[close_label]), float(close_price))
                        st.success("ペーパートレードの決済として記録しました。")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))


if __name__ == "__main__":
    main()
