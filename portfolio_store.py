"""Persistent, validated cash-account portfolio storage for Project X."""

from pathlib import Path
import re

import pandas as pd


PORTFOLIO_PATH = Path(__file__).resolve().parent / "project_x_portfolio.csv"
COLUMNS = ["コード", "銘柄名", "取引区分", "信用期限", "株数", "取得単価", "損切り価格"]
TRADE_TYPES = ("現物", "信用買い", "信用売り")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _code(value: object) -> str:
    result = str(value).strip().upper()
    if result.endswith(".0"):
        result = result[:-2]
    if not re.fullmatch(r"(?:\d{4}|\d{3}[A-Z])", result):
        raise ValueError("コードは4桁数字、または485Aのような英数字4文字で入力してください。")
    return result


def load_portfolio() -> pd.DataFrame:
    """Load the saved portfolio. A missing file is an empty portfolio, not an error."""
    if not PORTFOLIO_PATH.exists():
        return _empty()
    try:
        data = pd.read_csv(PORTFOLIO_PATH, encoding="utf-8-sig", dtype={"コード": "string"})
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"ポートフォリオを読み込めませんでした: {exc}") from exc
    for column in COLUMNS:
        if column not in data.columns:
            data[column] = "現物" if column == "取引区分" else (0 if column not in ("銘柄名", "信用期限") else "")
    return data[COLUMNS].fillna({"銘柄名": "", "取引区分": "現物", "信用期限": "", "損切り価格": 0})


def save_portfolio(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and save holdings. Blank editor rows are ignored deliberately."""
    if not set(COLUMNS).issubset(data.columns):
        raise ValueError("ポートフォリオの必要な列が不足しています。")
    cleaned = data[COLUMNS].copy()
    cleaned["コード"] = cleaned["コード"].fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned["コード"] != ""].copy()
    if cleaned.empty:
        _empty().to_csv(PORTFOLIO_PATH, index=False, encoding="utf-8-sig")
        return _empty()
    cleaned["コード"] = cleaned["コード"].map(_code)
    cleaned["銘柄名"] = cleaned["銘柄名"].fillna("").astype(str).str.strip()
    cleaned["取引区分"] = cleaned["取引区分"].fillna("現物").astype(str).str.strip()
    if not cleaned["取引区分"].isin(TRADE_TYPES).all():
        raise ValueError("取引区分は「現物」「信用買い」「信用売り」から選択してください。")
    expiry = pd.to_datetime(cleaned["信用期限"], errors="coerce")
    credit = cleaned["取引区分"] != "現物"
    if expiry[credit].isna().any():
        raise ValueError("信用取引は信用期限の登録が必須です。")
    cleaned["信用期限"] = expiry.dt.strftime("%Y-%m-%d").fillna("")
    for column in ("株数", "取得単価", "損切り価格"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    if cleaned[["株数", "取得単価"]].isna().any().any() or (cleaned["株数"] <= 0).any() or (cleaned["取得単価"] <= 0).any():
        raise ValueError("株数と取得単価には0より大きい数値を入力してください。")
    if (cleaned["損切り価格"].fillna(0) < 0).any():
        raise ValueError("損切り価格は0以上で入力してください。未設定の場合は0にしてください。")
    # The same code may be saved as multiple lots.  This preserves separate
    # entry prices, credit expiries and stop levels for additional purchases.
    cleaned["株数"] = cleaned["株数"].astype(int)
    cleaned[["取得単価", "損切り価格"]] = cleaned[["取得単価", "損切り価格"]].fillna(0).round(2)
    cleaned.to_csv(PORTFOLIO_PATH, index=False, encoding="utf-8-sig")
    return cleaned.reset_index(drop=True)


def add_holding(holding: pd.DataFrame) -> pd.DataFrame:
    """Append one or more holdings without replacing the saved portfolio."""
    return save_portfolio(pd.concat([load_portfolio(), holding], ignore_index=True))
