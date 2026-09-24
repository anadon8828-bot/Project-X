"""Official JPX universe loader for Project X all-TSE scanning."""

from pathlib import Path
import re

import pandas as pd


JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
CACHE_PATH = Path(__file__).resolve().parent / "tse_domestic_common_stocks.csv"
MARKETS = ("プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）")


def normalize_code(value: object) -> str:
    code = str(value).strip().upper()
    if code.endswith(".0"):
        code = code[:-2]
    return code


def refresh_universe() -> pd.DataFrame:
    """Download JPX's current listing master and keep domestic common stocks only."""
    source = pd.read_excel(JPX_LIST_URL, dtype={"コード": "string"})
    required = {"コード", "銘柄名", "市場・商品区分", "33業種区分", "規模区分"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"JPX一覧の必要な列が見つかりません: {sorted(missing)}")
    data = source[source["市場・商品区分"].isin(MARKETS)].copy()
    data["コード"] = data["コード"].map(normalize_code)
    data = data[data["コード"].str.fullmatch(r"(?:\d{4}|\d{3}[A-Z])", na=False)].copy()
    data = data.rename(columns={"市場・商品区分": "市場", "33業種区分": "業種", "規模区分": "規模"})
    data = data[["コード", "銘柄名", "市場", "業種", "規模"]].drop_duplicates("コード").sort_values("コード")
    if len(data) < 1000:
        raise ValueError("取得した東証普通株の件数が異常に少ないため保存しません。")
    data.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
    return data.reset_index(drop=True)


def load_universe(refresh: bool = False) -> pd.DataFrame:
    if refresh or not CACHE_PATH.exists():
        return refresh_universe()
    data = pd.read_csv(CACHE_PATH, encoding="utf-8-sig", dtype={"コード": "string"})
    return data[["コード", "銘柄名", "市場", "業種", "規模"]]
