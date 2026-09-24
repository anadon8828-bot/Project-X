"""Shared fail-closed freshness and candidate explanations."""
import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=1)
def tokyo_calendar():
    import exchange_calendars as xcals
    return xcals.get_calendar('XTKS')


def expected_price_day(now=None):
    now = now or pd.Timestamp.now(tz='Asia/Tokyo')
    local = now.tz_convert('Asia/Tokyo')
    cal = tokyo_calendar()
    day = pd.Timestamp(local.date())
    session = cal.date_to_session(day,direction='previous')
    if session.date()==local.date() and now.tz_convert('UTC') < cal.session_open(session):
        session = cal.previous_session(session)
    return str(session.date())


def fresh_rows(rows, now=None):
    now = now or pd.Timestamp.now(tz="Asia/Tokyo")
    if not {"取得日時","株価基準日"}.issubset(rows.columns):
        return pd.DataFrame()
    fetched = pd.to_datetime(rows["取得日時"],utc=True,errors="coerce")
    age = now.tz_convert("UTC")-fetched
    return rows[(rows["株価基準日"].astype(str)==expected_price_day(now)) & age.between(pd.Timedelta(0),pd.Timedelta(minutes=30))].copy()


def valid_materials(snapshot, now=None):
    now = now or pd.Timestamp.now(tz="Asia/Tokyo")
    if snapshot.get('state') != 'COMPLETED':
        return []
    checked = pd.to_datetime(snapshot.get("checked"),utc=True,errors="coerce")
    if pd.isna(checked) or not pd.Timedelta(0) <= now.tz_convert("UTC")-checked <= pd.Timedelta(minutes=30):
        return []
    result = []
    for item in snapshot.get("records",[]):
        published = pd.to_datetime(item.get("published"),utc=True,errors="coerce")
        if pd.notna(published) and pd.Timedelta(0) <= now.tz_convert("UTC")-published <= pd.Timedelta(hours=72):
            result.append(item)
    return result


def assessment(row, materials):
    reasons = []
    if pd.notna(row.get("テクニカル一致数")) and float(row["テクニカル一致数"])>0:
        reasons.append(str(row.get("候補根拠") or f"テクニカル条件{int(row['テクニカル一致数'])}/4項目一致"))
    positives = [m for m in materials if m.get("positive") is True]
    reasons.extend(m["title"] for m in positives)
    cautions = ["売買ルール未検証のため注文推奨ではありません", "開示内容の業績影響・織り込み済みかを原文と価格で確認"]
    if any(m.get("category")=="注意材料" for m in materials):
        cautions.append("同じ銘柄に注意材料の開示あり")
    return {"候補理由":"／".join(reasons) or "プラス条件なし", "現在の判断":"買い検討候補・売買条件未確認" if reasons else "候補条件未達", "注意点":"／".join(cautions),"材料候補":bool(positives)}
