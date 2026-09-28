"""Fresh, dated price-only research scan. Never reuses old candidate rows."""
import json
import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from verified_materials import collect_materials
from research_rules import expected_price_day
from tse_universe import load_universe

ROOT = Path(__file__).resolve().parent


def acquire_lock(path, stale_after=pd.Timedelta(minutes=5)):
    """Create a portable process lock that works on Windows and Render Linux."""
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        except OSError:
            return None
        if age <= stale_after:
            return None
        try:
            path.unlink()
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, OSError):
            return None
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    return descriptor


def release_lock(path, descriptor):
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def write_candidates(rows):
    path = ROOT / "watchlist_fresh_candidates.csv"
    temp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def prioritize_universe(universe):
    """Scan previously liquid names first, while still completing all TSE names."""
    path = ROOT / "tse_all_scan_candidates.csv"
    if not path.exists():
        return universe
    try:
        prior = pd.read_csv(path, encoding="utf-8-sig", dtype={"コード": str})
        if "コード" not in prior:
            return universe
        turnover = pd.to_numeric(prior.get("平均売買代金(百万円)"), errors="coerce")
        priority = prior.assign(_turnover=turnover).sort_values("_turnover", ascending=False)["コード"].dropna().astype(str)
        order = {code: position for position, code in enumerate(priority)}
        ordered = universe.copy()
        ordered["_priority"] = ordered["コード"].astype(str).map(order).fillna(len(order) + 1)
        return ordered.sort_values(["_priority", "コード"]).drop(columns="_priority").reset_index(drop=True)
    except (OSError, ValueError, pd.errors.ParserError):
        return universe


def status(**values):
    values["heartbeat"] = pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
    path = ROOT / "watchlist_update_status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def main():
    lock = ROOT / "watchlist_update.lock"
    descriptor = acquire_lock(lock)
    if descriptor is None:
        return
    started = pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
    rows, failed, excluded = [], 0, 0
    try:
        status(state="RUNNING", started=started, processed=0)
        universe = prioritize_universe(load_universe(refresh=True))
        materials = {"state":"RUNNING","checked":pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),"records":[]}
        news_path = ROOT / "verified_materials.json"
        news_temp = news_path.with_suffix(".tmp")
        news_temp.write_text(json.dumps(materials,ensure_ascii=False),encoding="utf-8")
        os.replace(news_temp,news_path)
        for offset in range(0, len(universe), 60):
            batch = universe.iloc[offset:offset+60]
            tickers = [f"{c}.T" for c in batch["コード"]]
            raw = yf.download(tickers, period="6mo", auto_adjust=True, group_by="ticker", threads=8, progress=False, timeout=15)
            for _, stock in batch.iterrows():
                code = str(stock["コード"])
                try:
                    frame = raw[f"{code}.T"].dropna(subset=["Close", "Volume"])
                    now = pd.Timestamp.now(tz="Asia/Tokyo")
                    if len(frame) < 76 or str(frame.index[-1].date()) != expected_price_day(now):
                        failed += 1
                        continue
                    close, volume = frame.Close, frame.Volume
                    ma25, ma75 = close.rolling(25).mean(), close.rolling(75).mean()
                    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
                    value = float((close * volume).tail(20).mean())
                    if value < 10_000_000:
                        excluded += 1
                        continue
                    signals = [close.iloc[-1] > ma25.iloc[-1], ma25.iloc[-1] > ma75.iloc[-1], macd.iloc[-1] > 0, volume.iloc[-1] > volume.iloc[-2]]
                    reasons = [label for flag,label in zip(signals,["株価が25日線より上","25日線が75日線より上","MACDがプラス","前日より出来高増加"]) if flag]
                    rows.append({**stock.to_dict(), "終値":float(close.iloc[-1]), "平均売買代金(百万円)":value/1e6, "テクニカル一致数":sum(signals), "候補根拠":"／".join(reasons), "株価基準日":str(frame.index[-1].date()), "取得日時":now.isoformat()})
                except (KeyError, ValueError, IndexError, TypeError):
                    failed += 1
            done = min(offset+60,len(universe))
            status(state="RUNNING",started=started,processed=done,total=len(universe),eligible=len(rows),failed=failed,excluded=excluded)
            # Publish an honest partial result quickly; the UI clearly shows RUNNING.
            if rows and (done <= 180 or done % 300 == 0):
                write_candidates(rows)
            print(f"取得 {done}/{len(universe)} 評価可能 {len(rows)} 取得不可・古い足 {failed}",flush=True)
        if not rows:
            raise RuntimeError("当日付の価格を確認できた候補がありません。休場日・取得障害も考えられます。")
        write_candidates(rows)
        try:
            records = collect_materials(set(universe["コード"].astype(str)))
            materials = {"state":"COMPLETED","checked":pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),"records":records}
        except Exception as exc:
            materials = {"state":"FAILED","checked":pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),"records":[],"error":str(exc)}
        news_temp.write_text(json.dumps(materials,ensure_ascii=False),encoding="utf-8")
        os.replace(news_temp,news_path)
        from candidate_watchlist import select_watchlist
        from candidate_history import record, settle
        from research_rules import fresh_rows, valid_materials
        record(ROOT,select_watchlist(fresh_rows(pd.DataFrame(rows)),materials=valid_materials(materials)))
        settle(ROOT)
        status(state="COMPLETED",started=started,finished=pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),processed=len(universe),total=len(universe),eligible=len(rows),failed=failed,excluded=excluded)
    except Exception as exc:
        status(state="FAILED",started=started,error=str(exc))
        raise
    finally:
        release_lock(lock, descriptor)


if __name__ == "__main__":
    main()
