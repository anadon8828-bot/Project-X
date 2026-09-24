"""Persistent, local-only trade-plan journal for Project X."""

from pathlib import Path

import pandas as pd


FILE = Path(__file__).resolve().parent / "project_x_trade_journal.csv"
COLUMNS = ["date", "mode", "code", "action", "status", "entry", "stop", "target", "shares", "max_loss_yen", "reason", "exit_price", "realized_return"]


def load_journal() -> pd.DataFrame:
    if not FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(FILE)
    return frame.reindex(columns=COLUMNS)


def journal_state(frame: pd.DataFrame) -> tuple[int, int]:
    if frame.empty:
        return 0, 0
    open_positions = int((frame.status == "OPEN").sum())
    closed = frame[frame.status == "CLOSED"].copy()
    streak = 0
    for value in pd.to_numeric(closed.realized_return, errors="coerce").dropna().iloc[::-1]:
        if value >= 0: break
        streak += 1
    return open_positions, streak


def journal_metrics(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Calculate performance from actual CLOSED trades only."""
    closed = frame[frame.status == "CLOSED"].copy()
    if closed.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    closed["realized_return"] = pd.to_numeric(closed["realized_return"], errors="coerce")
    closed["entry"] = pd.to_numeric(closed["entry"], errors="coerce")
    closed["exit_price"] = pd.to_numeric(closed["exit_price"], errors="coerce")
    closed["shares"] = pd.to_numeric(closed["shares"], errors="coerce")
    closed["date"] = pd.to_datetime(closed["date"], errors="coerce")
    closed = closed.dropna(subset=["realized_return", "entry", "exit_price", "shares", "date"]).sort_values("date")
    if closed.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    returns = closed["realized_return"]
    gains, losses = returns[returns > 0], returns[returns < 0]
    streak = current = 0
    for value in returns:
        if value < 0:
            current += 1
            streak = max(streak, current)
        else:
            current = 0
    equity = (1 + returns / 100).cumprod()
    drawdown = (equity / equity.cummax() - 1).min() * 100
    total_yen = ((closed["exit_price"] - closed["entry"]) * closed["shares"]).sum()
    metrics = {
        "取引回数": int(len(closed)),
        "勝率": float((returns > 0).mean() * 100),
        "平均利益": float(gains.mean()) if not gains.empty else 0.0,
        "平均損失": float(losses.mean()) if not losses.empty else 0.0,
        "期待値": float(returns.mean()),
        "Profit Factor": float(gains.sum() / abs(losses.sum())) if not losses.empty and losses.sum() else None,
        "最大連敗": int(streak),
        "最大DD": float(drawdown),
        "実現損益": float(total_yen),
    }
    closed["月"] = closed["date"].dt.strftime("%Y-%m")
    monthly = closed.groupby("月").agg(取引回数=("realized_return", "size"), 勝率=("realized_return", lambda values: round((values > 0).mean() * 100, 1)), 平均リターン=("realized_return", "mean"), 実現損益=("realized_return", lambda values: 0.0)).reset_index()
    monthly_yen = closed.assign(実現損益円=(closed["exit_price"] - closed["entry"]) * closed["shares"]).groupby("月")["実現損益円"].sum()
    monthly["実現損益"] = monthly["月"].map(monthly_yen).round(0)
    stock = closed.groupby("code").agg(取引回数=("realized_return", "size"), 勝率=("realized_return", lambda values: round((values > 0).mean() * 100, 1)), 平均リターン=("realized_return", "mean"), 実現損益=("realized_return", lambda values: 0.0)).reset_index().rename(columns={"code": "コード"})
    stock_yen = closed.assign(実現損益円=(closed["exit_price"] - closed["entry"]) * closed["shares"]).groupby("code")["実現損益円"].sum()
    stock["実現損益"] = stock["コード"].map(stock_yen).round(0)
    return metrics, monthly, stock


def add_plan(code: str, plan, mode: str = "PAPER") -> None:
    if mode not in {"PAPER", "LIVE_RESEARCH"}:
        raise ValueError("記録モードが不正です。")
    frame = load_journal()
    row = pd.DataFrame([{
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "code": code,
        "mode": mode,
        "action": plan.action, "status": "PLANNED",
        "entry": round(plan.entry, 2), "stop": round(plan.stop, 2), "target": round(plan.target, 2),
        "shares": plan.suggested_shares, "max_loss_yen": round(plan.maximum_loss_yen, 0),
        "reason": plan.reason, "exit_price": "", "realized_return": "",
    }])
    pd.concat([frame, row], ignore_index=True).to_csv(FILE, index=False, encoding="utf-8-sig")


def open_plan(row_index: int) -> None:
    frame = load_journal()
    if row_index not in frame.index:
        raise ValueError("取引記録が見つかりません。")
    if frame.loc[row_index, "status"] != "PLANNED":
        raise ValueError("この記録は開始できる状態ではありません。")
    if frame.loc[row_index, "action"] != "買い候補":
        raise ValueError("見送り・監視の計画は保有開始できません。")
    frame.loc[row_index, "status"] = "OPEN"
    frame.to_csv(FILE, index=False, encoding="utf-8-sig")


def close_plan(row_index: int, exit_price: float) -> None:
    frame = load_journal()
    if row_index not in frame.index:
        raise ValueError("取引記録が見つかりません。")
    if frame.loc[row_index, "status"] != "OPEN":
        raise ValueError("この記録はすでに決済済みです。")
    entry = float(frame.loc[row_index, "entry"])
    if exit_price <= 0 or entry <= 0:
        raise ValueError("価格は0より大きい数値にしてください。")
    frame.loc[row_index, "status"] = "CLOSED"
    frame.loc[row_index, "exit_price"] = round(exit_price, 2)
    frame.loc[row_index, "realized_return"] = round((exit_price / entry - 1) * 100, 4)
    frame.to_csv(FILE, index=False, encoding="utf-8-sig")


def position_status(current_price: float, stop: float, target: float) -> str:
    if current_price <= stop:
        return "損切りライン到達"
    if current_price >= target:
        return "利確ライン到達"
    return "保有継続"
