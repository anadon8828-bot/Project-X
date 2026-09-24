"""Small local settings store for Project X risk controls."""

import json
from pathlib import Path


PATH = Path(__file__).resolve().parent / "project_x_settings.json"
DEFAULTS = {"capital_yen": 1_000_000, "max_positions": 3, "risk_per_trade": 0.01}


def load_settings() -> dict:
    if not PATH.exists():
        return DEFAULTS.copy()
    try:
        saved = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULTS.copy()
    result = DEFAULTS.copy()
    result.update({key: saved[key] for key in DEFAULTS if key in saved})
    return result


def save_settings(capital_yen: float, max_positions: int, risk_per_trade: float) -> dict:
    if capital_yen < 100_000:
        raise ValueError("運用資金は10万円以上にしてください。")
    if not 1 <= max_positions <= 10:
        raise ValueError("最大保有数は1〜10で設定してください。")
    if risk_per_trade not in (.005, .01, .015, .02):
        raise ValueError("1取引の損失上限は0.5%、1.0%、1.5%、2.0%から選択してください。")
    result = {"capital_yen": int(capital_yen), "max_positions": int(max_positions), "risk_per_trade": float(risk_per_trade)}
    PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
