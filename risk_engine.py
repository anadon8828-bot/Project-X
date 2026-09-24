"""Conservative trade and position-sizing rules for Project X.

This module produces a research plan, not an order.  A plan is eligible only
when AI, expected return, technical confirmation and portfolio risk agree.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TradePlan:
    action: str
    reason: str
    entry: float
    stop: float
    target: float
    risk_per_share: float
    reward_per_share: float
    risk_reward: float
    suggested_shares: int
    maximum_loss_yen: float


def make_trade_plan(
    price: float,
    atr: float,
    ai_probability: float,
    probability_2pct: float,
    expected_return: float | None,
    technical_score: int,
    capital_yen: float = 1_000_000,
    current_positions: int = 0,
    recent_loss_streak: int = 0,
    max_positions: int = 3,
    risk_per_trade: float = .01,
) -> TradePlan:
    """Create a cash-only, capped-risk trade plan.

    A losing streak halves risk after two losses. No margin/leverage is used.
    """
    stop_distance = max(atr * 1.5, price * .03)
    expected = expected_return if expected_return is not None else 0.0
    target_distance = max(price * max(expected, .02), stop_distance * 2)
    stop, target = price - stop_distance, price + target_distance
    rr = target_distance / stop_distance
    allowed_risk = capital_yen * risk_per_trade * (.5 if recent_loss_streak >= 2 else 1)
    shares_by_risk = int(allowed_risk // stop_distance)
    shares_by_capital = int((capital_yen / max_positions) // price)
    shares = max(0, min(shares_by_risk, shares_by_capital))

    if current_positions >= max_positions:
        action, reason = "見送り", f"最大保有数（{max_positions}）に達しています"
    elif ai_probability < .60 or probability_2pct < .55:
        action, reason = "見送り", "AI確率が最低条件を満たしません"
    elif expected_return is not None and expected_return <= 0:
        action, reason = "見送り", "予測期待リターンが正ではありません"
    elif technical_score < 2:
        action, reason = "見送り", "テクニカル確認が不足しています"
    elif rr < 2 or shares == 0:
        action, reason = "見送り", "リスクリワードまたは許容損失の条件を満たしません"
    else:
        action, reason = "買い候補", "AI・期待値・テクニカル・資金管理の条件を満たしました"
    return TradePlan(action, reason, price, stop, target, stop_distance, target_distance, rr, shares, shares * stop_distance)
