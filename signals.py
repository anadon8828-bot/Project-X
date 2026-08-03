# =========================
# Project X 売買シグナル計算
# =========================


def calculate_rebound_score(data, latest):

    score = 0


    # RSI 売られすぎ判定
    if latest["RSI"] < 30:

        score += 40


    # 移動平均からの乖離
    if latest["Close"] < latest["MA25"] * 0.95:

        score += 30


    # MACD改善
    if latest["MACD"] > latest["Signal"]:

        score += 20


    # 出来高増加
    volume_average = (
        data["Volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )


    if latest["Volume"] > volume_average:

        score += 10


    return score



def create_signal(final_score):


    if final_score >= 80:

        return "🔴 BUY"


    elif final_score >= 60:

        return "🟡 HOLD"


    elif final_score >= 40:

        return "⚪ WAIT"


    else:

        return "🟢 SELL"


def create_decision(final_score):


    if final_score >= 80:

        return "🔴 BUY"


    elif final_score >= 60:

        return "🟡 HOLD"


    elif final_score >= 40:

        return "⚪ WAIT"


    else:

        return "🟢 SELL"