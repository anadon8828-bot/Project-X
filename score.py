def calculate_score(latest, probability):

    final_score = 0


    # AI評価（最大40点）

    final_score += probability * 40


    # トレンド（最大30点）

    if latest["Close"] > latest["MA25"]:

        final_score += 15


    if latest["MA25"] > latest["MA75"]:

        final_score += 15



    # MACD（15点）

    if latest["MACD"] > latest["Signal"]:

        final_score += 15



    # RSI（10点）

    if 40 <= latest["RSI"] <= 70:

        final_score += 10


    final_score = int(
        min(final_score,100)
    )


    return final_score