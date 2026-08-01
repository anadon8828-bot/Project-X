import yfinance as yf



def get_market_score():

    markets = {
        "日経平均": "^N225",
        "NASDAQ": "^IXIC",
        "S&P500": "^GSPC"
    }


    score = 50

    result = {}


    for name, code in markets.items():

        try:

            data = yf.Ticker(code).history(
                period="1mo"
            )


            ma5 = data["Close"].rolling(
                5
            ).mean().iloc[-1]


            close = data["Close"].iloc[-1]


            if close > ma5:

                result[name] = "🟢 強気"

                score += 10


            else:

                result[name] = "🔴 弱気"

                score -= 10


        except:

            result[name] = "判定不可"



    score = max(
        0,
        min(
            score,
            100
        )
    )


    return score, result