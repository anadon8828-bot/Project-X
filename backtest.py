import yfinance as yf
import pandas as pd


def run_backtest(
    model,
    code,
    period="3y",
    future_days=5
):

    data = yf.Ticker(
        code + ".T"
    ).history(
        period=period
    )


    if len(data) < 100:

        return pd.DataFrame()


    # テクニカル作成

    data["MA25"] = (
        data["Close"]
        .rolling(25)
        .mean()
    )


    data["MA75"] = (
        data["Close"]
        .rolling(75)
        .mean()
    )


    delta = data["Close"].diff()


    gain = delta.clip(
        lower=0
    )


    loss = -delta.clip(
        upper=0
    )


    rs = (
        gain.rolling(14).mean()
        /
        loss.rolling(14).mean()
    )


    data["RSI"] = (
        100 -
        100/(1+rs)
    )


    ema12 = (
        data["Close"]
        .ewm(span=12)
        .mean()
    )


    ema26 = (
        data["Close"]
        .ewm(span=26)
        .mean()
    )


    data["MACD"] = (
        ema12 - ema26
    )


    results = []


    for i in range(
        75,
        len(data)-future_days
    ):


        row = data.iloc[i]


        features = pd.DataFrame(
            [[
                row["MA25"],
                row["MA75"],
                row["RSI"],
                row["MACD"],
                row["Volume"]
            ]],
            columns=[
                "MA25",
                "MA75",
                "RSI",
                "MACD",
                "Volume"
            ]
        )


        probability = (
            model.predict_proba(
                features
            )[0][1]
        )


        future_price = (
            data["Close"]
            .iloc[i+future_days]
        )


        change = (
            future_price
            -
            row["Close"]
        ) / row["Close"] * 100


        results.append(
            {
                "日付":
                    data.index[i],

                "AI確率":
                    round(
                        probability*100,
                        1
                    ),

                "5日後変化率":
                    round(
                        change,
                        2
                    ),

                "勝敗":
                    "WIN"
                    if change > 0
                    else
                    "LOSE"
            }
        )


    return pd.DataFrame(results)
