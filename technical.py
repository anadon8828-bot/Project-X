import pandas as pd


def calculate_indicators(data):

    df = data.copy()

    # 移動平均
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA25"] = df["Close"].rolling(25).mean()
    df["MA75"] = df["Close"].rolling(75).mean()

    # RSI
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ボリンジャーバンド
    std = df["Close"].rolling(20).std()

    df["BB_Upper"] = df["MA25"] + std * 2
    df["BB_Lower"] = df["MA25"] - std * 2

    # MA乖離率
    df["MA25_Diff"] = (
        (df["Close"] - df["MA25"])
        / df["MA25"]
        * 100
    )

    df["MA75_Diff"] = (
        (df["Close"] - df["MA75"])
        / df["MA75"]
        * 100
    )

    # 5日騰落率
    df["Return5"] = (
        df["Close"].pct_change(5)
        * 100
    )

    # 出来高比率
    volume_avg = df["Volume"].rolling(20).mean()

    df["Volume_Ratio"] = (
        df["Volume"]
        / volume_avg
    )

    # ATR
    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] - df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"] - df["Close"].shift()
    ).abs()

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    return df