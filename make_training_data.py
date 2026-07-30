import yfinance as yf
import pandas as pd


# 対象銘柄
code = "7203"

print("データ取得中...")


ticker = yf.Ticker(code + ".T")

data = ticker.history(period="3y")


# 移動平均

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


# RSI

delta = data["Close"].diff()

gain = delta.where(delta > 0, 0)

loss = -delta.where(delta < 0, 0)


avg_gain = gain.rolling(14).mean()

avg_loss = loss.rolling(14).mean()


rs = avg_gain / avg_loss


data["RSI"] = 100 - (100 / (1 + rs))


# MACD

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


data["MACD"] = ema12 - ema26


# 翌日の株価

data["Tomorrow"] = (
    data["Close"]
    .shift(-1)
)


# 翌日上昇なら1

data["Target"] = (
    data["Tomorrow"] > data["Close"]
).astype(int)


# 不要な空白削除

data = data.dropna()


# 保存

data.to_csv(
    "training_data.csv"
)


print("完成！")
print(data.tail())