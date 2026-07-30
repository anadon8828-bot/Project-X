import yfinance as yf
import pandas as pd
import joblib


# 銘柄
code = "7203"


print("🤖 Project X AI予測開始")


# 株価取得

ticker = yf.Ticker(code + ".T")

data = ticker.history(period="1y")


# 指標計算

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


# 最新データ

latest = data.iloc[-1]


# AI入力

X = pd.DataFrame([[
    latest["MA25"],
    latest["MA75"],
    latest["RSI"],
    latest["MACD"],
    latest["Volume"]
]],
columns=[
    "MA25",
    "MA75",
    "RSI",
    "MACD",
    "Volume"
])


# AI読み込み

model = joblib.load(
    "model.pkl"
)


# 予測

result = model.predict(X)

probability = model.predict_proba(X)


up_probability = probability[0][1]


print("--------------------")

print(f"銘柄：{code}")

print(
    f"上昇確率：{up_probability*100:.2f}%"
)


if result[0] == 1:

    print("判定：🟢 BUY")

else:

    print("判定：🟡 WAIT")