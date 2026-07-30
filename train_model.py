import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


print("🚀 Project X AI学習開始")


CODES = [
    "7203",  # トヨタ
    "6758",  # ソニーG
    "9984",  # ソフトバンクG
    "8306",  # 三菱UFJ
    "9432",  # NTT
    "7011",  # 三菱重工
    "6857",  # アドバンテスト
    "6146",  # ディスコ
    "8035",  # 東京エレクトロン
    "4063",  # 信越化学
    "6501",  # 日立
    "6098",  # リクルート
    "8058",  # 三菱商事
    "8001",  # 伊藤忠商事
    "8766",  # 東京海上
    "8411",  # みずほFG
    "4502",  # 武田薬品
    "5108",  # ブリヂストン
    "6367",  # ダイキン
    "9433"   # KDDI
]


all_data = []


print("学習銘柄数:", len(CODES))
print("対象:", CODES)


for code in CODES:

    print("取得中:", code)

    df = yf.download(
        code + ".T",
        period="5y",
        auto_adjust=True,
        progress=False
    )


    if df.empty:
        continue


    # yfinance新仕様対応
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)


    df["CODE"] = code

    all_data.append(df)



data = pd.concat(all_data)


print("合計取得件数:", len(data))


# =========================
# 特徴量作成
# =========================


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


gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)


avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()


rs = avg_gain / avg_loss


data["RSI"] = (
    100 -
    (100/(1+rs))
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


data["MACD"] = ema12 - ema26



# 翌日の株価上昇判定

data["Target"] = (
    data["Close"]
    .shift(-1)
    >
    data["Close"]
).astype(int)



data = data.dropna()



features = [
    "MA25",
    "MA75",
    "RSI",
    "MACD",
    "Volume"
]


X = data[features]

y = data["Target"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)



print("学習:",len(X_train))
print("テスト:",len(X_test))


# =========================
# AI学習
# =========================


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


model.fit(
    X_train,
    y_train
)



pred = model.predict(X_test)



accuracy = accuracy_score(
    y_test,
    pred
)



print("====================")
print(
    "AI予測精度:",
    round(accuracy*100,2),
    "%"
)


print("====================")



importance = pd.DataFrame({

    "feature":features,

    "importance":
    model.feature_importances_

})


print(importance)



joblib.dump(
    model,
    "model.pkl"
)


print("====================")
print("✅ model.pkl 保存完了")
print("🚀 Project X AI学習終了")