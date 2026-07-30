import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import joblib


print("🤖 AI学習開始")


# データ読み込み

data = pd.read_csv(
    "training_data.csv"
)


# 学習に使う項目

features = [
    "MA25",
    "MA75",
    "RSI",
    "MACD",
    "Volume"
]


X = data[features]

y = data["Target"]


# 学習用とテスト用に分割

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)


# AIモデル作成

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 学習

model.fit(
    X_train,
    y_train
)


# 精度確認

prediction = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    prediction
)


print(
    f"AI精度：{accuracy*100:.2f}%"
)


# AI保存

joblib.dump(
    model,
    "model.pkl"
)


print("✅ AI保存完了")